"""Hyperparameter tuning for the GBT rainfall model — local, cheap, exploratory.

Separate from train.py (the clean final-run script) on purpose. Tunes on a 1%
sample, selecting by PR-AUC on a VALIDATION year carved from the training years
(fit <= tune_train_max_year, validate on the next year). 2024 (the real test
year) is never read here, so model selection can't leak into the final metrics.

Run:
    JAVA_HOME=$PWD/.runtime/java-home PYSPARK_SUBMIT_ARGS="--driver-memory 12g pyspark-shell" \
        uv run python scripts/tune.py --sample-frac 0.01
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT = "data/train_table"
TUNE_TRAIN_MAX_YEAR = 2022      # fit on <= this; validate on the next year (2023)
MAX_BINS = 128
MISSING_SENTINEL = -1.0

NUMERIC_FEATURES = [
    "lat", "lon",
    "rain_t", "rain_t5", "rain_t10", "rain_t15", "rain_t20",
    "isl_frac_wet", "isl_mean", "isl_max", "isl_std", "isl_n_stations",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "threshold",
]
CATEGORICAL = "station_id"
# Rank configs on the mid thresholds; 1% leaves too few positives at 10mm to trust.
RANK_THRESHOLDS = [0.2, 1.0, 2.0]

# Learning-rate zoom at the locked depth=5. stepSize and maxIter are tuned JOINTLY
# (a lower rate needs more trees to reach the same fit, so pairing them is the fair
# comparison). Explicit (maxDepth, maxIter, stepSize) configs, not a cross-product.
CONFIGS = [
    {"maxDepth": 5, "maxIter": 60,  "stepSize": 0.1},   # current best
    {"maxDepth": 5, "maxIter": 100, "stepSize": 0.1},   # more trees, same rate
    {"maxDepth": 5, "maxIter": 120, "stepSize": 0.05},  # half rate, ~double trees
    {"maxDepth": 5, "maxIter": 60,  "stepSize": 0.2},   # faster rate
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT)
    p.add_argument("--sample-frac", type=float, default=0.01)
    args = p.parse_args()

    spark = SparkSession.builder.appName("rainfall-gbt-tune").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.window import Window

    df = spark.read.parquet(args.input)

    # Split WITHIN the training years: fit < (val_year − 30min gap), validate on val_year.
    # 2024 is never referenced here.
    # Explicit SGT boundaries keep validation independent of the Python host timezone.
    val_start = datetime(TUNE_TRAIN_MAX_YEAR + 1, 1, 1, tzinfo=ZoneInfo("Asia/Singapore"))
    val_end = datetime(TUNE_TRAIN_MAX_YEAR + 2, 1, 1, tzinfo=ZoneInfo("Asia/Singapore"))
    fit_end = val_start - timedelta(minutes=30)
    tr = df.filter(F.col("timestamp") < F.lit(fit_end))
    va = df.filter((F.col("timestamp") >= F.lit(val_start)) & (F.col("timestamp") < F.lit(val_end)))

    # Sample whole station-timestamp observations (keep all thresholds together).
    keys = tr.select("station_id", "timestamp").distinct().sample(False, args.sample_frac, seed=42)
    tr = tr.join(keys, ["station_id", "timestamp"], "left_semi")

    FILL = {c: MISSING_SENTINEL for c in ["rain_t5", "rain_t10", "rain_t15", "rain_t20", "isl_std"]}
    tr, va = tr.fillna(FILL), va.fillna(FILL)
    tr = tr.cache()
    va = va.cache()
    print(f">>> tune train rows: {tr.count():,}   validate rows: {va.count():,}", flush=True)

    indexer = StringIndexer(inputCol=CATEGORICAL, outputCol="station_idx", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=["station_idx"] + NUMERIC_FEATURES,
                                outputCol="features", handleInvalid="keep")
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prob_vec",
                                            metricName="areaUnderPR")
    to_vec = F.udf(lambda pr: Vectors.dense([1.0 - pr, pr]), VectorUDT())
    prob1 = F.udf(lambda v: float(v[1]), "double")

    results = []
    for i, hp in enumerate(CONFIGS, 1):
        print(f">>> [{i}/{len(CONFIGS)}] fitting {hp} ...", flush=True)
        gbt = GBTClassifier(featuresCol="features", labelCol="label", maxBins=MAX_BINS, **hp)
        model = Pipeline(stages=[indexer, assembler, gbt]).fit(tr)
        scored = (model.transform(va)
                  .select("threshold", "label", prob1("probability").alias("p"))
                  .withColumn("prob_vec", to_vec("p")))
        # Mean validation PR-AUC over the rankable mid thresholds.
        prs = []
        for x in RANK_THRESHOLDS:
            sub = scored.filter(F.col("threshold") == x)
            prs.append(pr_eval.evaluate(sub))
        mean_pr = sum(prs) / len(prs)
        results.append({"params": hp, "pr_auc_by_threshold": dict(zip(RANK_THRESHOLDS, [round(v, 4) for v in prs])),
                        "mean_pr_auc": round(mean_pr, 4)})
        print(f"    mean PR-AUC (0.2/1/2mm) = {mean_pr:.4f}", flush=True)

    results.sort(key=lambda r: r["mean_pr_auc"], reverse=True)
    print("\n>>> RANKED RESULTS (best first):")
    print(json.dumps(results, indent=2))
    print("\n>>> BEST:", json.dumps(results[0]["params"]))
    spark.stop()


if __name__ == "__main__":
    main()
