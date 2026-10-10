"""Train the 30-minute rainfall-exceedance model on Dataproc (batch job).

Reads the threshold-expanded feature table built by the notebook (Part 2.2),
trains one LightGBM classifier that takes the threshold as a feature, and writes
the fitted pipeline plus a per-threshold metrics report.

This is a self-contained PySpark script that runs ON the cluster. It does not
import config.py (that file isn't shipped to the cluster) — pass paths via flags.

LightGBM comes from SynapseML, loaded as a job package. Match the Scala suffix to
the cluster's Spark/Scala build (2.12 for a Spark 3.5 image), and use the SynapseML
build documented for that Spark version (1.1.3 for Spark 3.5). Example submit:

    gcloud dataproc jobs submit pyspark scripts/train.py \
        --project=$PROJECT --region=$REGION --cluster=$CLUSTER \
        --properties=^#^spark.jars.packages=com.microsoft.azure:synapseml_2.12:1.1.3#spark.jars.repositories=https://mmlspark.blob.core.windows.net/maven \
        -- \
        --input gs://$BUCKET/data/train_table \
        --output gs://$BUCKET/models/run-001

(The ^#^ sets # as the delimiter so the two spark.jars.* properties can be passed
together without the commas inside them being split.)
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- Defaults; override with the matching CLI flag ---------------------------
BUCKET = "spark-rainfall-9486f2d0"
INPUT = f"gs://{BUCKET}/data/train_table"      # threshold-expanded feature table (Parquet)
OUTPUT = f"gs://{BUCKET}/models/run-001"        # model + metrics destination
TRAIN_MAX_YEAR = 2023                            # train on <= this year, test on the rest
NUM_ITERATIONS = 200
LEARNING_RATE = 0.05
NUM_LEAVES = 63
MONOTONE_THRESHOLD = True   # enforce P(exceed) non-increasing in the threshold feature
# -----------------------------------------------------------------------------

# Must match the notebook's MODEL_FEATURES (FEATURES + "threshold").
NUMERIC_FEATURES = [
    "lat", "lon",
    "rain_t", "rain_t5", "rain_t10", "rain_t15", "rain_t20",
    "isl_frac_wet", "isl_mean", "isl_max", "isl_std", "isl_n_stations",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "threshold",
]
CATEGORICAL = "station_id"
THRESHOLDS = [0.2, 1.0, 2.0, 5.0, 10.0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=INPUT)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--train-max-year", type=int, default=TRAIN_MAX_YEAR)
    p.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="Fraction of TRAIN rows to sample (for a quick local smoke test).")
    args = p.parse_args()

    spark = SparkSession.builder.appName("rainfall-30min-exceedance").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    from datetime import datetime, timedelta

    from pyspark import StorageLevel
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from synapse.ml.lightgbm import LightGBMClassifier

    df = spark.read.parquet(args.input)

    # Validate the schema up front — fail loudly rather than deep inside .fit().
    required = set(NUMERIC_FEATURES + [CATEGORICAL, "timestamp", "year", "label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")

    # Chronological split — never random: the five threshold rows of one observation,
    # and the overlapping future windows of adjacent timestamps, must not straddle it.
    # A row's label depends on rainfall up to t+30min, so the last training row at the
    # year boundary would otherwise peek into the test period. Hold out a 30-minute gap:
    # train on t < (cutoff − 30min), test on t >= cutoff. (A single held-out year is a
    # coarse-but-ample test — millions of rows, all seasons. If Dataproc cost allows,
    # rolling-origin CV over several years would give a cross-year variance estimate.)
    cutoff = datetime(args.train_max_year + 1, 1, 1)
    train_end = cutoff - timedelta(minutes=30)
    train = df.filter(F.col("timestamp") < F.lit(train_end))
    test = df.filter(F.col("timestamp") >= F.lit(cutoff))

    if args.sample_frac < 1.0:
        # Sample whole station-timestamp observations so all five thresholds stay together.
        keys = (train.select("station_id", "timestamp").distinct()
                .sample(False, args.sample_frac, seed=42))
        train = train.join(keys, ["station_id", "timestamp"], "left_semi")

    n_train, n_test = train.count(), test.count()
    print(f"train rows: {n_train:,}   test rows: {n_test:,}")
    if n_train == 0 or n_test == 0:
        raise ValueError("train or test split is empty — check --train-max-year and the data range")

    # station_id -> numeric index, declared categorical to LightGBM by slot INDEX (0),
    # which doesn't depend on the assembler preserving the column name in metadata.
    # handleInvalid="keep" buckets unseen stations rather than erroring (they then lean
    # on lat/lon and the other features).
    indexer = StringIndexer(inputCol=CATEGORICAL, outputCol="station_idx", handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=["station_idx"] + NUMERIC_FEATURES,
        outputCol="features",
        handleInvalid="keep",   # pass NaN lags/isl_std through to LightGBM's native missing handling
    )

    lgbm_kwargs = dict(
        featuresCol="features",
        labelCol="label",
        objective="binary",
        numIterations=args.num_iterations,
        learningRate=LEARNING_RATE,
        numLeaves=NUM_LEAVES,
        categoricalSlotIndexes=[0],         # station_idx is slot 0 of the assembled vector
        dataTransferMode="streaming",       # lower memory for the Spark->native handoff
    )
    if MONOTONE_THRESHOLD:
        # One entry per assembled feature; -1 on `threshold` forces P(exceed) to be
        # non-increasing as the threshold rises (P(>2) <= P(>1)), the rest unconstrained.
        cols = ["station_idx"] + NUMERIC_FEATURES
        lgbm_kwargs["monotoneConstraints"] = [-1 if c == "threshold" else 0 for c in cols]

    lgbm = LightGBMClassifier(**lgbm_kwargs)
    model = Pipeline(stages=[indexer, assembler, lgbm]).fit(train)

    pred = (model.transform(test).select("threshold", "label", "probability", "prediction")
            .persist(StorageLevel.MEMORY_AND_DISK))

    # Per-threshold confusion counts in ONE aggregation (the class balance differs a lot
    # across thresholds, so an aggregate number would mislead). PR-AUC/ROC-AUC still need
    # a per-subset evaluator pass. ROC/PR-AUC are undefined when a subset is single-class.
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability",
                                            metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability",
                                             metricName="areaUnderROC")
    conf = (
        pred.groupBy("threshold").agg(
            F.count("*").alias("n_test"),
            F.sum((F.col("label") == 1).cast("long")).alias("positives"),
            F.sum(((F.col("prediction") == 1) & (F.col("label") == 1)).cast("long")).alias("tp"),
            F.sum(((F.col("prediction") == 1) & (F.col("label") == 0)).cast("long")).alias("fp"),
            F.sum(((F.col("prediction") == 0) & (F.col("label") == 1)).cast("long")).alias("fn"),
        ).collect()
    )

    report = {"train_max_year": args.train_max_year, "num_iterations": args.num_iterations,
              "n_train": n_train, "n_test": n_test, "per_threshold": []}
    for row in sorted(conf, key=lambda r: r["threshold"]):
        x = float(row["threshold"])
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        positives, negatives = row["positives"], row["n_test"] - row["positives"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        both_classes = positives > 0 and negatives > 0  # AUC undefined otherwise
        sub = pred.filter(F.col("threshold") == x) if both_classes else None
        report["per_threshold"].append({
            "threshold": x,
            "n_test": row["n_test"],
            "positives": positives,
            "pr_auc": pr_eval.evaluate(sub) if both_classes else None,
            "roc_auc": roc_eval.evaluate(sub) if both_classes else None,
            "f1": f1, "precision": precision, "recall": recall,
        })

    config = {
        "model_features": ["station_idx"] + NUMERIC_FEATURES,
        "categorical_slot": 0,
        "thresholds": THRESHOLDS,
        "monotone_threshold": MONOTONE_THRESHOLD,
        "synapseml": "see submit --properties (synapseml_2.12:1.1.3)",
    }
    print(json.dumps({"report": report, "config": config}, indent=2))
    model.write().overwrite().save(args.output + "/model")
    (spark.createDataFrame([(json.dumps({"report": report, "config": config}),)], ["report"])
     .coalesce(1).write.mode("overwrite").text(args.output + "/metrics"))
    pred.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
