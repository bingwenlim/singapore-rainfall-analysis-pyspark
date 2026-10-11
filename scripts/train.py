"""Train the 30-minute rainfall-exceedance model (Spark MLlib GBTClassifier).

Reads the threshold-expanded feature table built by the notebook (Part 2.2),
trains one gradient-boosted-tree classifier that takes the threshold as a
feature, and writes the fitted pipeline plus a per-threshold metrics report.

Self-contained PySpark script — pass paths via flags. Runs on plain Spark MLlib
(no third-party packages), so it works on local Spark or Dataproc unchanged.

Why GBT and not LightGBM: SynapseML's LightGBM native library segfaults on this
environment (and won't load on Apple-Silicon locally at all). MLlib GBT is pure
JVM — no native lib, no worker socket mesh — so it runs anywhere Spark runs, at
the cost of two features LightGBM had (handled below):
  - No native NaN handling   -> we fill missing features with a -1 sentinel.
  - No monotone constraints   -> we enforce P(>X) non-increasing in X *post-hoc*
                                 at prediction time instead of during training.

Runs the same either way — --input/--output take local paths or gs:// URIs.

Local (what we've used; sample the train set to keep it quick):
    PYSPARK_SUBMIT_ARGS="--driver-memory 12g pyspark-shell" \
    uv run python scripts/train.py \
        --input data/train_table --output models/run-local \
        --sample-frac 0.1

    (Needs a JDK 11: either installed system-wide with JAVA_HOME set, or extracted
    project-locally to .runtime/java-home — this script auto-detects the latter.)

Cloud (Dataproc, full data — no extra packages/network setup; GBT is built-in):
    gcloud dataproc jobs submit pyspark scripts/train.py \
        --project=$PROJECT --region=$REGION --cluster=$CLUSTER \
        -- \
        --input gs://$BUCKET/data/train_table \
        --output gs://$BUCKET/models/run-001
"""

import argparse
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- Model hyperparameters (tuned; see scripts/tune.py) ----------------------
# --input / --output are REQUIRED and location-agnostic: pass local paths to run
# locally (e.g. data/train_table) or gs:// paths to run on Dataproc. Spark reads
# both identically, and nothing else in this script is cloud-specific.
TRAIN_MAX_YEAR = 2023                            # train on <= this year, test on the rest
NUM_ITERATIONS = 60                              # GBT boosting iterations (maxIter); tuned
MAX_DEPTH = 5                                    # tuned (depth 3 underfits, 7 overfits)
STEP_SIZE = 0.1                                  # GBT learning rate; tuned (0.05/0.1/0.2 tied)
MAX_BINS = 128                                   # must be >= station cardinality (67) + buffer
MISSING_SENTINEL = -1.0                          # fill value for missing features (rain >= 0,
                                                 # so -1 is an unambiguous "was missing" marker)
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
    p.add_argument("--input", required=True,
                   help="Parquet feature table (local path or gs:// URI)")
    p.add_argument("--output", required=True,
                   help="where to write model + metrics (local path or gs:// URI)")
    p.add_argument("--train-max-year", type=int, default=TRAIN_MAX_YEAR)
    p.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="Fraction of TRAIN rows to sample (for a quick local smoke test).")
    args = p.parse_args()

    # If no JAVA_HOME is set but the project-local JDK exists (.runtime/java-home,
    # created by extracting a Temurin 11 build there), use it — so the run command is
    # the same whether Java is installed system-wide or project-locally. On Dataproc
    # JAVA_HOME is already set, so this is a no-op there. Mirrors spark_project.session.
    if "JAVA_HOME" not in os.environ:
        bundled = Path(__file__).resolve().parents[1] / ".runtime" / "java-home"
        if (bundled / "bin/java").exists():
            os.environ["JAVA_HOME"] = str(bundled)

    spark = SparkSession.builder.appName("rainfall-30min-exceedance").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    # Pin the session timezone so timestamps render consistently (SGT locally, UTC on
    # Dataproc otherwise). Note this alone does NOT fix the year-boundary cutoff below:
    # a naive Python datetime passed to F.lit() is converted using the *driver process's*
    # local timezone before the session setting applies, so the cutoff must also be made
    # timezone-aware (see below). The two together make the split the same instant anywhere.
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
    # coarse-but-ample test — millions of rows, all seasons. Rolling-origin CV over
    # several years would give a cross-year variance estimate if desired.)
    # Timezone-aware cutoff: an aware datetime carries its own offset, so F.lit() converts
    # it to the same instant regardless of the driver process's local timezone (a naive
    # datetime would be read in the process tz — SGT locally, UTC on Dataproc — shifting the
    # boundary 8 hours between environments). Combined with the session tz above, both
    # environments split 2024 at the same Singapore calendar-year boundary.
    cutoff = datetime(args.train_max_year + 1, 1, 1, tzinfo=ZoneInfo("Asia/Singapore"))
    train_end = cutoff - timedelta(minutes=30)
    train = df.filter(F.col("timestamp") < F.lit(train_end))
    test = df.filter(F.col("timestamp") >= F.lit(cutoff))

    if args.sample_frac < 1.0:
        # Sample whole station-timestamp observations so all five thresholds stay together.
        keys = (train.select("station_id", "timestamp").distinct()
                .sample(False, args.sample_frac, seed=42))
        train = train.join(keys, ["station_id", "timestamp"], "left_semi")

    # Cheap emptiness guard (take(1) short-circuits instead of counting ~195M rows).
    if not train.take(1) or not test.take(1):
        raise ValueError("train or test split is empty — check --train-max-year and the data range")

    # GBT has no native NaN handling, so fill the missing features (gap-aware lags and
    # isl_std) with a -1 sentinel. Rainfall is always >= 0, so -1 stays an unambiguous
    # "was missing" marker, distinct from an observed zero — preserving the point of the
    # gap-aware design that filling with 0 would destroy.
    FILL = {c: MISSING_SENTINEL for c in ["rain_t5", "rain_t10", "rain_t15", "rain_t20", "isl_std"]}
    train = train.fillna(FILL)
    test = test.fillna(FILL)

    # station_id -> numeric index. Treated as an ordinary numeric feature here (not a
    # declared categorical): MLlib loses per-slot categorical metadata through the
    # VectorAssembler, and lat/lon already carry station geography. handleInvalid="keep"
    # buckets unseen stations instead of erroring.
    indexer = StringIndexer(inputCol=CATEGORICAL, outputCol="station_idx", handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=["station_idx"] + NUMERIC_FEATURES,
        outputCol="features",
        handleInvalid="keep",
    )
    # maxBins must be >= the station cardinality (~67); the default 32 would error.
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=args.num_iterations,
                        maxDepth=MAX_DEPTH, maxBins=MAX_BINS, stepSize=STEP_SIZE)

    print(">>> fitting GBT pipeline...", flush=True)
    model = Pipeline(stages=[indexer, assembler, gbt]).fit(train)
    print(">>> fit done; scoring test set...", flush=True)

    # Extract P(label=1) from the probability vector.
    prob1 = F.udf(lambda v: float(v[1]), "double")
    scored = model.transform(test).select(
        "station_id", "timestamp", "threshold", "label",
        prob1("probability").alias("p"),
    )

    # Post-hoc monotonicity: for one (station, timestamp), P(>X) must not increase with X.
    # GBT can't enforce this at train time, so take a running max of p over DESCENDING
    # threshold within each observation — i.e. P(>X) >= P(>any larger threshold). This
    # monotonized probability (p_mono) is the model's final probability output; the hard
    # prediction thresholds it at 0.5.
    w = (Window.partitionBy("station_id", "timestamp")
         .orderBy(F.desc("threshold"))
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    to_vec = F.udf(lambda p: Vectors.dense([1.0 - p, p]), VectorUDT())
    pred = (scored
            .withColumn("p_mono", F.max("p").over(w))
            .withColumn("prediction", (F.col("p_mono") >= 0.5).cast("int"))
            .withColumn("prob_vec", to_vec("p_mono"))
            .select("threshold", "label", "prediction", "prob_vec")
            .persist())

    # Confusion counts for all thresholds in ONE aggregation (class balance differs a lot
    # across thresholds, so a single aggregate would mislead).
    print(">>> computing per-threshold metrics...", flush=True)
    conf = {r["threshold"]: r for r in (
        pred.groupBy("threshold").agg(
            F.count("*").alias("n_test"),
            F.sum((F.col("label") == 1).cast("long")).alias("positives"),
            F.sum(((F.col("prediction") == 1) & (F.col("label") == 1)).cast("long")).alias("tp"),
            F.sum(((F.col("prediction") == 1) & (F.col("label") == 0)).cast("long")).alias("fp"),
            F.sum(((F.col("prediction") == 0) & (F.col("label") == 1)).cast("long")).alias("fn"),
        ).collect()
    )}

    # PR-AUC (primary, under heavy imbalance) and ROC-AUC per threshold, on the predicted
    # probability. Undefined when a threshold's test slice is single-class -> report None.
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prob_vec",
                                            metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prob_vec",
                                             metricName="areaUnderROC")

    report = {"train_max_year": args.train_max_year, "num_iterations": args.num_iterations,
              "per_threshold": []}
    for x in sorted(conf):
        row = conf[x]
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        positives, negatives = row["positives"], row["n_test"] - row["positives"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        both = positives > 0 and negatives > 0
        sub = pred.filter(F.col("threshold") == x) if both else None
        report["per_threshold"].append({
            "threshold": float(x),
            "n_test": row["n_test"],
            "positives": positives,
            "pr_auc": pr_eval.evaluate(sub) if both else None,
            "roc_auc": roc_eval.evaluate(sub) if both else None,
            "f1": f1, "precision": precision, "recall": recall,
        })
    pred.unpersist()

    config = {
        "model": "pyspark.ml GBTClassifier",
        "model_features": ["station_idx"] + NUMERIC_FEATURES,
        "missing_sentinel": MISSING_SENTINEL,
        "thresholds": THRESHOLDS,
        "monotonicity": "enforced post-hoc (running max of P over descending threshold)",
        "max_bins": MAX_BINS, "max_depth": MAX_DEPTH, "step_size": STEP_SIZE,
        "primary_metric": "pr_auc",
        "deferred": "probability calibration (Brier/reliability); full-data (unsampled) run",
    }
    print(json.dumps({"report": report, "config": config}, indent=2))
    print(">>> saving model + metrics to", args.output, flush=True)
    model.write().overwrite().save(args.output + "/model")
    (spark.createDataFrame([(json.dumps({"report": report, "config": config}),)], ["report"])
     .coalesce(1).write.mode("overwrite").text(args.output + "/metrics"))
    print(">>> done.", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
