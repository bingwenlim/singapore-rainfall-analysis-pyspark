"""Score the trained rainfall model.

Two modes:

1. Single prediction (the task's stated interface) --- give a station, a prediction
   time, and a threshold; the script derives that row's features from the cleaned
   rainfall data, runs the model, and prints the exceedance probability:

       uv run python scripts/predict.py single \
           --station S08 --time "2021-06-01 15:00:00" --threshold 5

2. Batch --- demonstrate scoring a sample from a pre-built feature table
   (e.g. data/train_table) and print probabilities; this does not compute metrics:

       uv run python scripts/predict.py batch --input data/train_table

In both modes the model's inputs are the engineered features (recent lags, island-wide
context at the prediction time, cyclical time, station location, and the threshold). The
features are *functions of* (station, time) applied to the rainfall data, so the data must
be available --- "rain 15 minutes ago" cannot be known without it. Single mode hides that
behind the three stated inputs by computing the features internally.

The saved model (models/run-001/model) is a Spark PipelineModel; the -1 missing-fill and
the monotonic-in-threshold adjustment live outside it and are applied here.
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODEL = "models/run-001/model"
PARQUET = "data/parquet"              # raw source, for single-mode feature derivation
MISSING = ["rain_t5", "rain_t10", "rain_t15", "rain_t20", "isl_std"]
SENTINEL = -1.0
THRESHOLDS = [0.2, 1.0, 2.0, 5.0, 10.0]


def start_spark():
    if "JAVA_HOME" not in os.environ:
        bundled = Path(__file__).resolve().parents[1] / ".runtime" / "java-home"
        if (bundled / "bin/java").exists():
            os.environ["JAVA_HOME"] = str(bundled)
    spark = SparkSession.builder.appName("rainfall-predict").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")  # match training
    return spark


def clean(df):
    """Same station cleaning + timestamp fix as the notebook/training."""
    q = df.groupBy("station_id").agg(
        F.count("*").alias("n"),
        ((F.max("timestamp").cast("long") - F.min("timestamp").cast("long")) / 300 + 1).alias("tn"),
        F.avg((F.col("reading_value") > 0).cast("int")).alias("wf"),
    ).withColumn("cp", F.col("n") / F.col("tn") * 100)
    keep = [r["station_id"] for r in
            q.filter((F.col("n") >= 1000) & (F.col("cp") >= 90) & (F.col("wf") > 0)).collect()]
    return (df.withColumn("timestamp", F.date_trunc("minute", F.col("timestamp") + F.expr("INTERVAL 1 SECOND")))
            .filter(F.col("station_id").isin(keep))
            .dropDuplicates(["station_id", "timestamp"]))


def build_features(dc):
    """Derive the model features for every row of a cleaned frame (mirrors notebook 2.2)."""
    import math
    base = dc.select("station_id", "timestamp",
                     F.col("reading_value").alias("rain_t"),
                     F.col("location_latitude").alias("lat"),
                     F.col("location_longitude").alias("lon"))
    feat = base.withColumn("ts", F.col("timestamp").cast("long"))
    for k, mins in [(1, 5), (2, 10), (3, 15), (4, 20)]:
        lag = base.select(F.col("station_id").alias("s_"),
                          (F.col("timestamp").cast("long") + k * 300).alias("tj_"),
                          F.col("rain_t").alias(f"rain_t{mins}"))
        feat = feat.join(lag, (feat.station_id == lag.s_) & (feat.ts == lag.tj_), "left").drop("s_", "tj_")
    isl = base.groupBy("timestamp").agg(
        F.avg((F.col("rain_t") > 0).cast("int")).alias("isl_frac_wet"),
        F.avg("rain_t").alias("isl_mean"), F.max("rain_t").alias("isl_max"),
        F.stddev("rain_t").alias("isl_std"), F.count("rain_t").alias("isl_n_stations"))
    feat = feat.join(isl, "timestamp")
    mod = F.hour("timestamp") * 60 + F.minute("timestamp")
    feat = (feat.withColumn("hour_sin", F.sin(2 * F.lit(math.pi) * mod / 1440))
            .withColumn("hour_cos", F.cos(2 * F.lit(math.pi) * mod / 1440))
            .withColumn("doy_sin", F.sin(2 * F.lit(math.pi) * F.dayofyear("timestamp") / 365))
            .withColumn("doy_cos", F.cos(2 * F.lit(math.pi) * F.dayofyear("timestamp") / 365)))
    return feat


def score(model, feat_with_threshold):
    """Fill missing, run the pipeline, return P(exceed) with the monotonic-in-threshold fix."""
    from pyspark.sql.window import Window
    df = feat_with_threshold.fillna({c: SENTINEL for c in MISSING})
    prob1 = F.udf(lambda v: float(v[1]), "double")
    scored = model.transform(df).withColumn("p_raw", prob1("probability"))
    w = (Window.partitionBy("station_id", "timestamp").orderBy(F.desc("threshold"))
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    return scored.withColumn("p_exceed", F.max("p_raw").over(w))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single", help="predict for one (station, time, threshold)")
    s.add_argument("--station", required=True)
    s.add_argument("--time", required=True, help='prediction time, e.g. "2021-06-01 15:00:00" (SGT)')
    s.add_argument("--threshold", type=float, required=True)
    s.add_argument("--parquet", default=PARQUET)
    s.add_argument("--model", default=MODEL)

    b = sub.add_parser("batch", help="score a pre-built feature table")
    b.add_argument("--input", required=True)
    b.add_argument("--model", default=MODEL)
    b.add_argument("--sample-frac", type=float, default=0.0001)
    b.add_argument("--show", type=int, default=20)
    args = p.parse_args()

    spark = start_spark()
    from pyspark.ml import PipelineModel
    model = PipelineModel.load(args.model)

    if args.mode == "single":
        # Derive features for the whole cleaned series (needed for lags/spatial), then pick
        # the one requested row. Score it at ALL thresholds in the grid so the monotonic
        # running-max is meaningful (with a single threshold it would be a no-op), then
        # report the requested one.
        feat = build_features(clean(spark.read.parquet(args.parquet)))
        row = feat.filter((F.col("station_id") == args.station) & (F.col("timestamp") == F.lit(args.time)))
        if not row.take(1):
            raise SystemExit(f"No reading for station {args.station} at {args.time} "
                             f"(must be an on-grid 5-minute SGT timestamp present in the data).")
        grid = spark.createDataFrame([(t,) for t in sorted(set(THRESHOLDS) | {args.threshold})], ["threshold"])
        expanded = row.crossJoin(F.broadcast(grid))
        scored = score(model, expanded)
        out = scored.filter(F.col("threshold") == args.threshold).select(
            "station_id", F.date_format("timestamp", "yyyy-MM-dd HH:mm:ss").alias("time_sgt"),
            "threshold", "p_exceed").first()
        print(f"\nP(station {out['station_id']} accumulates > {out['threshold']} mm "
              f"in the 30 min after {out['time_sgt']} SGT) = {out['p_exceed']:.3f}\n")
    else:
        df = spark.read.parquet(args.input)
        if 0 < args.sample_frac < 1.0:
            df = df.sample(False, args.sample_frac, seed=0)
        cols = ["station_id", "timestamp", "threshold", "p_exceed"]
        if "label" in df.columns:
            cols.append("label")
        score(model, df).select(*cols).orderBy("station_id", "timestamp", "threshold").show(args.show, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
