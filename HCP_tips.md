# Running on the Server: Essentials Only

## 1. Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync                  # requires JDK 11 (check with: java -version)
```

## 2. Data

Put the 8 CSV files in `data/`. File names must be `HistoricalRainfallacrossSingapore<2017..2024>.csv`.

```bash
uv run python scripts/csv_to_parquet.py     # fails if data/parquet already exists; delete it first
```

## 3. Required notebook change: the Spark session cell

In `notebooks/rainfall_eda.ipynb`, change the cell that creates the Spark session to:

```python
spark = create_local_spark(
    app_name="rainfall-eda",
    master="local[8]",          # originally local[*], which takes all 140 cores of the login node
    driver_memory="14g",
    conf={
        "spark.sql.session.timeZone": "Asia/Singapore",
        "spark.local.dir": "/scratch/<username>/spark_tmp",   # added
    },
)
```

After editing, restart the kernel and run all cells from the top. Select `<project dir>/.venv/bin/python` as the kernel.

## 4. Training (using 10% of the training data)

`train.py` sets neither the master nor the spill directory, so both must be supplied through environment variables:

```bash
export SPARK_LOCAL_DIRS=/scratch/<username>/spark_tmp
mkdir -p $SPARK_LOCAL_DIRS

# --sample-frac 0.1: use only 10% of the training data (sampled by station-timestamp, so the five thresholds stay together)
nohup env PYSPARK_SUBMIT_ARGS="--master local[8] --driver-memory 12g pyspark-shell" \
  uv run python scripts/train.py \
    --input data/train_table --output models/run-001 \
    --sample-frac 0.1 --num-iterations 40 > train.log 2>&1 &
```

## 5.Evaluation

```bash
PYSPARK_SUBMIT_ARGS="--master local[4] --driver-memory 4g pyspark-shell" \
  uv run python scripts/predict.py batch --input data/train_table
```



## 6. Limitation

Full-data training on the login node gets killed by the system.