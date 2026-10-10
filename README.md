# Singapore Rainfall Analysis and Prediction (PySpark)

Analysis of Singapore's 5-minute rainfall gauge data (91 stations, 2017–2024) in
two parts:

- **Part 1 — EDA / weather patterns** (`notebooks/rainfall_eda.ipynb`): data-quality
  checks, cleaning (keeps 67 of 91 stations), and the within-year / across-year /
  within-day rainfall patterns.
- **Part 2 — model** (`scripts/train.py`): estimates the probability a station
  accumulates more than a given threshold of rainfall in the **next 30 minutes**,
  taking station, prediction time, and threshold as inputs.

See **[HANDOFF.md](HANDOFF.md)** for the full state, every design decision, and the
history (including why the model is Spark MLlib GBT rather than LightGBM).

## Setup

Uses [`uv`](https://docs.astral.sh/uv/). A JDK is bundled under `.runtime/java-home`.

```
uv sync
```

Source CSVs go in `data/`; convert them to the partitioned Parquet the notebook
reads with:

```
uv run python scripts/csv_to_parquet.py
```

## Part 1 — EDA notebook

```
uv run jupyter lab notebooks
```

Open `rainfall_eda.ipynb` and run top to bottom (local Spark).

## Part 2 — train the model

The notebook's Part 2 writes the feature table to `data/train_table`. Then
`scripts/train.py` trains a GBT classifier. `--input`/`--output` take **local
paths or `gs://` URIs** — the same script runs locally or on Dataproc.

**Locally** (sample the training set to keep it quick; the test year is always full):

```
JAVA_HOME=$PWD/.runtime/java-home \
PYSPARK_SUBMIT_ARGS="--driver-memory 12g pyspark-shell" \
uv run python scripts/train.py \
    --input data/train_table --output models/run-local \
    --sample-frac 0.1
```

Writes `models/run-local/model` (fitted `PipelineModel`) and `models/run-local/metrics`
(per-threshold PR-AUC / ROC-AUC / F1 / precision / recall).

**On Dataproc** (full data; no extra packages or network setup — GBT is built in):
see the "Run on Dataproc" recipe in [HANDOFF.md](HANDOFF.md).

Hyperparameters are tuned via `scripts/tune.py` (local, validates on 2023 so the
2024 test year is never used for model selection).

## Layout

```
notebooks/rainfall_eda.ipynb   Part 1 EDA + Part 2 feature engineering
scripts/csv_to_parquet.py      one-time CSV -> Parquet (SGT) conversion
scripts/train.py               GBT training job (local or Dataproc)
scripts/tune.py                hyperparameter tuning harness
src/spark_project/session.py   local SparkSession helper
config.example.py              copy to config.py (gitignored) for GCP values
```

`data/` and `config.py` are gitignored.
