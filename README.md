# Singapore Rainfall Analysis and Prediction (PySpark)

Analysis of Singapore's 5-minute rainfall gauge data (91 stations, 2017–2024) in two parts:

- **Part 1 — EDA / weather patterns** (`notebooks/rainfall_eda.ipynb`): data-quality checks, cleaning (keeps 67 of 91 stations), and the within-year / across-year / within-day rainfall patterns.
- **Part 2 — model** (`scripts/train.py`): estimates the probability a station accumulates more than a given threshold of rainfall in the **next 30 minutes**, taking station, prediction time, and threshold as inputs.

## Setup

**Python deps** — uses [`uv`](https://docs.astral.sh/uv/) (install it first if you don't have it). `uv sync` creates the venv and fetches Python 3.11 (pinned) plus the dependencies:

```
uv sync
```

**Java** — Spark needs a **JDK 11**. Two ways; the scripts work with either:

- *System-wide:* install any JDK 11 (e.g. `brew install --cask temurin@11`, or SDKMAN, or your distro's package) and make sure `JAVA_HOME` points to it. `scripts/train.py` and the notebook use whatever `JAVA_HOME` you have set.
- *Project-local (no system install):* download an [Eclipse Temurin JDK 11](https://github.com/adoptium/temurin11-binaries/releases) build for your OS/arch, verify its SHA-256 against the published checksum, extract it into `.runtime/jdk/`, and symlink `.runtime/java-home` to the extracted JDK's home (`…/Contents/Home` on macOS). The scripts auto-detect `.runtime/java-home` when `JAVA_HOME` isn't set, so nothing else is needed. (This repo was developed with the macOS/arm64 Temurin 11 build this way; `.runtime/` is gitignored, so each clone sets up its own.)

**Data** — the source is Singapore's historical rainfall dataset on [data.gov.sg (collection 2279)](https://data.gov.sg/collections/2279/view). Download the yearly CSVs (2017–2024) and put them in `data/`, named `HistoricalRainfallacrossSingapore<year>.csv` (the pattern `scripts/csv_to_parquet.py` globs for). Then convert them to the partitioned Parquet the notebook reads:

```
uv run python scripts/csv_to_parquet.py
```

This writes `data/parquet/` (partitioned by year); your original CSVs are left untouched.

**Times are interpreted in Singapore time (SGT).** Conversion to Parquet preserves each observation’s moment: `08:00+08:00` in the CSV represents `00:00 UTC` and displays as `08:00` when read in SGT. The notebook and scripts configure Singapore time for analysis and prediction; training and validation cutoffs also explicitly specify SGT.

## Part 1 — EDA notebook

Open `notebooks/rainfall_eda.ipynb` in any Jupyter frontend (VS Code, JupyterLab, …) and run it top to bottom (local Spark). The notebook covers both Part 1 (the EDA) and Part 2's data preparation — running it through also writes the model's feature table to `data/train_table`. Use the project's `uv` virtualenv (`.venv`) as the kernel so PySpark (and your JDK 11) are picked up. For example, to launch JupyterLab:

```
uv run jupyter lab notebooks
```

## Part 2 — train the model

The notebook's Part 2 writes the feature table to `data/train_table`. Then `scripts/train.py` trains a GBT classifier. `--input`/`--output` take **local paths or `gs://` URIs** — the same script runs locally or on Dataproc.

**Locally** (sample the training set to keep it quick; the test year is always full):

```
PYSPARK_SUBMIT_ARGS="--driver-memory 12g pyspark-shell" \
uv run python scripts/train.py \
    --input data/train_table --output models/run-local \
    --sample-frac 0.1 --num-iterations 40
```

(The tuned default is 60 iterations, as used for the full run; `--num-iterations 40` reproduces the saved 10%-sample baseline.) Writes `models/run-local/model` (fitted `PipelineModel`) and `models/run-local/metrics` (per-threshold PR-AUC / ROC-AUC / F1 / precision / recall).

**On Dataproc** (full data; no extra packages or network setup — GBT is built in). Create a cluster, submit the job, then download results and delete the cluster:

```
# 1. upload the feature table (once)
gsutil -m cp -r data/train_table gs://$BUCKET/data/train_table

# 2. create a cluster (fixed 4 workers; any x86 image is fine)
gcloud dataproc clusters create rainfall-train \
    --project=$PROJECT --region=$REGION \
    --master-machine-type=n4-standard-4 --master-boot-disk-type=hyperdisk-balanced \
    --num-workers=4 --worker-machine-type=n4-standard-4 \
    --worker-boot-disk-type=hyperdisk-balanced --image-version=2.3-debian12 \
    --max-idle=30m --scopes=https://www.googleapis.com/auth/cloud-platform

# 3. submit the full-data run (drop --sample-frac to use all rows)
gcloud dataproc jobs submit pyspark scripts/train.py \
    --project=$PROJECT --region=$REGION --cluster=rainfall-train \
    -- --input gs://$BUCKET/data/train_table --output gs://$BUCKET/models/run-001

# 4. download results and DELETE the cluster (it bills while alive)
gsutil -m cp -r gs://$BUCKET/models/run-001 ./models/run-001
gcloud dataproc clusters delete rainfall-train --region=$REGION
```

First-time project setup (once): `gcloud services enable dataproc.googleapis.com` and grant the compute service account `roles/dataproc.worker`. Set `PROJECT`, `REGION`, `BUCKET` in your shell for the commands above — copy `config.example.py` to `config.py` and fill them in for reference.

Hyperparameters are tuned via `scripts/tune.py` (local, validates on 2023 so the 2024 test year is never used for model selection).

## Using the trained model

`scripts/predict.py` scores the saved model. It has two modes.

**Single prediction** — the task's stated interface: give a station, a prediction time, and a threshold, and get the exceedance probability. The script derives that row's features (recent lags, island-wide context at the time, cyclical time, location) from the cleaned rainfall data internally, so you only pass the three inputs:

```
uv run python scripts/predict.py single --station S08 --time "2021-06-01 15:00:00" --threshold 5
# -> P(station S08 accumulates > 5.0 mm in the 30 min after 2021-06-01 15:00:00) = ...
```

(The time must be an on-grid 5-minute timestamp, in Singapore time, present in `data/parquet`. The rainfall data is needed to compute "rain 15 min ago" and the island-wide context — those features are functions of the station and time applied to the data.)

**Batch** — a demonstration that scores rows from a pre-built feature table (e.g. `data/train_table`) and prints their probabilities:

```
uv run python scripts/predict.py batch --input data/train_table
```

This is for illustration, not metric reproduction: it prints predictions (and the true label if present) but does not compute PR-AUC/F1, and its random `--sample-frac` can separate the five threshold rows that belong to one station-time. The reported evaluation metrics come from `scripts/train.py`'s own test-set scoring, not from this command.

Under the hood: the saved model is a Spark ML `PipelineModel` **directory** (StringIndexer + VectorAssembler + GBT), not a single `.pkl`, so loading needs Spark. Two steps `train.py` applies live *outside* the pipeline and `predict.py` reproduces them — filling missing features (`rain_t5/10/15/20`, `isl_std`) with the `-1` sentinel before `transform`, and forcing the exceedance probability to be non-increasing in the threshold afterwards (a running max over descending threshold per station-time). Single mode scores the five training thresholds plus the requested threshold, applies the adjustment, and returns the requested probability. In batch mode, consistency is enforced only among thresholds retained together for a station-time; sampling can leave an incomplete grid.

## Layout

```
notebooks/rainfall_eda.ipynb   Part 1 EDA + Part 2 feature engineering
scripts/csv_to_parquet.py      one-time CSV -> Parquet conversion; analysed in SGT
scripts/train.py               GBT training job (local or Dataproc)
scripts/tune.py                hyperparameter tuning harness
scripts/predict.py             load the saved model and score rows
src/spark_project/session.py   local SparkSession helper
config.example.py              copy to config.py (gitignored) for GCP values
archive/                       archived project notes and handoff documents
```

`data/`, `config.py`, `.runtime/`, and `models/` are gitignored.
