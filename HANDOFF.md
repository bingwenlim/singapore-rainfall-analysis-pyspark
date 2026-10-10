# Handoff — Singapore rainfall analysis (PySpark)

Status as of this commit. Read this first if you're picking the project up.

## What this project is

Two deliverables over Singapore 5-minute rainfall data (91 stations, 2017–2024):

1. **Part 1 — EDA / weather patterns** (done, in the notebook).
2. **Part 2 — ML model**: probability a station accumulates **> X mm in the next
   30 minutes**, given station + prediction time + threshold X. Data prep is done
   in the notebook; **training runs on GCP Dataproc and has not been run yet.**

Everything local runs on local Spark via `uv`. The notebook is the report;
`scripts/` holds the batch jobs.

## Repo layout

- `notebooks/rainfall_eda.ipynb` — the main report. Part 1 (sections 1–3 + Task 1
  plots A–D) and Part 2 data prep (2.1 accumulation + threshold EDA, 2.2 feature
  table). Runs top-to-bottom on local Spark.
- `scripts/csv_to_parquet.py` — one-time CSV→Parquet conversion. Already run.
- `scripts/train.py` — the Dataproc training job (LightGBM via SynapseML). **Not
  run yet.**
- `src/spark_project/session.py` — `create_local_spark()` helper (bundled JDK, env).
- `config.py` — real GCP values (**gitignored**). `config.example.py` is the template.
- `data/` — **gitignored.** `data/parquet` (51 MB, source) and `data/train_table`
  (1.1 GB, the training table) are local only, not in git.

## Environment

- `uv` manages deps (`pyproject.toml` / `uv.lock`). Run things with `uv run ...`.
- Java: a JDK is bundled at `.runtime/java-home`. `create_local_spark()` sets
  `JAVA_HOME` automatically. If you run a raw script that builds its own
  SparkSession (e.g. the smoke tests), export it first:
  `export JAVA_HOME="$PWD/.runtime/java-home"`.
- Open the notebook: `uv run jupyter lab notebooks`.

## Key data facts (established, don't re-derive)

- Grain: `(station_id, timestamp)` is the primary key. Unique after cleaning.
- **Timestamps are Singapore local time (SGT).** The CSVs carry `+08:00`; the
  Parquet is stored SGT, and the notebook's Spark session uses
  `spark.sql.session.timeZone = "Asia/Singapore"`, so `F.hour()`/`F.month()` are
  already local — **no manual +8 anywhere.** (The data genuinely starts
  2017-01-01 08:04:59 SGT; that's in the source, not a bug.)
- 2017 has a quirk: readings land at `:04:59` not `:00`. Cleaning adds 1s and
  truncates to the minute. This can collapse the 2017 boundary pair into one slot,
  so cleaning also `dropDuplicates(["station_id","timestamp"])` (~58 rows).
- Cleaning keeps **67 of 91 stations** (drops: n<1000, completeness<90% over the
  station's own lifespan, or stuck-at-zero `wet_frac==0`). `df_clean` in the notebook.
- Rainfall is ~97% zeros per 5-min slot; the 30-min accumulation is ~94% dry.
- Threshold grid chosen from the accumulation distribution: **0.2, 1, 2, 5, 10 mm**
  (positive rates 4.65% → 0.40%). Above 10 mm too few positives.

## Part 2 pipeline (how it fits together)

1. Notebook §2.1 builds `accum30` = sum of the next six 5-min readings (t+5…t+30),
   **only on valid windows** (all six future slots present, non-null, non-negative).
2. Notebook §2.2 builds features (all at or before t, no leakage):
   - lags `rain_t, rain_t5…rain_t20` — **gap-aware** (join on `timestamp − k·5min`,
     so a missing past slot is NaN, not a wrong value).
   - island-wide spatial context at t: `isl_frac_wet, isl_mean, isl_max, isl_std,
     isl_n_stations`.
   - cyclical time: sin/cos of **minute-of-day** and day-of-year.
   - `station_id` (categorical), `lat`, `lon`.
   Missing values are kept as **NaN** (LightGBM handles them natively; a gap ≠ an
   observed zero). Then each row is replicated over the 5 thresholds, label =
   `accum30 > X`, written to `data/train_table` partitioned by year.
3. `scripts/train.py` trains ONE LightGBM classifier with `threshold` as a feature
   (so one model answers any X), chronological split, per-threshold evaluation.

## Modelling decisions (settled, with reasons)

- **Binary classifier, threshold as a feature** (not quantile regression; not
  one-model-per-threshold). The task wants P(exceed X) with X as an input.
- **Monotone constraint** (−1) on the threshold feature so P(>X) is non-increasing
  in X.
- **Chronological split with a 30-min boundary gap**: train on
  `timestamp < 2024-01-01 − 30min`, test on `timestamp >= 2024-01-01`. The gap
  stops a training row's forward label window from peeking into the test period.
  (Single held-out year 2024 — ample but coarse. Rolling-origin CV over several
  years is a possible upgrade **if Dataproc cost is low**; noted in the script.)
- **Evaluate per threshold**: PR-AUC (primary, heavy imbalance), ROC-AUC, F1.
  Never accuracy, never a single aggregate across thresholds.
- Known **local training limitation**: SynapseML's LightGBM native lib does not load
  on Apple-Silicon macOS. The pipeline *shape* was validated locally with MLlib GBT
  (PR-AUC ~0.75 at 0.2mm down to ~0.35 at 10mm on a 2% sample — sanity only). The
  real LightGBM fit must run on Dataproc (Linux).

## What's left: run Part 2 on Dataproc (NOT done)

GCP values are in `config.py`:
- PROJECT = `project-9486f2d0-6b86-4bff-891`
- REGION = `asia-southeast1`
- BUCKET = `spark-rainfall-9486f2d0`

Steps (you drive the GCP CLI; `! <cmd>` in the Claude session runs it inline):

1. **Auth / config** (once):
   ```
   gcloud auth login
   gcloud config set project project-9486f2d0-6b86-4bff-891
   ```
2. **Upload the training table** (1.1 GB):
   ```
   gsutil -m cp -r data/train_table gs://spark-rainfall-9486f2d0/data/train_table
   ```
3. **Create a Dataproc cluster — MUST be Spark 3.5 / Scala 2.12** (SynapseML 1.1.3
   has no Spark-3.5/Scala-2.13 build; mismatch → `LightGBMClassifier does not exist
   in the JVM`). Verify the image's Scala version before relying on it. If the
   cluster nodes are internal-IP-only, you need Cloud NAT so Ivy can fetch the
   SynapseML jar from `mmlspark.blob.core.windows.net` (not on Maven Central).
4. **Submit the job** (the `^#^` sets `#` as delimiter so the two spark.jars.*
   properties aren't split on their internal commas):
   ```
   gcloud dataproc jobs submit pyspark scripts/train.py \
     --project=project-9486f2d0-6b86-4bff-891 --region=asia-southeast1 --cluster=$CLUSTER \
     --properties=^#^spark.jars.packages=com.microsoft.azure:synapseml_2.12:1.1.3#spark.jars.repositories=https://mmlspark.blob.core.windows.net/maven \
     -- \
     --input gs://spark-rainfall-9486f2d0/data/train_table \
     --output gs://spark-rainfall-9486f2d0/models/run-001
   ```
   (Re-check the exact SynapseML coordinate/version at run time; versions drift.)
5. **Download results and DELETE the cluster** (it bills while alive):
   ```
   gsutil -m cp -r gs://spark-rainfall-9486f2d0/models/run-001 ./models/run-001
   gcloud dataproc clusters delete $CLUSTER --region=asia-southeast1
   ```

`train.py` writes `run-001/model` (the fitted pipeline) and `run-001/metrics`
(JSON report: per-threshold PR-AUC/ROC-AUC/F1/precision/recall + config).

Open decisions for whoever runs it:
- Dataproc **cluster** (Compute Engine) is the lower-risk path over Serverless for
  the SynapseML Scala-version reason above.
- `--num-iterations` / `learningRate` / `numLeaves` are defaults in `train.py`;
  tune after a first baseline, not before.
- Calibration (Brier/reliability) and threshold-specific class weighting are
  deliberately deferred until a baseline exists.

## Minor known loose ends (cosmetic, not blocking)

- Notebook cell `e47370e6` one-liner says the four zero-stations are "all with
  very low n" — S24B actually has ~33k rows (it's the stuck sensor). The §2d /
  window cells say this correctly; only that one-liner is stale.
- Notebook outputs include folium maps (interactive HTML) — they bloat the .ipynb
  and don't survive `nbconvert --execute` cleanly. Fine for viewing in Jupyter.
