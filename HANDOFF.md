# Handoff — Singapore rainfall analysis (PySpark)

Status as of this commit. Read this first if you're picking the project up.

## What this project is

Two deliverables over Singapore 5-minute rainfall data (91 stations, 2017–2024):

1. **Part 1 — EDA / weather patterns** (done, in the notebook).
2. **Part 2 — ML model**: probability a station accumulates **> X mm in the next
   30 minutes**, given station + prediction time + threshold X. Data prep is done
   in the notebook; **the model (`train.py`) trains locally with Spark MLlib GBT —
   a working run is done (10% sample). See the Part 2 sections below.**

Everything local runs on local Spark via `uv`. The notebook is the report;
`scripts/` holds the batch jobs.

## Repo layout

- `notebooks/rainfall_eda.ipynb` — the main report. Part 1 (sections 1–3 + Task 1
  plots A–D) and Part 2 data prep (2.1 accumulation + threshold EDA, 2.2 feature
  table). Runs top-to-bottom on local Spark.
- `scripts/csv_to_parquet.py` — one-time CSV→Parquet conversion. Already run.
- `scripts/train.py` — the training job: Spark MLlib `GBTClassifier`, runs locally
  (pure JVM, no cloud). Pivoted here from LightGBM/SynapseML — see "The LightGBM →
  GBT pivot" below. Hyperparameters are tuned (see `tune.py`).
- `scripts/tune.py` — hyperparameter tuning harness (local, 1% sample, validates on
  2023 so the 2024 test year stays untouched). Found GBT insensitive here; best was
  maxDepth 5 / maxIter 60 / stepSize 0.1 (all configs clustered ~0.681 val PR-AUC).
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
   - `station_id`, `lat`, `lon`.
   Missing values are kept as **NaN** in the table (a gap ≠ an observed zero; the
   table stays model-neutral). Then each row is replicated over the 5 thresholds,
   label = `accum30 > X`, written to `data/train_table` partitioned by year.
   (The notebook feature engineering is DONE and unchanged — the model pivot below
   is all consumer-side in `train.py`.)
3. `scripts/train.py` trains ONE classifier with `threshold` as a feature (so one
   model answers any X), chronological split, per-threshold evaluation. **Now a
   Spark MLlib `GBTClassifier`** (see the pivot below), run **locally**.

## Modelling decisions (settled, with reasons)

- **Binary classifier, threshold as a feature** (not quantile regression; not
  one-model-per-threshold). The task wants P(exceed X) with X as an input.
- **Chronological split with a 30-min boundary gap**: train on
  `timestamp < 2024-01-01 − 30min`, test on `timestamp >= 2024-01-01`. The gap
  stops a training row's forward label window from peeking into the test period.
  Single held-out year 2024 — ample but coarse; rolling-origin CV is a noted upgrade.
- **Evaluate per threshold**: PR-AUC (primary, heavy imbalance), ROC-AUC, plus
  F1/precision/recall at the 0.5 cutoff. Never accuracy, never one aggregate.
- **Monotonicity** (P(>X) non-increasing in X): enforced **post-hoc** (running max
  of the probability over descending threshold per station-time), since MLlib GBT
  has no monotone constraint.
- **Missing features → −1 sentinel** before the vector assembler (GBT has no native
  NaN handling; rainfall ≥ 0 so −1 is an unambiguous "was missing" marker, distinct
  from an observed 0 — preserving the gap-aware intent).
- **`station_id` as a numeric index feature** (not a declared categorical): MLlib
  loses per-slot categorical metadata through VectorAssembler, and lat/lon already
  carry station geography. `maxBins=128` (must exceed the ~67 station cardinality).

## The LightGBM → GBT pivot (why, so nobody re-litigates it)

The original plan was LightGBM via SynapseML on Dataproc. We **tried it on Dataproc
and abandoned it** after peeling through this failure chain (each real, each fixed,
until a wall):

1. Jar wouldn't download on internal-IP nodes → fixed with **Cloud NAT**.
2. LightGBM driver↔worker **socket handshake failed** → fixed with
   `useBarrierExecutionMode=True`.
3. Barrier mode conflicts with Spark **dynamic allocation** → set
   `spark.dynamicAllocation.enabled=false`.
4. Single-node barrier run **OOM'd**; 4-worker run then hit a native **SIGSEGV** in
   `lib_lightgbm` (confirmed via YARN container logs — not memory, not firewall).

The SIGSEGV is a native-library incompatibility (SynapseML 1.1.3 / lightgbmlib
3.3.510 on Dataproc 2.3), not something our config could fix — and LightGBM's native
lib won't load on Apple-Silicon locally either. So we **switched to MLlib
`GBTClassifier`**: pure JVM (no native lib, no worker socket mesh), runs anywhere
Spark runs. Slightly weaker than LightGBM, and it costs the two features handled
above (sentinel for NaN, post-hoc for monotonicity), but it **actually runs and
produces the deliverable**. All the Dataproc resources were deleted.

## How to run Part 2 (GBT, locally — DONE, reproducible)

No cloud needed. From the repo root:

```
JAVA_HOME=$PWD/.runtime/java-home PYSPARK_SUBMIT_ARGS="--driver-memory 12g pyspark-shell" \
  uv run python scripts/train.py \
    --input data/train_table --output models/run-local \
    --sample-frac 0.1 --num-iterations 40
```

- `--sample-frac` samples **whole station-timestamp observations** of the TRAIN set
  (all 5 thresholds kept together). The TEST set is always full 2024 (~28M rows).
- Writes `models/run-local/model` (fitted `PipelineModel`) and `models/run-local/metrics`
  (JSON: per-threshold pr_auc/roc_auc/f1/precision/recall + config).
- `train.py` is pure MLlib, so the SAME script also runs on a Dataproc cluster if a
  bigger run is ever wanted (no packages/NAT/barrier needed now).

Results on the 10% train sample (full 2024 test), per threshold — PR-AUC (primary),
ROC-AUC, then F1/precision/recall at the 0.5 cutoff:

| threshold | PR-AUC | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| >0.2 mm | 0.75 | 0.95 | 0.66 | 0.82 | 0.56 |
| >1 mm  | 0.67 | 0.95 | 0.59 | 0.79 | 0.47 |
| >2 mm  | 0.59 | 0.95 | 0.52 | 0.74 | 0.40 |
| >5 mm  | 0.47 | 0.96 | 0.38 | 0.73 | 0.25 |
| >10 mm | 0.36 | 0.97 | 0.29 | 0.65 | 0.18 |

PR-AUC far exceeds the positive base rate at every threshold (~4.6% at 0.2mm down to
~0.4% at 10mm), so the model has real signal — strongest for common rain, declining
(as expected) for rarer heavy events. The low recall at the 0.5 cutoff reflects a
conservative operating point, tunable later. Full numbers in `models/run-local/metrics`.

Open follow-ups (deferred, not blocking):
- **Full-data run** (drop `--sample-frac`): memory doesn't scale 10× (Spark streams
  partitions; eval is already full-2024 every run) — the fit just gets slower and
  is the one OOM-risk. Step up gradually (25/50%) rather than jumping to 100%.
- **Probability calibration** (Brier/reliability) — GBT probabilities are poorly
  calibrated; worth adding for a probability-estimation task.
- **0.5 cutoff is arbitrary** for rare events (drives low recall); tune the operating
  point on a validation period, not the test year.
- Rolling-origin CV instead of the single 2024 test year.

## Minor known loose ends (cosmetic, not blocking)

- Notebook cell `e47370e6` one-liner says the four zero-stations are "all with
  very low n" — S24B actually has ~33k rows (it's the stuck sensor). The §2d /
  window cells say this correctly; only that one-liner is stale.
- Notebook outputs include folium maps (interactive HTML) — they bloat the .ipynb
  and don't survive `nbconvert --execute` cleanly. Fine for viewing in Jupyter.
