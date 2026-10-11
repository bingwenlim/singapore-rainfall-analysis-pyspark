# Project notes — decisions, findings, and rationale

A record of what was done and *why*, for the write-up and for future reference.
Reproduction mechanics (env, exact commands) live in `README.md`;
this doc is the reasoning. Dates are the dataset's, not calendar.

## The data

Singapore rainfall, 5-minute tipping-bucket gauges, 91 stations, 2017–2024.

- **Grain:** `(station_id, timestamp)` is the primary key — one reading per station
  per 5-minute slot.
- **Timezone — a real trap we caught:** the CSVs carry `+08:00` (SGT). The first
  CSV→Parquet pass stored them as UTC, which silently shifted everything −8h (the
  year appeared to start at `00:04:59` instead of the true `08:04:59` SGT). Fixed by
  converting with the session TZ set to `Asia/Singapore`, so `F.hour()`/`F.month()`
  read as local time with no manual offset. Verified against the raw CSV.
- **2017 quirk:** 2017 readings land at `:04:59` not `:00`. Cleaning adds 1 second and
  truncates to the minute. That truncation can merge the 2017 boundary pair into one
  slot, so cleaning also de-duplicates `(station_id, timestamp)` (~58 rows).
- **Rainfall is sparse and quantized:** ~97% of 5-min readings are exactly 0; the
  smallest nonzero reading is ~0.2 mm — the tipping-bucket resolution. So a 0.2 mm
  threshold already *is* the "any detectable rain" signal; sub-0.2 thresholds (0.1,
  0.01) are below sensor resolution and meaningless.

## Cleaning — which stations to keep

Kept **67 of 91** stations. A station is dropped if it has too little data
(`n < 1000`), poor completeness over its own lifespan (`< 90%`), or a stuck sensor
(`wet_frac == 0`). Key reasoning:

- **Completeness is measured over each station's own active span, not the full 8
  years** — otherwise late-joining stations (a cohort started 2020-06-15) would look
  "incomplete" when they're actually fine for the time they existed.
- The stuck-sensor check (`wet_frac == 0`) catches S24B: ~33k readings but never once
  nonzero. Confirmed it's a dead sensor, not a dry location, by checking that every
  other station in the same window recorded rain.

## Part 1 — weather patterns (findings)

All metrics use a **spatial-first average** (average across stations present in each
5-min slot, then aggregate over time) so the changing station roster doesn't bias
results. This was a correction — an earlier per-station-then-average approach let a
station's *lifespan* leak into annual totals (partial years summed as if full).

- **Within a year:** rain is most *frequent* in the Nov–Jan wet season (Northeast
  Monsoon); the Southwest Monsoon (Jun–Sep) is not a distinct wet season here.
  Rain *intensity* runs opposite to frequency — the wet season rains often but light;
  the heaviest per-slot rain is in the Mar–Apr inter-monsoon transition. So seasonal
  wetness is a frequency phenomenon, not an intensity one.
- **Across years:** annual totals swing 1,700–2,900 mm (2019 dry, 2021–22 wet) with
  **no monotonic trend** over 8 years — natural variability, not a detectable trend.
  (Deliberately did not fit a regression to 8 points.)
- **Within a day:** the strongest pattern by far — a sharp mid-afternoon peak (~16%
  chance of a rainy hour at 15:00 vs ~3.5% overnight, ~4.5× swing). Hour of day is the
  biggest single driver of rainfall frequency, ahead of season.

## Part 2 — the model

**Task:** probability a station accumulates > X mm in the next 30 min, given station,
time, and threshold X.

**Design (settled, with reasons):**
- **Binary classifier with the threshold as a feature**, not quantile regression and
  not one-model-per-threshold. Each training row is replicated over the threshold grid
  (0.2/1/2/5/10 mm, chosen from the accumulation distribution) with label
  `accum30 > X`, so one model answers any X. Quantile regression was rejected: it needs
  multiple models + interpolation to recover a probability a classifier gives directly.
- **Label built only on fully-observed future windows** — all six of the next 5-min
  slots present, non-null, non-negative. Offline future windows are dropped, never
  labelled "no rain".
- **Features (all ≤ t, no leakage):** gap-aware lags (join on `timestamp − k·5min`, so
  a missing slot is genuinely missing, not a wrong value); island-wide spatial context
  at t (fraction wet, mean, max, std, station count); cyclical minute-of-day and
  day-of-year; station id + lat/lon. Lean on purpose — no rolling averages (redundant
  with raw lags for trees), no month (redundant with day-of-year).
- **Split:** chronological, train ≤2023 / test 2024, with a 30-minute gap so a training
  row's forward label can't peek into the test period. Single test year — ample
  (~28M rows, all seasons) but coarse; rolling-origin CV noted as an upgrade.
- **Metrics per threshold:** PR-AUC (primary, heavy imbalance), ROC-AUC, plus
  F1/precision/recall at the 0.5 cutoff. Never accuracy, never one aggregate.

**Hyperparameters (tuned, `scripts/tune.py`):** validated on 2023 (2024 never touched,
to avoid selection leakage). GBT proved *insensitive* here — depth 3 underfits, 7
overfits; learning-rate configs all tied ~0.681 val PR-AUC. Locked the cheapest
tied-best config: **maxDepth 5, maxIter 60, stepSize 0.1**. The honest finding is a
negative one: performance is driven by features/data, not these knobs.

## The LightGBM → GBT pivot (the engineering story)

Originally planned LightGBM (via SynapseML) for the model. Tried it on Dataproc and
abandoned it after a chain of failures — each real, each fixed, until a hard wall:

1. Jars wouldn't download on internal-IP nodes → fixed with Cloud NAT.
2. LightGBM's driver↔worker socket handshake failed → fixed with barrier execution mode.
3. Barrier mode conflicts with Spark dynamic allocation → disabled dynamic allocation.
4. Single-node run OOM'd; a 4-worker run then hit a native **SIGSEGV in `lib_lightgbm`**
   (confirmed from the YARN container logs — not memory, not firewall).

The SIGSEGV is a native-library incompatibility (SynapseML 1.1.3 / lightgbmlib 3.3.510
on this stack), and LightGBM's native lib also won't load on Apple-Silicon locally.
**Lesson:** SynapseML LightGBM carries a fragile native dependency and a worker socket
mesh that fight managed Spark environments.

Switched to **Spark MLlib `GBTClassifier`** — pure JVM, no native lib, no socket mesh,
so it runs anywhere Spark runs (local or Dataproc, unchanged). It costs two LightGBM
features, each handled:
- No native NaN handling → fill missing features with a **−1 sentinel** (rain ≥ 0, so
  −1 is unambiguously "missing", distinct from observed 0).
- No monotone constraint → enforce P(>X) non-increasing in X **post-hoc** (running max
  of the probability over descending threshold per station-time).
- Also: `station_id` as a numeric index (MLlib loses categorical metadata through the
  assembler), `maxBins=128` for the ~67-station cardinality.

## Results

**Local, 10% train sample, full 2024 test** (the proven baseline):

| threshold | PR-AUC | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| >0.2 mm | 0.75 | 0.95 | 0.66 | 0.82 | 0.56 |
| >1 mm  | 0.67 | 0.95 | 0.59 | 0.79 | 0.47 |
| >2 mm  | 0.59 | 0.95 | 0.52 | 0.74 | 0.40 |
| >5 mm  | 0.47 | 0.96 | 0.38 | 0.73 | 0.25 |
| >10 mm | 0.36 | 0.97 | 0.29 | 0.65 | 0.18 |

PR-AUC far exceeds the positive base rate at every threshold (~4.6% down to ~0.4%), so
the model has real signal — strongest for common rain, declining (as expected) for
rarer heavy events. Low recall at the 0.5 cutoff is a conservative operating point,
tunable.

**Cloud, full data (all ~160M train rows, full 2024 test):** GBT ran cleanly on
Dataproc (a 1% smoke test passed first, where every LightGBM attempt had failed). The
full run was memory-bound and spilled to disk on a fixed 4×n4-standard-4 cluster — slow
(>10× the sampled run, the spill tax), but it completed.

| threshold | PR-AUC | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| >0.2 mm | 0.756 | 0.95 | 0.67 | 0.81 | 0.56 |
| >1 mm  | 0.675 | 0.95 | 0.59 | 0.80 | 0.47 |
| >2 mm  | 0.600 | 0.96 | 0.53 | 0.74 | 0.41 |
| >5 mm  | 0.476 | 0.96 | 0.40 | 0.72 | 0.27 |
| >10 mm | 0.367 | 0.97 | 0.29 | 0.64 | 0.19 |

**Key comparison — full data barely beats the 10% sample.** PR-AUC rose only
~0.005–0.01 at every threshold (e.g. 0.75→0.756 at 0.2mm, 0.36→0.367 at 10mm) — within
noise. So the 10% sample was already sufficient; 10× the data (and a much slower,
spilling cloud run) bought negligible accuracy. For this problem the signal saturates
well before full data. The cloud run was still worth doing to prove GBT runs on Dataproc
and to confirm the sample was enough — but it isn't needed for the result.

## Known limitations / follow-ups

- Full data gives essentially the same accuracy as a 10% sample (above), so the sample
  is a legitimate, far cheaper training set for this problem.
- The 0.5 decision cutoff is arbitrary for rare events (drives low recall) — tune the
  operating point on a validation period, not the test year.
- GBT probabilities aren't calibrated — a Brier/reliability check would be worth adding
  for a probability-estimation task.
- Single 2024 test year; rolling-origin CV would give a cross-year variance estimate.
- The full-data run spilled (memory-bound on 4×n4-standard-4) — a high-memory node type
  (e.g. n4-highmem) or more shuffle partitions would cut the spill if re-run at scale.
- Minor: notebook cell `e47370e6` one-liner mis-describes S24B as "low n" (it's the
  stuck sensor with ~33k rows); other cells state it correctly.
