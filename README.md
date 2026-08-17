# federated-ueba

Communication-efficient federated UEBA for insider threat detection on CERT r4.2.
A BiLSTM autoencoder is trained across 50 simulated organisations with FedAvg,
and the cost of shrinking what each client uploads is measured against the
detection quality that shrinking buys or loses.

## Layout

Every path below is relative to the project root, and every script expects to be
run from there: the configuration resolves data and artefact paths against the
working directory, not against the file's own location.

```
config_manager.py               all configuration, read from pyproject.toml
federated_insider_detection.py  the federated runner and the sweep
train_centralized.py            the single-machine baseline
feature_extraction.py           inherited from Le & Zincir-Heywood, unmodified
temporal.py                     inherited, unmodified

run_sweep.ps1                   the sweep

federated_ueba/                 the pipeline: model, client, server, strategy,
                                scaling, scoring, compression, seeding
analysis/                       reads results, produces numbers
tests/
```

## Prepare the data

- The feature extraction script needs a Linux environment.
- Download the CERT dataset and move the `r4.2` folder under `dataset/`.
- Run `feature_extraction.py`, then `temporal.py`.

Both scripts come from Le & Zincir-Heywood and are kept unmodified.

**What `temporal.py` produces is not raw counts.** It replaces each value with
that day's percentile within the *same user's* preceding 30 days, centred at
zero, so every value runs from -50 to +50 and a negative one means "below this
user's own normal". Two things follow: the first 30 days of any user cannot be
scored at all, and the effective temporal context of the model is 44 days, not
the 14 of its input window.

## Install

```bash
pip install -e .
```

## Run

Experiments live in `pyproject.toml` under `[tool.fueba.experiments.*]`, each
layered over the base `[tool.fueba]` settings by `config_manager.py`. The runner
spawns one subprocess per experiment so each starts from a clean config.

Every artefact is written under a **seed-qualified run id** such as
`baseline__seed1`: weights in `model_pickle/`, scalers in `scaler_data/`,
reports in `federated_evaluation_reports/`. Two seeds of the same experiment
therefore never overwrite each other.

The full multi-seed sweep, which is what the reported results come from:

```bash
powershell -NoProfile -File run_sweep.ps1
```

A single experiment:

```bash
python federated_insider_detection.py --experiment baseline --mode full --seed 1
```

A named subset, which is easier to read than a long deny-list:

```bash
python federated_insider_detection.py --experiment all --seeds 1,2,3 --only baseline,top-k-0.1
```

The `ablation-*` configs rescore an already-trained experiment instead of
training their own, using the weights named by `scoring.ablation_source`
at the same seed. They are ordered after the experiments they depend on and
refuse to train even if asked to.

The centralized reference model, once per seed:

```bash
SEED=2 python train_centralized.py
```

### Stopping a sweep

Killing the shell does not kill the flower-simulation and Ray workers it
spawned; they keep training and keep writing into the same directories, so a
second sweep started afterwards races the first. Kill the tree by command line:

```bash
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*run_sweep*' } | ForEach-Object { taskkill /F /PID \$_.ProcessId /T }"
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

399 tests. `tests/test_regression.py` rescores a saved checkpoint against the
report it originally produced, so it needs the real dataset and a finished run;
without them it skips itself. It is what makes a refactor safe, so do not leave
it skipped before merging a structural change.

## Analyses that need no retraining

These read a finished run and are the answers to three of the correction items:

```bash
python analysis/analyze_features.py            feature table: source, meaning, distribution
python analysis/analyze_payload_codecs.py      lossless payload encodings, measured
python analysis/analyze_detection_latency.py   how many days before an insider is caught
```

`analysis/analyze_detection_latency.py` replays each user day by day, scoring with only
the windows available by that day. Scoring prospectively is the point: a
detector that recognises an insider only after the campaign is over is not
useful, and the usual metrics cannot tell the difference.

## Reading the numbers

Do not trust an on-disk figure without checking which run produced it. Three
guards exist because each of these silently corrupted earlier results:

- **`pipeline_version`.** Every summary records it. `analysis/compare_experiments.py`
  refuses to put runs of different versions in one table, and the sweep re-runs
  rather than resumes anything stamped with an older one. Result directories
  outlive the code that made them, and a stale number that still parses is far
  more dangerous than one that fails to load.
- **`communication_log_matches_run`.** When `false`, the communication log
  survived from a run with a different round or client count, so the MB totals
  describe that other run.
- **`Runs` against `Seeds`** in the comparison table. When they disagree, a
  legacy directory from before seeding is being averaged in.

Two figures describe convergence and they answer different questions.
`convergence_round` is the thesis definition, the first round reaching 95% of
the peak validation PR-AUC; it is sensitive to a single lucky round.
`plateau_round` is the first round after which nothing more is gained. The gap
between them is how many rounds were paid for and wasted.

Reported uncertainty also comes in two forms and they are not interchangeable.
The bootstrap interval resamples users within one run; the across-seed standard
deviation resamples the run itself. A difference between two configurations is
only worth claiming if it survives both.

### What selects what

Nothing that is reported is allowed to have chosen itself. The threshold is
chosen on a validation half of the users and the threshold-dependent metrics are
reported on the test half. PR-AUC needs no threshold and is reported over all
1000 users, because a single half is both noisier and biased: the two halves are
complementary draws of the 70 insiders, so the half that takes the hard cases
makes the other one easier. The federated arm selects no round at all; it reports
the mean of the last 20. The per-client validation loss printed during training is a different
thing: it splits *windows*, which overlap by 13 of 14 days, so it is optimistic
and nothing depends on it.

## Configuration worth knowing about

In `pyproject.toml`, under `[tool.fueba]`:

- `feature_transform`, how a percentile value is prepared. `log1p_positive` is
  the default and the original behaviour; it clips negatives to zero, which
  collapses 20% of all cells. `signed_log1p` keeps below-normal days
  distinguishable, `raw` passes the percentile through.
- `drop_constant_features`, a variance filter. Four of the 50 features are zero
  on every row, so it drops them, taking the input from 50 to 46 and the
  parameter count from 450,258 to 447,694. Off by default because that count is
  what every communication figure derives from.
- `payload_codec` / `entropy_coder`, lossless encodings of a payload whose
  contents are already decided. They change the reported byte count and nothing
  else.
- `sparsification_ratio`, the fraction of the model's weights that survives
  sparsification. Unrelated to `[tool.fueba.scoring] top_k_features`, which is a
  count of features used when scoring a window. The two were routinely confused.

## Comparing results

```bash
python analysis/compare_experiments.py
```

Writes `experiment_comparison_summary.csv` (one row per experiment and seed) and
`experiment_comparison_by_seed.csv` (aggregated), plus a forest plot of the
bootstrap intervals.
