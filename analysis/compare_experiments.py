"""Collect every experiment run into comparison tables.

The reported metric of a federated run is the mean over its last
`plateau_window_rounds` rounds, not the single round with the best validation
PR-AUC. The single-round rule is still computed and still in each run's summary,
and it appears here as `PR-AUC_Argmax` so the difference stays visible; it is a
sensitivity analysis rather than the headline. Why: after round 6 the validation
curve is flat to about 0.005, so the argmax was choosing among rounds that do not
differ, and it was choosing the luckiest. That gift was measured at +0.0032 for
the baseline and +0.008 to +0.020 for the sparse arms, whose validation curves
are four times noisier, so the rule was quietly discounting the cost of the very
compression this work is arguing for.

None of this needs a re-run. Everything here is recomputed from the round CSVs
that finished runs already wrote, which is why the rule could change while a
sweep was in progress.

Two different uncertainties are reported and they answer different questions:

  bootstrap CI   resamples users within one run. "How much does this number
                 depend on which users happen to be in the dataset?"
  across-seed sd resamples the run itself. "How much does this number depend on
                 initialisation and partitioning?"

A difference between two configurations is only meaningful if it survives both.
Reporting one alone was what made the single-seed ablation deltas hard to defend.

Runs produced before the global scaler fix are skipped, because their ranking is
not a noisier version of the corrected one but a different quantity. Pass
`--include-stale` to look at them anyway.

Outputs:
  experiment_comparison_summary.csv   one row per run (experiment x seed)
  experiment_comparison_by_seed.csv   one row per experiment, aggregated
  generated_visuals/bootstrap_pr_auc  forest plot of the intervals
"""

# This file lives in a subdirectory, so Python puts that subdirectory on
# sys.path rather than the project root and `import config_manager` fails.
# Adding the root explicitly keeps `python analysis/compare_experiments.py` working from the project
# root, which is how every path in the configuration is resolved anyway.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config_manager import PIPELINE_VERSION

# Force UTF-8 on stdout/stderr: the emoji in the progress messages raise
# UnicodeEncodeError under the legacy Windows code page used when output is
# redirected to a log file.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_FEDERATED_REPORT_DIR = "federated_evaluation_reports"
CENTRALIZED_REPORT_DIR = "centralized_evaluation_reports"
FIGURE_PATH = "generated_visuals/bootstrap_pr_auc.png"

RUN_ID_RE = re.compile(r"^(?P<experiment>.+?)__seed(?P<seed>\d+)$")

# Metrics aggregated across seeds.
AGG_METRICS = ["PR-AUC", "Max-F1", "Balanced_Acc", "Precision", "Recall"]

CENTRALIZED_COLOUR = "#B85042"
FEDERATED_COLOUR = "#065A82"

# The experiment whose peak defines "converged" for every other experiment.
SHARED_CONVERGENCE_REFERENCE = "baseline"


# --- Helpers ---------------------------------------------------------------

def parse_run_id(run_id):
    """Split `baseline__seed42` into ("baseline", 42).

    Directories written before seeds were introduced have no suffix; they are
    reported with seed None so they stay visible instead of being dropped.
    """
    match = RUN_ID_RE.match(run_id)
    if match:
        return match.group("experiment"), int(match.group("seed"))
    return run_id, None


def _round(value):
    return round(value, 4) if value is not None and not np.isnan(value) else None


def bootstrap_pr_auc(y_true, y_scores, n_iterations=None, seed=42,
                     confidence=None):
    """Bootstrap a PR-AUC confidence interval by resampling users.

    Resampling users rather than windows, because the reported metric ranks
    people: a user is one observation however many days they contributed.

    `confidence` is the central mass of the interval, so 0.95 puts 2.5% in each
    tail. The interval is a percentile range of the bootstrap distribution and is
    not symmetric about the point estimate, which is why the figure draws it with
    explicit bounds rather than as a plus-or-minus.
    """
    if n_iterations is None or confidence is None:
        from config_manager import config
        if n_iterations is None:
            n_iterations = config.get("evaluation", "bootstrap_iterations")
        if confidence is None:
            confidence = config.get("evaluation", "bootstrap_confidence")

    tail = (1.0 - confidence) / 2.0 * 100.0

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        # A resample that drew no insider has no precision-recall curve to
        # average. Skipped rather than counted as zero, which would drag the
        # interval down with a value that is undefined rather than low.
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(average_precision_score(y_true[idx], y_scores[idx]))
    if not scores:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(scores)), float(np.std(scores)),
            float(np.percentile(scores, tail)),
            float(np.percentile(scores, 100.0 - tail)))


def bootstrap_plateau_pr_auc(frames, n_iterations=None, seed=42, confidence=None):
    """Bootstrap the plateau mean, resampling users once for all rounds.

    The statistic is the mean PR-AUC over the plateau rounds, so the interval has
    to be built around that statistic rather than around any one round. Each
    iteration draws one set of users and scores every plateau round on that same
    set, which is what keeps the rounds paired: they describe the same 1000
    people, and resampling them independently per round would average away
    exactly the user-level variation the interval is meant to show.

    `frames` is a list of (y_true, y_scores) arrays, one per plateau round, all
    in the same user order.
    """
    if n_iterations is None or confidence is None:
        from config_manager import config
        if n_iterations is None:
            n_iterations = config.get("evaluation", "bootstrap_iterations")
        if confidence is None:
            confidence = config.get("evaluation", "bootstrap_confidence")

    if not frames:
        return np.nan, np.nan, np.nan, np.nan

    tail = (1.0 - confidence) / 2.0 * 100.0
    rng = np.random.RandomState(seed)
    n = len(frames[0][0])

    means = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        per_round = []
        for y_true, y_scores in frames:
            drawn = y_true[idx]
            # A resample with no insider has no precision-recall curve. Skipped
            # for the same reason as in the single-round bootstrap: undefined is
            # not zero.
            if len(np.unique(drawn)) < 2:
                continue
            per_round.append(average_precision_score(drawn, y_scores[idx]))
        if per_round:
            means.append(float(np.mean(per_round)))

    if not means:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(means)), float(np.std(means)),
            float(np.percentile(means, tail)),
            float(np.percentile(means, 100.0 - tail)))


def test_half(results_df, seed=None):
    """The half of the users the reported metrics describe.

    Every per-user CSV on disk holds all 1000 users, and the reported PR-AUC is
    the test half of them, so anything computed from the whole frame describes a
    different population than the number it sits next to.

    `seed` is accepted and ignored; the split comes from `split_seed`, which is
    the same for every run. It used to be the run seed, and that made two runs of
    one configuration report on two different sets of 35 insiders.
    """
    if "user" not in results_df.columns:
        return results_df
    from config_manager import config
    from federated_ueba import scoring
    _, test_users = scoring.split_users(
        results_df["user"].tolist(), results_df["is_actual_insider"].tolist(),
        seed=config.get("evaluation", "split_seed"),
        validation_fraction=config.get("evaluation", "validation_fraction"))
    return results_df[results_df["user"].isin(test_users)]


def _bootstrap_frame(results_df, label, seed=None):
    """Bootstrap a per-user results frame, reporting what it found."""
    if results_df is None or "is_actual_insider" not in results_df.columns:
        return np.nan, np.nan, np.nan, np.nan

    results_df = test_half(results_df, seed)
    ci = bootstrap_pr_auc(results_df["is_actual_insider"], results_df["max_z_score"])
    print(f"  Bootstrap PR-AUC [{label}]: {ci[0]:.4f} ± {ci[1]:.4f} "
          f"(95% CI: [{ci[2]:.4f}, {ci[3]:.4f}])")
    return ci


def _empty_row(experiment, seed, run_type):
    """Every row has the same columns, whichever collector produced it."""
    return {
        "Experiment": experiment, "Seed": seed, "Type": run_type,
        # PR-AUC and the four metrics beside it are plateau means: the average
        # over the last `plateau_window_rounds` rounds. PR-AUC_Argmax is the old
        # single-round rule, kept as the sensitivity analysis rather than
        # deleted, and Best_Round is the round it chose. See plateau_metrics.
        "Best_Round": None, "PR-AUC": None,
        "PR-AUC_Argmax": None, "PR-AUC_Plateau_SD": None, "Plateau_Rounds": None,
        "PR-AUC_Bootstrap_Mean": None, "PR-AUC_Std": None,
        "PR-AUC_CI_Lo": None, "PR-AUC_CI_Hi": None,
        "Max-F1": None, "Balanced_Acc": None, "Precision": None, "Recall": None,
        "Total_Comm_MB": None, "Upload_MB": None, "Download_MB": None,
        # Two convergence definitions, both reported: Conv_Round is the thesis
        # one (95% of the peak), Plateau_Round the round after which nothing more
        # is gained. See find_convergence_round / find_plateau_round.
        # Conv_Round_Shared measures every run against the same absolute target
        # instead of against its own peak, which is the only one of the three
        # that may be compared across experiments. See shared_convergence_target.
        "Comm_Log_OK": None, "Conv_Round": None, "Plateau_Round": None,
        "Conv_Round_Shared": None,
    }


def _read_json(path, description):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {description}. Skipping.")
        return None


# --- Federated runs --------------------------------------------------------

def load_best_round_results(exp_dir, best_round):
    """Per-user scores for the best round of a federated run."""
    path = os.path.join(exp_dir, "round_by_round_results",
                        f"round_{best_round}_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  ⚠️  {path} missing; cannot bootstrap this run.")
    return None


def plateau_rounds(rounds, window=None):
    """The rounds the reported metric averages over: the last `window` of them.

    Expressed in rounds rather than in scored checkpoints so that changing
    `checkpoint_stride` does not silently change how much averaging happens.
    """
    if window is None:
        from config_manager import config
        window = config.get("evaluation", "plateau_window_rounds")
    if not rounds:
        return []
    cutoff = max(rounds) - window
    return [r for r in rounds if r > cutoff]


def plateau_metrics(df, window=None):
    """Mean of every numeric metric over the plateau rounds, and its spread.

    Returns (means, sd, rounds_used). `sd` is the spread of PR-AUC across the
    rounds in the window, which is the part a single-round number hides: an arm
    that is still climbing and an arm that is decaying can share a mean.
    """
    rounds = plateau_rounds(df["Round"].tolist(), window)
    if not rounds:
        return {}, None, []

    window_df = df[df["Round"].isin(rounds)]
    means = {column: float(window_df[column].mean())
             for column in window_df.columns
             if column != "Round" and pd.api.types.is_numeric_dtype(window_df[column])}
    sd = (float(window_df["PR-AUC"].std(ddof=1)) if len(window_df) > 1 else None)
    return means, sd, rounds


def load_round_scores(exp_dir, round_number, seed=None):
    """Per-user scores for one round, restricted to the reported test half.

    The round CSV holds all 1000 users, and bootstrapping it produced an interval
    around PR-AUC over all users while the reported point estimate was the test
    half: measured at 0.8204 against a reported 0.8796 for the baseline, an
    interval that did not contain the number it was printed beside. The split is
    reproduced here through `scoring.split_users`, the same function the
    evaluation used, rather than by rewriting the rule.

    `seed` is the run seed, because the split depends on it. Without one the
    frame is returned whole, which is only right for a fixture that has no split.
    """
    path = os.path.join(exp_dir, "round_by_round_results",
                        f"round_{round_number}_results.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    if not {"user", "is_actual_insider", "max_z_score"} <= set(frame.columns):
        return None
    # Sorted rather than trusted: the bootstrap pairs the rounds by position, so
    # one round written in a different user order would silently pair each
    # user's label with another user's score.
    frame = frame.sort_values("user")

    if seed is not None:
        frame = test_half(frame).sort_values("user")

    return (frame["is_actual_insider"].to_numpy(),
            frame["max_z_score"].to_numpy())


def load_plateau_frames(exp_dir, rounds, seed=None):
    """Per-user scores for every plateau round that was actually written."""
    frames = []
    for round_number in rounds:
        scores = load_round_scores(exp_dir, round_number, seed=seed)
        if scores is not None:
            frames.append(scores)
    if frames and len({len(y_true) for y_true, _ in frames}) > 1:
        print(f"  ⚠️  {exp_dir}: plateau rounds cover different numbers of "
              f"users; no interval for this run.")
        return []
    return frames


def load_rounds_frame(exp_dir):
    """This run's round-by-round metrics, or None if it never wrote them."""
    path = os.path.join(exp_dir, "federated_rounds_comparison.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if "Round" in df.columns else None


def load_validation_curve(exp_dir):
    """Round-by-round validation PR-AUC for one run, or None if it has none.

    The validation half, not the test half, for the same reason checkpoint
    selection reads it: convergence is a choice about which round to look at, and
    the reported half must not influence any choice.
    """
    df = load_rounds_frame(exp_dir)
    if df is None or "PR-AUC_val" not in df.columns:
        return None

    # Round 0 is dropped. It is the model before any training, but it is scored
    # against the reference error distribution the clients measured with their
    # *trained* local models, which is a product of training that had not
    # happened yet. The effect is not small: the baseline's round 0 scores 0.8972
    # against 0.8810 at round 50, so an untrained model appears to beat a trained
    # one, and every run then "converges" at round 0. Keeping it would make the
    # convergence column report an artefact rather than a measurement.
    df = df[df["Round"] > 0]
    if df.empty:
        return None
    return df["Round"].tolist(), df["PR-AUC_val"].tolist()


def shared_convergence_target(runs, fraction=None):
    """The absolute validation PR-AUC that counts as converged, from the baseline.

    Why this exists. The `convergence_round` in each summary is the first round
    reaching 95% of *that run's own* peak, and so it describes the shape of one
    curve rather than comparing two. Measured, it inverts the ranking it appears
    to give: baseline converged at round 6, top-k-0.1 at 24 and top-k-0.05 at 14,
    which reads as the worst configuration converging in the middle. It is not
    faster; its ceiling is lower, so 95% of that lower ceiling arrives sooner.

    Against one shared target the number means what a reader assumes it means:
    how many rounds of communication this configuration needs to reach the
    quality the baseline reaches. A configuration that never reaches it is
    reported as "never", which is the honest answer and one the per-run
    definition cannot express.

    Averaged over the baseline seeds rather than taken from one, so the target
    does not inherit a single run's luck. Returns None when no baseline run is
    available, which makes the column empty rather than silently rescaled.
    """
    if fraction is None:
        from config_manager import config
        fraction = config.get("evaluation", "convergence_peak_fraction")

    peaks = []
    for summary, run_id, exp_path in runs:
        experiment, _ = parse_run_id(summary.get("run_id", run_id))
        if experiment != SHARED_CONVERGENCE_REFERENCE:
            continue
        curve = load_validation_curve(exp_path)
        if curve is not None and curve[1]:
            peaks.append(max(curve[1]))

    if not peaks:
        print(f"  ⚠️  No '{SHARED_CONVERGENCE_REFERENCE}' run found; the shared "
              f"convergence column will be empty. Only the per-run "
              f"convergence_round is available, and it is not comparable "
              f"across experiments.")
        return None

    target = float(np.mean(peaks)) * fraction
    print(f"📐 Shared convergence target: {target:.4f} validation PR-AUC "
          f"({fraction:.0%} of the mean peak over {len(peaks)} "
          f"'{SHARED_CONVERGENCE_REFERENCE}' run(s)).")
    return target


def find_shared_convergence_round(rounds, scores, target):
    """First round reaching an absolute target, or "never" if none does.

    A string rather than None for the miss, because None already means "this run
    has no curve to read" and the two must not be confused in a results table:
    one is a missing measurement, the other is a measurement whose value is that
    the configuration never got there.
    """
    if target is None or not rounds:
        return None
    for round_number, score in zip(rounds, scores):
        if score >= target:
            return int(round_number)
    return "never"


def _check_communication_log(summary, run_id):
    """Warn when the MB totals came from a different run than the metrics."""
    comm_ok = summary.get("communication_log_matches_run")
    if comm_ok is False:
        print(f"  ⚠️  {run_id}: communication log does not match this run; "
              f"its MB figures are not trustworthy.")
    return comm_ok


def _federated_row(summary, run_id, exp_path, shared_target=None):
    """One table row for one federated run, with its bootstrap interval.

    The reported metrics are plateau means, read from the round CSVs rather than
    from `best_metrics` in the summary. That is what lets the rule change without
    re-running anything: the summary keeps recording the old single-round choice,
    and this layer decides what goes in the thesis table.
    """
    experiment, seed = parse_run_id(summary.get("run_id", run_id))
    best_metrics = summary.get("best_metrics", {})
    best_round = summary.get("best_round")

    rounds_df = load_rounds_frame(exp_path)
    if rounds_df is None:
        means, plateau_sd, window = {}, None, []
    else:
        means, plateau_sd, window = plateau_metrics(rounds_df)

    frames = load_plateau_frames(exp_path, window, seed=summary.get("seed", seed))
    if frames:
        ci_mean, ci_std, ci_lo, ci_hi = bootstrap_plateau_pr_auc(frames)
        print(f"  Bootstrap plateau PR-AUC [{run_id}]: {ci_mean:.4f} ± "
              f"{ci_std:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}], "
              f"{len(frames)} rounds)")
    else:
        # Falls back to the single round rather than reporting nothing, so a run
        # whose round files are missing still gets an interval, clearly labelled.
        ci_mean, ci_std, ci_lo, ci_hi = _bootstrap_frame(
            load_best_round_results(exp_path, best_round), f"{run_id} (argmax)",
            seed=summary.get("seed", seed))

    curve = load_validation_curve(exp_path)
    shared_convergence = (find_shared_convergence_round(*curve, shared_target)
                          if curve is not None else None)

    row = _empty_row(experiment, summary.get("seed", seed), "Federated")
    row.update({
        "Best_Round": best_round,
        "PR-AUC": _round(means.get("PR-AUC")),
        "PR-AUC_Argmax": best_metrics.get("PR-AUC"),
        "PR-AUC_Plateau_SD": _round(plateau_sd),
        "Plateau_Rounds": len(window) if window else None,
        "PR-AUC_Bootstrap_Mean": _round(ci_mean),
        "PR-AUC_Std": _round(ci_std),
        "PR-AUC_CI_Lo": _round(ci_lo),
        "PR-AUC_CI_Hi": _round(ci_hi),
        "Max-F1": _round(means.get("Max-F1")),
        "Balanced_Acc": _round(means.get("Balanced_Accuracy")),
        "Precision": _round(means.get("Precision")),
        "Recall": _round(means.get("Recall")),
        "Total_Comm_MB": summary.get("total_communication_mb"),
        "Upload_MB": summary.get("total_upload_mb"),
        "Download_MB": summary.get("total_download_mb"),
        "Comm_Log_OK": _check_communication_log(summary, run_id),
        "Conv_Round": summary.get("convergence_round"),
        "Plateau_Round": summary.get("plateau_round"),
        "Conv_Round_Shared": shared_convergence,
    })
    return row


def is_current_pipeline(summary):
    """Whether this run was produced by the pipeline version now in the code."""
    return summary.get("pipeline_version") == PIPELINE_VERSION


def _report_skipped(stale):
    if stale:
        print(f"\n⚠️  {len(stale)} run(s) skipped: produced by an older pipeline "
              f"version, so their numbers are not comparable with current runs.\n"
              f"    {', '.join(stale)}\n    Pass --include-stale to see them "
              f"anyway. They must not go into the thesis.")


def collect_federated(include_stale=False, seeds=None):
    """One row per federated run directory, skipping runs of an older pipeline.

    `seeds` restricts the set to those seed numbers. Without it every finished
    run is read, which quietly produces an unbalanced table mid-sweep: on
    2026-08-12 `baseline` had five seeds while every other arm had four, and
    since baseline is the reference for every comparison, the reference arm was
    averaged over a different number of runs than the arms it was compared with.
    """
    if not os.path.exists(BASE_FEDERATED_REPORT_DIR):
        print(f"Warning: '{BASE_FEDERATED_REPORT_DIR}' not found.")
        return []

    runs = []
    stale = []
    for run_id in sorted(os.listdir(BASE_FEDERATED_REPORT_DIR)):
        exp_path = os.path.join(BASE_FEDERATED_REPORT_DIR, run_id)
        summary_file = os.path.join(exp_path, "experiment_summary.json")
        if not os.path.isdir(exp_path) or not os.path.exists(summary_file):
            continue

        if seeds is not None and parse_run_id(run_id)[1] not in seeds:
            continue

        summary = _read_json(summary_file, summary_file)
        if summary is None:
            continue
        if not is_current_pipeline(summary) and not include_stale:
            stale.append(run_id)
            continue
        runs.append((summary, run_id, exp_path))

    _report_skipped(stale)

    # Collected first and turned into rows second, because the shared
    # convergence target comes from the baseline runs and so cannot be known
    # while the first row is still being built.
    target = shared_convergence_target(runs)
    return [_federated_row(summary, run_id, exp_path, target)
            for summary, run_id, exp_path in runs]


# --- Centralized runs ------------------------------------------------------

def find_centralized_reports():
    """Seed subdirectories first, then the legacy report at the root."""
    if not os.path.isdir(CENTRALIZED_REPORT_DIR):
        return []

    found = []
    for entry in sorted(os.listdir(CENTRALIZED_REPORT_DIR)):
        sub = os.path.join(CENTRALIZED_REPORT_DIR, entry)
        if os.path.isdir(sub) and os.path.exists(
                os.path.join(sub, "centralized_experiment_summary.json")):
            found.append(sub)

    legacy = os.path.join(CENTRALIZED_REPORT_DIR,
                          "centralized_experiment_summary.json")
    if os.path.exists(legacy):
        found.append(CENTRALIZED_REPORT_DIR)
    return found


def _centralized_bootstrap(report_dir, seed):
    """The per-user scores must describe the checkpoint the summary reports.

    Restricted to the test half like the federated ones, and for the same
    reason: this file holds all 1000 users while `pr_auc` in the summary is the
    test half. The centralized arm is the comparison the whole thesis turns on,
    so an interval that described a different population than the point would
    have mattered more here than anywhere else.
    """
    results_path = os.path.join(report_dir, "centralized_insider_results.csv")
    if not os.path.exists(results_path):
        print(f"  ⚠️  {results_path} missing; no bootstrap interval for this "
              f"centralized run. Re-run train_centralized.py to produce it.")
        return np.nan, np.nan, np.nan, np.nan

    results = pd.read_csv(results_path)
    if not {"is_actual_insider", "max_z_score"} <= set(results.columns):
        return np.nan, np.nan, np.nan, np.nan
    return _bootstrap_frame(results, f"Centralized {report_dir}", seed=seed)


def _centralized_row(summary, report_dir):
    """One table row for one centralized run, in the same columns as a federated one."""
    ad = summary.get("anomaly_detection_metrics", {})
    ci_mean, ci_std, ci_lo, ci_hi = _centralized_bootstrap(
        report_dir, summary.get("seed"))

    row = _empty_row("Centralized", summary.get("seed"), "Centralized")
    row.update({
        "PR-AUC": ad.get("pr_auc"),
        "PR-AUC_Bootstrap_Mean": _round(ci_mean),
        "PR-AUC_Std": _round(ci_std),
        "PR-AUC_CI_Lo": _round(ci_lo),
        "PR-AUC_CI_Hi": _round(ci_hi),
        "Max-F1": ad.get("best_f1"),
        "Balanced_Acc": ad.get("balanced_accuracy_at_best_f1"),
        "Precision": ad.get("precision_at_best_f1"),
        "Recall": ad.get("recall_at_best_f1"),
    })
    return row


def collect_centralized(include_stale=False):
    """One row per centralized run, under the same version rule as the federated ones."""
    reports = find_centralized_reports()
    if not reports:
        print(f"Warning: no centralized summary under '{CENTRALIZED_REPORT_DIR}'. "
              f"Run train_centralized.py first.")
        return []

    rows = []
    stale = []
    for report_dir in reports:
        summary = _read_json(
            os.path.join(report_dir, "centralized_experiment_summary.json"),
            report_dir)
        if summary is None:
            continue
        if not is_current_pipeline(summary) and not include_stale:
            stale.append(report_dir)
            continue
        rows.append(_centralized_row(summary, report_dir))

    _report_skipped(stale)
    return rows


# --- Aggregation -----------------------------------------------------------

def aggregate_across_seeds(df):
    """Mean, sd and n over the runs of each experiment.

    `Runs` counts every row that feeds the mean and sd, including any legacy
    directory that predates seeding and therefore has no seed. `Seeds` counts
    the distinct seeds among them. When the two disagree, an un-seeded legacy
    run is being mixed in and the aggregate should not be trusted.
    """
    records = []
    for experiment, group in df.groupby("Experiment", sort=False):
        record = {"Experiment": experiment,
                  "Type": group["Type"].iloc[0],
                  "Runs": len(group),
                  "Seeds": int(group["Seed"].nunique())}

        for metric in AGG_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = (round(float(values.mean()), 4)
                                        if len(values) else None)
            # ddof=1 needs at least two runs; a single run has no spread.
            record[f"{metric}_sd"] = (round(float(values.std(ddof=1)), 4)
                                      if len(values) > 1 else None)

        upload = pd.to_numeric(group["Upload_MB"], errors="coerce").dropna()
        record["Upload_MB_mean"] = (round(float(upload.mean()), 1)
                                    if len(upload) else None)
        records.append(record)

    return (pd.DataFrame(records)
            .sort_values("PR-AUC_mean", ascending=False)
            .reset_index(drop=True))


# --- Figure ----------------------------------------------------------------

def _plot_frame(df):
    """Runs that have an interval, labelled and ordered for the forest plot."""
    plot_df = df.dropna(subset=["PR-AUC_CI_Lo", "PR-AUC_CI_Hi", "PR-AUC"]).copy()
    if plot_df.empty:
        return plot_df

    plot_df["label"] = [
        f"{r.Experiment} (s{int(r.Seed)})" if pd.notna(r.Seed) else str(r.Experiment)
        for r in plot_df.itertuples()
    ]
    return plot_df.sort_values("PR-AUC")


def _draw_forest(ax, plot_df):
    y = np.arange(len(plot_df))
    point = plot_df["PR-AUC"].to_numpy(dtype=float)
    lo = plot_df["PR-AUC_CI_Lo"].to_numpy(dtype=float)
    hi = plot_df["PR-AUC_CI_Hi"].to_numpy(dtype=float)
    colours = [CENTRALIZED_COLOUR if t == "Centralized" else FEDERATED_COLOUR
               for t in plot_df["Type"]]

    # Drawn as hlines rather than errorbar: the interval is a percentile range,
    # so it is asymmetric about the point estimate, and errorbar takes only one
    # colour for the whole series.
    ax.hlines(y, lo, hi, colors=colours, linewidth=1.6, alpha=0.85)
    cap = 0.22
    ax.vlines(lo, y - cap, y + cap, colors=colours, linewidth=1.2, alpha=0.85)
    ax.vlines(hi, y - cap, y + cap, colors=colours, linewidth=1.2, alpha=0.85)
    ax.scatter(point, y, s=26, c=colours, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("PR-AUC (nokta tahmini ve %95 bootstrap guven araligi)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_bootstrap_intervals(df, output_path=FIGURE_PATH):
    """Forest plot of PR-AUC with its bootstrap confidence interval.

    Each row is one run: the marker is the point estimate, the bar spans the 95%
    interval from resampling users. Reading configurations off this plot answers
    the question a table of point estimates cannot: whether two configurations
    actually differ, or whether their intervals overlap so heavily that the gap
    is noise. That distinction is what the ablation claims rest on.
    """
    plot_df = _plot_frame(df)
    if plot_df.empty:
        print("  No bootstrap intervals available; skipping figure.")
        return None

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.34 * len(plot_df) + 1.2)), dpi=200)
    _draw_forest(ax, plot_df)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stem = os.path.splitext(output_path)[0]
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bootstrap interval figure to '{output_path}' (+ pdf, svg)")
    return output_path


# --- Orchestration ---------------------------------------------------------

def _print_section(title, df):
    print("\n" + "=" * 120)
    print(f"                    {title}")
    print("=" * 120)
    print(df.to_string(index=False))


def _warn_about_data_quality(by_seed):
    """Flag aggregates that cannot support a claim about a difference."""
    single = by_seed[by_seed["Runs"] <= 1]["Experiment"].tolist()
    if single:
        print(f"\n⚠️  Only one run for: {', '.join(single)}. No spread can be "
              f"reported for these; run more seeds before claiming a difference.")

    mixed = by_seed[by_seed["Runs"] != by_seed["Seeds"]]["Experiment"].tolist()
    if mixed:
        print(f"\n⚠️  Un-seeded legacy runs mixed into: {', '.join(mixed)}. "
              f"Remove the legacy report directories or re-run these.")


def compare_experiments(include_stale=False, seeds=None, figure_path=FIGURE_PATH):
    """Build both comparison tables and the forest plot from everything on disk.

    `figure_path` is configurable because the default writes into the results
    tree, and a caller regenerating a thesis figure may want it somewhere else.
    """
    print("📊 Comparing experiment results (federated & centralized)...")
    if seeds is not None:
        print(f"   Restricted to seeds {sorted(seeds)}.")

    rows = (collect_federated(include_stale, seeds)
            + collect_centralized(include_stale))
    if not rows:
        print("No experiment summaries found to compare.")
        return

    comparison_df = (pd.DataFrame(rows)
                     .sort_values(by="PR-AUC", ascending=False)
                     .reset_index(drop=True))

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    _print_section("PER-RUN RESULTS (experiment x seed)", comparison_df)
    comparison_df.to_csv("experiment_comparison_summary.csv", index=False)
    print("\nSaved 'experiment_comparison_summary.csv'")

    by_seed = aggregate_across_seeds(comparison_df)
    _print_section("AGGREGATED ACROSS SEEDS (mean ± sd)", by_seed)
    _warn_about_data_quality(by_seed)

    by_seed.to_csv("experiment_comparison_by_seed.csv", index=False)
    print("Saved 'experiment_comparison_by_seed.csv'")

    plot_bootstrap_intervals(comparison_df, figure_path)


def _seeds_from_argv(argv):
    """`--seeds 1,2,3,4` or `--seeds=1,2,3,4`, or None for everything on disk."""
    for i, arg in enumerate(argv):
        raw = None
        if arg.startswith("--seeds="):
            raw = arg.split("=", 1)[1]
        elif arg == "--seeds" and i + 1 < len(argv):
            raw = argv[i + 1]
        if raw is not None:
            return {int(s) for s in raw.replace(" ", "").split(",") if s}
    return None


def _figure_from_argv(argv):
    """`--figure PATH` or `--figure=PATH`, else the configured default."""
    for i, arg in enumerate(argv):
        if arg.startswith("--figure="):
            return arg.split("=", 1)[1]
        if arg == "--figure" and i + 1 < len(argv):
            return argv[i + 1]
    return FIGURE_PATH


if __name__ == "__main__":
    compare_experiments(include_stale="--include-stale" in sys.argv,
                        seeds=_seeds_from_argv(sys.argv[1:]),
                        figure_path=_figure_from_argv(sys.argv[1:]))
