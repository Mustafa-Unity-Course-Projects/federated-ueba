"""Detection broken down by CERT insider scenario.

The `insider` column of the processed data is not a flag, it is the scenario
number: 1, 2 or 3, with 30, 30 and 10 users. Everything downstream binarises it
with `!= 0`, correctly, and reports one number over all 70. That number hides
which kind of insider the pipeline finds, and the three kinds are not variations
of one behaviour:

  1  removable media and uploads after hours, over a sustained period
  2  a job seeker gathering data for a competitor
  3  a system administrator sabotaging after being let go

The temporal persistence stage takes the mean of a user's top three windows, so
it rewards behaviour that repeats. That should suit scenarios 1 and 2 and work
against 3 if its damage is concentrated in a short burst. This module measures
whether it does.

Per scenario it reports the ranking quality against the same population of
normal users, so the three numbers answer "how well does this pipeline separate
*this kind* of insider from everyone who is not an insider at all".

Reads the per-user CSVs the runs already wrote; nothing is re-scored, and no
experiment has to be re-run.

    python analysis/scenario_breakdown.py
    python analysis/scenario_breakdown.py --runs baseline delta-0.1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from federated_ueba import scoring

FEDERATED_DIR = "federated_evaluation_reports"
CENTRALIZED_DIR = "centralized_evaluation_reports"

SCENARIO_NAMES = {
    1: "1 - removable media, after hours",
    2: "2 - job seeker, data gathering",
    3: "3 - admin sabotage after dismissal",
}


def scenario_by_user(data_path):
    """Each user's scenario number, 0 for the 930 who are not insiders."""
    df = pd.read_csv(data_path, low_memory=False,
                     usecols=["user", "insider"])
    return df.groupby("user")["insider"].max().astype(int)


def load_federated(run_id, best_round):
    path = os.path.join(FEDERATED_DIR, run_id, "round_by_round_results",
                        f"round_{best_round}_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def summary_of(run_id):
    path = os.path.join(FEDERATED_DIR, run_id, "experiment_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def reported_round(run_id):
    """The round this run is read at, and the threshold that goes with it.

    Not `best_round` from the summary, which still selects over round 0 and
    picks it for 21 of the 63 finished runs. Round 0 is the model before any
    training, scored against a reference the clients measured with their trained
    local models, so a scenario table built on it describes the scoring pipeline
    rather than the federated model.
    """
    path = os.path.join(FEDERATED_DIR, run_id, "federated_rounds_comparison.csv")
    if not os.path.exists(path):
        return None, None
    frame = pd.read_csv(path)
    frame = frame[frame["Round"] > 0]
    if frame.empty:
        return None, None
    row = frame.loc[frame["PR-AUC_val"].idxmax()]
    return int(row["Round"]), float(row["Optimal-Threshold"])


def per_scenario(results, scenarios, threshold=None):
    """One row per scenario, ranked against the users who are not insiders.

    PR-AUC is computed with only this scenario's insiders as positives and every
    non-insider as negative. Insiders of the *other* scenarios are dropped rather
    than counted as negatives: calling a real insider a false positive would
    punish the pipeline for being right.
    """
    merged = results.merge(scenarios.rename("scenario"), left_on="user",
                           right_index=True, how="inner")
    normals = merged[merged["scenario"] == 0]

    rows = []
    for scenario in sorted(s for s in merged["scenario"].unique() if s > 0):
        group = merged[merged["scenario"] == scenario]
        subset = pd.concat([group, normals])
        y = (subset["scenario"] > 0).astype(int).to_numpy()
        s = subset["max_z_score"].to_numpy()

        # Where this scenario's insiders sit in the ranking of everyone, as a
        # percentile: 100 means top of the list.
        order = merged["max_z_score"].rank(pct=True) * 100
        percentiles = order[merged["scenario"] == scenario]

        row = {
            "scenario": SCENARIO_NAMES.get(scenario, str(scenario)),
            "insiders": len(group),
            "PR-AUC": round(float(average_precision_score(y, s)), 4),
            "median_percentile": round(float(percentiles.median()), 1),
            "worst_percentile": round(float(percentiles.min()), 1),
        }
        if threshold is not None:
            caught = int((group["max_z_score"] >= threshold).sum())
            row["caught"] = f"{caught}/{len(group)}"
            row["recall"] = round(caught / len(group), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def report(label, results, scenarios, threshold):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    if threshold is not None:
        print(f"operating threshold {threshold:.3f} "
              f"(chosen on the validation half, as reported)")
    table = per_scenario(results, scenarios, threshold)
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=["baseline"],
                        help="federated experiment names (seed 1)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    from config_manager import config
    data_path = args.data or config.get("data", "processed_data_path")
    scenarios = scenario_by_user(data_path)

    counts = scenarios[scenarios > 0].value_counts().sort_index()
    print("Insider users per scenario in the processed data:")
    for scenario, count in counts.items():
        print(f"  {SCENARIO_NAMES.get(scenario, scenario)}: {count}")
    print(f"  total: {int(counts.sum())} insiders among "
          f"{len(scenarios)} users")

    # Centralized first: it is the strongest model, so it shows what the
    # scenarios cost when the model is not the limiting factor.
    central = os.path.join(CENTRALIZED_DIR, f"seed{args.seed}",
                           "centralized_insider_results.csv")
    if os.path.exists(central):
        summary_path = os.path.join(CENTRALIZED_DIR, f"seed{args.seed}",
                                    "centralized_experiment_summary.json")
        threshold = None
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                threshold = json.load(f).get(
                    "anomaly_detection_metrics", {}).get("optimal_threshold")
        report("Centralized model", pd.read_csv(central), scenarios, threshold)

    for name in args.runs:
        run_id = f"{name}__seed{args.seed}"
        summary = summary_of(run_id)
        if summary is None:
            print(f"\n{run_id}: no summary; skipped.")
            continue
        round_number, threshold = reported_round(run_id)
        if round_number is None:
            print(f"\n{run_id}: no round table; skipped.")
            continue
        results = load_federated(run_id, round_number)
        if results is None:
            print(f"\n{run_id}: no per-user results; skipped.")
            continue
        report(f"Federated: {name} (seed {args.seed}, round {round_number})",
               results, scenarios, threshold)


if __name__ == "__main__":
    main()
