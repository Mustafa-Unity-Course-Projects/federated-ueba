"""Which uncertainty dominates a reported PR-AUC: the users, or the seed.

`compare_experiments.py` already bootstraps each run by resampling users. That
answers "how much does this number depend on which 1000 people we evaluated on",
and it is the interval J17 asks for. It cannot answer the question that decides
whether one configuration beats another, because it holds the training run fixed.

Two runs of the same configuration differ for reasons that have nothing to do
with the evaluation set: which 25 of 50 clients each round sampled, and how the
weights were initialised. The baseline measured that at sd 0.0083 across five
seeds. The sparsified arms looked far wider on the first two seeds, and if that
holds it is the more important number of the two.

So this reports three figures per experiment:

  within-run     mean over seeds of the user-resampling sd. Evaluation-set noise.
  across-seed    sd of the per-seed point estimates. Training noise.
  hierarchical   resample seeds with replacement, then users within the drawn
                 seed. Both sources at once, and the interval a claim like
                 "top-k 0.1 costs nothing" has to survive.

It also reports each experiment at two round choices, because the choice is not
innocent (see Y5/Y9 in ieee/juri_duzeltme_listesi.md):

  own-best       each seed evaluated at its own argmax round. What we report
                 today, and optimistic: the argmax is partly noise, and picking
                 it per seed picks that noise five times.
  shared-best    one round chosen for the whole experiment, the one maximising
                 the mean across seeds. The selection happens once instead of
                 five times, so it cannot inflate the spread.

Usage:

    python analysis/seed_variance.py                 every experiment on disk
    python analysis/seed_variance.py --only baseline,top-k-0.1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import PIPELINE_VERSION, config  # noqa: E402

REPORTS_DIR = "federated_evaluation_reports"
OUTPUT_CSV = "generated_visuals/seed_variance.csv"


def parse_run_id(run_id):
    """`baseline__seed3` into `("baseline", 3)`."""
    experiment, _, seed = run_id.partition("__seed")
    return experiment, int(seed) if seed.isdigit() else None


def load_runs(only=None, seeds=None, version=PIPELINE_VERSION):
    """Every finished current-pipeline run on disk, grouped by experiment name.

    Unfinished runs are skipped rather than half-read: a run without `done` may
    have a partial round_by_round directory, and averaging over it would quietly
    mix a 12-round run into a table describing 50-round ones.

    `seeds` restricts to a set of run seeds. A partly finished sweep otherwise
    gives one experiment four seeds and the rest three, and an arm averaged over
    more runs than the arm it is compared against is not a paired comparison.

    `version` drops runs scored by an older pipeline. Without it the directory
    listing is the definition of the experiment set, and dropping an arm from
    the configuration does not drop its directories: after the v8 flip the five
    stale `ablation-no-diversity` runs were still read, and because that v7 arm
    and the v8 default are the same three stages, the diversity row compared an
    arm against itself and reported the multiplier as making no difference.
    `compare_experiments.py` already refused those runs by name; this is the
    same guard one layer down. Pass `version=None` to read everything.
    """
    runs = {}
    skipped = set()
    for path in sorted(glob.glob(f"{REPORTS_DIR}/*/experiment_summary.json")):
        with open(path) as f:
            summary = json.load(f)
        if not summary.get("done"):
            continue
        experiment, seed = parse_run_id(summary.get("run_id", ""))
        if only and experiment not in only:
            continue
        if seeds and seed not in seeds:
            continue
        if version is not None and summary.get("pipeline_version") != version:
            skipped.add(experiment)
            continue
        runs.setdefault(experiment, []).append(
            (summary, os.path.dirname(path)))
    if skipped:
        print(f"!  Atlandi (pipeline surumu {version} degil): "
              f"{', '.join(sorted(skipped))}")
    return runs


def round_scores(exp_dir, round_number):
    """Per-user scores and labels for one round, or None if not scored."""
    path = os.path.join(exp_dir, "round_by_round_results",
                        f"round_{round_number}_results.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    return (frame["is_actual_insider"].to_numpy(),
            frame["max_z_score"].to_numpy())


def available_rounds(exp_dir):
    """Round numbers this run actually scored, in order.

    Round 0 is excluded everywhere in this module. It is the model before any
    training, but it is scored against the reference error distribution the
    clients measured with their *trained* local models, so it is not a round the
    run ever passed through. Left in, it wins the argmax outright: 21 of the 63
    finished runs record `best_round = 0`, including all three baseline seeds,
    which would have put an untrained model on the reference side of every
    comparison below.
    """
    pattern = os.path.join(exp_dir, "round_by_round_results", "round_*_results.csv")
    rounds = []
    for path in glob.glob(pattern):
        stem = os.path.basename(path)
        number = int(stem[len("round_"):-len("_results.csv")])
        if number > 0:
            rounds.append(number)
    return sorted(rounds)


def plateau_frames(exp_dir, window=None):
    """Per-user scores for each round the reported metric averages over.

    This is the statistic the thesis reports (Y5): the mean over the last
    `plateau_window_rounds` rounds, not the value at one selected round. A run is
    therefore represented here by several frames rather than one, and every
    consumer below averages across them.
    """
    rounds = available_rounds(exp_dir)
    if not rounds:
        return []
    if window is None:
        window = config.get("evaluation", "plateau_window_rounds")
    cutoff = max(rounds) - window
    frames = []
    for round_number in rounds:
        if round_number <= cutoff:
            continue
        scored = round_scores(exp_dir, round_number)
        if scored is not None:
            frames.append(scored)
    return frames


def plateau_pr_auc(frames):
    """A run's reported number: mean PR-AUC over its plateau rounds."""
    if not frames:
        return float("nan")
    return float(np.mean([pr_auc(labels, scores) for labels, scores in frames]))


def own_best_round(exp_dir):
    """The argmax round on the validation half, round 0 excluded.

    Recomputed rather than read from `experiment_summary.json`, whose
    `best_round` still selects over round 0. Kept only so the plateau metric can
    be compared against what a per-seed argmax would have reported.
    """
    path = os.path.join(exp_dir, "federated_rounds_comparison.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["Round"] > 0]
    if frame.empty or "PR-AUC_val" not in frame.columns:
        return None
    return int(frame.loc[frame["PR-AUC_val"].idxmax(), "Round"])


def pr_auc(labels, scores):
    """PR-AUC over every evaluated user, insiders and non-insiders alike.

    Deliberately the whole population rather than one half of the user split.
    The two halves are complementary draws from 35 insiders, so a half that gets
    the hard cases makes the other half easy; measured across five baseline
    seeds their correlation is -0.955 and each half carries sd ~0.048 while the
    union carries 0.008. The union is the stable quantity.
    """
    return float(average_precision_score(labels, scores))


def shared_best_round(runs):
    """The round maximising the mean PR-AUC across seeds.

    Chosen once for the experiment, so unlike a per-seed argmax it cannot borrow
    a different seed's noise for every row. Only rounds that every seed scored
    are eligible; a round missing from one seed would otherwise be compared on a
    smaller set of runs than the rest.
    """
    per_seed = []
    common = None
    for _, exp_dir in runs:
        rounds = available_rounds(exp_dir)
        common = set(rounds) if common is None else common & set(rounds)
    if not common:
        return None

    for round_number in sorted(common):
        column = []
        for _, exp_dir in runs:
            scored = round_scores(exp_dir, round_number)
            if scored is None:
                column = []
                break
            column.append(pr_auc(*scored))
        if column:
            per_seed.append((float(np.mean(column)), round_number))
    return max(per_seed)[1] if per_seed else None


def bootstrap_users(labels, scores, iterations, rng):
    """PR-AUC over `iterations` user resamples of one run."""
    values = []
    n = len(labels)
    for _ in range(iterations):
        index = rng.randint(0, n, n)
        if len(np.unique(labels[index])) < 2:
            continue
        values.append(average_precision_score(labels[index], scores[index]))
    return np.asarray(values)


def hierarchical_bootstrap(per_seed_data, iterations, rng):
    """Resample seeds with replacement, then users within the seed drawn.

    Two-level because the two sources compound: a claim about a configuration
    has to survive both a different draw of clients and a different draw of
    users. Resampling only users would report an interval that shrinks with the
    evaluation set while the real spread across reruns stays where it was.
    """
    values = []
    seeds = len(per_seed_data)
    for _ in range(iterations):
        labels, scores = per_seed_data[rng.randint(0, seeds)]
        index = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        values.append(average_precision_score(labels[index], scores[index]))
    return np.asarray(values)


def paired_difference(runs_a, runs_b, iterations, confidence, seed=42):
    """Bootstrap the difference between two configurations, users paired.

    The absolute interval on a single configuration is wide, because PR-AUC over
    1000 users rests on 35 insiders and resampling them moves the number by about
    0.055. Reading that interval as "nothing can be distinguished" would be
    wrong: both configurations are scored on the *same* people, so a resample
    that happens to draw the hard insiders drags both arms down together. That
    part of the noise is common and cancels in the difference.

    So the user index is drawn once per iteration and applied to both arms, while
    the seed is drawn independently for each: two training runs really are
    independent, and pairing them would hide the training noise this whole
    analysis exists to expose.

    Returns the mean difference, its interval, and the share of iterations that
    landed on the same side as the mean. That last number is the honest answer to
    "is A better than B", and it is not a p-value.
    """
    rng = np.random.RandomState(seed)
    tail = (1.0 - confidence) / 2.0 * 100.0
    differences = []

    def resampled_mean(frames, index):
        """The run's plateau statistic on one draw of users."""
        values = []
        for labels, scores in frames:
            drawn = labels[index]
            # A resample holding no insider has no precision-recall curve.
            # Skipped rather than counted as zero, as everywhere else.
            if len(np.unique(drawn)) < 2:
                continue
            values.append(average_precision_score(drawn, scores[index]))
        return float(np.mean(values)) if values else None

    for _ in range(iterations):
        frames_a = runs_a[rng.randint(0, len(runs_a))]
        frames_b = runs_b[rng.randint(0, len(runs_b))]
        # Same users for both arms, and the same users across the plateau rounds
        # within an arm. The two runs scored the same population, so the index is
        # meaningful in both; a run scored on a different user set would make
        # this comparison invalid and is rejected by the caller.
        n = len(frames_a[0][0])
        index = rng.randint(0, n, n)
        mean_a = resampled_mean(frames_a, index)
        mean_b = resampled_mean(frames_b, index)
        if mean_a is None or mean_b is None:
            continue
        differences.append(mean_a - mean_b)

    differences = np.asarray(differences)
    mean = float(differences.mean())
    same_side = float((np.sign(differences) == np.sign(mean)).mean())
    return {
        "Mean_Difference": round(mean, 4),
        "CI_Lo": round(float(np.percentile(differences, tail)), 4),
        "CI_Hi": round(float(np.percentile(differences, 100.0 - tail)), 4),
        "Share_Same_Sign": round(same_side, 3),
    }


def compare_pairs(runs, pairs, iterations, confidence):
    """Paired difference for each requested `A:B` comparison."""
    rows = []
    for pair in pairs:
        name_a, _, name_b = pair.partition(":")
        missing = [n for n in (name_a, name_b) if n not in runs]
        if missing:
            print(f"  {pair}: no finished runs for {missing}, skipped")
            continue

        collected = {}
        for name in (name_a, name_b):
            scored = [plateau_frames(exp_dir) for _, exp_dir in runs[name]]
            collected[name] = [frames for frames in scored if frames]

        sizes = {len(labels) for arm in collected.values()
                 for frames in arm for labels, _ in frames}
        if len(sizes) != 1:
            print(f"  {pair}: arms scored different numbers of users {sizes}; "
                  f"a paired comparison would be meaningless, skipped")
            continue

        thin = [name for name, arm in collected.items() if len(arm) < 2]
        if thin:
            # Drawing seeds with replacement from a single run always draws that
            # run, so the interval collapses onto evaluation noise and reports a
            # confidence the evidence does not support.
            print(f"  ⚠️  {pair}: {thin} has one finished seed; its training "
                  f"noise is not in this interval.")

        row = {"Comparison": f"{name_a} - {name_b}"}
        row.update(paired_difference(collected[name_a], collected[name_b],
                                     iterations, confidence))
        rows.append(row)
    return rows


def summarise(experiment, runs, iterations, confidence, seed=42):
    """One row per (experiment, round choice)."""
    rng = np.random.RandomState(seed)
    tail = (1.0 - confidence) / 2.0 * 100.0
    rows = []

    # "plateau" is what the thesis reports; the other two are kept as the Y5
    # sensitivity analysis, showing what a round-selection rule would have bought.
    choices = {"plateau": "plateau",
               "own-best": None,
               "shared-best": shared_best_round(runs)}
    for choice, fixed_round in choices.items():
        collected = []
        single = []
        for summary, exp_dir in runs:
            if choice == "plateau":
                frames = plateau_frames(exp_dir)
                if frames:
                    collected.append(frames)
                    single.append(frames[-1])
                continue
            round_number = (fixed_round if fixed_round is not None
                            else own_best_round(exp_dir))
            scored = round_scores(exp_dir, round_number) if round_number else None
            if scored is not None:
                collected.append([scored])
                single.append(scored)

        if len(collected) < 2:
            continue

        points = np.array([plateau_pr_auc(frames) for frames in collected])
        # Evaluation noise is read from one frame per run rather than the whole
        # window: the plateau mean already averages the rounds, so bootstrapping
        # every round would report a narrower spread than any single reported
        # number has.
        within = np.mean([bootstrap_users(labels, scores, iterations, rng).std()
                          for labels, scores in single])
        combined = hierarchical_bootstrap(single, iterations, rng)

        rows.append({
            "Experiment": experiment,
            "Round_Choice": choice,
            "Round": fixed_round if fixed_round is not None else "per-seed",
            "Seeds": len(collected),
            "Mean_PR-AUC": round(float(points.mean()), 4),
            "Across_Seed_SD": round(float(points.std(ddof=1)), 4),
            "Within_Run_SD": round(float(within), 4),
            "Hier_CI_Lo": round(float(np.percentile(combined, tail)), 4),
            "Hier_CI_Hi": round(float(np.percentile(combined, 100.0 - tail)), 4),
            "Min": round(float(points.min()), 4),
            "Max": round(float(points.max()), 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="",
                        help="comma separated experiment names")
    parser.add_argument("--compare", default="",
                        help="comma separated A:B pairs, e.g. "
                             "delta-0.1:top-k-0.1")
    parser.add_argument("--seeds", default="",
                        help="comma separated run seeds, e.g. 1,2,3")
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    only = {name.strip() for name in args.only.split(",") if name.strip()}
    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
    iterations = config.get("evaluation", "bootstrap_iterations")
    confidence = config.get("evaluation", "bootstrap_confidence")

    runs = load_runs(only or None, seeds or None)
    if not runs:
        raise SystemExit("No finished runs found. Nothing to compare.")

    rows = []
    for experiment in sorted(runs):
        if len(runs[experiment]) < 2:
            print(f"  {experiment}: 1 seed only, skipped "
                  f"(variance needs at least two runs)")
            continue
        rows.extend(summarise(experiment, runs[experiment], iterations, confidence))

    if not rows:
        raise SystemExit("Every experiment had fewer than two finished seeds.")

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved to '{args.output}'")

    own = table[table["Round_Choice"] == "own-best"]["Mean_PR-AUC"].mean()
    shared = table[table["Round_Choice"] == "shared-best"]["Mean_PR-AUC"].mean()
    plateau = table[table["Round_Choice"] == "plateau"]["Mean_PR-AUC"].mean()
    print(f"\nRound selection is worth {own - shared:+.4f} PR-AUC against a "
          f"shared round and {own - plateau:+.4f} against the reported plateau "
          f"mean. That is the optimism a per-seed argmax buys.")

    pairs = [p.strip() for p in args.compare.split(",") if p.strip()]
    if pairs:
        comparisons = compare_pairs(runs, pairs, iterations, confidence)
        if comparisons:
            print("\nPaired differences (same users in both arms, seeds drawn "
                  "independently):")
            print(pd.DataFrame(comparisons).to_string(index=False))
            print("\nShare_Same_Sign is the fraction of resamples agreeing with "
                  "the mean's direction. An interval spanning zero means the "
                  "two configurations are not separated by this evidence.")


if __name__ == "__main__":
    main()
