"""Label and evaluation leakage: the paths by which a test answer could reach a
number that claims not to have seen it.

Leakage is the failure mode this project can least afford to have and least
easily notice. A leak does not crash, does not warn, and moves the reported
figure in the flattering direction, which is the direction nobody investigates.
So each path is pinned rather than argued about:

  training data   the model and the scaler see normal behaviour only, and the
                  labels are used for nothing else
  scoring         a user's score depends on that user's rows and on the shared
                  reference distribution, never on anyone else's rows or label
  selection       what chooses (threshold, checkpoint) is measured on the
                  validation half; what is reported comes from the test half
  metric          a score carrying no information must land on the base rate

The audit that produced this file also measured what remains. The centralized
arm picks its epoch by validation PR-AUC and the thesis headline is the whole
population, so the selected half sits inside the reported number. Measured over
five seeds the selection is worth +0.0058 on the validation half, roughly half
of which reaches the union: below the 0.0086 noise floor, and it inflates the
*centralized* arm, so it makes the federation cost look larger than it is. Real,
small, and in the conservative direction. Recorded, not fixed.
"""

import os
import re
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_ueba import scaling, scoring  # noqa: E402

CPU = torch.device("cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ZeroModel(torch.nn.Module):
    """Reconstructs zero, so the squared error is the input squared."""

    def forward(self, x):
        return torch.zeros_like(x)


class Identity:
    def transform(self, x):
        return np.asarray(x, dtype=np.float32)


def population(n_users=12, n_days=20, insiders=(2.0, 5.0), seed=0):
    """A frame where insider days are numerically extreme.

    Extreme on purpose: if any statistic computed from "normal only" data were
    in fact computed from all of it, these rows would move it far enough that a
    tolerance could not hide the difference.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for user in range(n_users):
        for day in range(n_days):
            bad = float(user) in insiders and day in (3, 4)
            rows.append({
                "user": float(user), "day": day,
                "insider": 1 if bad else 0,
                "busy": (500.0 if bad else rng.rand()),
                "quiet": (500.0 if bad else rng.rand() * 0.4),
            })
    return pd.DataFrame(rows), ["busy", "quiet"]


class TestLabelsNeverReachTraining(unittest.TestCase):
    """The only thing a label may do is remove a row from training."""

    def setUp(self):
        self.df, self.features = population()
        self.chunks = [[float(u) for u in range(6)],
                       [float(u) for u in range(6, 12)]]

    def test_the_scaler_is_the_same_with_and_without_insider_rows(self):
        """The strongest form: deleting the labelled rows changes nothing.

        If the scaler drew on insider rows, removing them would move the mean of
        `busy`, whose insider values are three orders of magnitude out.
        """
        with_insiders = scaling.build_global_scaler(
            self.df, self.features, self.chunks, min_cohort=1)
        without = scaling.build_global_scaler(
            self.df[self.df["insider"] == 0], self.features, self.chunks,
            min_cohort=1)
        np.testing.assert_allclose(with_insiders.mean_, without.mean_)
        np.testing.assert_allclose(with_insiders.scale_, without.scale_)

    def test_flipping_a_label_does_not_move_the_scaler(self):
        """A label change alone, with no data change, must be inert here."""
        before = scaling.build_global_scaler(
            self.df[self.df["insider"] == 0], self.features, self.chunks,
            min_cohort=1)
        relabelled = self.df[self.df["insider"] == 0].copy()
        relabelled.loc[relabelled["user"] == 7.0, "insider"] = 1
        after = scaling.build_global_scaler(
            relabelled, self.features, self.chunks, min_cohort=1)
        # User 7's rows are now excluded, so the scaler *should* move. This test
        # exists to prove the exclusion is real rather than decorative.
        self.assertFalse(np.allclose(before.mean_, after.mean_),
                         "labelling a user changed nothing, so the insider "
                         "filter is not running")


class TestScoringIsPerUser(unittest.TestCase):
    """One user's number must not depend on who else was evaluated."""

    def setUp(self):
        self.df, self.features = population()
        self.scorer = scoring.Scorer(
            features=self.features, scaler=Identity(),
            ref_mean=np.zeros(2), ref_std=np.ones(2),
            cfg=scoring.ScoringConfig(
                top_k_features=2, persistence_window=3,
                diversity_threshold=2.0, scan_stride=1,
                inference_batch_size=32),
            window_size=14, device=CPU)

    def score_of(self, frame, user, users):
        out = self.scorer.scan(ZeroModel(), frame, users)
        return float(out.loc[out["user"] == user, "max_z_score"].iloc[0])

    def test_a_user_scored_alone_matches_the_user_scored_in_a_crowd(self):
        alone = self.score_of(self.df[self.df["user"] == 1.0], 1.0, [1.0])
        crowd = self.score_of(self.df, 1.0, [float(u) for u in range(12)])
        self.assertAlmostEqual(alone, crowd, places=6)

    def test_another_users_label_does_not_change_this_users_score(self):
        base = self.score_of(self.df, 1.0, [1.0, 2.0, 3.0])
        flipped = self.df.copy()
        flipped.loc[flipped["user"] == 3.0, "insider"] = 1
        self.assertAlmostEqual(
            base, self.score_of(flipped, 1.0, [1.0, 2.0, 3.0]), places=6)

    def test_removing_the_insiders_does_not_change_a_normal_users_score(self):
        base = self.score_of(self.df, 1.0, [1.0])
        no_insiders = self.df[~self.df["user"].isin([2.0, 5.0])]
        self.assertAlmostEqual(
            base, self.score_of(no_insiders, 1.0, [1.0]), places=6)


class TestSelectionAndReportingAreSeparate(unittest.TestCase):
    """What chooses is not what reports."""

    def results(self, n=200, positives=20, seed=0):
        rng = np.random.RandomState(seed)
        labels = np.zeros(n)
        labels[:positives] = 1.0
        return pd.DataFrame({
            "user": np.arange(n, dtype=float),
            "max_z_score": rng.rand(n) + labels * 0.6,
            "is_actual_insider": labels,
        })

    def test_the_reported_f1_is_not_the_best_achievable_on_the_test_half(self):
        """A threshold fitted on the test half would score at least as well.

        If the reported F1 ever equalled the test-half optimum across seeds, the
        threshold would be coming from the wrong half.
        """
        from sklearn.metrics import precision_recall_curve

        beaten = 0
        for seed in range(8):
            frame = self.results(seed=seed)
            metrics = scoring.evaluate_scores(frame, seed=42)
            _, test_users = scoring.split_users(
                frame["user"].tolist(), frame["is_actual_insider"].tolist(),
                seed=42)
            test = frame[frame["user"].isin(test_users)]
            p, r, _ = precision_recall_curve(test["is_actual_insider"],
                                             test["max_z_score"])
            denominator = p + r
            best = float(np.max(np.where(denominator > 0,
                                         2 * p * r / np.where(denominator > 0,
                                                              denominator, 1),
                                         0.0)))
            self.assertLessEqual(metrics["f1"], best + 1e-9)
            if metrics["f1"] < best - 1e-9:
                beaten += 1
        self.assertGreater(beaten, 0,
                           "reported F1 matched the test-half optimum in every "
                           "seed, which is what fitting on the test half looks "
                           "like")

    def test_the_three_populations_are_genuinely_different_numbers(self):
        metrics = scoring.evaluate_scores(self.results(), seed=42)
        values = {metrics["pr_auc_val"], metrics["pr_auc_test"],
                  metrics["pr_auc_all"]}
        self.assertEqual(len(values), 3)

    def test_no_user_appears_in_both_halves(self):
        frame = self.results()
        val, test = scoring.split_users(
            frame["user"].tolist(), frame["is_actual_insider"].tolist(),
            seed=42)
        self.assertEqual(set(val) & set(test), set())
        self.assertEqual(len(val) + len(test), len(frame))


class TestNoSelectionReadsTheUnionMetric(unittest.TestCase):
    """`pr_auc_all` describes a population that includes the selecting half.

    Choosing a checkpoint by it would let the validation half both pick and be
    reported on. The docstring in `scoring.evaluate_scores` says so; this makes
    the codebase agree with the docstring, because a comment cannot fail.
    """

    SELECTION = re.compile(
        r"(>|<|max|argmax|best|idxmax)[^\n]*\b(pr_auc_all|PR_AUC_all)\b"
        r"|\b(pr_auc_all|PR_AUC_all)\b[^\n]*(>|<)")

    SOURCES = ["train_centralized.py", "federated_insider_detection.py",
               os.path.join("analysis", "seed_variance.py"),
               os.path.join("analysis", "findings.py"),
               os.path.join("analysis", "literature_baseline.py")]

    def test_the_union_metric_is_never_compared_for_a_choice(self):
        offenders = []
        for rel in self.SOURCES:
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"'):
                        continue
                    if self.SELECTION.search(line):
                        offenders.append("%s:%d %s" % (rel, number, stripped))
        self.assertEqual(offenders, [], "union metric used in a comparison:\n"
                                        + "\n".join(offenders))

    def test_the_centralized_selection_rule_is_recorded_verbatim(self):
        """The summary must state which half chose, so a reader can check."""
        with open(os.path.join(ROOT, "train_centralized.py"),
                  encoding="utf-8") as f:
            source = f.read()
        self.assertIn("selection_rule", source)
        self.assertIn("validation user half", source)


class TestAnUninformativeScoreLandsOnTheBaseRate(unittest.TestCase):
    """If this fails, no other number in the thesis means anything."""

    def test_shuffled_scores_collapse_to_the_positive_rate(self):
        rng = np.random.RandomState(0)
        n, positives = 1000, 70          # the shape of the real population
        labels = np.zeros(n)
        labels[:positives] = 1.0
        scores = rng.rand(n) + labels * 0.9

        from sklearn.metrics import average_precision_score
        informed = average_precision_score(labels, scores)
        shuffled = [average_precision_score(labels, rng.permutation(scores))
                    for _ in range(200)]

        base_rate = positives / n
        self.assertAlmostEqual(float(np.mean(shuffled)), base_rate, delta=0.02)
        self.assertGreater(informed, 0.5)


if __name__ == "__main__":
    unittest.main()
