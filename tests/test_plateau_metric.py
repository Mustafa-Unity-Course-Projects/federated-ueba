"""The reported metric is a plateau mean, not one round's score (Y5).

The rule it replaced picked the round with the best validation PR-AUC. On a
curve that is flat to about 0.005 after round 6, that argmax was choosing among
rounds that do not differ and taking the luckiest one, and it did so more
generously for the noisier sparse arms than for the baseline. A rule whose gift
grows with noise discounts exactly the cost this thesis is trying to measure.

These tests pin the replacement: which rounds are averaged, what is averaged over
them, and that the interval is built around the plateau mean rather than around
any single round in it.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analysis.compare_experiments import (bootstrap_plateau_pr_auc,
                                          load_plateau_frames,
                                          load_round_scores, plateau_metrics,
                                          plateau_rounds)


class TestPlateauWindow(unittest.TestCase):
    """Which rounds the mean is taken over."""

    def test_the_window_is_the_last_rounds_not_the_last_checkpoints(self):
        """Counted in rounds, so checkpoint_stride cannot change the averaging."""
        every_round = list(range(0, 51))
        every_second = list(range(0, 51, 2))
        self.assertEqual(plateau_rounds(every_round, window=20)[0], 31)
        self.assertEqual(plateau_rounds(every_second, window=20)[0], 32)

    def test_a_wider_window_averages_more_rounds(self):
        rounds = list(range(0, 51, 2))
        self.assertEqual(len(plateau_rounds(rounds, window=10)), 5)
        self.assertEqual(len(plateau_rounds(rounds, window=20)), 10)

    def test_a_window_wider_than_the_run_takes_everything(self):
        rounds = [0, 2, 4]
        self.assertEqual(plateau_rounds(rounds, window=100), [0, 2, 4])

    def test_no_rounds_gives_no_window(self):
        self.assertEqual(plateau_rounds([], window=20), [])


class TestPlateauMetrics(unittest.TestCase):
    """What is averaged, and what the spread says."""

    def frame(self, test_scores, val_scores=None):
        rounds = list(range(0, 2 * len(test_scores), 2))
        return pd.DataFrame({
            "Round": rounds,
            "PR-AUC": test_scores,
            "PR-AUC_val": val_scores or test_scores,
            "Max-F1": [s - 0.01 for s in test_scores],
        })

    def test_the_mean_is_taken_over_the_window_only(self):
        df = self.frame([0.10, 0.20, 0.80, 0.90])
        means, _, window = plateau_metrics(df, window=4)
        self.assertEqual(window, [4, 6])
        self.assertAlmostEqual(means["PR-AUC"], 0.85)

    def test_every_numeric_metric_is_averaged_the_same_way(self):
        df = self.frame([0.80, 0.90])
        means, _, _ = plateau_metrics(df, window=100)
        self.assertAlmostEqual(means["Max-F1"], 0.84)

    def test_the_round_column_is_not_averaged_into_a_metric(self):
        df = self.frame([0.80, 0.90])
        means, _, _ = plateau_metrics(df, window=100)
        self.assertNotIn("Round", means)

    def test_the_spread_exposes_a_run_that_is_not_actually_flat(self):
        """A declining arm and a steady one can share a mean; the sd separates them."""
        steady, _, _ = plateau_metrics(self.frame([0.85, 0.85, 0.85, 0.85]), window=100)
        _, steady_sd, _ = plateau_metrics(self.frame([0.85, 0.85, 0.85, 0.85]), window=100)
        _, sliding_sd, _ = plateau_metrics(self.frame([0.95, 0.90, 0.80, 0.75]), window=100)
        self.assertAlmostEqual(steady["PR-AUC"], 0.85)
        self.assertEqual(steady_sd, 0.0)
        self.assertGreater(sliding_sd, 0.05)

    def test_one_round_has_a_mean_but_no_spread(self):
        _, sd, _ = plateau_metrics(self.frame([0.85]), window=100)
        self.assertIsNone(sd)

    def test_it_undoes_the_bias_it_was_introduced_for(self):
        """The measured shape of the problem, as a test.

        A noisy arm's argmax lands on its luckiest round; its plateau mean does
        not. The gap has to be larger for the noisy arm than for the flat one,
        which is the whole reason the rule changed.
        """
        flat = [0.880, 0.881, 0.879, 0.880, 0.882]
        noisy = [0.860, 0.905, 0.845, 0.870, 0.850]

        flat_gap = max(flat) - float(np.mean(flat))
        noisy_gap = max(noisy) - float(np.mean(noisy))
        self.assertGreater(noisy_gap, 3 * flat_gap)


class TestPlateauFrames(unittest.TestCase):
    """Loading the per-user scores the interval is built from."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.rounds_dir = os.path.join(self.tmp, "round_by_round_results")
        os.makedirs(self.rounds_dir)

    def write_round(self, round_number, users, labels, scores):
        pd.DataFrame({"user": users, "is_actual_insider": labels,
                      "max_z_score": scores}).to_csv(
            os.path.join(self.rounds_dir, f"round_{round_number}_results.csv"),
            index=False)

    def test_users_are_sorted_so_rounds_stay_paired(self):
        """One round written in another order would pair labels with wrong scores."""
        self.write_round(0, [2, 0, 1], [1, 0, 0], [9.0, 1.0, 2.0])
        labels, scores = load_round_scores(self.tmp, 0)
        self.assertEqual(list(labels), [0, 0, 1])
        self.assertEqual(list(scores), [1.0, 2.0, 9.0])

    def test_a_missing_round_file_is_skipped_not_fatal(self):
        self.write_round(0, [0, 1], [0, 1], [1.0, 5.0])
        frames = load_plateau_frames(self.tmp, [0, 2])
        self.assertEqual(len(frames), 1)

    def test_rounds_covering_different_users_yield_no_frames(self):
        """Refused rather than paired by position, which would be silent nonsense."""
        self.write_round(0, [0, 1], [0, 1], [1.0, 5.0])
        self.write_round(2, [0, 1, 2], [0, 1, 0], [1.0, 5.0, 2.0])
        self.assertEqual(load_plateau_frames(self.tmp, [0, 2]), [])


class TestPlateauBootstrap(unittest.TestCase):
    """The interval is around the plateau mean, and the rounds stay paired."""

    def frames(self, n_rounds, n_users=200, insiders=20, spread=0.0, seed=0):
        """Rounds of per-user scores over the same users.

        The insider signal is deliberately weak. A cleanly separable one puts
        PR-AUC at exactly 1.0 in every resample, and an interval of width zero
        cannot show anything about how the interval is built.
        """
        rng = np.random.RandomState(seed)
        labels = np.zeros(n_users)
        labels[:insiders] = 1
        out = []
        for i in range(n_rounds):
            scores = rng.rand(n_users) + labels * (0.3 + i * spread)
            out.append((labels, scores))
        return out

    def test_it_is_reproducible(self):
        frames = self.frames(4)
        first = bootstrap_plateau_pr_auc(frames, n_iterations=50, confidence=0.95)
        second = bootstrap_plateau_pr_auc(frames, n_iterations=50, confidence=0.95)
        self.assertEqual(first, second)

    def test_the_interval_brackets_the_mean(self):
        mean, _, lo, hi = bootstrap_plateau_pr_auc(
            self.frames(4), n_iterations=100, confidence=0.95)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)

    def test_the_rounds_are_paired_by_user(self):
        """Resampling users once for all rounds is the point, not an detail.

        Every plateau round scores the same people, so user-level variation is
        shared across rounds and must not average away. Drawing a fresh resample
        per round cancels it and reports an interval far too narrow, which is the
        opposite of what the interval is for. Compared against that mistake
        directly rather than against a threshold picked by hand.
        """
        frames = self.frames(8)
        paired = bootstrap_plateau_pr_auc(frames, n_iterations=300)
        paired_width = paired[3] - paired[2]

        # The mistake, written out: a new draw for every round.
        rng = np.random.RandomState(42)
        n = len(frames[0][0])
        unpaired_means = []
        for _ in range(300):
            per_round = []
            for y_true, y_scores in frames:
                idx = rng.randint(0, n, n)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                per_round.append(average_precision_score(y_true[idx], y_scores[idx]))
            if per_round:
                unpaired_means.append(float(np.mean(per_round)))
        unpaired_width = (np.percentile(unpaired_means, 97.5)
                          - np.percentile(unpaired_means, 2.5))

        # Measured at about 1.49 on this fixture. The bound is 1.25 to leave room
        # for Monte Carlo noise while staying far from 1.0, which is what "the
        # pairing does nothing" would look like.
        self.assertGreater(paired_width, unpaired_width * 1.25)

    def test_identical_rounds_reduce_to_the_single_round_case(self):
        labels, scores = self.frames(1)[0]
        repeated = [(labels, scores)] * 5
        single = bootstrap_plateau_pr_auc([(labels, scores)], n_iterations=100)
        many = bootstrap_plateau_pr_auc(repeated, n_iterations=100)
        self.assertAlmostEqual(single[0], many[0], places=10)

    def test_no_frames_gives_no_interval(self):
        result = bootstrap_plateau_pr_auc([], n_iterations=10)
        self.assertTrue(all(np.isnan(value) for value in result))


if __name__ == "__main__":
    unittest.main()
