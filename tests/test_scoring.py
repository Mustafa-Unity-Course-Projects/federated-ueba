"""Scoring pipeline: the four stages, the ablation switches, and metric hygiene.

The pipeline is the thesis's third contribution, and the ablation table is the
evidence for it. If a switch silently stopped doing anything the table would
still look plausible, so each stage is checked for having a real effect rather
than merely being wired up.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_ueba import scoring  # noqa: E402

CPU = torch.device("cpu")


def cfg(**overrides):
    base = dict(top_k_features=5, persistence_window=3, diversity_threshold=2.0,
                scan_stride=1, inference_batch_size=128)
    base.update(overrides)
    return scoring.ScoringConfig(**base)


def scanner(scoring_cfg=None, features=None, scaler=None):
    """A Scorer over the fixture frame. Tests name only what they vary."""
    return scoring.Scorer(
        features=features, scaler=scaler,
        ref_mean=np.zeros(2), ref_std=np.ones(2),
        cfg=scoring_cfg or cfg(), window_size=14, device=CPU)


class TestWindowScore(unittest.TestCase):
    """One window: per-feature error to a single number."""

    def setUp(self):
        self.n = 10
        self.mean = np.full(self.n, 1.0)
        self.std = np.full(self.n, 1.0)

    def test_zscore_uses_the_reference_distribution(self):
        # Error exactly at the reference mean is not anomalous at all.
        at_mean = scoring.window_score(self.mean.copy(), self.mean, self.std,
                                       cfg().without("diversity"), self.n)
        self.assertEqual(at_mean, 0.0)

        elevated = scoring.window_score(self.mean + 3.0, self.mean, self.std,
                                        cfg().without("diversity"), self.n)
        self.assertAlmostEqual(elevated, 3.0, places=5)

    def test_raw_error_without_calibration_is_not_normalised(self):
        sq_err = np.full(self.n, 4.0)
        calibrated = scoring.window_score(sq_err, self.mean, self.std,
                                          cfg().without("diversity"), self.n)
        raw = scoring.window_score(sq_err, self.mean, self.std,
                                   cfg().without("zscore", "diversity"), self.n)
        self.assertAlmostEqual(calibrated, 3.0, places=5)   # (4 - 1) / 1
        self.assertAlmostEqual(raw, 4.0, places=5)
        self.assertNotAlmostEqual(calibrated, raw)

    def test_negative_deviation_is_clipped(self):
        """Reconstructing better than usual is not evidence of an anomaly."""
        sq_err = np.zeros(self.n)                   # well below the reference
        score = scoring.window_score(sq_err, self.mean, self.std,
                                     cfg().without("diversity"), self.n)
        self.assertEqual(score, 0.0)

    def test_topk_focus_ignores_the_quiet_features(self):
        sq_err = self.mean.copy()
        sq_err[:5] += 10.0                          # five loud features
        focused = scoring.window_score(sq_err, self.mean, self.std,
                                       cfg().without("diversity"), self.n)
        diffuse = scoring.window_score(sq_err, self.mean, self.std,
                                       cfg().without("topk", "diversity"), self.n)
        # Averaging over all ten features halves the signal. Tolerance is 4
        # places because the z-score denominator carries a +1e-6 guard against
        # division by zero, which shifts large scores in the fifth decimal.
        self.assertAlmostEqual(focused, 10.0, places=4)
        self.assertAlmostEqual(diffuse, 5.0, places=4)

    def test_diversity_multiplier_rewards_simultaneous_deviation(self):
        one_feature = self.mean.copy()
        one_feature[0] += 10.0
        many_features = self.mean + 10.0

        without = cfg().without("diversity")
        with_div = cfg()

        # A single loud feature: only 1 of 10 crosses the threshold.
        self.assertAlmostEqual(
            scoring.window_score(one_feature, self.mean, self.std, with_div, self.n)
            / scoring.window_score(one_feature, self.mean, self.std, without, self.n),
            1.1, places=5)

        # All ten loud: the multiplier doubles the score.
        self.assertAlmostEqual(
            scoring.window_score(many_features, self.mean, self.std, with_div, self.n)
            / scoring.window_score(many_features, self.mean, self.std, without, self.n),
            2.0, places=5)


class TestTemporalPersistence(unittest.TestCase):
    """Many windows: a user's windows collapse to one threat score."""

    def test_persistence_averages_the_highest_windows(self):
        windows = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        self.assertAlmostEqual(
            scoring.aggregate_windows(windows, cfg(persistence_window=3)),
            11.0)

    def test_without_persistence_a_single_spike_is_diluted(self):
        windows = [0.0] * 99 + [100.0]
        persistent = scoring.aggregate_windows(windows, cfg(persistence_window=3))
        averaged = scoring.aggregate_windows(
            windows, cfg().without("persistence"))
        self.assertAlmostEqual(persistent, 100.0 / 3)
        self.assertAlmostEqual(averaged, 1.0)
        self.assertGreater(persistent, averaged)

    def test_selection_is_by_score_not_by_time(self):
        """The top windows are the highest-scoring ones, wherever they occur."""
        early_spike = [9.0, 9.0, 9.0] + [0.0] * 10
        late_spike = [0.0] * 10 + [9.0, 9.0, 9.0]
        self.assertAlmostEqual(
            scoring.aggregate_windows(early_spike, cfg(persistence_window=3)),
            scoring.aggregate_windows(late_spike, cfg(persistence_window=3)))

    def test_empty_input_scores_zero(self):
        self.assertEqual(scoring.aggregate_windows([], cfg()), 0.0)


class ConstantModel(torch.nn.Module):
    """Reconstructs everything as zero, so the error is the input itself."""

    def forward(self, x):
        return torch.zeros_like(x)


class TestScanUsers(unittest.TestCase):
    def setUp(self):
        rows = []
        for user in [3.0, 1.0, 2.0]:                # deliberately unsorted
            for day in range(20):
                rows.append({"user": user, "day": day,
                             "f0": user, "f1": 0.0,
                             "insider": 1 if (user == 2.0 and day == 5) else 0})
        self.df = pd.DataFrame(rows)
        self.features = ["f0", "f1"]

        class Identity:
            def transform(self, x):
                return np.asarray(x, dtype=np.float32)

        self.scaler = Identity()

    def test_output_is_sorted_by_user(self):
        """Downstream splits must not depend on the order users were scanned.

        The user/test split is drawn from this frame's row order, so an
        order-dependent scan would silently change which users are reported on.
        """
        out = scanner(features=self.features, scaler=self.scaler).scan(
            ConstantModel(), self.df, [3.0, 1.0, 2.0])
        self.assertEqual(list(out["user"]), [1.0, 2.0, 3.0])

    def test_scan_order_does_not_change_results(self):
        scan = scanner(features=self.features, scaler=self.scaler)
        a = scan.scan(ConstantModel(), self.df, [1.0, 2.0, 3.0])
        b = scan.scan(ConstantModel(), self.df, [3.0, 2.0, 1.0])
        pd.testing.assert_frame_equal(a, b)

    def test_a_user_is_positive_if_any_day_is_flagged(self):
        out = scanner(features=self.features, scaler=self.scaler).scan(
            ConstantModel(), self.df, [1.0, 2.0, 3.0])
        labels = dict(zip(out["user"], out["is_actual_insider"]))
        self.assertEqual(labels[2.0], 1.0)
        self.assertEqual(labels[1.0], 0.0)

    def test_user_with_too_few_days_scores_zero(self):
        short = self.df[self.df["day"] < 5]
        out = scanner(features=self.features, scaler=self.scaler).scan(
            ConstantModel(), short, [1.0])
        self.assertEqual(float(out["max_z_score"].iloc[0]), 0.0)


class TestEvaluateScores(unittest.TestCase):
    """Metric hygiene: what selects must never be what reports."""

    def make_results(self, n=200, positives=20, seed=0):
        rng = np.random.RandomState(seed)
        labels = np.zeros(n)
        labels[:positives] = 1.0
        scores = rng.rand(n) + labels * 0.6
        return pd.DataFrame({"user": np.arange(n, dtype=float),
                             "max_z_score": scores,
                             "is_actual_insider": labels}).sort_values("user")

    def test_reports_the_test_half(self):
        m = scoring.evaluate_scores(self.make_results(), seed=42)
        self.assertEqual(m["pr_auc"], m["pr_auc_test"])

    def test_val_test_and_all_are_distinct_measurements(self):
        m = scoring.evaluate_scores(self.make_results(), seed=42)
        self.assertNotAlmostEqual(m["pr_auc_val"], m["pr_auc_test"], places=6)
        for key in ("pr_auc_val", "pr_auc_test", "pr_auc_all"):
            self.assertGreater(m[key], 0.0)
            self.assertLessEqual(m[key], 1.0)

    def test_threshold_is_not_the_one_that_maximises_test_f1(self):
        """The reported F1 is charged at a threshold fitted elsewhere.

        If these coincided, the threshold would have been fitted on the users it
        is scored against, and the reported F1 would be an upper bound rather
        than an estimate.
        """
        from sklearn.metrics import f1_score

        results = self.make_results(seed=3)
        m = scoring.evaluate_scores(results, seed=42)

        # Best achievable F1 on the reported half, had we been allowed to cheat.
        from sklearn.model_selection import train_test_split
        _, test_users = train_test_split(
            results["user"].tolist(), test_size=0.5,
            stratify=results["is_actual_insider"].tolist(), random_state=42)
        test_df = results[results["user"].isin(test_users)]

        oracle = max(
            f1_score(test_df["is_actual_insider"],
                     (test_df["max_z_score"] >= t).astype(int), zero_division=0)
            for t in np.unique(test_df["max_z_score"]))

        self.assertLessEqual(m["f1"], oracle + 1e-9)
        self.assertLess(m["f1"], oracle,
                        "reported F1 matched the oracle; the threshold leaked")

    def test_confusion_counts_cover_the_test_half_only(self):
        results = self.make_results(n=200)
        m = scoring.evaluate_scores(results, seed=42)
        self.assertEqual(m["tp"] + m["fp"] + m["tn"] + m["fn"], 100)


class TestScoringConfig(unittest.TestCase):
    def test_all_stages_run_by_default(self):
        self.assertEqual(scoring.ScoringConfig().stages, scoring.STAGES)

    def test_without_removes_only_what_it_names(self):
        remaining = scoring.ScoringConfig().without("diversity")
        self.assertFalse(remaining.uses("diversity"))
        self.assertTrue(remaining.uses("zscore"))
        self.assertTrue(remaining.uses("topk"))
        self.assertTrue(remaining.uses("persistence"))

    def test_an_unknown_stage_name_is_refused(self):
        """A typo must stop the run, not silently produce a shorter pipeline."""
        with self.assertRaises(ValueError):
            scoring.ScoringConfig(stages=("zscore", "topk_focus"))

    def test_stage_order_in_the_file_does_not_change_the_pipeline(self):
        """The order is fixed by the maths, so listing it differently is the
        same scorer rather than a different one."""
        class Reordered:
            def get(self, section, key):
                if (section, key) == ("scoring", "stages"):
                    return ["persistence", "diversity", "topk", "zscore"]
                return config.get(section, key)

        from config_manager import config
        self.assertEqual(scoring.ScoringConfig.from_config(Reordered()).stages,
                         scoring.STAGES)

    def test_from_config_reads_every_setting_from_the_config(self):
        """No key may fall back to the dataclass default any more.

        The dataclass still carries defaults so that a test can build one
        directly, but `from_config` must take every value from the file: a stage
        silently left on because its key was missing is exactly the failure the
        no-defaults rule exists to prevent.
        """
        from config_manager import config

        # An arm whose stage list differs from the default in both directions:
        # it turns the diversity multiplier on, which the default leaves off.
        # A `from_config` that fell back to the dataclass default would get the
        # multiplier wrong here in the direction the old test could not see.
        config.set_experiment("ablation-with-diversity")
        try:
            cfg_obj = scoring.ScoringConfig.from_config(config)

            self.assertTrue(cfg_obj.uses("diversity"))
            self.assertTrue(cfg_obj.uses("zscore"))
            self.assertEqual(cfg_obj.top_k_features,
                             config.get("scoring", "top_k_features"))
            self.assertEqual(cfg_obj.persistence_window,
                             config.get("scoring", "persistence_window"))
            self.assertEqual(cfg_obj.zscore_epsilon,
                             config.get("scoring", "zscore_epsilon"))
        finally:
            config.set_experiment("baseline")


if __name__ == "__main__":
    unittest.main()
