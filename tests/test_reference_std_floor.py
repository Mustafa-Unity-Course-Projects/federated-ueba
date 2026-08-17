"""A near-constant feature must not be able to take over the ranking (Y15).

Stage one of the scoring divides by the reference standard deviation of each
feature. Four of the fifty features have a reference std around 3e-5, so
dividing by `std + zscore_epsilon` with an epsilon of 1e-6 multiplied any
deviation on them by roughly thirty thousand. Stage two then takes the largest
features per window, which meant it took those same four for every user, and the
ranking stopped describing behaviour.

Most runs never showed it, because their models reconstruct those features
exactly and a zero numerator hides any denominator. Whether a run survived
therefore depended on something no one would choose to rank models by. Measured
on one unchanged checkpoint, the difference was PR-AUC 0.2394 against 0.8215.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from federated_ueba import scoring


class TestFloorReferenceStd(unittest.TestCase):
    def test_a_degenerate_feature_is_raised_to_the_floor(self):
        ref_std = np.array([1.0, 1.0, 1.0, 3e-5])
        floored = scoring.floor_reference_std(ref_std, 0.01)
        self.assertAlmostEqual(floored[3], 0.01)

    def test_features_with_normal_spread_are_untouched(self):
        ref_std = np.array([1.0, 2.0, 0.5, 3e-5])
        floored = scoring.floor_reference_std(ref_std, 0.01)
        np.testing.assert_allclose(floored[:3], [1.0, 2.0, 0.5])

    def test_the_floor_follows_the_scale_of_the_data(self):
        """Relative, not absolute: the same guard on a differently scaled run."""
        small = scoring.floor_reference_std(np.array([1e-3, 1e-3, 1e-9]), 0.01)
        large = scoring.floor_reference_std(np.array([1e3, 1e3, 1e-9]), 0.01)
        self.assertAlmostEqual(small[2], 1e-5)
        self.assertAlmostEqual(large[2], 10.0)

    def test_the_real_shape_of_the_reference_is_guarded(self):
        """Four degenerate features among fifty, which is the measured case."""
        ref_std = np.concatenate([np.full(46, 0.7), np.full(4, 3e-5)])
        floored = scoring.floor_reference_std(ref_std, 0.01)
        self.assertAlmostEqual(floored[-1], 0.007)
        self.assertAlmostEqual(floored[0], 0.7)

    def test_a_mostly_degenerate_reference_guards_little(self):
        """A documented limit, pinned so it cannot become a surprise.

        The floor is a share of the median, and if most features are degenerate
        they set the median themselves. No rule separating flat from not-flat
        avoids this without a threshold there is nothing to fit, so the
        behaviour is recorded rather than worked around.
        """
        ref_std = np.array([1e-5] * 6 + [1.0, 1.0, 1.0])
        floored = scoring.floor_reference_std(ref_std, 0.01)
        self.assertAlmostEqual(floored[0], 1e-5)

    def test_a_zero_std_is_raised_too(self):
        floored = scoring.floor_reference_std(np.array([1.0, 0.0]), 0.01)
        self.assertAlmostEqual(floored[1], 0.01)

    def test_a_fraction_of_zero_disables_the_floor(self):
        ref_std = np.array([1.0, 3e-5])
        np.testing.assert_allclose(
            scoring.floor_reference_std(ref_std, 0.0), ref_std)

    def test_an_all_degenerate_reference_is_left_alone(self):
        """Nothing to measure a floor against; refused rather than invented."""
        ref_std = np.zeros(4)
        np.testing.assert_allclose(
            scoring.floor_reference_std(ref_std, 0.01), ref_std)

    def test_no_reference_is_not_an_error(self):
        self.assertIsNone(scoring.floor_reference_std(None, 0.01))


class TestScorerAppliesTheFloor(unittest.TestCase):
    """Applied in the Scorer, so every scoring path gets it or none does."""

    def scorer(self, ref_std, fraction=0.01):
        cfg = scoring.ScoringConfig(zscore_std_floor_fraction=fraction)
        return scoring.Scorer(
            features=["a", "b", "c"], scaler=None,
            ref_mean=np.zeros(3), ref_std=np.asarray(ref_std, dtype=np.float64),
            cfg=cfg, window_size=14, device="cpu")

    def test_construction_floors_the_reference(self):
        s = self.scorer([1.0, 1.0, 3e-5])
        self.assertAlmostEqual(s.ref_std[2], 0.01)

    def test_the_amplifier_is_bounded_after_flooring(self):
        """The quantity that actually caused the failure: 1 / (std + eps)."""
        cfg = scoring.ScoringConfig()
        raw = 1.0 / (3e-5 + cfg.zscore_epsilon)
        s = self.scorer([1.0, 1.0, 3e-5])
        floored = 1.0 / (s.ref_std[2] + cfg.zscore_epsilon)

        self.assertGreater(raw, 30000)
        self.assertLess(floored, 101)

    def test_a_scorer_over_healthy_features_is_unchanged(self):
        """The floor must not move any run that was never at risk."""
        original = [0.8, 1.2, 1.0]
        s = self.scorer(original)
        np.testing.assert_allclose(s.ref_std, original)


class TestWindowScoreWithTheFloor(unittest.TestCase):
    """End to end through stage one, on the shape of the real failure."""

    def test_a_tiny_error_on_a_flat_feature_no_longer_dominates(self):
        cfg = scoring.ScoringConfig(top_k_features=2)
        ref_mean = np.array([0.5, 0.5, 2e-5])
        # The real reference: two features with spread, one flat.
        raw_std = np.array([0.7, 0.7, 3e-5])
        floored = scoring.floor_reference_std(raw_std,
                                              cfg.zscore_std_floor_fraction)

        # A window that is unremarkable everywhere, including a small absolute
        # miss on the flat feature.
        sq_err = np.array([0.5, 0.5, 1e-3])

        before = scoring.window_score(sq_err, ref_mean, raw_std, cfg, 3)
        after = scoring.window_score(sq_err, ref_mean, floored, cfg, 3)

        self.assertGreater(before, 10)
        self.assertLess(after, 1)


if __name__ == "__main__":
    unittest.main()
