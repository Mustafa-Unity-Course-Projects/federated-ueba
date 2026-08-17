"""The shared scaler: correctness, the variance floor, and partition independence.

The bug this replaces was not a wrong number, it was a wrong *ranking*: a feature
that happened to be constant on one client got a scale of 3e-15, and dividing by
that put fourteen users at the top of the results on a numerical artifact. So the
floor is tested directly, and so is the property that made the old design wrong,
namely that the scaler must not depend on how users were split.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import config  # noqa: E402
from federated_ueba import scaling  # noqa: E402


def frame(n_users=40, n_days=20, seed=0):
    """Users differ in scale, which is what broke per-client scalers."""
    rng = np.random.RandomState(seed)
    rows = []
    for user in range(n_users):
        heavy = user < n_users // 2
        for day in range(n_days):
            rows.append({
                "user": float(user), "day": day, "insider": 0,
                # Group-dependent spread: a per-client scaler would disagree.
                "busy": rng.rand() * (10.0 if heavy else 0.5),
                "quiet": rng.rand() * 0.4,
                # Constant everywhere. The old code divided by its float noise.
                "flat": 3.0,
            })
    return pd.DataFrame(rows), ["busy", "quiet", "flat"]


class TestGlobalScaler(unittest.TestCase):
    def setUp(self):
        self.df, self.features = frame()
        self.users = sorted(self.df["user"].unique())

    def test_matches_a_scaler_fitted_on_everything_at_once(self):
        """Summing per-client statistics must equal fitting on the pooled data."""
        chunks = np.array_split(self.users, 8)
        federated = scaling.build_global_scaler(self.df, self.features, chunks)
        pooled = scaling.build_global_scaler(self.df, self.features, [self.users])

        np.testing.assert_allclose(federated.mean_, pooled.mean_, atol=1e-9)
        np.testing.assert_allclose(federated.scale_, pooled.scale_, atol=1e-9)

    def test_is_independent_of_how_users_are_split(self):
        """The property the per-client design lacked, within its honest scope.

        `min_cohort=1` because this is a statement about the aggregation, not
        about the privacy floor: summing counts, sums and sums of squares gives
        the same answer however the rows are grouped. The floor narrows where
        that holds, and `test_the_privacy_floor_costs_partition_independence`
        below pins the boundary rather than leaving it implied.
        """
        even = scaling.build_global_scaler(
            self.df, self.features, np.array_split(self.users, 8), min_cohort=1)
        lopsided = scaling.build_global_scaler(
            self.df, self.features,
            [self.users[:1], self.users[1:3], self.users[3:]], min_cohort=1)

        np.testing.assert_allclose(even.mean_, lopsided.mean_, atol=1e-9)
        np.testing.assert_allclose(even.scale_, lopsided.scale_, atol=1e-9)

    def test_the_privacy_floor_costs_partition_independence(self):
        """The tension, stated rather than hidden.

        Excluding small cohorts means the scaler now depends on the partition
        after all, because a lopsided split has clients to exclude and an even
        one does not. That is a real cost of the privacy floor and it is the one
        property the global scaler was introduced to gain.

        It is accepted because of where it binds. On the real partition the IID
        scaler is unchanged to the bit, since no IID client is small, and the
        non-IID scaler moves by at most 2.4% on a feature's scale. The exposure
        it removes is a client whose statistics describe one person.
        """
        even = scaling.build_global_scaler(
            self.df, self.features, np.array_split(self.users, 4), min_cohort=1)
        lopsided_guarded = scaling.build_global_scaler(
            self.df, self.features,
            [self.users[:1], self.users[1:]], min_cohort=5)

        self.assertFalse(np.allclose(even.mean_, lopsided_guarded.mean_, atol=1e-9))

    def test_a_single_user_client_cannot_distort_the_scaler(self):
        """Under Dirichlet partitioning many clients hold exactly one user.

        About the arithmetic, so the floor is lifted here: even when every client
        holds one user, summing sufficient statistics reproduces the pooled
        scaler exactly. The floor keeps those clients out for privacy, not
        because their numbers would be wrong.
        """
        singles = [[u] for u in self.users]
        one_shot = scaling.build_global_scaler(
            self.df, self.features, [self.users], min_cohort=1)
        split = scaling.build_global_scaler(
            self.df, self.features, singles, min_cohort=1)
        np.testing.assert_allclose(split.scale_, one_shot.scale_, atol=1e-9)


class TestVarianceFloor(unittest.TestCase):
    def setUp(self):
        self.df, self.features = frame()
        self.users = sorted(self.df["user"].unique())
        self.scaler = scaling.build_global_scaler(self.df, self.features, [self.users])

    def test_constant_feature_is_left_unscaled(self):
        flat = self.features.index("flat")
        self.assertEqual(self.scaler.scale_[flat], 1.0)

    def test_no_scale_is_small_enough_to_explode(self):
        self.assertTrue((self.scaler.scale_ >= scaling.constant_feature_tolerance()).all())

    def test_transformed_values_stay_in_a_sane_range(self):
        """The old scaler produced 1e15 here; anything similar must fail."""
        values = scaling.prepare_features(self.df, self.features)
        z = self.scaler.transform(values)
        self.assertLess(np.abs(z).max(), 100)

    def test_an_unseen_value_in_a_constant_feature_stays_bounded(self):
        """The exact case that broke: scoring a user the scaler never saw."""
        values = scaling.prepare_features(self.df, self.features)
        values.loc[0, "flat"] = 9.0      # never observed
        z = self.scaler.transform(values)
        self.assertLess(np.abs(z).max(), 100)

    def test_variance_is_never_negative(self):
        """Cancellation in sum-of-squares can push it below zero."""
        self.assertTrue((self.scaler.var_ >= 0).all())


class TestPrepareFeatures(unittest.TestCase):
    def test_applies_log1p_after_clipping_negatives(self):
        df = pd.DataFrame({"a": [0.0, np.e - 1, -5.0]})
        np.testing.assert_allclose(
            scaling.prepare_features(df, ["a"]).to_numpy().ravel(),
            [0.0, 1.0, 0.0], atol=1e-6)

    def test_missing_column_becomes_zero(self):
        df = pd.DataFrame({"a": [1.0]})
        out = scaling.prepare_features(df, ["a", "absent"])
        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(out.loc[0, "absent"], 0.0)

    def test_non_numeric_becomes_zero(self):
        df = pd.DataFrame({"a": ["x"]})
        self.assertEqual(scaling.prepare_features(df, ["a"]).loc[0, "a"], 0.0)

    def test_feature_names_survive_to_the_scaler(self):
        """The reason this returns a frame: `transform` verifies the names.

        The model's inputs are positional, so a column arriving in the wrong
        order would feed the wrong feature to the wrong input and fail silently.
        """
        df, features = frame(n_users=4, n_days=20)
        out = scaling.prepare_features(df, features)
        self.assertEqual(list(out.columns), features)

    def test_the_scaler_rejects_features_in_the_wrong_order(self):
        df, features = frame(n_users=4, n_days=20)
        # Four users, below the configured cohort floor. Lifted here because the
        # subject is column ordering, not the privacy guard.
        scaler = scaling.build_global_scaler(
            df, features, [sorted(df["user"].unique())], min_cohort=1)

        shuffled = scaling.prepare_features(df, features)[features[::-1]]
        with self.assertRaises(ValueError):
            scaler.transform(shuffled)


class TestFeatureTransforms(unittest.TestCase):
    """The three ways a percentile value can be prepared for the model.

    The processed dataset holds percentile deviations in [-50, +50], not counts,
    so what happens to the negative half is a methodology decision rather than a
    detail. The original transform discards it.
    """

    def setUp(self):
        # One below-normal day, one ordinary day, one above-normal day.
        self.df = pd.DataFrame({"a": [-40.0, 0.0, 40.0]})

    def _values(self, transform):
        return scaling.prepare_features(
            self.df, ["a"], transform=transform)["a"].tolist()

    def test_the_default_is_the_original_behaviour(self):
        """Changing the default would silently change every reported number."""
        self.assertEqual(config.get("data", "feature_transform"), "log1p_positive")

    def test_log1p_positive_discards_the_below_normal_day(self):
        """The behaviour to be aware of: -40 and 0 become the same input."""
        below, ordinary, above = self._values("log1p_positive")
        self.assertEqual(below, 0.0)
        self.assertEqual(ordinary, 0.0)
        self.assertGreater(above, 0.0)

    def test_signed_log1p_keeps_the_below_normal_day_distinguishable(self):
        below, ordinary, above = self._values("signed_log1p")
        self.assertLess(below, ordinary)
        self.assertLess(ordinary, above)
        # Symmetric: equal deviations either way get equal magnitude.
        self.assertAlmostEqual(below, -above, places=5)

    def test_raw_passes_the_percentile_through(self):
        self.assertEqual(self._values("raw"), [-40.0, 0.0, 40.0])

    def test_all_three_compress_the_scale_except_raw(self):
        """log1p exists to compress; on bounded percentiles that is a choice."""
        self.assertLess(max(self._values("signed_log1p")), 5.0)
        self.assertEqual(max(self._values("raw")), 40.0)

    def test_an_unknown_transform_is_refused(self):
        """Silently falling back would produce results nobody could explain."""
        with self.assertRaises(ValueError):
            scaling.prepare_features(self.df, ["a"], transform="whatever")

    def test_every_named_transform_works(self):
        for transform in scaling.FEATURE_TRANSFORMS:
            with self.subTest(transform=transform):
                out = scaling.prepare_features(self.df, ["a"], transform=transform)
                self.assertEqual(len(out), 3)
                self.assertFalse(out.isna().any().any())


class TestEmptyInput(unittest.TestCase):
    def test_raises_when_there_is_nothing_to_fit(self):
        df = pd.DataFrame({"user": [], "insider": [], "a": []})
        with self.assertRaises(ValueError):
            scaling.build_global_scaler(df, ["a"], [[]])

    def test_ignores_clients_with_no_normal_rows(self):
        df, features = frame(n_users=10)
        users = sorted(df["user"].unique())
        df.loc[df["user"] == users[0], "insider"] = 1

        with_all = scaling.build_global_scaler(df, features, [users])
        without = scaling.build_global_scaler(df, features, [users[1:]])
        np.testing.assert_allclose(with_all.mean_, without.mean_, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
