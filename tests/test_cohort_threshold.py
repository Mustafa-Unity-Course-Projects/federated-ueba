"""The minimum cohort behind the global scaler's statistics.

The scaler is built from three numbers per feature per client: a count, a sum
and a sum of squares. Those disclose nothing about a person as long as the cohort
behind them is not a person. At a cohort of one, `sum / count` is that user's own
per-feature mean and the server can read it directly.

Measured on this dataset: the Dirichlet partition at alpha = 0.5 puts 1 user on
its smallest client and two or fewer on 16 of 50, while the IID partition never
drops below 16. So the floor binds only where it is needed, and these tests pin
both halves of that: it must exclude the small cohorts, and it must leave the IID
scaler untouched to the bit.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config_manager import config
from federated_ueba import scaling

FEATURES = ["a", "b"]


def frame(users_and_values):
    """One row per (user, value) pair, two features, no insiders."""
    rows = [{"user": user, "a": float(value), "b": float(value) * 2.0,
             "insider": 0}
            for user, values in users_and_values.items()
            for value in values]
    return pd.DataFrame(rows)


class TestSmallCohortsAreExcluded(unittest.TestCase):
    def setUp(self):
        # Client A holds five users, client B holds one. Their values are far
        # apart so that including B is visible in the aggregate.
        self.df = frame({f"a{i}": [1.0, 1.0] for i in range(5)}
                        | {"lonely": [1000.0, 1000.0]})
        self.big = [f"a{i}" for i in range(5)]
        self.small = ["lonely"]

    def test_a_single_user_client_does_not_reach_the_scaler(self):
        guarded = scaling.build_global_scaler(
            self.df, FEATURES, [self.big, self.small], min_cohort=5)
        # Only the cohort of five contributed, so the mean is theirs alone.
        # Compared against log1p(1.0) rather than 1.0: prepare_features applies
        # the configured transform before any statistic is taken, so the scaler
        # describes transformed values throughout.
        self.assertAlmostEqual(guarded.mean_[0], float(np.log1p(1.0)), places=6)

    def test_without_the_floor_the_lone_user_moves_the_mean(self):
        """Shows the guard is doing something rather than being a no-op here."""
        open_scaler = scaling.build_global_scaler(
            self.df, FEATURES, [self.big, self.small], min_cohort=1)
        self.assertGreater(open_scaler.mean_[0], 1.0)

    def test_a_cohort_exactly_at_the_floor_is_accepted(self):
        """Off-by-one here would silently drop a legitimate client."""
        counted = scaling.build_global_scaler(
            self.df, FEATURES, [self.big], min_cohort=5)
        self.assertEqual(counted.n_samples_seen_, 10)

    def test_excluding_everything_fails_loudly(self):
        """A scaler built from nothing would be silently meaningless."""
        with self.assertRaises(ValueError) as caught:
            scaling.build_global_scaler(
                self.df, FEATURES, [self.small], min_cohort=5)
        self.assertIn("minimum_cohort_size", str(caught.exception))

    def test_cohort_size_counts_users_not_rows(self):
        """One user with many days is still one user, and the whole point is
        how many people stand behind the numbers."""
        chatty = frame({"one_user": [1.0] * 500})
        with self.assertRaises(ValueError):
            scaling.build_global_scaler(
                chatty, FEATURES, [["one_user"]], min_cohort=5)


class TestTheFloorIsConfigured(unittest.TestCase):
    def test_it_is_read_from_configuration(self):
        self.assertEqual(scaling.minimum_cohort_size(),
                         config.get("data", "minimum_cohort_size"))

    def test_the_configured_floor_is_above_two(self):
        """Two users can still recover each other's share by subtraction."""
        self.assertGreater(config.get("data", "minimum_cohort_size"), 2)


class TestTheIidScalerIsUnchanged(unittest.TestCase):
    """The reason the IID results did not have to be recomputed for privacy."""

    def test_an_even_partition_is_bitwise_identical_with_and_without_the_floor(self):
        users = {f"u{i}": [float(i), float(i) + 0.5] for i in range(50)}
        df = frame(users)
        # Ten clients of five users each: every cohort clears a floor of five.
        chunks = [[f"u{i}" for i in range(start, start + 5)]
                  for start in range(0, 50, 5)]

        without = scaling.build_global_scaler(df, FEATURES, chunks, min_cohort=1)
        with_floor = scaling.build_global_scaler(df, FEATURES, chunks, min_cohort=5)

        self.assertTrue(np.array_equal(without.mean_, with_floor.mean_))
        self.assertTrue(np.array_equal(without.scale_, with_floor.scale_))
        self.assertEqual(without.n_samples_seen_, with_floor.n_samples_seen_)


if __name__ == "__main__":
    unittest.main()
