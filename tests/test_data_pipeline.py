"""Windowing and user partitioning: the arithmetic the jury asked about.

Two of the correction items are answered by numbers this file pins:

  J6   the thesis says both "10 days" and "14 days", and never states how many
       windows a user actually produces
  J18  the non-IID split has to be a documented Dirichlet partition rather than
       a claim, and it has to lose no users

Nothing here trains anything. These are the pure functions around the model, and
they are the ones whose behaviour ends up quoted in the text.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from config_manager import config  # noqa: E402
from federated_ueba import task  # noqa: E402


class TestWindowArithmetic(unittest.TestCase):
    """How many windows a user of N days produces, and of what shape."""

    def test_the_configured_window_is_fourteen_days_with_stride_one(self):
        """The thesis contradicts itself on this; the code does not."""
        self.assertEqual(config.get("model", "window_size"), 14)
        self.assertEqual(config.get("model", "stride"), 1)
        self.assertEqual(task.WINDOW_SIZE, 14)
        self.assertEqual(task.STRIDE, 1)

    def test_a_user_of_n_days_produces_n_minus_thirteen_windows(self):
        """The formula to quote: windows = days - window_size + 1, at stride 1."""
        for days in (14, 15, 30, 100, 540):
            values = np.zeros((days, 50))
            self.assertEqual(len(task.build_windows(values)),
                             days - task.WINDOW_SIZE + 1,
                             f"{days} days")

    def test_a_user_with_too_few_days_produces_nothing(self):
        """Fewer than fourteen days cannot fill a window, so the user is dropped."""
        for days in (0, 1, 13):
            self.assertEqual(task.build_windows(np.zeros((days, 50))), [])

    def test_exactly_fourteen_days_produces_exactly_one_window(self):
        windows = task.build_windows(np.zeros((14, 50)))
        self.assertEqual(len(windows), 1)

    def test_every_window_is_the_input_matrix_the_thesis_describes(self):
        """14 x 50, that is 700 values per window."""
        windows = task.build_windows(np.zeros((20, 50)))
        for window in windows:
            self.assertEqual(window.shape, (14, 50))
            self.assertEqual(window.size, 700)

    def test_windows_are_in_chronological_order(self):
        """Day number stored in column 0, so the ordering is checkable."""
        values = np.arange(20).reshape(20, 1).repeat(50, axis=1).astype(float)
        windows = task.build_windows(values)
        first_days = [w[0, 0] for w in windows]
        self.assertEqual(first_days, sorted(first_days))
        self.assertEqual(first_days[0], 0.0)


class TestWindowOverlap(unittest.TestCase):
    """Stride 1 means consecutive windows share all but one day.

    This is not a defect of the windowing, which is meant to be dense. It matters
    because `load_partitioned_data` then splits those windows at random into the
    client's train and validation sets, so the two sides hold near-duplicates of
    each other. The resulting per-client validation loss, which is what Flower
    aggregates and reports as the distributed loss each round, is therefore not
    an independent measurement and cannot be used as evidence about overfitting.
    """

    def test_consecutive_windows_share_all_but_one_day(self):
        values = np.arange(20).reshape(20, 1).repeat(50, axis=1).astype(float)
        windows = task.build_windows(values)

        overlap = np.intersect1d(windows[0][:, 0], windows[1][:, 0])
        self.assertEqual(len(overlap), task.WINDOW_SIZE - 1)

    def test_a_random_split_of_these_windows_cannot_be_independent(self):
        """Any split leaves a validation window 13/14 identical to a train one."""
        values = np.arange(40).reshape(40, 1).repeat(50, axis=1).astype(float)
        windows = task.build_windows(values)

        rng = np.random.RandomState(0)
        order = rng.permutation(len(windows))
        train, validation = order[:20], order[20:]

        worst_overlap = max(
            len(np.intersect1d(windows[v][:, 0], windows[t][:, 0]))
            for v in validation for t in train)
        self.assertEqual(worst_overlap, task.WINDOW_SIZE - 1)


class PartitionTestCase(unittest.TestCase):
    """A frame of 100 users, which is what both partitioners take."""

    def setUp(self):
        self._non_iid = config.get("data", "is_non_iid")
        self.users = [f"USER{i:04d}" for i in range(100)]
        self.df = pd.DataFrame({"user": self.users})

    def tearDown(self):
        config._active_config["data"]["is_non_iid"] = self._non_iid

    def _set_non_iid(self, enabled, alpha=0.5):
        config._active_config["data"]["is_non_iid"] = enabled
        config._active_config["data"]["non_iid_alpha"] = alpha


class TestIIDPartition(PartitionTestCase):
    def setUp(self):
        super().setUp()
        self._set_non_iid(False)

    def test_every_user_lands_in_exactly_one_partition(self):
        chunks = task.partition_users(self.df, 10)
        assigned = [user for chunk in chunks for user in chunk]

        self.assertEqual(len(assigned), len(self.users))
        self.assertEqual(sorted(assigned), sorted(self.users))

    def test_partitions_are_even(self):
        chunks = task.partition_users(self.df, 10)
        self.assertEqual({len(chunk) for chunk in chunks}, {10})

    def test_the_assignment_is_deterministic_and_alphabetical(self):
        """Client 0 always holds the first users by ID, never a random draw.

        Worth knowing rather than assuming: an earlier version calibrated the
        Z-score reference from client 0 alone, which under this partition is a
        fixed slice of the user list, not a sample of it.
        """
        chunks = task.partition_users(self.df, 10)
        self.assertEqual(list(chunks[0]), self.users[:10])
        self.assertEqual(list(chunks[-1]), self.users[-10:])

    def test_an_uneven_user_count_loses_nobody(self):
        df = pd.DataFrame({"user": [f"U{i}" for i in range(97)]})
        chunks = task.partition_users(df, 10)
        self.assertEqual(sum(len(c) for c in chunks), 97)


class TestDirichletPartition(PartitionTestCase):
    def setUp(self):
        super().setUp()
        self._set_non_iid(True, alpha=0.5)

    def test_every_user_lands_in_exactly_one_partition(self):
        chunks = task.partition_users(self.df, 10)
        assigned = [user for chunk in chunks for user in chunk]

        self.assertEqual(len(assigned), len(self.users))
        self.assertEqual(sorted(assigned), sorted(self.users))

    def test_no_client_is_left_without_users(self):
        """A client with nothing to train on would silently contribute zeros."""
        for alpha in (0.1, 0.5, 1.0):
            chunks = task._dirichlet_partition(self.users, 10, alpha=alpha, seed=1)
            self.assertTrue(all(len(chunk) >= 1 for chunk in chunks), f"alpha {alpha}")

    def test_the_same_seed_reproduces_the_same_partition(self):
        first = task._dirichlet_partition(self.users, 10, alpha=0.5, seed=7)
        second = task._dirichlet_partition(self.users, 10, alpha=0.5, seed=7)
        for a, b in zip(first, second):
            self.assertEqual(list(a), list(b))

    def test_a_different_seed_gives_a_different_partition(self):
        first = task._dirichlet_partition(self.users, 10, alpha=0.5, seed=7)
        second = task._dirichlet_partition(self.users, 10, alpha=0.5, seed=8)
        self.assertNotEqual([list(c) for c in first], [list(c) for c in second])

    def test_a_lower_alpha_makes_the_split_more_uneven(self):
        """This is the whole point of the parameter, so it is worth checking."""
        def spread(alpha):
            sizes = [len(c) for c in
                     task._dirichlet_partition(self.users, 10, alpha=alpha, seed=3)]
            return np.std(sizes)

        self.assertGreater(spread(0.1), spread(100.0))

    def test_a_high_alpha_approaches_an_even_split(self):
        sizes = [len(c) for c in
                 task._dirichlet_partition(self.users, 10, alpha=1000.0, seed=3)]
        self.assertLess(np.std(sizes), 3.0)


class TestFeatureSelection(unittest.TestCase):
    def test_the_configured_fifty_features_are_used(self):
        """The count the parameter arithmetic and the input matrix both rest on."""
        self.assertEqual(len(config.get("data", "selected_features")), 50)

    def test_metadata_columns_never_become_features(self):
        columns = list(config.get("data", "selected_features")) + [
            "user", "day", "insider", "week"]
        df = pd.DataFrame(columns=columns)

        features = task.select_features(df)
        self.assertEqual(len(features), 50)
        for metadata in ("user", "day", "insider", "week"):
            self.assertNotIn(metadata, features)

    def test_a_missing_column_is_dropped_rather_than_faked(self):
        """Silently filling a missing feature with zeros would shift every score."""
        configured = list(config.get("data", "selected_features"))
        df = pd.DataFrame(columns=configured[:-1] + ["user"])

        features = task.select_features(df)
        self.assertEqual(len(features), 49)
        self.assertNotIn(configured[-1], features)


class TestVarianceFilter(unittest.TestCase):
    """The systematic feature selection the jury asked for (J5).

    Off by default, because dropping features changes the input dimension and
    therefore the parameter count every communication figure derives from.
    """

    def setUp(self):
        self._original = config.get("data", "drop_constant_features")
        # 'flat' never varies; 'live' does.
        self.df = pd.DataFrame({"live": [1.0, 5.0, 9.0], "flat": [3.0, 3.0, 3.0]})

    def tearDown(self):
        config._active_config["data"]["drop_constant_features"] = self._original

    def test_a_feature_with_no_spread_is_dropped(self):
        kept = task.drop_constant_features(self.df, ["live", "flat"])
        self.assertEqual(kept, ["live"])

    def test_float_noise_does_not_save_a_constant_feature(self):
        """The exact shape of the four real ones: zero plus 7.1e-15."""
        df = pd.DataFrame({"noisy": [0.0, 7.105427357601002e-15, 0.0]})
        self.assertEqual(task.drop_constant_features(df, ["noisy"]), [])

    def test_the_filter_agrees_with_the_scaler_on_what_is_constant(self):
        """Two definitions of constant would put the two out of step."""
        from federated_ueba import scaling
        borderline = pd.DataFrame({"x": [0.0, scaling.constant_feature_tolerance() * 10, 0.0]})
        self.assertEqual(task.drop_constant_features(borderline, ["x"]), ["x"])

    def test_the_filter_is_off_by_default(self):
        self.assertFalse(config.get("data", "drop_constant_features"))


if __name__ == "__main__":
    unittest.main()
