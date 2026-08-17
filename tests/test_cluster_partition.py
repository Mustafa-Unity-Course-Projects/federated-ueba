"""Non-IID has to mean something, and quantity skew does not mean enough.

Raised on 2026-08-05: FedAvg showed no degradation under the non-IID partition,
which contradicts the federated learning literature. It turned out the literature
was not being tested. `_dirichlet_partition` skews how many users a client holds,
not which kind, so every client stays a random sample of the same distribution
and there is no client drift for FedAvg to handle badly.

Trying to fix it uncovered something else. Users cannot be told apart by their
mean feature vector at all: the processed dataset holds percentile deviations
from each user's own baseline, so every signed mean sits near zero and k-means
puts 996 of 1000 users in one cluster. What does separate them is how far they
typically stray from that baseline, which is why the profile is a mean absolute
deviation.

These tests pin both halves: the partition is a real partition, and it produces
genuinely different client populations rather than differently sized ones.
"""

import unittest

import numpy as np
import pandas as pd

from federated_ueba.task import (
    _cluster_dirichlet_partition, _dirichlet_partition, _role_labels,
    _user_behaviour_clusters,
)

# Real configured feature names, not invented ones. `select_features` intersects
# the frame's columns with the 50 named in pyproject.toml, so a synthetic frame
# with made-up column names yields no features at all and the clustering fails
# on an empty array rather than on anything the test meant to check.
def _configured_features(count=3):
    from config_manager import config
    return list(config.get("data", "selected_features"))[:count]


FEATURES = _configured_features()


def synthetic_frame(seed=0):
    """Users of three volatilities, all with a signed mean near zero.

    Built to look like the real thing in the way that matters: nobody is
    distinguishable by their average, everybody is distinguishable by their
    spread. `role` tracks the volatility group, which is what the real dataset
    does loosely and what makes the two label sources comparable here.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for user in range(90):
        group = user % 3
        scale = [0.5, 3.0, 12.0][group]
        for _ in range(40):
            values = rng.normal(0.0, scale, size=len(FEATURES))
            rows.append({"user": user, "role": float(group),
                         **dict(zip(FEATURES, values))})
    return pd.DataFrame(rows)


def mixture_divergence(chunks, labels, users, num_clusters):
    """Size-weighted total variation between client and global cluster mixtures.

    Weighted, because a client holding one user always looks extreme by sampling
    noise alone; an unweighted average would report quantity skew as if it were
    distribution skew.
    """
    index = {user: label for user, label in zip(users, labels)}
    globally = np.bincount(labels, minlength=num_clusters) / len(users)

    distances, weights = [], []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        local = np.bincount([index[u] for u in chunk],
                            minlength=num_clusters) / len(chunk)
        distances.append(0.5 * np.abs(local - globally).sum())
        weights.append(len(chunk))
    return float(np.average(distances, weights=weights))


class ClusteringUsers(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_frame()
        self.users = sorted(self.df["user"].unique())

    def test_volatility_groups_are_recovered(self):
        labels = _user_behaviour_clusters(self.df, self.users, 3, seed=1)
        sizes = sorted(np.bincount(labels, minlength=3))
        # Three equal groups by construction; k-means should not collapse them
        # into one cluster and two singletons the way it does on signed means.
        self.assertGreater(sizes[0], len(self.users) // 6)

    def test_a_label_is_returned_for_every_user_in_order(self):
        labels = _user_behaviour_clusters(self.df, self.users, 3, seed=1)
        self.assertEqual(len(labels), len(self.users))


class ClusterPartition(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_frame()
        self.users = sorted(self.df["user"].unique())

    def partition(self, alpha=0.5, clients=10, seed=1):
        return _cluster_dirichlet_partition(
            self.df, self.users, clients, alpha, 3, seed=seed)

    def test_every_user_is_assigned_exactly_once(self):
        chunks = self.partition()
        assigned = np.concatenate(chunks)
        self.assertEqual(sorted(assigned), self.users)

    def test_no_client_is_left_with_nothing(self):
        """An empty client trains on nothing and averages an untrained model in."""
        for alpha in (0.5, 0.05):
            chunks = self.partition(alpha=alpha, clients=20)
            self.assertTrue(all(len(chunk) > 0 for chunk in chunks),
                            f"empty client at alpha {alpha}")

    def test_the_same_seed_gives_the_same_partition(self):
        first = [list(c) for c in self.partition(seed=7)]
        second = [list(c) for c in self.partition(seed=7)]
        self.assertEqual(first, second)

    def test_different_seeds_give_different_partitions(self):
        first = [list(c) for c in self.partition(seed=1)]
        second = [list(c) for c in self.partition(seed=2)]
        self.assertNotEqual(first, second)

    def test_clients_hold_different_populations_not_just_different_sizes(self):
        """The whole point, stated as a number.

        On the real dataset the size-weighted divergence is 0.119 for quantity
        skew and 0.434 for cluster skew at the same alpha.
        """
        labels = _user_behaviour_clusters(self.df, self.users, 3, seed=1)

        quantity = _dirichlet_partition(self.users, 10, 0.5, seed=1)
        cluster = self.partition(alpha=0.5, clients=10)

        self.assertGreater(
            mixture_divergence(cluster, labels, self.users, 3),
            mixture_divergence(quantity, labels, self.users, 3) * 1.5)

    def test_role_labels_come_from_the_data_not_from_clustering(self):
        """The natural partition, which is the one the literature prefers."""
        labels = _role_labels(self.df, self.users)
        self.assertEqual(len(labels), len(self.users))
        self.assertEqual(len(set(labels)), 3)

    def test_a_role_partition_skews_the_role_mixture(self):
        labels = _role_labels(self.df, self.users)
        by_role = _cluster_dirichlet_partition(
            self.df, self.users, 10, 0.5, 3, seed=1, labels=labels)
        quantity = _dirichlet_partition(self.users, 10, 0.5, seed=1)

        self.assertGreater(
            mixture_divergence(by_role, labels, self.users, 3),
            mixture_divergence(quantity, labels, self.users, 3))

    def test_a_smaller_alpha_skews_harder(self):
        labels = _user_behaviour_clusters(self.df, self.users, 3, seed=1)
        mild = mixture_divergence(self.partition(alpha=1.0, clients=10),
                                  labels, self.users, 3)
        harsh = mixture_divergence(self.partition(alpha=0.05, clients=10),
                                   labels, self.users, 3)
        self.assertGreater(harsh, mild)


if __name__ == "__main__":
    unittest.main()
