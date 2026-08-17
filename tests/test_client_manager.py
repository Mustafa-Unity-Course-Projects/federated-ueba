"""Client sampling has to be reproducible, and for a while it was not.

Recorded as a reproducibility limit (Y4) on the reasoning that Flower samples
inside its own client manager and the seed cannot reach it. Half right: the seed
does reach it, but `SimpleClientManager.sample` draws from `list(self.clients)`,
whose order is the order Ray's actors registered in. Two runs of the same seed
sample from differently ordered lists and get different clients, no matter how
the generator is seeded.

These tests pin both halves of the fix: the candidate order no longer depends on
registration order, and the draw no longer shares the module-level generator with
whatever else runs in the server process.
"""

import random
import unittest
from unittest.mock import MagicMock

from federated_ueba.client_manager import SeededClientManager


def manager_with(cids, seed=1, partitions=None):
    """A manager holding these clients, registered in the order given.

    `cids` stand in for Flower's node ids, which come from `os.urandom` and so
    are different on every run; `partitions` is what stays the same. By default
    the partition is the position in `cids`, mimicking Flower assigning them in
    node creation order.
    """
    manager = SeededClientManager(seed)
    for position, cid in enumerate(cids):
        proxy = MagicMock()
        proxy.cid = cid
        partition = position if partitions is None else partitions[cid]
        proxy.get_properties.return_value = MagicMock(
            properties={"partition_id": partition})
        manager.clients[cid] = proxy
    return manager


def sampled_cids(manager, count):
    return [proxy.cid for proxy in manager.sample(count)]


class SamplingIsReproducible(unittest.TestCase):
    def test_the_same_seed_samples_the_same_clients(self):
        cids = [f"client{i}" for i in range(20)]
        first = sampled_cids(manager_with(cids, seed=3), 5)
        second = sampled_cids(manager_with(cids, seed=3), 5)
        self.assertEqual(first, second)

    def test_different_seeds_sample_differently(self):
        cids = [f"client{i}" for i in range(20)]
        first = sampled_cids(manager_with(cids, seed=1), 5)
        second = sampled_cids(manager_with(cids, seed=2), 5)
        self.assertNotEqual(first, second)

    def test_registration_order_does_not_change_which_partitions_are_drawn(self):
        """The actual bug, and why sorting node ids could not have fixed it.

        Two runs see the same partitions behind different node ids, registered
        in a different order. The partitions drawn must match; the node ids
        naturally will not.
        """
        cids = [f"client{i}" for i in range(20)]
        partitions = {cid: i for i, cid in enumerate(cids)}

        renamed = [f"node{i}" for i in range(20)]
        # Same partitions, different ids, registered in a different order.
        renamed_partitions = {cid: i for i, cid in enumerate(renamed)}
        shuffled = list(renamed)
        random.Random(99).shuffle(shuffled)

        first = manager_with(cids, seed=4, partitions=partitions)
        second = manager_with(shuffled, seed=4, partitions=renamed_partitions)

        drawn_first = [p.get_properties.return_value.properties["partition_id"]
                       for p in first.sample(5)]
        drawn_second = [p.get_properties.return_value.properties["partition_id"]
                        for p in second.sample(5)]
        self.assertEqual(drawn_first, drawn_second)

    def test_the_module_generator_cannot_shift_the_draw(self):
        """Anything else in the server process may consume randomness."""
        cids = [f"client{i}" for i in range(20)]

        random.seed(0)
        first = sampled_cids(manager_with(cids, seed=5), 5)

        random.seed(0)
        for _ in range(37):
            random.random()
        second = sampled_cids(manager_with(cids, seed=5), 5)

        self.assertEqual(first, second)

    def test_successive_rounds_draw_different_subsets(self):
        """Reproducible must not mean identical every round."""
        manager = manager_with([f"client{i}" for i in range(20)], seed=6)
        rounds = [sampled_cids(manager, 5) for _ in range(4)]
        self.assertGreater(len({tuple(r) for r in rounds}), 1)

    def test_the_sequence_of_rounds_repeats_across_runs(self):
        cids = [f"client{i}" for i in range(20)]
        first = [sampled_cids(m, 5) for m in [manager_with(cids, seed=7)] * 1]
        one = manager_with(cids, seed=7)
        two = manager_with(cids, seed=7)
        self.assertEqual([sampled_cids(one, 5) for _ in range(3)],
                         [sampled_cids(two, 5) for _ in range(3)])
        self.assertTrue(first)


class SamplingContract(unittest.TestCase):
    """What the replacement must keep doing, since it does not call super()."""

    def test_the_requested_number_is_returned(self):
        manager = manager_with([f"client{i}" for i in range(20)], seed=1)
        self.assertEqual(len(manager.sample(7)), 7)

    def test_too_few_clients_yields_nothing(self):
        """Flower's behaviour: a failed round, not a quietly smaller one."""
        manager = manager_with([f"client{i}" for i in range(3)], seed=1)
        self.assertEqual(manager.sample(10, min_num_clients=1), [])

    def test_a_criterion_filters_the_candidates(self):
        manager = manager_with([f"client{i}" for i in range(10)], seed=1)
        criterion = MagicMock()
        criterion.select = lambda proxy: proxy.cid.endswith(("0", "1", "2"))

        chosen = [p.cid for p in manager.sample(2, min_num_clients=1,
                                                criterion=criterion)]
        self.assertTrue(all(cid.endswith(("0", "1", "2")) for cid in chosen))

    def test_proxies_are_returned_not_identifiers(self):
        manager = manager_with([f"client{i}" for i in range(10)], seed=1)
        for proxy in manager.sample(3):
            self.assertTrue(hasattr(proxy, "cid"))


if __name__ == "__main__":
    unittest.main()
