"""A client manager whose sampling is reproducible.

This closes the last unseeded source of randomness in the pipeline, and it was
recorded as unfixable long enough to be worth explaining why it is not.

Flower's `SimpleClientManager.sample` does this:

    available_cids = list(self.clients)
    ...
    sampled_cids = random.sample(available_cids, num_clients)

`self.clients` is a dict filled as clients register, and under Ray they register
in whatever order the actors happen to come up. So the *list* differs between two
runs of the same seed, and sampling from differently ordered lists gives
different clients however well the random number generator is seeded. Seeding
alone could never have fixed it, which is why seeding alone did not.

Two changes make it deterministic:

  sorted order   the candidate list is sorted before sampling, so the sequence
                 the generator draws from no longer depends on actor startup
  private stream the draw uses this object's own `random.Random` rather than the
                 module-level one, so anything else in the server process that
                 consumes randomness cannot shift which clients a round gets

The private stream is seeded once and then used for every call, so reproducibility
depends on the *sequence* of calls being the same between runs. It is: one
`configure_fit` and one `configure_evaluate` per round, in that order, for a
fixed number of rounds.
"""

import random

from flwr.common import GetPropertiesIns
from flwr.server.client_manager import SimpleClientManager

from federated_ueba.seeding import derive_seed

PARTITION_KEY = "partition_id"


class SeededClientManager(SimpleClientManager):
    """`SimpleClientManager`, with a reproducible `sample`."""

    def __init__(self, seed, timeout=None):
        super().__init__()
        self.seed = seed
        self.timeout = timeout
        self._rng = random.Random(derive_seed(seed))
        self._partitions = {}

    def _partition_of(self, cid):
        """This client's partition, asked once and remembered.

        Sorting by `cid` would be enough if a cid meant anything, and it does
        not: Flower draws node ids from `os.urandom`, so they are different on
        every run and an ordering over them is arbitrary in a way no seed can
        reach. The partition is the identifier that names the same data twice,
        so it is the one worth asking for.

        Cached because it cannot change: a partition is fixed at node
        registration, and asking again every round would put 50 extra
        round-trips into every round of every experiment.
        """
        if cid not in self._partitions:
            response = self.clients[cid].get_properties(
                GetPropertiesIns(config={}), timeout=self.timeout, group_id=None)
            self._partitions[cid] = int(response.properties[PARTITION_KEY])
        return self._partitions[cid]

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        """Draw `num_clients`, identically for a given seed.

        Deliberately not calling `super().sample`: the ordering and the generator
        are exactly what has to change, and they are both inside that method.
        Everything else here mirrors it, including waiting for enough clients to
        connect and honouring a criterion.
        """
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids
                              if criterion.select(self.clients[cid])]

        if num_clients > len(available_cids):
            # Matches Flower's behaviour: too few clients yields nothing rather
            # than a short round, so the caller sees a failed round instead of a
            # quietly smaller one.
            return []

        # Ordered by partition, which is the same sequence on every run. This is
        # the fix; the seeded generator alone was never enough, because it was
        # drawing from a list whose order came from Ray's actor scheduling.
        available_cids.sort(key=self._partition_of)

        sampled_cids = self._rng.sample(available_cids, num_clients)
        # Returned in partition order too, so that anything downstream which
        # sums over the results does it in the same sequence every run. Floating
        # point addition is not associative, and Ray returns results in whatever
        # order they finish.
        sampled_cids.sort(key=self._partition_of)
        return [self.clients[cid] for cid in sampled_cids]
