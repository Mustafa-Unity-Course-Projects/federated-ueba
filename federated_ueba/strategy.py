"""Server-side FedAvg, extended with checkpointing and the server plugin step.

Two things are added to plain FedAvg:

  checkpointing   every round's aggregated model is written to disk, because the
                  round that detects best is chosen afterwards by rescoring each
                  one, not during training
  server plugins  whatever the clients did to shrink their upload has to be
                  undone before the next round goes out, which is what
                  `apply_on_server` does (fp16 back to fp32, for instance)

Aggregation itself is untouched. That is deliberate: every compression variant
hands FedAvg a complete model, so the averaging the thesis describes is Flower's
own implementation rather than something reimplemented here.
"""

import json
import pickle
from logging import INFO
from pathlib import Path

import flwr
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log

from config_manager import config
from federated_ueba.efficiency_plugins import get_downlink_codec, get_plugin

# Key the checkpoint dictionary is stored under. `federated_insider_detection`
# reads it back by the same name; the file holds a plain list of arrays in
# state_dict order, which is the form Flower aggregates.
CHECKPOINT_KEY = "global_parameters"


class ModelSavingMixin:
    """Checkpointing and the two efficiency hooks, independent of the base rule.

    A mixin rather than a subclass of one strategy, because FedAvg and FedProx are
    separate classes in Flower and both need identical checkpointing, identical
    uplink undo and identical downlink compression. Subclassing one of them and
    copying the body into the other is how the two would drift, and a drift here
    would make the FedAvg and FedProx runs incomparable for a reason that has
    nothing to do with either aggregation rule.

    Mixed in ahead of the strategy in the MRO, so every `super()` call below
    reaches the aggregation rule that follows it.
    """

    def __init__(self, save_path, settings, *args, **kwargs):
        """`settings` is the configuration to read, passed in rather than imported.

        It used to read the module-level singleton, which made the `config`
        argument of `build_strategy` a lie: a caller could hand it one
        configuration and get a strategy built from another. Nothing in a run hit
        that, because the singleton was the only configuration there was, but it
        meant the compression settings could not be exercised without mutating
        global state.
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        # No partition_id: this is the server side, which has no residual state
        # of its own. Only the client-side error feedback keeps per-client files.
        self.efficiency_plugin = get_plugin(settings)
        self.downlink = get_downlink_codec(settings)
        super().__init__(*args, **kwargs)

    def _compress_downlink(self, parameters):
        """Shrink the broadcast the same way for training and for evaluation.

        Both directions of every round go through here, so the reported download
        figure covers all of it. Evaluation is the larger half: at fraction_fit
        0.5 a round broadcasts to 25 clients to train and to all 50 to evaluate.
        """
        ndarrays = self.downlink.compress(parameters_to_ndarrays(parameters))
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round, parameters, client_manager):
        return super().configure_fit(
            server_round, self._compress_downlink(parameters), client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return super().configure_evaluate(
            server_round, self._compress_downlink(parameters), client_manager)

    FAILURE_LOG = "client_failures.json"

    def _record_failures(self, server_round, results, failures):
        """Write down any round that lost clients, for the runner to check.

        Flower treats a failed client as a client that did not participate: the
        round still aggregates, the run still finishes, and the summary still
        says `done`. That is the wrong default here. A round that averaged 5
        clients instead of 25 is not the configuration the thesis describes, and
        nothing downstream could tell the difference afterwards. One sweep lost
        20 of 25 clients for ten consecutive rounds to Ray actors dying on
        Windows, and the only evidence was a line buried in a 5 MB log.

        Written to disk rather than returned, because the strategy runs inside
        `flower-simulation`, a separate interpreter from the runner that decides
        whether the run counts.
        """
        if not failures:
            return
        path = self.save_path / self.FAILURE_LOG
        record = {}
        if path.exists():
            with open(path) as f:
                record = json.load(f)
        record[str(server_round)] = {
            "results": len(results), "failures": len(failures)}
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        log(INFO, f"Round {server_round} lost {len(failures)} of "
                  f"{len(results) + len(failures)} clients; recorded in {path}")

    @staticmethod
    def _in_partition_order(results):
        """Fix the order the updates are summed in.

        Ray hands back results as the actors finish, so the same round can
        aggregate the same 25 updates in 25 different orders. Floating point
        addition is not associative: the sums differ in their last bits, the next
        round starts from a slightly different model, and over 50 rounds two runs
        of the same seed diverge visibly. Sorting by partition removes it.

        Clients that did not report a partition sort last rather than raising, so
        an older checkpoint or a third-party client still aggregates.
        """
        return sorted(
            results,
            key=lambda pair: int(pair[1].metrics.get("partition_id", 1 << 62)))

    def aggregate_fit(self, server_round, results, failures):
        """Average the client updates, then undo their transport encoding."""
        self._record_failures(server_round, results, failures)
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, self._in_partition_order(results), failures)

        # None when every client in the round failed; there is nothing to undo.
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            ndarrays = self.efficiency_plugin.apply_on_server(ndarrays)
            aggregated_parameters = ndarrays_to_parameters(ndarrays)

        return aggregated_parameters, metrics

    def evaluate(self, server_round, parameters):
        """Save this round's global model, then evaluate as the base rule would.

        The checkpoint holds the server's own model, not the compressed
        broadcast: `downlink_plugins` is a transport decision and never reaches
        this, so quantizing the downlink leaves the checkpoint untouched.

        `active_plugins` is a different matter and the dtype here does not tell
        the whole story. Under uplink quantization the clients return float16
        arrays, and FedAvg's weighted average keeps numpy's float16 all the way
        through, so the aggregate is computed in fp16 arithmetic; the cast in
        `aggregate_fit` widens the container back to float32 but not the
        precision. Every value in a checkpoint from that arm is exactly fp16
        representable, which is also why its broadcast entropy-codes so much
        smaller than the dense baseline's. Uplink quantization is therefore an
        aggregation decision as well as a transport one, and the thesis has to
        describe it that way.

        Checkpointing hangs off `evaluate` because it is the one hook that runs
        once per round on the server with the current global model in hand,
        including round 0, which is the untrained starting point. Detection
        quality is not measured here: it needs the full scoring pipeline over all
        1000 users, which happens after training in a separate pass.
        """
        self._save_global_model(server_round, parameters)
        return super().evaluate(server_round, parameters)

    def _save_global_model(self, server_round, parameters):
        """Write one round's global model, named by round number."""
        # Recreated defensively: the directory is cleared at the start of a run,
        # and a checkpoint failing to save would only surface much later, as a
        # gap in the round-by-round curve.
        self.save_path.mkdir(exist_ok=True, parents=True)
        filename = self.save_path / f"parameters_round_{server_round}.pkl"

        with open(filename, "wb") as f:
            pickle.dump({CHECKPOINT_KEY: parameters_to_ndarrays(parameters)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        log(INFO, f"Checkpoint saved to: {filename}")


class FedAvgWithModelSaving(ModelSavingMixin, flwr.server.strategy.FedAvg):
    """Plain FedAvg. The rule every reported result in the thesis uses."""


class FedProxWithModelSaving(ModelSavingMixin, flwr.server.strategy.FedProx):
    """FedProx, which differs from FedAvg only on the client.

    Its aggregation is FedAvg's. What Flower's FedProx adds on the server is
    sending `proximal_mu` down in the fit configuration; the client reads it and
    adds the penalty to its loss. See `federated_ueba.proximal`.
    """


# One name per aggregation rule, resolved from [tool.fueba.federation] strategy.
# A registry rather than an if-else so that adding a rule is adding an entry, and
# so that an unknown name fails with the list of known ones.
STRATEGIES = {
    "fedavg": FedAvgWithModelSaving,
    "fedprox": FedProxWithModelSaving,
}


def build_strategy(settings, **kwargs):
    """Construct the configured strategy, reading `settings` and nothing global.

    `proximal_mu` is passed only to FedProx, because FedAvg does not accept it
    and silently dropping it would let a run configured for FedProx train as
    FedAvg while its summary recorded a mu it never applied.
    """
    name = settings.get("federation", "strategy")
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Expected one of "
                         f"{sorted(STRATEGIES)}.")

    if name == "fedprox":
        kwargs["proximal_mu"] = settings.get("federation", "proximal_mu")

    return STRATEGIES[name](settings=settings, **kwargs)
