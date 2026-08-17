"""The Flower server: clears the previous run, then drives the rounds.

Entry point for `flower-simulation`, which the experiment runner launches as a
subprocess. Everything it needs comes from the environment through
`config_manager`, because a subprocess cannot be handed a Python object.
"""

import os
import shutil
import sys
from typing import Dict, List, Tuple

from flwr.common import Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from config_manager import config
from federated_ueba.client_manager import SeededClientManager
from federated_ueba.seeding import seed_everything
from federated_ueba.strategy import build_strategy

# Safety net for a direct `flower-simulation` invocation, which does not go
# through the runner and so never sees PYTHONIOENCODING. Without this, the emoji
# in cleanup() below raise UnicodeEncodeError inside the ServerApp thread and the
# whole simulation aborts before round 1.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

NUM_ROUNDS = config.get("federation", "num_rounds")
FRACTION_FIT = config.get("federation", "fraction_fit")
MIN_FIT_CLIENTS = config.get("federation", "min_fit_clients")
MIN_AVAILABLE_CLIENTS = config.get("federation", "min_available_clients")
SAVE_PATH = config.get("federation", "save_path")
SCALER_DIR = config.get("data", "scaler_dir")
# Only the comm/ subdirectory of this is touched here; the reports themselves are
# written by the evaluation pass and must survive.
FEDERATED_EVAL_REPORTS_DIR = "federated_evaluation_reports"

def cleanup():
    """Clear the training artefacts of this run before it starts.

    Single authority for training-side cleanup: the checkpoints, the per-client
    scalers and the communication log. Evaluation artefacts belong to
    federated_insider_detection.py and are left alone here.

    SAVE_PATH and SCALER_DIR are already seed- and experiment-qualified by
    config_manager, so this can only ever delete the current run's own data.
    A stale communication log is what previously caused runs to report the MB
    total of an entirely different run, so it is always removed.
    """
    print(f"🧹 Clearing training artefacts for run_id: {config.run_id}...")

    for folder in [SAVE_PATH, SCALER_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  Removed folder: {folder}")

    comm_dir = os.path.join(FEDERATED_EVAL_REPORTS_DIR, str(config.run_id), "comm")
    if os.path.exists(comm_dir):
        shutil.rmtree(comm_dir)
        print(f"  Removed stale communication logs: {comm_dir}")

def aggregate_evaluate_metrics(
    results: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Average the clients' evaluation metrics, weighted by how much data each saw.

    Weighted rather than a plain mean, because clients hold different numbers of
    windows, and under a Dirichlet partition they differ a great deal. An
    unweighted average would let a client with a handful of windows count as much
    as one with thousands.

    This produces the per-round loss curve Flower reports. Detection quality is
    not measured here; see `federated_insider_detection.run_evaluation`.
    """
    if not results:
        return {}

    weighted_totals: Dict[str, float] = {}
    total_examples = 0

    for num_examples, metrics in results:
        total_examples += num_examples
        for key, value in metrics.items():
            weighted_totals[key] = weighted_totals.get(key, 0.0) + float(value) * num_examples

    return {key: total / total_examples for key, total in weighted_totals.items()}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Per-round settings sent to every client selected to train.

    The round number is what lets a client seed itself reproducibly: Ray does not
    guarantee which actor handles which partition, so a client keys its random
    stream on (run seed, partition, round) rather than on anything about the
    actor it happens to be running in. See `federated_ueba.seeding`.
    """
    return {"server_round": server_round}


def initial_parameters():
    """The untrained global model every run starts from.

    Imported here rather than at module scope because `federated_ueba.task` reads
    configuration when it loads, and this module is imported by the simulation
    before that configuration is necessarily settled.
    """
    from flwr.common import ndarrays_to_parameters

    from federated_ueba import task

    # Sized the way the clients size theirs, not from the configured feature
    # list. The two differ whenever a filter is on: `features-filtered` drops
    # four all-zero features, so the clients build a 46-wide model while the
    # configured list still names 50. Reading the list here made every client in
    # that experiment reject the broadcast with a shape mismatch on every round.
    model = task.LSTMAutoencoder(
        input_dim=task.input_dimension(),
        hidden_dim=config.get("model", "hidden_dim"))
    return ndarrays_to_parameters(
        [value.detach().cpu().numpy() for value in model.state_dict().values()])


def server_fn(context):
    """Assemble the strategy and round configuration for one simulation."""
    # Runs once, before round 1, and only ever on this run's own directories.
    cleanup()

    # This process chooses which clients each round samples, so it needs the run
    # seed too. Without it the sampled subset differed between runs of the same
    # seed, which alone was enough to make results irreproducible.
    seed_everything(config.seed)

    strategy = build_strategy(
        config,
        on_fit_config_fn=fit_config,
        # Built here rather than fetched from a client. Flower's default is to
        # ask one arbitrary client for its freshly initialised weights, and which
        # client that is depends on the order Ray happens to register actors in,
        # so round 0 differed between runs of the same seed. Constructing the
        # model in this seeded process removes that source entirely.
        initial_parameters=initial_parameters(),
        # Fraction of the federation sampled to train each round. At 50 nodes and
        # 0.5 that is 25 clients per round, which is what the communication
        # figures assume.
        fraction_fit=FRACTION_FIT,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        save_path=SAVE_PATH,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    # Named to avoid shadowing the imported `config`, which this function needs.
    server_config = ServerConfig(num_rounds=NUM_ROUNDS)

    # Flower's default manager samples from `list(self.clients)`, whose order is
    # Ray's actor registration order, so the same seed drew different clients on
    # different runs. See `federated_ueba.client_manager`.
    return ServerAppComponents(strategy=strategy, config=server_config,
                               client_manager=SeededClientManager(config.seed))


app = ServerApp(server_fn=server_fn)
