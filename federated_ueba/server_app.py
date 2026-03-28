from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from config_manager import config
from federated_ueba.strategy import FedAvgWithModelSaving


NUM_ROUNDS = config.get("federation", "num_rounds")
FRACTION_FIT = config.get("federation", "fraction_fit")
MIN_FIT_CLIENTS = config.get("federation", "num_clients")
MIN_AVAILABLE_CLIENTS = config.get("federation", "num_clients")
SAVE_PATH = config.get("federation", "save_path")


def server_fn(context):
    # 1. Define Strategy (Aggregates local LSTM weights into a global model)
    strategy = FedAvgWithModelSaving(
        fraction_fit=FRACTION_FIT,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        save_path=SAVE_PATH,
    )

    # 2. Configure the server rounds
    config_ = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config_)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)
