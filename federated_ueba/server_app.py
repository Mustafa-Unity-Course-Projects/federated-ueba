from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from federated_ueba.strategy import FedAvgWithModelSaving


NUM_ROUNDS = 20


def server_fn(context):
    # 1. Define Strategy (Aggregates local LSTM weights into a global model)
    strategy = FedAvgWithModelSaving(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        save_path="./model_pickles"
    )

    # 2. Configure the server rounds
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)