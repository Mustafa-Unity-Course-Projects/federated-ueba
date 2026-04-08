import os
import shutil
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar

from config_manager import config
from federated_ueba.strategy import FedAvgWithModelSaving

NUM_ROUNDS = config.get("federation", "num_rounds")
FRACTION_FIT = config.get("federation", "fraction_fit")
MIN_FIT_CLIENTS = config.get("federation", "min_fit_clients") or 2
MIN_AVAILABLE_CLIENTS = config.get("federation", "min_available_clients") or 2
SAVE_PATH = config.get("federation", "save_path")
SCALER_DIR = config.get("data", "scaler_dir") or "scaler_data"

def cleanup():
    """Removes previous run data: model_pickle, scaler_data, and communication logs for the CURRENT run_id."""
    print(f"🧹 Cleaning up previous run data for run_id: {config.run_id}...")
    
    # Remove Directories
    for folder in [SAVE_PATH, SCALER_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  Removed folder: {folder}")
            
    # Remove Log Files - make it run specific
    log_file = f"communication_log_{config.run_id}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"  Removed log file: {log_file}")

def aggregate_evaluate_metrics(
    results: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics by averaging them."""
    if not results:
        return {}
    
    aggregated_metrics: Dict[str, float] = {}
    total_examples = 0
    
    for num_examples, metrics in results:
        total_examples += num_examples
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += float(value) * num_examples

    return {key: value / total_examples for key, value in aggregated_metrics.items()}


def server_fn(context):
    # Perform cleanup only once at the start of the server
    cleanup()

    # 1. Define Strategy (Aggregates local LSTM weights into a global model)
    strategy = FedAvgWithModelSaving(
        fraction_fit=FRACTION_FIT,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        save_path=SAVE_PATH,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    # 2. Configure the server rounds
    config_ = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config_)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)
