import torch
import pandas as pd
from config_manager import config
from federated_ueba.task import LSTMAutoencoder  # Adjust path if needed


def run_baseline():
    # Load the WHOLE dataset (no sharding)
    df = pd.read_csv(config.get("data", "processed_data_path"))

    # Pre-process using the window/stride from settings.toml
    # (Use your existing sequence creation logic here)

    # Train for X epochs
    # Save the model as 'centralized_model.pth'
    print("Baseline training complete. Metrics recorded.")


if __name__ == "__main__":
    run_baseline()