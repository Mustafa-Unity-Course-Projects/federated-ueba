from collections import OrderedDict
import flwr as fl
import torch
import pickle
import os
import numpy as np

from config_manager import config
from federated_ueba import task
from federated_ueba.efficiency_plugins import get_plugin

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, partition_id, trainloader, valloader, model):
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model.to(task.DEVICE)
        self.efficiency_plugin_manager = get_plugin(config)

    def get_parameters(self, config):
        params = [val.detach().cpu().numpy() for val in self.model.parameters()]
        processed_params = self.efficiency_plugin_manager.apply_on_client(params)
        bytes_cost = self.efficiency_plugin_manager.measure_transport_size(processed_params)
        mb = bytes_cost / (1024 * 1024)
        
        # Use run-specific log file
        log_file = f"communication_log_{config.run_id}.csv"
        with open(log_file, "a") as f:
            f.write(f"upload,{mb:.4f}\n")
        print(f"DEBUG: Client transmitting {mb:.2f} MB for run {config.run_id}")
        
        return processed_params

    def set_parameters(self, parameters):
        processed_params = self.efficiency_plugin_manager.apply_on_server(parameters)
        params_dict = zip(self.model.state_dict().keys(), processed_params)
        state_dict = OrderedDict({k: torch.tensor(v).to(task.DEVICE) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config_dict):
        self.set_parameters(parameters)
        local_epochs = config.get("federation", "local_epochs") or 3
        task.train(self.model, self.trainloader, self.valloader, epochs=local_epochs)
        
        mean_err, std_err = task.get_error_distribution(self.model, self.trainloader)
        stats = {"mean_per_feature": mean_err, "std_per_feature": std_err}
        
        scaler_dir = config.get("data", "scaler_dir")
        # Ensure scaler_dir exists (config.get should have appended the run_id already)
        os.makedirs(scaler_dir, exist_ok=True)
        stats_path = os.path.join(scaler_dir, f"error_stats_client_{self.partition_id}.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)
            
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config_dict):
        self.set_parameters(parameters)
        loss = task.test(self.model, self.valloader)
        return float(loss), len(self.valloader), {"mse": float(loss)}

def client_fn(context):
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)
    trainloader, valloader, detected_dim = task.load_partitioned_data(
        config.get("data", "processed_data_path"),
        partition_id,
        num_partitions
    )
    hidden_dim = config.get("model", "hidden_dim") or 128
    model = task.LSTMAutoencoder(input_dim=detected_dim, hidden_dim=hidden_dim).to(task.DEVICE)
    return FlowerClient(partition_id, trainloader, valloader, model).to_client()

app = fl.client.ClientApp(client_fn=client_fn)
