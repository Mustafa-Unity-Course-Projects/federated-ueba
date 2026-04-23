from collections import OrderedDict
import flwr as fl
import torch
import pickle
import os
import numpy as np

from config_manager import config as global_config
from federated_ueba import task
from federated_ueba.efficiency_plugins import get_plugin

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, partition_id, trainloader, valloader, model):
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model.to(task.DEVICE)
        self.efficiency_plugin_manager = get_plugin(global_config)

    def get_parameters(self, config):
        """
        config: Dict[str, Scalar] passed from Flower.
        We use global_config for our application settings.
        """
        params = [val.detach().cpu().numpy() for val in self.model.parameters()]
        processed_params = self.efficiency_plugin_manager.apply_on_client(params)
        bytes_cost = self.efficiency_plugin_manager.measure_transport_size(processed_params)
        mb = bytes_cost / (1024 * 1024)
        
        # Use run-specific log file inside the report directory
        report_dir = os.path.join("federated_evaluation_reports", global_config.run_id)
        os.makedirs(report_dir, exist_ok=True)
        log_file = os.path.join(report_dir, "communication_log.csv")

        with open(log_file, "a") as f:
            f.write(f"upload,{mb:.4f}\n")
        print(f"DEBUG: Client transmitting {mb:.2f} MB for run {global_config.run_id}")

        return processed_params

    def set_parameters(self, parameters):
        processed_params = self.efficiency_plugin_manager.apply_on_server(parameters)
        params_dict = zip(self.model.state_dict().keys(), processed_params)
        state_dict = OrderedDict({k: torch.tensor(v).to(task.DEVICE) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config_dict):
        self.set_parameters(parameters)
        local_epochs = global_config.get("federation", "local_epochs") or 3
        task.train(self.model, self.trainloader, self.valloader, epochs=local_epochs)
        
        mean_err, std_err = task.get_error_distribution(self.model, self.trainloader)
        stats = {"mean_per_feature": mean_err, "std_per_feature": std_err}
        
        scaler_dir = global_config.get("data", "scaler_dir")
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
        global_config.get("data", "processed_data_path"),
        partition_id,
        num_partitions
    )
    hidden_dim = global_config.get("model", "hidden_dim") or 128
    model = task.LSTMAutoencoder(input_dim=detected_dim, hidden_dim=hidden_dim).to(task.DEVICE)
    return FlowerClient(partition_id, trainloader, valloader, model).to_client()

app = fl.client.ClientApp(client_fn=client_fn)
