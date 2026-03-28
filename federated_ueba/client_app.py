from collections import OrderedDict
import flwr as fl
import torch

from config_manager import config
from federated_ueba import task

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, model):
        self.trainloader = trainloader
        self.model = model.to(task.DEVICE)
        self.criterion = torch.nn.MSELoss()

    def get_parameters(self, config):
        # 1. Get parameters as NumPy arrays (using detach() to handle tensors with grad)
        params = [val.detach().cpu().numpy() for val in self.model.parameters()]

        # 2. Calculate Size (The "Communication Cost" Metric)
        total_bytes = sum(p.nbytes for p in params)
        mb = total_bytes / (1024 * 1024)

        # 3. Log it to a file
        with open("communication_log.csv", "a") as f:
            f.write(f"upload,{mb:.4f}\n")

        print(f"DEBUG: Client transmitting {mb:.2f} MB")
        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(task.DEVICE) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # task.train already handles moving batches to task.DEVICE
        task.train(self.model, self.trainloader, epochs=3)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # task.test already handles moving batches to task.DEVICE
        loss = task.test(self.model, self.trainloader)
        return float(loss), len(self.trainloader), {"mse": float(loss)}

def client_fn(context):
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)

    # Load data and get the count of features
    trainloader, detected_dim = task.load_partitioned_data(
        config.get("data", "processed_data_path"),
        partition_id,
        num_partitions
    )

    # Initialize model and move it to the configured device immediately
    model = task.LSTMAutoencoder(input_dim=detected_dim, hidden_dim=128).to(task.DEVICE)

    return FlowerClient(trainloader, model).to_client()

app = fl.client.ClientApp(client_fn=client_fn)
