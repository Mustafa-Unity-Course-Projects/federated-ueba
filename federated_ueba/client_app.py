from collections import OrderedDict
import flwr as fl
import torch
from federated_ueba import task

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, model):
        self.trainloader = trainloader
        self.model = model
        self.criterion = torch.nn.MSELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(task.DEVICE) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # We pass the model and loader to the task.train function
        # Using 3-5 epochs per round is a good balance for CERT data
        task.train(self.model, self.trainloader, epochs=3)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # loss represents the "Reconstruction Error"
        loss = task.test(self.model, self.trainloader)
        return float(loss), len(self.trainloader), {"mse": float(loss)}

def client_fn(context):
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)

    # Load data and get the count of features (minus the 'insider' column)
    trainloader, detected_dim = task.load_partitioned_data(
        "cert_data.csv",
        partition_id,
        num_partitions
    )

    # model now dynamically adjusts to 500+ features
    model = task.LSTMAutoencoder(input_dim=detected_dim, hidden_dim=128)
    return FlowerClient(trainloader, model).to_client()

app = fl.client.ClientApp(client_fn=client_fn)