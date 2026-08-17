"""Flower client: trains one partition and reports its communication cost.

Every transfer is appended to `federated_evaluation_reports/<run_id>/comm/client_<id>.csv`
as `direction,phase,mb`. Both directions are compressed and both are measured the
same way, by encoding the payload rather than by counting the array bytes.

They are still reported separately, because they are not the same quantity and
one dominates. Per round the server broadcasts to 25 clients to train and to all
50 to evaluate, while only the 25 upload, so the download is roughly three times
the upload before any compression. See `DownlinkCodec` for why the two directions
cannot use the same compression.
"""

import os
import pickle
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from config_manager import config as global_config
from federated_ueba import task
from federated_ueba.efficiency_plugins import get_downlink_codec, get_plugin
from federated_ueba.seeding import seed_everything


class FlowerClient(fl.client.NumPyClient):
    """One participant: holds a slice of users, trains on it, uploads its weights.

    Rebuilt from scratch every round. Flower does not keep client objects alive
    between rounds, which is why anything that has to persist across them (the
    error-feedback residual) is written to disk rather than kept as an attribute.
    """

    def __init__(self, partition_id, trainloader, valloader, model):
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model.to(task.DEVICE)
        # partition_id lets the error-feedback variant find its own residual file
        self.efficiency_plugin_manager = get_plugin(global_config,
                                                    partition_id=partition_id)
        # No partition_id: the downlink is a broadcast, identical for everyone.
        self.downlink = get_downlink_codec(global_config)
        # The global weights this round started from. Delta sparsification needs
        # them to compute the update; None until the server has sent a model.
        self._round_reference = None

    def _log_transfer(self, direction, phase, mb):
        """Append one transfer record to this client's own log file.

        One file per partition, never a shared one. Under Ray the 25 sampled
        clients of a round run as separate processes, and Windows does not give
        `open(path, "a")` the atomic append POSIX does: seek-to-end and write are
        two steps, so concurrent writers overwrite each other's lines. That
        silently dropped up to 79% of the records in an earlier sweep, which
        directly understated the headline communication-cost numbers.

        Only this partition ever writes this file, and never twice at once, so
        the record count is exact. `read_communication_log` sums the parts.
        """
        comm_dir = os.path.join("federated_evaluation_reports",
                                global_config.run_id, "comm")
        os.makedirs(comm_dir, exist_ok=True)
        path = os.path.join(comm_dir, f"client_{self.partition_id}.csv")
        with open(path, "a") as f:
            f.write(f"{direction},{phase},{mb:.4f}\n")

    def get_parameters(self, config, phase="fit"):
        """Compress the local weights and report the resulting upload size.

        Reads from `state_dict()` rather than `parameters()` so the ordering
        matches `set_parameters`, which zips against `state_dict().keys()`. The
        two agree for this model only because it has no buffers; going through
        `state_dict` on both sides removes that dependency.
        """
        params = [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
        processed_params = self.efficiency_plugin_manager.apply_on_client(
            params, reference=self._round_reference)
        mb = self.efficiency_plugin_manager.measure_transport_size(processed_params) / (1024 * 1024)
        self._log_transfer("upload", phase, mb)
        return processed_params

    def set_parameters(self, parameters, phase="fit"):
        """Install the global model, charging and then undoing the downlink codec.

        Measured on what arrived, before decompressing, so the figure is the
        bytes that crossed the wire rather than the size of the arrays once the
        client has expanded them back to fp32. This used to count `p.nbytes` on
        the expanded arrays, which charged the downlink at raw dense fp32 while
        the uplink was charged through an entropy coder, and so reported a
        download cost that no implementation would actually pay.

        Undoing the uplink plugins here would be wrong and used to happen: what
        arrives is the server's broadcast, and the server already undid the
        uplink encoding when it aggregated.
        """
        self._log_transfer("download", phase,
                           self.downlink.measure(parameters) / (1024 * 1024))

        processed_params = self.downlink.decompress(parameters)
        # Record the starting point of this round. It is the same model the
        # server holds, so a sparse delta measured against it is exactly what the
        # server needs to reconstruct the client's update.
        self._round_reference = [np.array(p, dtype=np.float32) for p in processed_params]
        params_dict = zip(self.model.state_dict().keys(), processed_params)
        state_dict = OrderedDict({k: torch.tensor(v).to(task.DEVICE) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_properties(self, config):
        """Report which partition this client holds.

        The server needs it to sample reproducibly. Flower identifies a client by
        a node id drawn from `os.urandom`, so node ids differ on every run and no
        amount of seeding makes an ordering over them stable; the partition is the
        only identifier that means the same thing twice. See
        `federated_ueba.client_manager`.
        """
        return {"partition_id": self.partition_id}

    def fit(self, parameters, config_dict):
        """One round of local training: take the global model, train, upload.

        Returns `(parameters, num_examples, metrics)`. The example count is what
        FedAvg weights this client by when averaging, so a client holding more
        data pulls the global model further.
        """
        self.set_parameters(parameters, phase="fit")

        # Seeded per (run, partition, round) rather than once per actor: Ray does
        # not guarantee which actor handles which partition, so seeding by actor
        # would make dropout masks, denoising noise and batch order depend on
        # scheduling. `server_round` arrives from the strategy's fit_config.
        seed_everything(global_config.seed, self.partition_id,
                        config_dict.get("server_round", 0))

        local_epochs = global_config.get("federation", "local_epochs")
        # Sent by the strategy, not read from configuration here. Flower's FedProx
        # puts it in the fit config, and FedAvg does not put it anywhere, so its
        # absence is what makes a FedAvg round train without the penalty. Reading
        # it from the file instead would apply FedProx under a FedAvg strategy.
        proximal_mu = float(config_dict.get("proximal_mu", 0.0))
        task.train(self.model, self.trainloader, self.valloader,
                   epochs=local_epochs, proximal_mu=proximal_mu)

        # Reference distribution for the Z-score stage of the scoring pipeline.
        # Written every round and overwritten in place; evaluation averages the
        # final copies of all clients that trained.
        mean_err, std_err = task.get_error_distribution(self.model, self.trainloader)
        scaler_dir = global_config.get("data", "scaler_dir")
        os.makedirs(scaler_dir, exist_ok=True)
        with open(os.path.join(scaler_dir, f"error_stats_client_{self.partition_id}.pkl"), "wb") as f:
            pickle.dump({"mean_per_feature": mean_err, "std_per_feature": std_err}, f)

        # The partition travels back with the update so the server can put the
        # results in a fixed order before averaging them. Ray returns them in
        # whatever order the actors finish, and floating point addition is not
        # associative, so an unordered sum differs in its last bits between runs
        # and the difference compounds over 50 rounds.
        # The partition travels back with the update so the server can put the
        # results in a fixed order before averaging them. Ray returns them in
        # whatever order the actors finish, and floating point addition is not
        # associative, so an unordered sum differs in its last bits between runs
        # and the difference compounds over 50 rounds.
        return (self.get_parameters(config={}, phase="fit"),
                len(self.trainloader), {"partition_id": self.partition_id})

    def evaluate(self, parameters, config_dict):
        """Report this client's reconstruction loss on the global model.

        Only the loss: detection quality needs the full scoring pipeline over all
        users, which runs after training rather than inside the federation. Note
        that this loss comes from a window split that overlaps the training
        windows, so it tracks training progress but is not an independent measure.
        """
        self.set_parameters(parameters, phase="eval")
        loss = task.test(self.model, self.valloader)
        return float(loss), len(self.valloader), {"mse": float(loss)}


def client_fn(context):
    """Build the client for one supernode. Called by Flower once per participation."""
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)

    # Seeded before anything random happens: `load_partitioned_data` shuffles the
    # training loader and the model below is initialised from the global RNG.
    # Keyed on the partition so two clients do not start from identical weights,
    # and reproducibly so the same run seed rebuilds the same client.
    seed_everything(global_config.seed, partition_id)

    trainloader, valloader, detected_dim = task.load_partitioned_data(
        global_config.get("data", "processed_data_path"), partition_id, num_partitions)

    # Under a heterogeneous partition a client can end up with too few users to
    # build a single window. Give it empty loaders so Flower keeps running; it
    # simply contributes nothing to the aggregate.
    if trainloader is None:
        dummy = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(1, task.WINDOW_SIZE, detected_dim)),
            batch_size=1)
        trainloader = valloader = dummy

    hidden_dim = global_config.get("model", "hidden_dim")
    model = task.LSTMAutoencoder(input_dim=detected_dim, hidden_dim=hidden_dim).to(task.DEVICE)
    return FlowerClient(partition_id, trainloader, valloader, model).to_client()


app = fl.client.ClientApp(client_fn=client_fn)
