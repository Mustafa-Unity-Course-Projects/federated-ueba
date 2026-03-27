from logging import INFO
import pickle
from pathlib import Path

import flwr
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays

class FedAvgWithModelSaving(flwr.server.strategy.FedAvg):
    """This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """
    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        # Create directory if needed
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""

        # convert parameters to list of NumPy arrays
        # this will make things easy if you want to load them into a
        # PyTorch or TensorFlow model later
        ndarrays = parameters_to_ndarrays(parameters)
        data = {'globa_parameters': ndarrays}
        filename = str(self.save_path/f"parameters_round_{server_round}.pkl")
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        # save the parameters to disk using a custom method
        self._save_global_model(server_round, parameters)

        # call the parent method so evaluation is performed as
        # FedAvg normally does.
        return super().evaluate(server_round, parameters)