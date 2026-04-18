from logging import INFO
import pickle
from pathlib import Path
import flwr
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from config_manager import config
from federated_ueba.efficiency_plugins import get_plugin

class FedAvgWithModelSaving(flwr.server.strategy.FedAvg):
    """A custom strategy for saving global checkpoints and handling communication plugins."""
    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.efficiency_plugin = get_plugin(config)
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate results and apply server-side plugins."""
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            processed_ndarrays = self.efficiency_plugin.apply_on_server(ndarrays)
            aggregated_parameters = ndarrays_to_parameters(processed_ndarrays)
            
        return aggregated_parameters, metrics

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""
        ndarrays = parameters_to_ndarrays(parameters)
        data = {'global_parameters': ndarrays}
        
        # Ensure save_path exists for the current run_id
        save_path_for_run = Path(self.save_path)
        save_path_for_run.mkdir(exist_ok=True, parents=True)
        
        filename = str(save_path_for_run/f"parameters_round_{server_round}.pkl")
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        self._save_global_model(server_round, parameters)
        return super().evaluate(server_round, parameters)
