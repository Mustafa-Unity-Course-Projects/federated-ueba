import numpy as np
from abc import ABC, abstractmethod

class CommunicationPlugin(ABC):
    """Base class for communication efficiency plugins."""
    @abstractmethod
    def apply_on_client(self, parameters):
        """Processes parameters on the client before uploading."""
        pass

    @abstractmethod
    def apply_on_server(self, parameters):
        """Processes parameters on the server after aggregation."""
        pass

class StandardPlugin(CommunicationPlugin):
    def apply_on_client(self, parameters): return parameters
    def apply_on_server(self, parameters): return parameters

class TopKPlugin(CommunicationPlugin):
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def apply_on_client(self, parameters):
        processed_params = []
        for p in parameters:
            p_flat = p.flatten()
            k = max(1, int(len(p_flat) * self.ratio))
            k = min(k, len(p_flat))
            indices = np.argpartition(np.abs(p_flat), -k)[-k:]
            p_sparse = np.zeros_like(p_flat)
            p_sparse[indices] = p_flat[indices]
            processed_params.append(p_sparse.reshape(p.shape))
        return processed_params

    def apply_on_server(self, parameters):
        return parameters

class QuantizationPlugin(CommunicationPlugin):
    def apply_on_client(self, parameters):
        return [p.astype(np.float16) for p in parameters]

    def apply_on_server(self, parameters):
        return [p.astype(np.float32) for p in parameters]

class PluginManager:
    """Manages a chain of efficiency plugins and calculates real transport size."""
    def __init__(self, plugins):
        self.plugins = plugins

    def apply_on_client(self, parameters):
        for plugin in self.plugins:
            parameters = plugin.apply_on_client(parameters)
        return parameters

    def apply_on_server(self, parameters):
        for plugin in reversed(self.plugins):
            parameters = plugin.apply_on_server(parameters)
        return parameters

    def measure_transport_size(self, processed_params):
        """
        Calculates the actual bandwidth used based on the final processed state.
        Handles both Dense and Sparse (Top-K) scenarios.
        """
        if not self.plugins:
            return sum(p.nbytes for p in processed_params)

        # Check if we intended to use Sparse transmission (Top-K)
        is_sparse = any(isinstance(p, TopKPlugin) for p in self.plugins)
        
        if is_sparse:
            # For sparse, we only send non-zero values and their indices
            total_non_zeros = sum(np.count_nonzero(p) for p in processed_params)
            # Size = (Index [4 bytes]) + (Value [depends on dtype])
            val_size = processed_params[0].itemsize if len(processed_params) > 0 else 4
            return total_non_zeros * (4 + val_size)
        else:
            # For dense (standard or just quantization), we send the full array
            return sum(p.nbytes for p in processed_params)

def get_plugin(config):
    active_modes = config.get("efficiency", "active_plugins") or ["standard"]
    plugin_list = []
    for mode in active_modes:
        if mode == "top_k":
            plugin_list.append(TopKPlugin(ratio=config.get("efficiency", "top_k_ratio") or 0.1))
        elif mode == "quantization":
            plugin_list.append(QuantizationPlugin())
    return PluginManager(plugin_list)
