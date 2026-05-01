import os
import tomllib
from pathlib import Path
from threading import Lock

class _ConfigManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(_ConfigManager, cls).__new__(cls)
                cls._instance._load_configs()
        return cls._instance

    def _load_configs(self):
        current_dir = Path(__file__).resolve().parent
        
        # Load settings.toml (Base settings)
        self._settings = self._load_toml(current_dir, "settings.toml")
        
        # Load pyproject.toml
        self._pyproject = self._load_toml(current_dir, "pyproject.toml")
        
        # Load experiments.toml if it exists
        self._experiments = self._load_toml(current_dir, "experiments.toml")
        
        # Get run_id from environment or default
        self._run_id = os.environ.get("RUN_ID", "default_run")
        
        # Current active experiment name
        self._experiment_name = os.environ.get("EXPERIMENT_NAME")
        self._active_config = self._settings.copy()
        
        if self._experiment_name:
            self.set_experiment(self._experiment_name)

    def _load_toml(self, directory, filename):
        path = directory / filename
        if not path.exists():
            path = directory.parent / filename
        
        if path.exists():
            with open(path, "rb") as f:
                return tomllib.load(f)
        else:
            return {}

    def set_experiment(self, experiment_name):
        """Activates a specific experiment configuration and overrides base settings."""
        # Reset to base settings before applying new experiment
        # We need a deep-ish copy here because we modify nested dicts
        import copy
        self._active_config = copy.deepcopy(self._settings)
        
        if experiment_name not in self._experiments:
            print(f"⚠️ Experiment '{experiment_name}' not found in experiments.toml. Using base settings.")
            self._experiment_name = experiment_name # Still set it for pathing
            return

        self._experiment_name = experiment_name
        exp_settings = self._experiments[experiment_name]
        
        # Merge experiment settings into active_config
        # Supports both flat keys with dots "federation.num_rounds" and nested dicts
        for key, value in exp_settings.items():
            if "." in key:
                parts = key.split(".")
                curr = self._active_config
                for part in parts[:-1]:
                    if part not in curr or not isinstance(curr[part], dict):
                        curr[part] = {}
                    curr = curr[part]
                curr[parts[-1]] = value
            else:
                if isinstance(value, dict) and key in self._active_config and isinstance(self._active_config[key], dict):
                    self._active_config[key].update(value)
                else:
                    self._active_config[key] = value

    def get(self, *keys, default=None):
        val = self._get_from_dict(self._active_config, keys, default)
        
        # Use experiment_name for paths if available, otherwise run_id
        current_id = self._experiment_name or self._run_id

        # Automatically inject ID into paths
        if isinstance(val, str) and ("model_pickle" in val or "scaler_data" in val or "evaluation_reports" in val):
            if "{run_id}" in val:
                return val.format(run_id=current_id)
            # If not formatted but matches, append it
            return os.path.join(val, current_id)
        return val

    def get_pyproject(self, *keys, default=None):
        return self._get_from_dict(self._pyproject, keys, default)

    def _get_from_dict(self, data, keys, default):
        curr = data
        for key in keys:
            if isinstance(curr, dict) and key in curr:
                curr = curr[key]
            else:
                return default
        return curr

    @property
    def run_id(self):
        return self._experiment_name or self._run_id

    @property
    def experiment_names(self):
        return list(self._experiments.keys())

config = _ConfigManager()
