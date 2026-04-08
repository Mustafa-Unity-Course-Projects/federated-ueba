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
        
        # Load settings.toml
        self._settings = self._load_toml(current_dir, "settings.toml")
        
        # Load pyproject.toml
        self._pyproject = self._load_toml(current_dir, "pyproject.toml")
        
        # Get run_id from environment or default
        self._run_id = os.environ.get("RUN_ID", "default_run")

    def _load_toml(self, directory, filename):
        path = directory / filename
        if not path.exists():
            path = directory.parent / filename
        
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
                return data
        else:
            return {}

    def get(self, *keys, default=None):
        val = self._get_from_dict(self._settings, keys, default)
        
        # Automatically inject run_id into paths
        if isinstance(val, str) and ("model_pickle" in val or "scaler_data" in val or "evaluation_reports" in val):
            if "{run_id}" in val:
                return val.format(run_id=self._run_id)
            # If not formatted but matches, append it
            return os.path.join(val, self._run_id)
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
        return self._run_id

config = _ConfigManager()
