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

    def _load_toml(self, directory, filename):
        path = directory / filename
        if not path.exists():
            path = directory.parent / filename
        
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
                print(f"--- Config Successfully Loaded from {path} ---")
                return data
        else:
            print(f"--- Warning: {filename} not found ---")
            return {}

    def get(self, *keys, default=None):
        """
        Access config data safely from settings.toml: 
        config.get('model', 'lr')
        Or multiple keys for nested access:
        config.get('tool', 'flwr', 'federations', 'local-simulation', 'options', 'num-supernodes')
        """
        return self._get_from_dict(self._settings, keys, default)

    def get_pyproject(self, *keys, default=None):
        """Access config data safely from pyproject.toml"""
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
    def settings(self):
        return self._settings

    @property
    def pyproject(self):
        return self._pyproject

config = _ConfigManager()
