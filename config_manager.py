import tomllib
from pathlib import Path
from threading import Lock

class _ConfigManager:
    _instance = None
    _lock = Lock()  # Makes it thread-safe for parallel client simulations

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(_ConfigManager, cls).__new__(cls)
                cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # This finds the directory where config_manager.py LIVES
        current_dir = Path(__file__).resolve().parent

        # 1. Try looking in the same folder as this script (Root)
        config_path = current_dir / "settings.toml"

        # 2. If not there, try one level up (if this script is in /federated_ueba/)
        if not config_path.exists():
            config_path = current_dir.parent / "settings.toml"

        if not config_path.exists():
            # Final fallback: print the attempted paths to help debugging
            raise FileNotFoundError(
                f"Config file not found. Checked:\n"
                f"1. {current_dir / 'settings.toml'}\n"
                f"2. {current_dir.parent / 'settings.toml'}"
            )

        with open(config_path, "rb") as f:
            self._data = tomllib.load(f)
            print(f"--- Config Successfully Loaded from {config_path} ---")

    def get(self, section, key=None, default=None):
        """Access config data safely: config.get('model', 'lr')"""
        try:
            section_data = self._data.get(section, {})
            if key:
                return section_data.get(key, default)
            return section_data
        except Exception:
            return default

    @property
    def all(self):
        return self._data

config = _ConfigManager()