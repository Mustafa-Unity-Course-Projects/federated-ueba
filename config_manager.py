"""The single source of truth for what an experiment run is configured to do.

Everything lives in `pyproject.toml` under `[tool.fueba.*]`, with per-experiment
overrides under `[tool.fueba.experiments.<name>]`. On top of that sit the
environment variables `EXPERIMENT_NAME` and `SEED`, which exist because
flower-simulation runs as a separate interpreter and cannot be handed a Python
object; the environment is the only channel a subprocess inherits. That is also
why the config is a singleton read at construction time, and why the runner sets
those variables before importing this module.

**There are no defaults.** Reading a key that the file does not define raises,
and `validate()` reports every missing or mistyped key at once before a run
starts. A silent fallback would let a run measure something other than what its
recorded configuration says, which is exactly how an earlier sweep came to report
figures nobody could reproduce.
"""

import copy
import os
import tomllib
from pathlib import Path
from threading import Lock

# Bumped whenever a change makes older results incomparable with new ones rather
# than merely different. Every run records it, and `analysis/compare_experiments.py`
# refuses to put runs of different versions in one table. This exists because
# result directories outlive the code that made them, and a stale number that
# still parses is far more dangerous than one that fails to load.
#
#   1  original pipeline
#   2  2026-07-29: one global scaler replaces the per-client scalers, and both
#      pipelines select their checkpoint on the validation user half. Version 1
#      rankings had fourteen users pinned to the top by a scaler whose scale was
#      about 3.3e-15, so they do not measure the same quantity.
#   3  2026-07-31: the simulation subprocess is seeded. Until now only the runner
#      was, so client model initialisation, dropout, denoising noise, batch order
#      and which clients a round sampled were all unseeded: two runs at the same
#      seed differed from round 0 onwards. Version 2 numbers are each one draw
#      from an unmeasured distribution rather than a reproducible result.
#   4  2026-07-31: the downlink is compressed and measured the same way the
#      uplink is. Version 3 charged downloads at raw dense fp32 while charging
#      uploads through an entropy coder, so its download and total figures are a
#      cost no implementation would pay and cannot be compared against these.
#      Detection quality is unaffected, but the two are recorded together and a
#      version is a property of the run, not of one column.
#   5  2026-07-31: a client must hold at least `minimum_cohort_size` users before
#      its statistics enter the global scaler. The IID scaler is unchanged to the
#      bit, because no IID client is that small; the non-IID scaler moves by up
#      to 2.4% on the scale of a feature, since 24 of its 50 clients fall below
#      the floor. Bumped even though only the non-IID runs move, for the reason
#      in the note above: a version describes the run.
#   6  2026-08-06: the reference standard deviation has a floor relative to the
#      median, so a feature the model reconstructs identically can no longer act
#      as a thirty-thousand-fold amplifier in the z-score. Version 5 divided by
#      `std + 1e-6` with four of the fifty features sitting at a std near 3e-5;
#      whether that destroyed a run's ranking depended on whether its model
#      happened to be exact on those features, which is not a property anyone
#      would choose to rank models by. Measured on one checkpoint, unchanged
#      model: PR-AUC 0.2394 before, 0.8215 after. Every version 5 score is a
#      draw from that lottery even where it came out fine, so none of them are
#      comparable with these.
#   7  2026-08-07: the validation/test user split no longer uses the run seed.
#      It used to, so every seed reported on a different set of 35 insiders, and
#      that dominated everything. Measured: one unchanged model scored against
#      twenty splits has a standard deviation of 0.0548 and a range of 0.75 to
#      0.91; two runs of the same configuration at seeds 1 and 2 differed by
#      0.0952 on their own splits and by 0.0069 on any shared one. Version 6
#      numbers each describe a different half of the population, so they are not
#      comparable with each other, let alone with these.
#   8  2026-08-14: the diversity multiplier is off by default, so the reported
#      score is the product of three scoring stages rather than four. Measured
#      across five seeds, removing it improved detection every time (+0.0078,
#      +0.0085, +0.0167, +0.0165, +0.0096; mean +0.0119), and the paired
#      interval never separated from zero in either direction, so the stage
#      could not earn its place. Every version 7 number includes it and is
#      therefore about 0.012 lower than the same run scored here; the paired
#      differences the thesis rests on are largely unaffected, because both arms
#      of every comparison carried the multiplier and it cancels.
#      The stage itself is kept and reachable through `ablation-with-diversity`,
#      so the measurement above stays reproducible.
PIPELINE_VERSION = 8


class ConfigError(Exception):
    """Raised for a missing, misspelled or mistyped configuration key."""


# Every key the code may read, with the type it must have. This is the contract:
# `validate()` checks the file against it and reports everything wrong at once,
# rather than failing one key per crash halfway through a sweep.
#
# It duplicates the key names that are also in pyproject.toml, deliberately. The
# file says what the values are; this says what the code requires. A key added to
# one and not the other is caught immediately instead of silently ignored.
SCHEMA = {
    "experiment": {"seed": int, "device": str},
    "data": {
        "dataset_path": str, "processed_data_path": str,
        "scaler_dir": str, "global_scaler_path": str,
        "test_size": float, "batch_size": int,
        "is_non_iid": bool, "non_iid_alpha": float,
        "non_iid_mode": str, "non_iid_clusters": int,
        "feature_transform": str, "drop_constant_features": bool,
        "constant_feature_tolerance": float, "minimum_cohort_size": int,
        "selected_features": list,
    },
    "model": {
        "window_size": int, "stride": int, "hidden_dim": int,
        "encoder_bidirectional": bool, "encoder_layers": int,
        "decoder_layers": int, "use_bottleneck": bool,
        "bottleneck_divisor": int, "dropout": float, "input_noise_std": float,
    },
    "training": {
        "learning_rate": float, "weight_decay": float, "grad_clip_norm": float,
    },
    "centralized": {
        "epochs": int, "early_stopping_patience": int,
        "lr_schedule_factor": float, "lr_schedule_patience": int,
    },
    "federation": {
        "strategy": str, "proximal_mu": float,
        "num_rounds": int, "local_epochs": int, "fraction_fit": float,
        "min_fit_clients": int, "min_available_clients": int, "save_path": str,
    },
    "efficiency": {
        "active_plugins": list, "sparsification_ratio": float,
        "payload_codec": str, "entropy_coder": str,
        "zlib_level": int, "lzma_preset": int,
        "downlink_plugins": list, "downlink_entropy_coder": str,
    },
    "scoring": {
        "top_k_features": int, "persistence_window": int,
        "diversity_threshold": float, "scan_stride": int,
        "zscore_epsilon": float, "zscore_std_floor_fraction": float,
        "stages": list,
        "ablation_source": str,
        "fallback_error_mean": float, "fallback_error_std": float,
    },
    "evaluation": {
        "checkpoint_stride": int, "validation_fraction": float,
        "split_seed": int,
        "convergence_peak_fraction": float, "plateau_tolerance": float,
        "plateau_window_rounds": int,
        "bootstrap_iterations": int, "bootstrap_confidence": float,
    },
    "sweep": {"experiments": list},
}

# Keys only some experiments declare, so their absence is not an error. The
# node-count experiments override the federation size; everything else inherits
# it from the Flower federation block.
OPTIONAL = {("federation", "num_supernodes"): int}

# Settings whose value is a path belonging to one specific run.
_RUN_SCOPED_PATH_MARKERS = ("model_pickle", "scaler_data", "evaluation_reports")


class _ConfigManager:
    """One instance per process. Use the module-level `config`, not this class."""

    _instance = None
    # Ray runs client actors as threads within a worker, so two of them can reach
    # the first construction at the same moment. Without the lock they would each
    # build a config and the second would discard the first mid-read.
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(_ConfigManager, cls).__new__(cls)
                cls._instance._load_configs()
        return cls._instance

    # --- construction ------------------------------------------------------

    def _load_configs(self):
        # Next to this file, so a run started from any working directory reads
        # the same configuration.
        project_dir = Path(__file__).resolve().parent
        self._pyproject = self._load_toml(project_dir, "pyproject.toml")

        fueba = self._pyproject.get("tool", {}).get("fueba")
        if not fueba:
            raise ConfigError(
                "pyproject.toml has no [tool.fueba] section. That block holds "
                "every experiment setting; without it nothing can run.")

        self._experiments = fueba.get("experiments", {})
        self._settings = {k: v for k, v in fueba.items() if k != "experiments"}

        # Only used when no experiment is named, which happens when a module is
        # imported outside a run (a test, or the comparison script).
        self._run_id = os.environ.get("RUN_ID", "default_run")

        # The environment wins over the file so that `--seeds 1,2,3` can drive
        # three runs from one configuration, and so flower-simulation inherits
        # the same value as the parent process.
        env_seed = os.environ.get("SEED")
        if env_seed is not None:
            self._seed = int(env_seed)
        else:
            self._seed = self._require(self._settings, "experiment", "seed")

        self._experiment_name = os.environ.get("EXPERIMENT_NAME")
        # A deep copy, because the overrides below write into nested sections and
        # must not reach back into the parsed file.
        self._active_config = copy.deepcopy(self._settings)
        self._record_seed()

        if self._experiment_name:
            self.set_experiment(self._experiment_name)

    def _load_toml(self, directory, filename):
        path = directory / filename
        if not path.exists():
            path = directory.parent / filename
        if not path.exists():
            raise ConfigError(f"{filename} not found next to {directory}.")
        with open(path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def _require(settings, section, key):
        """Read during construction, before `get` and its path handling exist."""
        try:
            return settings[section][key]
        except KeyError:
            raise ConfigError(
                f"[tool.fueba.{section}] {key} is not defined in pyproject.toml."
            ) from None

    def _record_seed(self):
        """Write the effective seed back into the active configuration.

        `experiment_summary.json` stores this dict as the record of what was run.
        A `--seeds 1,2,3` sweep overrides the seed at runtime, so without this the
        stored record would claim every one of those runs used the file's seed.
        """
        self._active_config.setdefault("experiment", {})["seed"] = self._seed

    # --- validation --------------------------------------------------------

    def validate(self):
        """Check the whole configuration, reporting every problem at once.

        Called once at the start of a run. Failing here costs seconds; failing on
        the twentieth experiment of a sweep costs hours, and failing silently
        costs the credibility of every number the run produced.

        Three separate questions, one per helper: is everything the code needs
        present and the right type, is anything present that the code does not
        know, and is any whole section unrecognised.
        """
        problems = (self._missing_or_mistyped()
                    + self._unknown_sections()
                    + self._unknown_keys())
        if problems:
            raise ConfigError(
                f"{len(problems)} configuration problem(s):\n  "
                + "\n  ".join(problems))

    def _type_problem(self, section, key, value, expected):
        """A description of the type mismatch, or None if the value is fine."""
        # bool is a subclass of int, so an accidental `true` where a count
        # belongs would otherwise pass a plain isinstance check.
        if expected is int and isinstance(value, bool):
            return f"[tool.fueba.{section}] {key} is a boolean, expected int"
        # An integer where a float belongs is harmless, and TOML makes it easy to
        # write `1` for `1.0`, so this is allowed.
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            return None
        if not isinstance(value, expected):
            return (f"[tool.fueba.{section}] {key} is {type(value).__name__}, "
                    f"expected {expected.__name__}")
        return None

    def _missing_or_mistyped(self):
        problems = []
        for section, keys in SCHEMA.items():
            if section not in self._active_config:
                problems.append(f"[tool.fueba.{section}] is missing entirely")
                continue
            for key, expected in keys.items():
                if key not in self._active_config[section]:
                    problems.append(f"[tool.fueba.{section}] {key} is missing")
                    continue
                problem = self._type_problem(
                    section, key, self._active_config[section][key], expected)
                if problem:
                    problems.append(problem)
        return problems

    def _unknown_sections(self):
        return [f"[tool.fueba.{section}] is not a section the code knows about"
                for section in sorted(set(self._active_config) - set(SCHEMA))]

    def _unknown_keys(self):
        """Unknown keys matter as much as missing ones.

        A renamed setting left behind under its old name reads as configured but
        is never applied, so the run silently uses something other than what the
        file appears to say. `top_k_ratio`, renamed to `sparsification_ratio`,
        was exactly that hazard.
        """
        problems = []
        for section, keys in SCHEMA.items():
            present = set(self._active_config.get(section, {}))
            allowed = set(keys) | {k for s, k in OPTIONAL if s == section}
            problems.extend(
                f"[tool.fueba.{section}] {key} is not a setting the code "
                f"reads; remove it or correct the name"
                for key in sorted(present - allowed))
        return problems

    # --- experiments -------------------------------------------------------

    def set_experiment(self, experiment_name):
        """Apply one experiment's overrides on top of the base settings."""
        # Reset first. Deep copy because the merge writes into nested sections.
        self._active_config = copy.deepcopy(self._settings)
        self._record_seed()

        if experiment_name not in self._experiments:
            known = ", ".join(sorted(self._experiments))
            raise ConfigError(
                f"Experiment {experiment_name!r} is not defined in "
                f"[tool.fueba.experiments]. Known: {known}")

        self._experiment_name = experiment_name
        for section, overrides in self._experiments[experiment_name].items():
            if section not in self._active_config:
                self._active_config[section] = {}
            self._active_config[section].update(overrides)

    # --- reading -----------------------------------------------------------

    def get(self, section, key):
        """Read a setting, qualifying run-scoped paths with the run id.

        Raises rather than returning a default. The qualification is what keeps
        two runs from overwriting each other: `model_pickle` becomes
        `model_pickle/baseline__seed3`, so the same experiment at three seeds
        writes to three directories.
        """
        try:
            value = self._active_config[section][key]
        except KeyError:
            if (section, key) in OPTIONAL:
                return None
            raise ConfigError(
                f"[tool.fueba.{section}] {key} is not defined. Every setting "
                f"must be declared in pyproject.toml; there are no defaults."
            ) from None

        if not isinstance(value, str):
            return value
        if not any(marker in value for marker in _RUN_SCOPED_PATH_MARKERS):
            return value

        # `scaler_dir = "scaler_data/{run_id}"` says where the id goes; a path
        # without the placeholder gets it appended.
        if "{run_id}" in value:
            return value.format(run_id=self.run_id)
        return os.path.join(value, self.run_id)

    def get_pyproject(self, *keys, default=None):
        """Read from pyproject.toml outside [tool.fueba], where Flower's own
        options live. Those are the framework's contract, not ours, so a default
        is allowed here."""
        current = self._pyproject
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    # --- identity ----------------------------------------------------------

    @property
    def seed(self):
        """Seeds partitioning, the initial model, client training and the splits."""
        return self._seed

    def set_seed(self, seed):
        """Change the active seed. Also exported so subprocesses inherit it."""
        self._seed = int(seed)
        os.environ["SEED"] = str(self._seed)
        self._record_seed()

    @property
    def run_id(self):
        """Identifier used for every on-disk artefact of this run.

        The seed is part of it so that repeated runs of the same experiment under
        different seeds cannot overwrite each other's weights, scalers or reports.
        """
        return self.qualify(self._experiment_name or self._run_id)

    def qualify(self, experiment_name):
        """`baseline` -> `baseline__seed3`.

        Also used to name an experiment other than the active one, which is how
        an ablation locates the weights of the run it rescores: it must read the
        source at the same seed, never at whichever seed happens to exist.
        """
        return f"{experiment_name}__seed{self._seed}"

    @property
    def experiment_names(self):
        """Every experiment declared in pyproject.toml, in declaration order."""
        return list(self._experiments)


# The single instance every module shares. Importing this builds it, which reads
# the environment, so anything that sets EXPERIMENT_NAME or SEED must do so
# before the first import anywhere in the process.
config = _ConfigManager()
