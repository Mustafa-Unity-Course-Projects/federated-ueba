"""Configuration: experiment layering, seed qualification and artefact paths.

Paths are what keep two runs from overwriting each other. A silent collision here
does not raise; it produces one directory holding a mixture of two runs, which is
exactly how an earlier sweep came to report the communication total of a
different experiment.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import config  # noqa: E402
from federated_insider_detection import NOT_IN_SWEEP  # noqa: E402


class ConfigTestCase(unittest.TestCase):
    """Restores the singleton, which every test in the process shares."""

    def setUp(self):
        self._seed = config.seed
        self._experiment = config._experiment_name

    def tearDown(self):
        # The singleton is shared by every test in the process, so leaving it on
        # another experiment changes what later tests measure.
        config.set_seed(self._seed)
        config.set_experiment(self._experiment or "baseline")


class TestSeedQualification(ConfigTestCase):
    def test_run_id_carries_the_seed(self):
        config.set_seed(7)
        config.set_experiment("baseline")
        self.assertEqual(config.run_id, "baseline__seed7")

    def test_artefact_paths_differ_between_seeds(self):
        config.set_experiment("baseline")

        config.set_seed(1)
        first = (config.get("federation", "save_path"),
                 config.get("data", "scaler_dir"))
        config.set_seed(2)
        second = (config.get("federation", "save_path"),
                  config.get("data", "scaler_dir"))

        self.assertNotEqual(first, second)
        self.assertIn("seed1", first[0])
        self.assertIn("seed2", second[0])

    def test_seed_is_exported_for_subprocesses(self):
        """flower-simulation is a separate interpreter and reads the env."""
        config.set_seed(11)
        self.assertEqual(os.environ.get("SEED"), "11")

    def test_qualify_resolves_an_ablation_source(self):
        """An ablation must read the weights of its source at the same seed."""
        config.set_seed(3)
        config.set_experiment("ablation-full")
        self.assertEqual(config.qualify("baseline"), "baseline__seed3")
        self.assertEqual(config.run_id, "ablation-full__seed3")


class TestExperimentLayering(ConfigTestCase):
    def test_dotted_keys_override_the_base_settings(self):
        config.set_experiment("top-k-0.1")
        self.assertEqual(config.get("efficiency", "active_plugins"), ["top_k"])
        # Untouched sections still come from the base configuration.
        self.assertEqual(config.get("model", "window_size"), 14)

    def test_switching_experiments_does_not_leak_state(self):
        """Each experiment must start from the base settings, not the previous one."""
        config.set_experiment("top-k-0.05")
        self.assertEqual(config.get("efficiency", "active_plugins"), ["top_k"])

        config.set_experiment("baseline")
        self.assertEqual(config.get("efficiency", "active_plugins"), [])

    def test_ablation_switches_layer_over_the_defaults(self):
        config.set_experiment("ablation-no-topk")
        stages = config.get("scoring", "stages")
        self.assertNotIn("topk", stages)
        # The other stages stay on. An ablation that dropped two would measure
        # the pair rather than the one it is named after.
        self.assertEqual(set(stages), {"zscore", "persistence"})
        self.assertEqual(config.get("scoring", "ablation_source"), "baseline")

    def test_the_default_pipeline_is_three_stages(self):
        """The diversity multiplier was measured and removed on 2026-08-14."""
        config.set_experiment("baseline")
        self.assertEqual(config.get("scoring", "stages"),
                         ["zscore", "topk", "persistence"])

    def test_one_ablation_adds_a_stage_instead_of_removing_one(self):
        """How the removed multiplier stays measurable.

        `ablation-full` is the pipeline as shipped; this arm is that plus the
        diversity multiplier, so the difference between them is the stage's
        contribution. Deleting the stage would have made the number the thesis
        reports impossible to reproduce.
        """
        config.set_experiment("ablation-with-diversity")
        stages = config.get("scoring", "stages")
        self.assertIn("diversity", stages)
        self.assertEqual(set(stages),
                         {"zscore", "topk", "diversity", "persistence"})
        self.assertEqual(config.get("scoring", "ablation_source"), "baseline")

        config.set_experiment("ablation-full")
        self.assertNotIn("diversity", config.get("scoring", "stages"))

    def test_node_experiments_declare_their_own_federation_size(self):
        """Without this they are an exact copy of baseline under another name."""
        config.set_experiment("nodes-10")
        self.assertEqual(config.get("federation", "num_supernodes"), 10)
        config.set_experiment("nodes-20")
        self.assertEqual(config.get("federation", "num_supernodes"), 20)
        config.set_experiment("baseline")
        self.assertIsNone(config.get("federation", "num_supernodes"))


class TestSeedIsRecorded(ConfigTestCase):
    """The stored config is the record of what ran, so it must carry the real seed."""

    def test_the_active_config_follows_the_seed_override(self):
        config.set_seed(23)
        self.assertEqual(config.get("experiment", "seed"), 23)

    def test_switching_experiments_does_not_restore_the_file_seed(self):
        """set_experiment resets to the file, which still says seed 42."""
        config.set_seed(23)
        config.set_experiment("baseline")
        self.assertEqual(config.get("experiment", "seed"), 23)


class TestRoundBudget(ConfigTestCase):
    # Not part of the reported suite: a tiny end-to-end check whose numbers mean
    # nothing, so it is allowed its own round count.
    # Taken from the runner rather than restated, so that adding a short
    # development configuration cannot fail this test and cannot be quietly
    # excused by editing a second list. The rule is "everything in the sweep runs
    # 50 rounds", and the sweep membership has one definition.
    EXCLUDED = NOT_IN_SWEEP

    # These exist precisely to vary the round count, so the rule below does not
    # apply to them. They have their own, stricter rule underneath.
    ROUND_EPOCH_TRADE_OFF = {"rounds-25-epochs-10", "rounds-10-epochs-25"}

    # 50 rounds x 5 local epochs. Holding this fixed is what makes the trade-off
    # experiments a comparison of communication rather than of training effort.
    LOCAL_EPOCH_BUDGET = 250

    def test_every_reported_experiment_uses_the_documented_budget(self):
        """Communication totals scale with rounds, so a stray value is not cosmetic."""
        for name in config.experiment_names:
            if name in self.EXCLUDED or name in self.ROUND_EPOCH_TRADE_OFF:
                continue
            config.set_experiment(name)
            rounds = config.get("federation", "num_rounds")
            self.assertEqual(rounds, 50, f"{name} runs {rounds} rounds, not 50")

    def test_the_trade_off_experiments_hold_the_compute_budget_fixed(self):
        """Otherwise fewer rounds would just mean less training, not less traffic."""
        for name in self.ROUND_EPOCH_TRADE_OFF:
            config.set_experiment(name)
            rounds = config.get("federation", "num_rounds")
            epochs = config.get("federation", "local_epochs")
            self.assertEqual(rounds * epochs, self.LOCAL_EPOCH_BUDGET,
                             f"{name}: {rounds} x {epochs} is not "
                             f"{self.LOCAL_EPOCH_BUDGET} local epochs")

    def test_the_baseline_sits_at_the_same_budget(self):
        """The point of comparison for the two above."""
        config.set_experiment("baseline")
        self.assertEqual(config.get("federation", "num_rounds")
                         * config.get("federation", "local_epochs"),
                         self.LOCAL_EPOCH_BUDGET)


if __name__ == "__main__":
    unittest.main()
