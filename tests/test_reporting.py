"""The guard that keeps results from an older pipeline out of the tables.

Result directories outlive the code that produced them. Everything on disk before
2026-07-29 was made with the per-client scaler, whose ranking is not a noisier
version of the current one but a different quantity, and those directories still
parse perfectly well. The version stamp is what makes them fail loudly instead.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.compare_experiments import (_seeds_from_argv,  # noqa: E402
                                          is_current_pipeline, parse_run_id)
from config_manager import PIPELINE_VERSION  # noqa: E402


class TestPipelineVersionGuard(unittest.TestCase):
    def test_a_run_of_the_current_version_is_accepted(self):
        self.assertTrue(is_current_pipeline({"pipeline_version": PIPELINE_VERSION}))

    def test_a_summary_without_the_stamp_is_rejected(self):
        """Every pre-2026-07-29 directory looks like this."""
        self.assertFalse(is_current_pipeline({"seed": 1, "best_round": 40}))

    def test_an_older_version_is_rejected(self):
        self.assertFalse(is_current_pipeline({"pipeline_version": PIPELINE_VERSION - 1}))

    def test_a_newer_version_is_also_rejected(self):
        """Reading a directory written by newer code is the same hazard."""
        self.assertFalse(is_current_pipeline({"pipeline_version": PIPELINE_VERSION + 1}))


class TestSweepResume(unittest.TestCase):
    """A resumed sweep must not mistake pre-fix results for finished work."""

    def setUp(self):
        import federated_insider_detection as runner
        self.runner = runner
        self.tmp = tempfile.mkdtemp()
        self._original_dir = runner.BASE_REPORT_DIR
        runner.BASE_REPORT_DIR = self.tmp

    def tearDown(self):
        self.runner.BASE_REPORT_DIR = self._original_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_summary(self, run_id, summary):
        os.makedirs(os.path.join(self.tmp, run_id), exist_ok=True)
        with open(os.path.join(self.tmp, run_id, "experiment_summary.json"),
                  "w") as f:
            json.dump(summary, f)

    def test_a_finished_current_run_is_not_repeated(self):
        self._write_summary("baseline__seed1",
                            {"done": True, "pipeline_version": PIPELINE_VERSION})
        self.assertTrue(self.runner.already_done("baseline__seed1"))

    def test_a_finished_run_of_an_older_pipeline_is_repeated(self):
        """The trap: these say done, and they are exactly what must be redone."""
        self._write_summary("baseline__seed1", {"done": True})
        self.assertFalse(self.runner.already_done("baseline__seed1"))

    def test_an_unfinished_run_is_repeated(self):
        self._write_summary("baseline__seed1",
                            {"done": False, "pipeline_version": PIPELINE_VERSION})
        self.assertFalse(self.runner.already_done("baseline__seed1"))

    def test_a_missing_run_is_repeated(self):
        self.assertFalse(self.runner.already_done("never-ran__seed1"))

    def test_a_corrupted_summary_is_repeated_rather_than_raising(self):
        """A sweep killed mid-write leaves truncated JSON behind."""
        os.makedirs(os.path.join(self.tmp, "baseline__seed1"))
        with open(os.path.join(self.tmp, "baseline__seed1",
                               "experiment_summary.json"), "w") as f:
            f.write('{"done": tru')
        self.assertFalse(self.runner.already_done("baseline__seed1"))

    def test_the_smoke_test_never_enters_a_sweep(self):
        self.assertIn("smoke-test", self.runner.NOT_IN_SWEEP)


class TestExperimentSelection(unittest.TestCase):
    """Which experiments a sweep script actually launches."""

    def setUp(self):
        from config_manager import config
        import federated_insider_detection as runner
        self.config = config
        self.runner = runner

    def test_only_restricts_the_sweep_to_the_named_experiments(self):
        chosen = self.runner.select_experiments(self.config, "baseline,top-k-0.1", "")
        self.assertEqual(chosen, ["baseline", "top-k-0.1"])

    def test_skip_removes_the_named_experiments(self):
        chosen = self.runner.select_experiments(self.config, "", "baseline")
        self.assertNotIn("baseline", chosen)
        self.assertIn("top-k-0.1", chosen)

    def test_the_smoke_test_is_dropped_even_without_being_named(self):
        chosen = self.runner.select_experiments(self.config, "", "")
        self.assertNotIn("smoke-test", chosen)

    def test_a_plain_sweep_runs_only_the_reported_set(self):
        """The default used to be every experiment defined, which reached 29.

        Most were exploratory and stayed in the sweep after answering their
        question, costing 33 minutes each on every re-run and leaving the thesis
        a result it has to defend. `[tool.fueba.sweep] experiments` is now the
        list, and a plain sweep may not exceed it.
        """
        reported = set(self.config.get("sweep", "experiments"))
        chosen = set(self.runner.select_experiments(self.config, "", ""))
        self.assertEqual(chosen, reported)

    def test_only_can_reach_an_experiment_outside_the_reported_set(self):
        """Leaving the list must not make a configuration unrunnable."""
        reported = set(self.config.get("sweep", "experiments"))
        outside = next(name for name in self.config.experiment_names
                       if name not in reported
                       and name not in self.runner.NOT_IN_SWEEP)
        self.assertEqual(
            self.runner.select_experiments(self.config, outside, ""), [outside])

    def test_a_typo_in_the_reported_set_stops_the_sweep(self):
        """Otherwise the list silently describes more work than it selects."""
        class ConfigWithTypo:
            experiment_names = self.config.experiment_names

            def get(self, section, key):
                return ["baseline", "delta-0.1", "no-such-experiment"]

        with self.assertRaises(SystemExit) as caught:
            self.runner.select_experiments(ConfigWithTypo(), "", "")
        self.assertIn("no-such-experiment", str(caught.exception))

    def test_an_empty_reported_set_sweeps_everything(self):
        """The documented escape hatch, so the old behaviour stays reachable."""
        class ConfigWithNoList:
            experiment_names = self.config.experiment_names

            def get(self, section, key):
                return []

        chosen = self.runner.select_experiments(ConfigWithNoList(), "", "")
        self.assertEqual(
            set(chosen),
            set(self.config.experiment_names) - self.runner.NOT_IN_SWEEP)

    def test_ablations_are_recognised_so_they_can_be_ordered_last(self):
        """The key is nested in TOML; reading the flat spelling found nothing."""
        self.assertTrue(self.runner.is_ablation(self.config, "ablation-full"))
        self.assertFalse(self.runner.is_ablation(self.config, "baseline"))

    def test_an_ablation_never_runs_before_the_experiment_it_rescores(self):
        chosen = self.runner.select_experiments(self.config, "", "")
        chosen.sort(key=lambda name: self.runner.is_ablation(self.config, name))
        first_ablation = next(i for i, name in enumerate(chosen)
                              if self.runner.is_ablation(self.config, name))
        self.assertLess(chosen.index("baseline"), first_ablation)


class TestBothRunnersStampTheVersion(unittest.TestCase):
    """A stamp only helps if the runners actually write it."""

    def _source(self, filename):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, filename), encoding="utf-8") as f:
            return f.read()

    def test_the_federated_runner_writes_it(self):
        self.assertIn('"pipeline_version": PIPELINE_VERSION',
                      self._source("federated_insider_detection.py"))

    def test_the_centralized_runner_writes_it(self):
        self.assertIn('"pipeline_version": PIPELINE_VERSION',
                      self._source("train_centralized.py"))


class TestSeedRestriction(unittest.TestCase):
    """Comparisons must be pinned to a seed set rather than reading the disk.

    Without this the table is silently unbalanced mid-sweep. On 2026-08-12
    `baseline` had finished its fifth seed while every other arm was still at
    four, and baseline is the reference for every comparison, so the reference
    arm was averaged over five runs and the arms it was compared against over
    four. The other two analysis entry points already took `--seeds`; this one
    did not.
    """

    def test_a_comma_list_is_parsed(self):
        self.assertEqual(_seeds_from_argv(["--seeds", "1,2,3,4"]), {1, 2, 3, 4})

    def test_the_equals_form_is_parsed(self):
        self.assertEqual(_seeds_from_argv(["--seeds=1,2,3,4,5"]), {1, 2, 3, 4, 5})

    def test_spaces_inside_the_list_are_tolerated(self):
        self.assertEqual(_seeds_from_argv(["--seeds", "1, 2 ,3"]), {1, 2, 3})

    def test_no_flag_means_no_restriction(self):
        """None, not an empty set: absent must read everything, not nothing."""
        self.assertIsNone(_seeds_from_argv(["--include-stale"]))

    def test_other_flags_are_not_mistaken_for_it(self):
        self.assertIsNone(_seeds_from_argv([]))

    def test_the_run_id_a_restriction_filters_on(self):
        """The filter keys off parse_run_id, so the two must agree."""
        self.assertEqual(parse_run_id("baseline__seed5"), ("baseline", 5))
        self.assertEqual(parse_run_id("delta-0.05-downlink-fp16__seed4"),
                         ("delta-0.05-downlink-fp16", 4))

    def test_an_unseeded_legacy_directory_is_excluded_by_any_restriction(self):
        """Its seed is None, which is in no seed set, so it drops out."""
        self.assertIsNone(parse_run_id("baseline")[1])


if __name__ == "__main__":
    unittest.main()
