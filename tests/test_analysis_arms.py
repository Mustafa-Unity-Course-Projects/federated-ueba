"""The analysis layer must not read arms that no longer mean what they say.

Two defects found on 2026-08-15, both of the shape this project keeps hitting:
a read that raises nothing and returns a wrong number.

`load_runs` filtered on `done` but not on pipeline version, so the directory
listing was the definition of the experiment set. Dropping an arm from the
configuration does not drop its directories, and the five stale
`ablation-no-diversity` runs stayed readable after the v8 flip.

`findings.ABLATIONS` still paired the diversity stage as
`ablation-full` minus `ablation-no-diversity`. Under v8 those are the same three
stages, so the row compared an arm against itself and reported the multiplier as
contributing -0.0003, indistinguishable. The real figure is -0.0117. Nothing
failed; the table simply printed a number that meant nothing.

The second test below is the general guard: it checks each pair against the
configuration's stage lists rather than trusting the arm names.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import findings, seed_variance  # noqa: E402
from config_manager import PIPELINE_VERSION, config  # noqa: E402


class TestPipelineVersionFilter(unittest.TestCase):
    """`load_runs` reads a directory tree, so the tree is built for real."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._reports_dir = seed_variance.REPORTS_DIR
        seed_variance.REPORTS_DIR = self.tmp

    def tearDown(self):
        seed_variance.REPORTS_DIR = self._reports_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_run(self, run_id, version=PIPELINE_VERSION, done=True):
        directory = os.path.join(self.tmp, run_id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "experiment_summary.json"), "w") as f:
            json.dump({"run_id": run_id, "done": done,
                       "pipeline_version": version}, f)

    def test_a_stale_arm_is_not_read(self):
        self.write_run("baseline__seed1")
        self.write_run("ablation-no-diversity__seed1",
                       version=PIPELINE_VERSION - 1)
        runs = seed_variance.load_runs()
        self.assertIn("baseline", runs)
        self.assertNotIn("ablation-no-diversity", runs)

    def test_a_stale_seed_of_a_live_arm_is_not_read(self):
        """The dangerous case: the arm survives, so nothing looks missing."""
        self.write_run("baseline__seed1")
        self.write_run("baseline__seed2", version=PIPELINE_VERSION - 1)
        runs = seed_variance.load_runs()
        self.assertEqual(len(runs["baseline"]), 1)

    def test_unfinished_runs_are_still_skipped(self):
        self.write_run("baseline__seed1", done=False)
        self.assertNotIn("baseline", seed_variance.load_runs())

    def test_the_filter_can_be_turned_off_deliberately(self):
        """Comparing pipeline versions to each other is a real, separate task."""
        self.write_run("baseline__seed1", version=PIPELINE_VERSION - 1)
        self.assertIn("baseline", seed_variance.load_runs(version=None))


class TestStagePairsMatchTheConfiguration(unittest.TestCase):
    """Each stage row must compare an arm that has the stage with one that lacks it."""

    def setUp(self):
        self._experiment = config._experiment_name

    def tearDown(self):
        config.set_experiment(self._experiment or "baseline")

    def stages(self, experiment):
        config.set_experiment(experiment)
        return set(config.get("scoring", "stages"))

    def test_every_pair_differs_by_exactly_its_own_stage(self):
        for stage, label, with_arm, without_arm in findings.ABLATIONS:
            with self.subTest(stage=stage):
                has = self.stages(with_arm)
                lacks = self.stages(without_arm)
                self.assertIn(stage, has, f"{label}: {with_arm} lacks {stage}")
                self.assertNotIn(stage, lacks,
                                 f"{label}: {without_arm} still has {stage}")
                # An ablation that differs by two stages measures the pair, not
                # the stage it is named after.
                self.assertEqual(has - lacks, {stage})
                self.assertEqual(lacks - has, set())

    def test_no_pair_compares_an_arm_with_itself(self):
        """What the old diversity row did, silently, for a full rescore."""
        for stage, label, with_arm, without_arm in findings.ABLATIONS:
            with self.subTest(stage=stage):
                self.assertNotEqual(with_arm, without_arm)
                self.assertNotEqual(self.stages(with_arm),
                                    self.stages(without_arm),
                                    f"{label}: both arms run the same stages")

    def test_every_named_arm_still_exists_in_the_configuration(self):
        for _, label, with_arm, without_arm in findings.ABLATIONS:
            for arm in (with_arm, without_arm):
                self.assertIn(arm, config.experiment_names,
                              f"{label} names '{arm}', which is not configured")

    def test_the_default_pipeline_arm_carries_the_shipped_stages(self):
        """`ablation-full` is the reference, so it must equal the default."""
        config.set_experiment("baseline")
        default = set(config.get("scoring", "stages"))
        self.assertEqual(self.stages(findings.DEFAULT_PIPELINE), default)


if __name__ == "__main__":
    unittest.main()
