"""One test per bug that actually happened, for the ones with no other home.

Every entry here is a real failure from this project, not a hypothetical. Bugs
whose subject has its own module are tested next to it instead, and are listed
below so this file works as the full ledger:

    scaler pathology, per-client scale of 3.3e-15    test_scaling
    partition independence, and what the privacy
      floor costs it                                 test_cohort_threshold
    server and clients sizing the model differently  test_input_dimension
    download charged raw while upload was coded      test_downlink
    build_strategy ignoring the config handed to it  test_strategy
    per-round MB divided by record count             test_communication_log
    convergence defined two ways                     test_convergence
    the reported numbers as a whole                  test_regression

What is left over are failures of plumbing rather than of a function: settings
that no code reads, a sweep that selects nothing, a sweep that fails silently.
Those share no module, and each one cost hours precisely because nothing failed.
"""

import io
import os
import re
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_manager
import federated_insider_detection as runner
from config_manager import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Genuinely read by nothing. Documented in pyproject.toml as recording which
# corpus the processed file came from; the raw tree is read by
# feature_extraction.py, which is inherited and must not be modified.
KEYS_NOTHING_READS = {("data", "dataset_path")}

# Read, but not through `config.get(section, key)`, so the textual scan below
# cannot see it. Each one is checked separately by a test rather than simply
# excused, because "the scan cannot see it" and "nothing reads it" look identical
# from here and only one of them is acceptable.
KEYS_READ_INDIRECTLY = {("experiment", "seed"): "config.seed"}


def modules_that_read_configuration():
    """Source of every module that may call config.get, as one string."""
    skip = {"archive", ".venv", "venv_wsl", "venv", "dataset", "ExtractedData",
            ".git", "__pycache__", "tests", "logs"}
    sources = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if not name.endswith(".py") or name == "config_manager.py":
                continue
            with open(os.path.join(root, name), encoding="utf-8-sig") as handle:
                sources.append(handle.read())
    return "\n".join(sources)


class TestEverySettingIsRead(unittest.TestCase):
    """A configured value nothing reads is worse than a hardcoded one.

    At one point 23 of 59 declared keys were dead: written in pyproject.toml,
    checked by the schema, documented in a comment, and ignored by the code,
    which kept its own literal. Setting `use_bottleneck = false` changed nothing
    and raised nothing. Two of the dead keys were hiding a real problem, because
    the centralized and federated arms had drifted onto different optimiser
    settings while the file showed one value.

    This was found three times with a throwaway script before it became a test.
    """

    def setUp(self):
        self.source = modules_that_read_configuration()
        self.reads = set(re.findall(
            r"""get\(\s*["']([a-z_]+)["']\s*,\s*["']([a-z_0-9]+)["']""",
            self.source))

    def test_no_declared_setting_is_ignored_by_the_code(self):
        declared = {(section, key)
                    for section, keys in config_manager.SCHEMA.items()
                    for key in keys}
        unread = (declared - self.reads - KEYS_NOTHING_READS
                  - set(KEYS_READ_INDIRECTLY))
        self.assertEqual(
            unread, set(),
            "declared in pyproject.toml and read nowhere; either wire it up or "
            "remove it, because a setting that changes nothing is a lie in the "
            "configuration file")

    def test_the_indirectly_read_settings_really_do_reach_the_code(self):
        """Otherwise this list becomes a way to hide a dead key from the test."""
        config.set_experiment("baseline")
        config.set_seed(4242)
        try:
            self.assertEqual(config.seed, 4242)
            self.assertEqual(config.get("experiment", "seed"), 4242)
        finally:
            config.set_seed(config_manager.config.seed)

    def test_the_documented_exception_really_is_unread(self):
        """Otherwise the allow-list quietly grows into a way to ignore this test."""
        for section, key in KEYS_NOTHING_READS:
            self.assertNotIn((section, key), self.reads,
                             f"{section}.{key} is read now; drop it from the "
                             f"exception list")

    def test_nothing_reads_a_setting_the_schema_does_not_declare(self):
        """The other direction: a read that validate() would never check."""
        declared = {(section, key)
                    for section, keys in config_manager.SCHEMA.items()
                    for key in keys}
        optional = set(config_manager.OPTIONAL)
        sections = set(config_manager.SCHEMA)

        undeclared = {(s, k) for s, k in self.reads
                      if s in sections and (s, k) not in declared
                      and (s, k) not in optional}
        self.assertEqual(undeclared, set(),
                         "read by the code but absent from SCHEMA, so validate() "
                         "cannot catch it going missing")


class TestASweepCannotSucceedWithoutWorking(unittest.TestCase):
    """Two ways a sweep reported success having done nothing."""

    def test_selecting_no_experiment_stops_rather_than_finishing(self):
        """`--only smoke-test` printed "Running 0 experiment(s)" and then "All
        experiments finished", exiting zero. In a log and in an exit code that is
        indistinguishable from a completed sweep."""
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                runner.select_experiments(config, "smoke-test", "")
        self.assertIn("nothing to sweep", str(caught.exception))

    def test_the_message_names_what_is_always_excluded(self):
        """Naming a smoke test is the likely mistake, so the error has to say so."""
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                runner.select_experiments(config, "smoke-test", "")
        for name in runner.NOT_IN_SWEEP:
            self.assertIn(name, str(caught.exception))

    def test_a_normal_selection_still_returns_work(self):
        """Guarding against nothing must not start rejecting everything."""
        with redirect_stdout(io.StringIO()):
            chosen = runner.select_experiments(config, "", "")
        self.assertGreater(len(chosen), 1)


class TestTheSweepStopsWhenEverythingFails(unittest.TestCase):
    """The collapse: 139 experiments spawned and dead in eight seconds.

    The parent was killed mid-sweep, so every subsequent subprocess died on
    startup. The loop ignored exit codes, so it ran to the end of the list,
    printed "All experiments finished" and exited zero. Four experiments had
    really completed; the rest had not, and nothing said so.
    """

    def sweep_with(self, codes):
        """Run the loop against a scripted sequence of exit codes."""
        attempts = []

        def fake_spawn(exp_name, mode, seed):
            attempts.append(f"{exp_name}__seed{seed}")
            return codes[len(attempts) - 1]

        with redirect_stdout(io.StringIO()):
            completed = runner.sweep(["a", "b", "c", "d"], [1], "train",
                                     spawn=fake_spawn)
        return completed, attempts

    def test_it_gives_up_after_consecutive_failures(self):
        completed, attempts = self.sweep_with([1, 1, 0, 0])
        self.assertFalse(completed)
        self.assertEqual(len(attempts), runner.MAX_CONSECUTIVE_FAILURES)

    def test_one_failure_between_successes_does_not_end_the_sweep(self):
        """A single flaky run is worth stepping over; one did recover on retry."""
        completed, attempts = self.sweep_with([0, 1, 0, 0])
        self.assertTrue(completed)
        self.assertEqual(len(attempts), 4)

    def test_the_counter_resets_on_success(self):
        """Without a reset, any two failures in a long sweep would stop it."""
        completed, attempts = self.sweep_with([1, 0, 1, 0])
        self.assertTrue(completed)
        self.assertEqual(len(attempts), 4)

    def test_a_clean_sweep_reports_completion(self):
        completed, attempts = self.sweep_with([0, 0, 0, 0])
        self.assertTrue(completed)
        self.assertEqual(len(attempts), 4)


class TestVersionDiscipline(unittest.TestCase):
    """The stamp that keeps two pipelines out of one table.

    Not bumping it after changing how communication was measured let the resume
    guard skip exactly the run that needed redoing, and a comparison was drawn
    between the old accounting and the new one. The reported saving was inflated
    by three points before it was caught.
    """

    def test_every_version_is_explained_in_the_source(self):
        """A number with no note is a number nobody can act on later."""
        with open(os.path.join(PROJECT_ROOT, "config_manager.py"),
                  encoding="utf-8-sig") as handle:
            head = handle.read().split("PIPELINE_VERSION")[0]
        for version in range(1, config_manager.PIPELINE_VERSION + 1):
            self.assertRegex(head, rf"#\s+{version}\s",
                             f"version {version} has no entry in the history "
                             f"comment above PIPELINE_VERSION")

    def test_a_result_from_another_version_is_not_treated_as_done(self):
        import json
        import shutil
        import tempfile

        temporary = tempfile.mkdtemp()
        run_id = "made-up-run__seed1"
        os.makedirs(os.path.join(temporary, run_id))
        summary = {"done": True,
                   "pipeline_version": config_manager.PIPELINE_VERSION - 1}
        with open(os.path.join(temporary, run_id, "experiment_summary.json"),
                  "w") as handle:
            json.dump(summary, handle)

        original = runner.BASE_REPORT_DIR
        try:
            runner.BASE_REPORT_DIR = temporary
            with redirect_stdout(io.StringIO()):
                self.assertFalse(runner.already_done(run_id))
        finally:
            runner.BASE_REPORT_DIR = original
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
