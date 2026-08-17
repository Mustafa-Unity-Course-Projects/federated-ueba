"""A round that lost clients must not reach a results table.

Found on 2026-08-05. Ray actors began dying mid-sweep on Windows (worker exit
code 10054, with memory, GPU, disk and ports all healthy), and ten consecutive
rounds of `delta-0.05__seed3` aggregated 5 clients instead of 25. Flower treats a
crashed client as an absent one, so the run kept going, would have reported 50
rounds, and would have been marked `done` like any other. The only trace was a
line inside a 5 MB log.

That is the same silent-failure shape as the sweep loop that once ignored
subprocess exit codes: the number that comes out is plausible, and nothing
downstream can tell it apart from a good one.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import federated_insider_detection as runner
from federated_ueba.strategy import ModelSavingMixin


class FailureRecording(unittest.TestCase):
    """The strategy writes down every round that lost a client."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.strategy = ModelSavingMixin.__new__(ModelSavingMixin)
        self.strategy.save_path = __import__("pathlib").Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def log_path(self):
        return os.path.join(self.tmp, ModelSavingMixin.FAILURE_LOG)

    def test_a_clean_round_writes_nothing(self):
        self.strategy._record_failures(1, [MagicMock()] * 25, [])
        self.assertFalse(os.path.exists(self.log_path()))

    def test_a_degraded_round_is_recorded_with_its_counts(self):
        self.strategy._record_failures(7, [MagicMock()] * 5, [MagicMock()] * 20)

        with open(self.log_path()) as f:
            record = json.load(f)
        self.assertEqual(record["7"], {"results": 5, "failures": 20})

    def test_several_rounds_accumulate_rather_than_overwrite(self):
        self.strategy._record_failures(7, [MagicMock()] * 5, [MagicMock()] * 20)
        self.strategy._record_failures(8, [MagicMock()] * 24, [MagicMock()])

        with open(self.log_path()) as f:
            record = json.load(f)
        self.assertEqual(sorted(record), ["7", "8"])


class RejectingDegradedRuns(unittest.TestCase):
    """The runner refuses to score what the strategy flagged."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.layout = MagicMock()
        self.layout.train_save_path = self.tmp
        self.layout.run_id = "delta-0.05__seed3"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_log(self, record):
        with open(os.path.join(self.tmp, ModelSavingMixin.FAILURE_LOG), "w") as f:
            json.dump(record, f)

    def test_no_log_means_the_run_is_fine(self):
        runner.reject_degraded_training(self.layout)

    def test_an_empty_log_means_the_run_is_fine(self):
        self.write_log({})
        runner.reject_degraded_training(self.layout)

    def test_a_recorded_failure_stops_the_run(self):
        self.write_log({"7": {"results": 5, "failures": 20}})
        with self.assertRaises(RuntimeError):
            runner.reject_degraded_training(self.layout)

    def test_the_message_names_the_run_and_the_worst_round(self):
        self.write_log({"7": {"results": 5, "failures": 20},
                        "8": {"results": 24, "failures": 1}})
        with self.assertRaises(RuntimeError) as caught:
            runner.reject_degraded_training(self.layout)

        message = str(caught.exception)
        self.assertIn("delta-0.05__seed3", message)
        self.assertIn("2 round(s)", message)
        # The worst round, not the last one, because that is what decides
        # whether the run is salvageable.
        self.assertIn("5 of 25", message)

    def test_even_one_lost_client_is_enough(self):
        """No threshold on purpose.

        A tolerance would need a defensible number, and there is none: a round
        missing one client is still not the configuration described, and the
        cost of re-running a pair is half an hour against a result nobody can
        audit afterwards.
        """
        self.write_log({"3": {"results": 24, "failures": 1}})
        with self.assertRaises(RuntimeError):
            runner.reject_degraded_training(self.layout)


if __name__ == "__main__":
    unittest.main()
