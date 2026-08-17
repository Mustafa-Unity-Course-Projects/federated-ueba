"""Communication accounting: the MB figures the thesis reports.

These numbers are the whole point of the work, and until now they came out of a
closure nested inside a 293-line function, so nothing could check them. Moving
the function out is what let this file exist, and writing it immediately found
that `per_round` was dividing by the number of transfer records rather than the
number of rounds.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_insider_detection import read_communication_log  # noqa: E402


class CommunicationLogTestCase(unittest.TestCase):
    """Builds the per-client log files a real run would leave behind."""

    def setUp(self):
        self.report_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.report_dir, ignore_errors=True)

    def write_log(self, partition_id, records):
        comm_dir = os.path.join(self.report_dir, "comm")
        os.makedirs(comm_dir, exist_ok=True)
        path = os.path.join(comm_dir, f"client_{partition_id}.csv")
        with open(path, "a") as f:
            for direction, phase, mb in records:
                f.write(f"{direction},{phase},{mb:.4f}\n")

    def simulate_run(self, rounds, clients_per_round, supernodes,
                     upload_mb=0.1, download_mb=1.7176):
        """The transfer pattern one federated run produces.

        Each round: the sampled clients download and upload during fit, then
        every client downloads again to evaluate. Evaluation does not upload
        weights, which is why uploads and downloads are counted separately.
        """
        for round_number in range(rounds):
            for client in range(clients_per_round):
                self.write_log(client, [("download", "fit", download_mb),
                                        ("upload", "fit", upload_mb)])
            for client in range(supernodes):
                self.write_log(client, [("download", "eval", download_mb)])


class TestTotals(CommunicationLogTestCase):
    def test_uploads_and_downloads_are_summed_separately(self):
        self.write_log(0, [("upload", "fit", 0.5), ("download", "fit", 1.0)])
        self.write_log(1, [("upload", "fit", 0.25), ("download", "eval", 2.0)])

        totals = read_communication_log(self.report_dir, expected_uploads=2,
                                        num_rounds=1)

        self.assertAlmostEqual(totals["upload"], 0.75, places=4)
        self.assertAlmostEqual(totals["download"], 3.0, places=4)
        self.assertAlmostEqual(totals["total"], 3.75, places=4)

    def test_every_client_file_is_counted(self):
        """One file per partition; missing one would understate the total."""
        for partition in range(25):
            self.write_log(partition, [("upload", "fit", 0.1)])

        totals = read_communication_log(self.report_dir, expected_uploads=25,
                                        num_rounds=1)

        self.assertAlmostEqual(totals["upload"], 2.5, places=4)
        self.assertEqual(totals["upload_records"], 25)

    def test_a_run_with_no_log_reports_zero_rather_than_failing(self):
        totals = read_communication_log(self.report_dir, expected_uploads=50,
                                        num_rounds=50)

        self.assertEqual(totals["total"], 0.0)
        self.assertIsNone(totals["log_consistent"])


class TestPerRound(CommunicationLogTestCase):
    """The field is named per_round, so it has to be per round."""

    def test_per_round_divides_by_rounds_not_by_records(self):
        self.simulate_run(rounds=10, clients_per_round=25, supernodes=50)

        totals = read_communication_log(self.report_dir, expected_uploads=250,
                                        num_rounds=10)

        self.assertAlmostEqual(totals["per_round"], totals["total"] / 10, places=6)

    def test_per_round_times_rounds_reconstructs_the_total(self):
        """The identity a reader of the thesis table will check."""
        self.simulate_run(rounds=50, clients_per_round=25, supernodes=50)

        totals = read_communication_log(self.report_dir, expected_uploads=1250,
                                        num_rounds=50)

        self.assertAlmostEqual(totals["per_round"] * 50, totals["total"], places=4)

    def test_the_old_record_based_figure_was_much_smaller(self):
        """Pins the size of the error, so the fix cannot be quietly undone.

        With 25 fit clients and 50 evaluating, one round writes 100 records, so
        dividing by records understated the per-round cost a hundredfold.
        """
        self.simulate_run(rounds=10, clients_per_round=25, supernodes=50)
        totals = read_communication_log(self.report_dir, expected_uploads=250,
                                        num_rounds=10)

        records = 10 * (25 * 2 + 50)
        old_figure = totals["total"] / records
        self.assertAlmostEqual(totals["per_round"] / old_figure, records / 10,
                               places=4)

    def test_zero_rounds_does_not_divide_by_zero(self):
        self.write_log(0, [("upload", "fit", 0.5)])
        totals = read_communication_log(self.report_dir, expected_uploads=1,
                                        num_rounds=0)
        self.assertAlmostEqual(totals["per_round"], 0.5, places=4)


class TestConsistencyCheck(CommunicationLogTestCase):
    """A log left over from a different run silently produced wrong totals."""

    def test_a_matching_log_is_accepted(self):
        self.simulate_run(rounds=50, clients_per_round=25, supernodes=50)

        totals = read_communication_log(self.report_dir, expected_uploads=1250,
                                        num_rounds=50)
        self.assertTrue(totals["log_consistent"])

    def test_the_extra_upload_flower_makes_at_startup_is_tolerated(self):
        """Flower asks one client for the initial parameters before round 1."""
        self.simulate_run(rounds=50, clients_per_round=25, supernodes=50)
        self.write_log(0, [("upload", "init", 1.7176)])

        totals = read_communication_log(self.report_dir, expected_uploads=1250,
                                        num_rounds=50)
        self.assertTrue(totals["log_consistent"])

    def test_a_log_from_a_longer_run_is_rejected(self):
        """100 rounds of records under a 50 round configuration."""
        self.simulate_run(rounds=100, clients_per_round=25, supernodes=50)

        totals = read_communication_log(self.report_dir, expected_uploads=1250,
                                        num_rounds=50)
        self.assertFalse(totals["log_consistent"])

    def test_a_truncated_log_is_rejected(self):
        """What the lost-records bug looked like: most uploads simply missing."""
        self.simulate_run(rounds=50, clients_per_round=5, supernodes=50)

        totals = read_communication_log(self.report_dir, expected_uploads=1250,
                                        num_rounds=50)
        self.assertFalse(totals["log_consistent"])


if __name__ == "__main__":
    unittest.main()
