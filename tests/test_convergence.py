"""The three convergence definitions reported for every federated run.

The jury asked what "converged" means and the thesis and the code had drifted
apart on the answer, so every definition is pinned here. They are pure functions
over the per-round validation scores; nothing about training is involved.

Two of them read each run against its own peak and belong to that run alone. The
third reads every run against one shared target and is the only one that may be
put in a column next to another experiment.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from analysis.compare_experiments import (find_shared_convergence_round,
                                          load_validation_curve,
                                          shared_convergence_target)
from federated_insider_detection import (UNTRAINED_ROUND,
                                         find_convergence_round,
                                         find_plateau_round,
                                         trained_rounds)


class TestConvergenceRound(unittest.TestCase):
    """Thesis section 4.3: first round reaching 95% of the peak PR-AUC."""

    def test_fires_when_the_score_first_gets_within_five_percent_of_the_peak(self):
        rounds = [0, 2, 4, 6, 8]
        scores = [0.10, 0.30, 0.50, 0.86, 0.90]
        # Peak 0.90, target 0.855: round 6 is the first to clear it.
        self.assertEqual(find_convergence_round(rounds, scores), 6)

    def test_an_early_peak_makes_it_fire_at_the_first_round(self):
        """The known weakness of this definition, and why the plateau is reported too."""
        rounds = [0, 2, 4]
        scores = [0.90, 0.50, 0.40]
        self.assertEqual(find_convergence_round(rounds, scores), 0)

    def test_a_still_improving_run_converges_only_at_the_end(self):
        rounds = [0, 2, 4, 6]
        scores = [0.10, 0.30, 0.50, 0.90]
        self.assertEqual(find_convergence_round(rounds, scores), 6)

    def test_no_rounds_gives_no_answer(self):
        self.assertIsNone(find_convergence_round([], []))


class TestPlateauRound(unittest.TestCase):
    """The stricter reading: the round after which nothing more is bought."""

    def test_fires_once_later_rounds_stop_improving(self):
        rounds = [0, 2, 4, 6, 8]
        scores = [0.50, 0.70, 0.72, 0.721, 0.719]
        # From round 4 on, no later score beats 0.72 by more than 0.005.
        self.assertEqual(find_plateau_round(rounds, scores), 4)

    def test_a_monotonically_improving_run_never_plateaus_early(self):
        rounds = [0, 2, 4, 6]
        scores = [0.10, 0.30, 0.50, 0.90]
        self.assertEqual(find_plateau_round(rounds, scores), 6)

    def test_a_decaying_run_plateaus_immediately(self):
        rounds = [0, 2, 4]
        scores = [0.90, 0.50, 0.40]
        self.assertEqual(find_plateau_round(rounds, scores), 0)

    def test_a_late_recovery_delays_the_plateau(self):
        rounds = [0, 2, 4, 6]
        scores = [0.50, 0.90, 0.40, 0.60]
        self.assertEqual(find_plateau_round(rounds, scores), 2)

    def test_the_two_definitions_disagree_on_a_slow_climb(self):
        """Reporting both is only worth the space because they differ."""
        rounds = list(range(0, 10, 2))
        scores = [0.10, 0.40, 0.70, 0.88, 0.90]
        self.assertEqual(find_convergence_round(rounds, scores), 6)
        self.assertEqual(find_plateau_round(rounds, scores), 8)


class TestSharedConvergenceRound(unittest.TestCase):
    """The cross-experiment definition: one absolute target for every run."""

    def test_fires_at_the_first_round_reaching_the_target(self):
        rounds = [0, 2, 4, 6, 8]
        scores = [0.10, 0.30, 0.50, 0.86, 0.90]
        self.assertEqual(find_shared_convergence_round(rounds, scores, 0.85), 6)

    def test_a_run_that_never_reaches_the_target_says_so(self):
        """The answer the per-run definition cannot give."""
        rounds = [0, 2, 4]
        scores = [0.40, 0.50, 0.55]
        self.assertEqual(find_shared_convergence_round(rounds, scores, 0.85),
                         "never")

    def test_a_missing_target_is_not_the_same_as_a_missed_one(self):
        """None means unmeasurable, "never" means measured and not reached."""
        self.assertIsNone(find_shared_convergence_round([0, 2], [0.1, 0.2], None))
        self.assertIsNone(find_shared_convergence_round([], [], 0.85))

    def test_it_reverses_the_ranking_the_per_run_definition_gives(self):
        """The measurement that made this column necessary (Y9).

        A weaker configuration reaches 95% of its own lower ceiling sooner, so
        the per-run number calls it the faster one. Against a shared target it
        is correctly reported as never getting there at all.
        """
        rounds = list(range(0, 10, 2))
        strong = [0.10, 0.40, 0.70, 0.88, 0.90]
        weak = [0.10, 0.58, 0.60, 0.60, 0.60]

        # Per-run: the weak run looks like it converged first.
        self.assertLess(find_convergence_round(rounds, weak),
                        find_convergence_round(rounds, strong))

        # Shared: 95% of the strong run's peak, which the weak run never reaches.
        target = max(strong) * 0.95
        self.assertEqual(find_shared_convergence_round(rounds, strong, target), 6)
        self.assertEqual(find_shared_convergence_round(rounds, weak, target),
                         "never")

    def test_the_reference_run_converges_by_its_own_target(self):
        """A sanity bound: the baseline must reach 95% of its own peak."""
        rounds = [0, 2, 4, 6]
        scores = [0.10, 0.30, 0.50, 0.90]
        target = max(scores) * 0.95
        self.assertEqual(find_shared_convergence_round(rounds, scores, target), 6)


class TestSharedConvergenceTarget(unittest.TestCase):
    """Where the shared target comes from: the baseline runs' own CSVs.

    Reading the round CSV rather than the summary is the point. It means the
    column can be added to runs that already finished, so closing Y9 costs no
    re-runs and cannot change a number a running experiment is reporting.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def _run(self, run_id, scores):
        exp_path = os.path.join(self.tmp, run_id)
        os.makedirs(exp_path)
        pd.DataFrame({"Round": list(range(len(scores))),
                      "PR-AUC_val": scores}).to_csv(
            os.path.join(exp_path, "federated_rounds_comparison.csv"), index=False)
        return {"run_id": run_id}, run_id, exp_path

    def test_the_target_is_a_fraction_of_the_baseline_peak(self):
        runs = [self._run("baseline__seed1", [0.1, 0.5, 0.8])]
        self.assertAlmostEqual(shared_convergence_target(runs, fraction=0.95),
                               0.8 * 0.95)

    def test_several_baseline_seeds_are_averaged(self):
        """So the target does not inherit one run's luck."""
        runs = [self._run("baseline__seed1", [0.1, 0.8]),
                self._run("baseline__seed2", [0.1, 0.6])]
        self.assertAlmostEqual(shared_convergence_target(runs, fraction=1.0), 0.7)

    def test_other_experiments_do_not_move_the_target(self):
        runs = [self._run("baseline__seed1", [0.1, 0.8]),
                self._run("top-k-0.05__seed1", [0.1, 0.2])]
        self.assertAlmostEqual(shared_convergence_target(runs, fraction=1.0), 0.8)

    def test_no_baseline_run_leaves_the_target_undefined(self):
        """Empty column rather than a target quietly taken from something else."""
        runs = [self._run("top-k-0.05__seed1", [0.1, 0.2])]
        self.assertIsNone(shared_convergence_target(runs, fraction=0.95))

    def test_a_run_without_a_round_csv_is_skipped_not_fatal(self):
        exp_path = os.path.join(self.tmp, "baseline__seed9")
        os.makedirs(exp_path)
        runs = [({"run_id": "baseline__seed9"}, "baseline__seed9", exp_path),
                self._run("baseline__seed1", [0.1, 0.8])]
        self.assertAlmostEqual(shared_convergence_target(runs, fraction=1.0), 0.8)

    def test_the_curve_read_is_the_validation_half(self):
        """Not the test half: convergence is a choice, and choices read val."""
        exp_path = os.path.join(self.tmp, "baseline__seed1")
        os.makedirs(exp_path)
        pd.DataFrame({"Round": [0, 1, 2],
                      "PR-AUC_val": [0.1, 0.8, 0.85],
                      "PR-AUC": [0.9, 0.95, 0.96]}).to_csv(
            os.path.join(exp_path, "federated_rounds_comparison.csv"), index=False)
        self.assertEqual(load_validation_curve(exp_path), ([1, 2], [0.8, 0.85]))

    def test_round_zero_is_excluded_from_the_curve(self):
        """It is the untrained model scored against a trained reference.

        The clients measure the reference error distribution with their locally
        trained models, so round 0 is judged against something that did not exist
        when it was saved. Measured on the real runs, the baseline scores 0.8972
        at round 0 and 0.8810 at round 50: keeping it would say the untrained
        model beat the trained one and would make every run converge at round 0.
        """
        exp_path = os.path.join(self.tmp, "baseline__seed1")
        os.makedirs(exp_path)
        pd.DataFrame({"Round": [0, 2, 4],
                      "PR-AUC_val": [0.99, 0.5, 0.8]}).to_csv(
            os.path.join(exp_path, "federated_rounds_comparison.csv"), index=False)

        rounds, scores = load_validation_curve(exp_path)
        self.assertNotIn(0, rounds)
        self.assertEqual(scores, [0.5, 0.8])

        # And so the target comes from a trained peak, not the untrained one.
        runs = [({"run_id": "baseline__seed1"}, "baseline__seed1", exp_path)]
        self.assertAlmostEqual(shared_convergence_target(runs, fraction=1.0), 0.8)

    def test_a_run_with_only_round_zero_has_no_curve(self):
        exp_path = os.path.join(self.tmp, "baseline__seed1")
        os.makedirs(exp_path)
        pd.DataFrame({"Round": [0], "PR-AUC_val": [0.9]}).to_csv(
            os.path.join(exp_path, "federated_rounds_comparison.csv"), index=False)
        self.assertIsNone(load_validation_curve(exp_path))


class TestUntrainedRoundIsNeverSelected(unittest.TestCase):
    """The summary writer must drop round 0 before it selects anything.

    The analysis layer already dropped it, but `experiment_summary.json` was
    still written with it in: 21 of the first 63 finished runs recorded
    `best_round = 0`, all three baseline seeds among them. Round 0 is the model
    before any training, scored against a reference distribution the clients
    measured with their trained local models, so it scores high without having
    earned it. These tests pin the exclusion at the point where it was missing.
    """

    def test_round_zero_is_dropped_and_the_rest_keep_their_order(self):
        rounds, scores = trained_rounds([0, 2, 4], [0.9, 0.3, 0.5])
        self.assertEqual(rounds, [2, 4])
        self.assertEqual(scores, [0.3, 0.5])

    def test_a_curve_of_only_round_zero_becomes_empty_rather_than_wrong(self):
        self.assertEqual(trained_rounds([0], [0.9]), ([], []))

    def test_convergence_ignores_an_inflated_untrained_round(self):
        """Without the filter this reports convergence before training began."""
        rounds, scores = trained_rounds([0, 2, 4, 6], [0.99, 0.30, 0.50, 0.86])
        # Peak among trained rounds is 0.86; 95% of it is 0.817, so round 6.
        self.assertEqual(find_convergence_round(rounds, scores, fraction=0.95), 6)

    def test_plateau_ignores_an_inflated_untrained_round(self):
        rounds, scores = trained_rounds([0, 2, 4, 6], [0.99, 0.30, 0.80, 0.80])
        # Round 0 would satisfy "nothing later improves on me" immediately.
        self.assertEqual(find_plateau_round(rounds, scores, tolerance=0.005), 4)

    def test_the_untrained_round_constant_is_zero(self):
        """Named rather than literal, so both call sites move together."""
        self.assertEqual(UNTRAINED_ROUND, 0)


if __name__ == "__main__":
    unittest.main()
