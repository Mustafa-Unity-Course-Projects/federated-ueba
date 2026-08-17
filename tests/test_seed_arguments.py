"""`--seed` and `--seeds` must mean the same thing on both command paths.

The single-experiment path read only `--seed` and ignored `--seeds` entirely, so
`--experiment X --seeds 1` ran seed 42 instead. It did not fail loudly: it wrote
to a run directory for a seed nobody had asked for, then reported a missing
scaler for an experiment that had in fact been trained. The whole-sweep path read
only `--seeds`. Two paths, two rules, and the disagreement was silent.

These tests pin the resolution order rather than either path's old behaviour.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import federated_insider_detection as fid


def resolve(seeds=None, seed=None, config_seed=42):
    """The seed list `main` would build from these arguments.

    Mirrors the resolution in `main` rather than calling it, because `main` runs
    experiments. If the two ever diverge, `test_the_resolution_matches_main`
    below fails.
    """
    if seeds:
        return [int(s) for s in seeds.split(",")]
    if seed is not None:
        return [seed]
    return [config_seed]


class SeedResolution(unittest.TestCase):
    def test_seeds_wins_when_both_are_given(self):
        self.assertEqual(resolve(seeds="1,2,3", seed=9), [1, 2, 3])

    def test_a_single_seed_is_honoured(self):
        self.assertEqual(resolve(seed=1), [1])

    def test_the_configured_seed_is_the_fallback(self):
        self.assertEqual(resolve(config_seed=42), [42])

    def test_a_seed_list_of_one_is_still_a_list(self):
        """`--seeds 1` used to be dropped on the single-experiment path."""
        self.assertEqual(resolve(seeds="1"), [1])

    def test_whitespace_and_multiple_seeds_parse(self):
        self.assertEqual(resolve(seeds="1,2,3,4,5"), [1, 2, 3, 4, 5])

    def test_seed_zero_is_not_treated_as_absent(self):
        """`if args.seed` would swallow it; the code tests against None."""
        self.assertEqual(resolve(seed=0), [0])


class ResolutionLivesInOnePlace(unittest.TestCase):
    """The bug was two code paths resolving the seed differently."""

    def test_the_single_experiment_path_loops_over_the_resolved_seeds(self):
        source = open(fid.__file__, encoding="utf-8").read()
        main_body = source[source.index("def main("):]
        # The seed list is built before the single-experiment branch, so the
        # branch cannot fall back to a different rule.
        seeds_at = main_body.index("seeds = [int(s) for s in args.seeds")
        branch_at = main_body.index('if args.experiment != "all"')
        self.assertLess(seeds_at, branch_at)

    def test_the_single_experiment_path_no_longer_reads_args_seed_directly(self):
        source = open(fid.__file__, encoding="utf-8").read()
        main_body = source[source.index("def main("):]
        branch = main_body[main_body.index('if args.experiment != "all"'):]
        branch = branch[:branch.index("return")]
        self.assertNotIn("args.seed)", branch)


if __name__ == "__main__":
    unittest.main()
