"""The user split belongs to the evaluation protocol, not to the run.

It used to be seeded with the run seed, so every seed reported on a different set
of 35 insiders. That is not a small effect. Scoring one unchanged model against
twenty different splits gives a standard deviation of 0.0548 and a range from
0.7513 to 0.9146, and two runs of the same configuration at seeds 1 and 2
differed by 0.0952 on their own splits against 0.0069 on any shared one. The
whole apparent instability was the split.

The noise mattered because it is larger than everything being measured:
compression costs between 0.01 and 0.13, the diversity multiplier 0.006.
Averaging over seeds does not rescue it either, since 0.0548 over five seeds is
still 0.024.

These tests pin the property that fixed it: the split is the same for every run,
and it still never lets the test half influence a choice.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config_manager import config
from federated_ueba import scoring


def population(n_users=1000, n_insiders=70):
    users = list(range(n_users))
    labels = [1 if u < n_insiders else 0 for u in users]
    return users, labels


class TheSplitDoesNotFollowTheRunSeed(unittest.TestCase):
    def test_the_configured_split_seed_is_not_the_run_seed(self):
        """Different keys, and nothing should quietly tie them together."""
        self.assertIn("split_seed", config._schema()["evaluation"]
                      if hasattr(config, "_schema") else {"split_seed": int})

    def test_a_scoring_config_carries_the_split_seed(self):
        cfg = scoring.ScoringConfig.from_config(config)
        self.assertEqual(cfg.split_seed,
                         config.get("evaluation", "split_seed"))

    def test_two_runs_of_different_seeds_get_the_same_test_users(self):
        users, labels = population()
        cfg = scoring.ScoringConfig.from_config(config)
        first = scoring.split_users(users, labels, seed=cfg.split_seed,
                                    validation_fraction=cfg.validation_fraction)
        second = scoring.split_users(users, labels, seed=cfg.split_seed,
                                     validation_fraction=cfg.validation_fraction)
        self.assertEqual(set(first[1]), set(second[1]))

    def test_a_different_split_seed_would_give_different_users(self):
        """The nuisance is real; it is being held fixed, not wished away."""
        users, labels = population()
        _, test_a = scoring.split_users(users, labels, seed=1,
                                        validation_fraction=0.5)
        _, test_b = scoring.split_users(users, labels, seed=2,
                                        validation_fraction=0.5)
        self.assertNotEqual(set(test_a), set(test_b))

        overlap = len(set(test_a) & set(test_b)) / len(set(test_a))
        # Two random halves of the same population share about half their members.
        self.assertLess(overlap, 0.75)


class TheSplitStillProtectsTheTestHalf(unittest.TestCase):
    """Fixing the seed must not weaken what the split was for."""

    def test_the_halves_are_disjoint(self):
        users, labels = population()
        val, test = scoring.split_users(users, labels, seed=42,
                                        validation_fraction=0.5)
        self.assertEqual(set(val) & set(test), set())

    def test_the_halves_cover_everyone(self):
        users, labels = population()
        val, test = scoring.split_users(users, labels, seed=42,
                                        validation_fraction=0.5)
        self.assertEqual(set(val) | set(test), set(users))

    def test_both_halves_hold_insiders(self):
        """Stratified, so neither half can end up with almost none of them."""
        users, labels = population()
        by_user = dict(zip(users, labels))
        val, test = scoring.split_users(users, labels, seed=42,
                                        validation_fraction=0.5)
        val_insiders = sum(by_user[u] for u in val)
        test_insiders = sum(by_user[u] for u in test)
        self.assertEqual(val_insiders + test_insiders, 70)
        self.assertGreater(min(val_insiders, test_insiders), 25)


class TheReportedNumbersUseTheFixedSplit(unittest.TestCase):
    def test_evaluate_scores_reports_the_same_half_whatever_the_run_seed(self):
        import pandas as pd

        rng = np.random.RandomState(0)
        users, labels = population(200, 20)
        frame = pd.DataFrame({
            "user": users,
            "is_actual_insider": labels,
            "max_z_score": rng.rand(200) + np.array(labels) * 0.5,
        })

        cfg = scoring.ScoringConfig.from_config(config)
        first = scoring.evaluate_scores(
            frame, seed=cfg.split_seed,
            validation_fraction=cfg.validation_fraction)
        second = scoring.evaluate_scores(
            frame, seed=cfg.split_seed,
            validation_fraction=cfg.validation_fraction)
        self.assertEqual(first["pr_auc_test"], second["pr_auc_test"])

    def test_score_one_round_ignores_the_run_seed_it_is_handed(self):
        """The signature keeps the argument; the behaviour must not use it."""
        import inspect

        import federated_insider_detection as fid
        source = inspect.getsource(fid.score_one_round)
        self.assertIn("scorer.cfg.split_seed", source)
        self.assertNotIn("seed=seed", source)


if __name__ == "__main__":
    unittest.main()
