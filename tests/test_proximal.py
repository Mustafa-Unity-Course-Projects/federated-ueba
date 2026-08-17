"""The FedProx proximal term.

The property that matters most is the identity at mu = 0: FedProx must reduce to
FedAvg exactly, so that adding the strategy to the code cannot change what the
FedAvg runs report. Everything else here is about the term being the one the
paper defines rather than the one Flower's docstring sketches.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from federated_ueba.proximal import proximal_penalty, snapshot_global


def tensors(*values):
    return [torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in values]


class TestProximalPenalty(unittest.TestCase):
    def test_zero_mu_is_plain_fedavg(self):
        """The identity that lets FedProx exist in the code without disturbing it."""
        local = tensors([1.0, 2.0], [3.0])
        reference = tensors([0.0, 0.0], [0.0])
        self.assertEqual(float(proximal_penalty(local, reference, mu=0.0)), 0.0)

    def test_no_penalty_when_the_client_has_not_moved(self):
        local = tensors([1.0, -2.0, 0.5])
        reference = [t.detach().clone() for t in local]
        self.assertAlmostEqual(
            float(proximal_penalty(local, reference, mu=1.0)), 0.0)

    def test_it_is_the_squared_norm_not_a_sum_of_norms(self):
        """Difference [3, 4] has norm 5 and squared norm 25.

        At mu = 2 the paper's definition gives (2/2) * 25 = 25. A sum of norms
        would give 5. This is the distinction Flower's docstring blurs.
        """
        local = tensors([3.0, 4.0])
        reference = tensors([0.0, 0.0])
        self.assertAlmostEqual(
            float(proximal_penalty(local, reference, mu=2.0)), 25.0, places=5)

    def test_tensors_are_summed_not_averaged(self):
        """A model is many tensors; the penalty is over all weights at once."""
        single = proximal_penalty(tensors([3.0, 4.0]), tensors([0.0, 0.0]), mu=2.0)
        split = proximal_penalty(tensors([3.0], [4.0]), tensors([0.0], [0.0]), mu=2.0)
        self.assertAlmostEqual(float(single), float(split), places=5)

    def test_the_penalty_grows_with_mu(self):
        local, reference = tensors([1.0, 1.0]), tensors([0.0, 0.0])
        weak = float(proximal_penalty(local, reference, mu=0.01))
        strong = float(proximal_penalty(local, reference, mu=1.0))
        self.assertLess(weak, strong)

    def test_a_negative_mu_is_refused(self):
        """It would reward drifting, which is the opposite of the point."""
        with self.assertRaises(ValueError):
            proximal_penalty(tensors([1.0]), tensors([0.0]), mu=-0.1)

    def test_shape_mismatch_is_refused_rather_than_broadcast(self):
        """Silent broadcasting would produce a plausible number from wrong pairing."""
        with self.assertRaises(ValueError):
            proximal_penalty(tensors([1.0, 2.0]), tensors([0.0]), mu=0.1)


class TestGradientFlow(unittest.TestCase):
    def test_the_gradient_pulls_the_local_weights_back(self):
        local = tensors([2.0])
        reference = tensors([0.0])
        proximal_penalty(local, reference, mu=1.0).backward()
        # d/dw of (1/2) w^2 is w, so the gradient at w = 2 is 2, pointing away
        # from the reference; the optimiser subtracts it and moves back toward 0.
        self.assertAlmostEqual(float(local[0].grad), 2.0, places=5)

    def test_no_gradient_reaches_the_global_model(self):
        """The reference is a target, not a parameter to be optimised."""
        local = tensors([2.0])
        reference = tensors([0.0])
        proximal_penalty(local, reference, mu=1.0).backward()
        self.assertIsNone(reference[0].grad)

    def test_it_can_be_added_to_a_loss(self):
        local = tensors([2.0])
        combined = torch.nn.functional.mse_loss(
            local[0], torch.tensor([1.0])) + proximal_penalty(
                local, tensors([0.0]), mu=1.0)
        combined.backward()
        self.assertIsNotNone(local[0].grad)


class TestSnapshot(unittest.TestCase):
    def test_the_snapshot_does_not_move_when_the_model_trains(self):
        """Referencing instead of cloning would make the penalty always zero."""
        model = torch.nn.Linear(3, 2)
        before = snapshot_global(model)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        penalty = proximal_penalty(list(model.parameters()), before, mu=1.0)
        self.assertGreater(float(penalty), 0.0)

    def test_the_snapshot_carries_no_gradient(self):
        model = torch.nn.Linear(3, 2)
        self.assertTrue(all(not t.requires_grad for t in snapshot_global(model)))


if __name__ == "__main__":
    unittest.main()
