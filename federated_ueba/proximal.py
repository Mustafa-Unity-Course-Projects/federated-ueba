"""The FedProx proximal term, which is the whole of what FedProx adds.

FedProx (Li et al., 2020, arXiv:1812.06127) changes nothing on the server: its
aggregation is FedAvg's. What it changes is each client's objective, by adding a
penalty for drifting away from the global model it started the round with:

    loss = MSE(reconstruction, input) + (mu / 2) * || w - w_global ||^2

The reason it exists is client drift. With 5 local epochs a client can travel a
long way from the shared model, and under a heterogeneous partition each one
travels somewhere different; averaging those apart-pulled models gives something
worse than any of them. The penalty is a leash whose length is `mu`. At mu = 0
FedProx is exactly FedAvg, which is the identity this module's tests pin.

Whether it helps here is an open question rather than an assumption. The measured
non-IID setting in this work is quantity skew, which FedAvg already resists, so
the honest expectation is a small effect. It is implemented so the comparison can
be made instead of asserted.

A note on the definition. Flower's own FedProx docstring shows the penalty built
from `.norm(2)` summed over tensors, which is a sum of norms rather than a
squared norm. That is not what the paper defines and it changes the gradient's
scale. This module implements the paper: the squared L2 norm of the concatenated
difference, which equals the sum of the per-tensor squared norms.
"""

import torch


def proximal_penalty(local_parameters, global_parameters, mu):
    """The (mu / 2) * ||w - w_global||^2 term, as a differentiable scalar.

    `local_parameters` are the live tensors being optimised, so the result keeps
    its graph and can be added straight to the loss. `global_parameters` are the
    round's starting point and are treated as constants: they are a fixed target,
    not something the optimiser may move.

    Returns a tensor rather than a float so that `loss + penalty` stays
    differentiable. At mu = 0 it returns a zero that still carries the right
    device and dtype, so callers need no special case.
    """
    if mu < 0:
        raise ValueError(f"proximal_mu must not be negative, got {mu}.")

    total = None
    for local, reference in zip(local_parameters, global_parameters):
        if local.shape != reference.shape:
            raise ValueError(
                f"Proximal term needs matching shapes, got {tuple(local.shape)} "
                f"and {tuple(reference.shape)}. The reference must be the global "
                f"model this round started from, in state_dict order.")
        # detach: the global model is where the client was told to stay near, so
        # gradient must flow to the local weights only. Without this the penalty
        # would also try to move a tensor that is not being trained.
        difference = local - reference.detach()
        squared = torch.sum(difference * difference)
        total = squared if total is None else total + squared

    if total is None:
        # No parameters at all. Returning a plain zero would break `loss + term`
        # on a non-CPU device, so this is only reachable for an empty model.
        return torch.zeros(())

    return (mu / 2.0) * total


def snapshot_global(model):
    """A detached copy of the weights a round starts from.

    Taken before local training, because after the first optimiser step the model
    no longer holds the global weights the penalty is measured against. Cloned
    rather than referenced for the same reason: the tensors are about to be
    updated in place.
    """
    return [p.detach().clone() for p in model.parameters()]
