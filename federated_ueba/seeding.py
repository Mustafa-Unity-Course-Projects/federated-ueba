"""Deterministic seeding, shared by the runner, the server and every client.

`flower-simulation` is a separate interpreter and its Ray actors are separate
processes again, so seeding the runner does nothing for them. Before this module
existed, only the runner was seeded, and two runs at the same seed produced
different weights: measurably so from round 0, the untrained starting model,
because client model initialisation was never seeded at all.

What the seed has to reach, and where each is handled:

    partitioning, evaluation splits   the runner process (already seeded)
    which clients a round samples     the server process
    model init, dropout, denoising,   each client actor
    batch shuffling

The client seed is derived from the run seed, the partition id and the round
number rather than being a single constant. That matters because Ray decides
which actor handles which partition, and that assignment is not stable between
runs: seeding once per actor would make the result depend on scheduling. Keying
on the partition and round instead makes a client's work identical no matter
which actor picks it up or in what order.
"""

import os
import random

# Must be set before any CUDA context exists, which is why it is at import
# time rather than inside the function. cuBLAS picks a workspace per stream
# otherwise, and the reduction order in an LSTM backward follows from it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def derive_seed(*parts):
    """Combine a run seed with identifiers into one reproducible 32-bit seed.

    Plain addition would collide: seed 1 with partition 2 and seed 2 with
    partition 1 would land on the same value, making two different situations
    share a random stream. Mixing each part into a hash avoids that.
    """
    value = 0
    for part in parts:
        # 1_000_003 is prime, so successive parts do not fold onto each other.
        value = (value * 1_000_003 + int(part)) % (2 ** 32)
    return value


def seed_everything(*parts):
    """Seed torch, numpy and random from the given identifiers.

    The comment here used to say deterministic cuDNN was not needed because the
    reported runs execute on the CPU. That was wrong: `[experiment] device` is
    "auto" and this machine has CUDA, so every run has been on the GPU. cuDNN
    picks nondeterministic algorithms for LSTM backward by default, and it was
    the last reason two runs of the same seed produced different weights after
    client sampling had been fixed.
    """
    seed = derive_seed(*parts)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Only when CUDA is already in use. Calling into torch.cuda here would build
    # a CUDA context inside every Ray worker even on a CPU run, which cost 3.6x
    # wall-clock when it was measured: 307s against 85s for the same two rounds.
    # `is_initialized` is false until something actually touches the device, so
    # this stays quiet on CPU runs and still seeds a genuine GPU run.
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)
        # deterministic picks a fixed algorithm; benchmark off stops cuDNN
        # choosing one by timing, which depends on what else the machine is
        # doing and so differs between runs of the same seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only, because an op without a deterministic kernel should degrade
        # to a warning rather than kill a 13-hour sweep.
        torch.use_deterministic_algorithms(True, warn_only=True)

    # PYTHONHASHSEED is deliberately not set here: CPython reads it once at
    # interpreter start, so assigning it to os.environ afterwards changes
    # nothing. Verified rather than assumed. The runner sets it in the
    # environment it hands to the simulation subprocess, which is a fresh
    # interpreter and therefore does honour it.
    return seed
