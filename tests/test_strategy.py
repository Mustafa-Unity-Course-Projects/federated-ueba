"""Strategy selection, and the checkpointing that has to behave the same in both.

FedAvg and FedProx are separate classes in Flower, and both need identical
checkpointing, identical undo of the uplink encoding and identical downlink
compression. That shared behaviour lives in a mixin rather than being copied,
and these tests pin that the mixin actually reaches both.

The property that protects every existing result: choosing FedProx must be the
only thing that changes anything. A FedAvg run after FedProx was added has to
train exactly as it did before, which is what the mu = 0 path is for.
"""

import os
import pickle
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flwr.common import ndarrays_to_parameters

from config_manager import config
from federated_ueba import strategy as strategy_module
from federated_ueba.strategy import (CHECKPOINT_KEY, FedAvgWithModelSaving,
                                     FedProxWithModelSaving, ModelSavingMixin,
                                     STRATEGIES, build_strategy)

MODEL = [np.zeros((2, 3), dtype=np.float32), np.ones(4, dtype=np.float32)]


def config_with(overrides):
    """The real configuration with a few keys replaced.

    A stub rather than `set_experiment`, so a test cannot leave the shared
    singleton pointing at another experiment for whatever runs next.
    """
    class Stub:
        def get(self, section, key):
            if (section, key) in overrides:
                return overrides[(section, key)]
            return config.get(section, key)
    return Stub()


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_configured_default_is_fedavg(self):
        """Every reported result is FedAvg; a silent switch would invalidate them."""
        built = build_strategy(config, save_path=self.tmp)
        self.assertIsInstance(built, FedAvgWithModelSaving)

    def test_fedprox_resolves_and_receives_its_mu(self):
        built = build_strategy(
            config_with({("federation", "strategy"): "fedprox"}),
            save_path=self.tmp)
        self.assertIsInstance(built, FedProxWithModelSaving)
        self.assertEqual(built.proximal_mu,
                         config.get("federation", "proximal_mu"))

    def test_an_unknown_strategy_names_the_known_ones(self):
        with self.assertRaises(ValueError) as caught:
            build_strategy(config_with({("federation", "strategy"): "fedat"}),
                           save_path=self.tmp)
        message = str(caught.exception)
        self.assertIn("fedavg", message)
        self.assertIn("fedprox", message)

    def test_fedavg_is_not_given_a_proximal_mu(self):
        """It does not accept one, and recording a mu it never applied would be
        worse than the TypeError."""
        self.assertFalse(hasattr(build_strategy(config, save_path=self.tmp),
                                 "proximal_mu"))


class TestMixinReachesBothStrategies(unittest.TestCase):
    def test_both_registered_strategies_carry_the_shared_behaviour(self):
        for name, cls in STRATEGIES.items():
            with self.subTest(strategy=name):
                self.assertTrue(issubclass(cls, ModelSavingMixin))

    def test_the_mixin_comes_first_so_its_hooks_win(self):
        """Behind the base rule in the MRO, its configure_fit would never run."""
        for name, cls in STRATEGIES.items():
            with self.subTest(strategy=name):
                mro = cls.__mro__
                self.assertLess(mro.index(ModelSavingMixin),
                                mro.index(strategy_module.flwr.server.strategy.FedAvg))

    def test_fedprox_still_aggregates_as_fedavg(self):
        """Its contribution is entirely on the client, so this must hold."""
        self.assertIs(
            FedProxWithModelSaving.aggregate_fit,
            FedAvgWithModelSaving.aggregate_fit)


class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_round_including_zero_is_saved(self):
        """Round 0 is the untrained model and the only before-training point."""
        built = build_strategy(config, save_path=self.tmp)
        for server_round in (0, 1):
            built.evaluate(server_round, ndarrays_to_parameters(MODEL))
            path = os.path.join(self.tmp, f"parameters_round_{server_round}.pkl")
            self.assertTrue(os.path.exists(path))

    def test_the_checkpoint_holds_the_uncompressed_model(self):
        """Quantizing the broadcast is a transport decision. What the federation
        holds after aggregation is the fp32 average, and that is what the
        evaluation pass rescores."""
        built = build_strategy(
            config_with({("efficiency", "downlink_plugins"): ["quantization"]}),
            save_path=self.tmp)
        built.evaluate(0, ndarrays_to_parameters(MODEL))

        with open(os.path.join(self.tmp, "parameters_round_0.pkl"), "rb") as f:
            saved = pickle.load(f)[CHECKPOINT_KEY]

        for original, stored in zip(MODEL, saved):
            self.assertEqual(stored.dtype, np.float32)
            self.assertTrue(np.array_equal(original, stored))


class StubClientManager:
    """One client, available immediately.

    Flower's own SimpleClientManager cannot be used here: `sample` waits for
    clients to register and its default timeout is a day, so a test that passes
    an empty manager hangs rather than failing. Nothing in `configure_fit` calls
    a method on the client objects, so a bare sentinel is enough to make it build
    the instructions this test wants to inspect.
    """

    CLIENT = object()

    def num_available(self):
        return 1

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return [self.CLIENT]


class TestDownlinkReachesTheBroadcast(unittest.TestCase):
    """configure_fit and configure_evaluate are the only paths to the clients."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.built = build_strategy(
            config_with({("efficiency", "downlink_plugins"): ["quantization"],
                         ("federation", "min_fit_clients"): 1,
                         ("federation", "min_available_clients"): 1}),
            save_path=self.tmp, min_fit_clients=1, min_available_clients=1,
            min_evaluate_clients=1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def instructions(self, hook):
        return hook(1, ndarrays_to_parameters(MODEL), StubClientManager())

    def test_what_reaches_the_client_is_quantized_on_both_paths(self):
        from flwr.common import parameters_to_ndarrays

        for hook in (self.built.configure_fit, self.built.configure_evaluate):
            with self.subTest(hook=hook.__name__):
                sent = self.instructions(hook)
                self.assertEqual(len(sent), 1)
                _, ins = sent[0]
                for array in parameters_to_ndarrays(ins.parameters):
                    self.assertEqual(array.dtype, np.float16)


class TestFedProxReachesTheClient(unittest.TestCase):
    """FedProx does its work through the fit config, so the mixin must not eat it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, name):
        return build_strategy(
            config_with({("federation", "strategy"): name}),
            save_path=self.tmp, on_fit_config_fn=lambda rnd: {"server_round": rnd},
            min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1)

    def fit_config(self, name):
        sent = self.build(name).configure_fit(
            1, ndarrays_to_parameters(MODEL), StubClientManager())
        return sent[0][1].config

    def test_fedprox_sends_its_mu_down_with_the_model(self):
        """The mixin overrides configure_fit; if it stopped delegating, FedProx
        would train exactly as FedAvg and the comparison would be empty."""
        self.assertEqual(self.fit_config("fedprox")["proximal_mu"],
                         config.get("federation", "proximal_mu"))

    def test_fedavg_sends_no_mu_at_all(self):
        """Its absence is what makes the client train without the penalty."""
        self.assertNotIn("proximal_mu", self.fit_config("fedavg"))

    def test_the_round_number_survives_alongside_it(self):
        """Per-round seeding reads this, so losing it would unseed the clients."""
        for name in ("fedavg", "fedprox"):
            with self.subTest(strategy=name):
                self.assertEqual(self.fit_config(name)["server_round"], 1)


if __name__ == "__main__":
    unittest.main()
