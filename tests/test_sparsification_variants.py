"""The two sparsification variants must stay distinguishable.

Found on 2026-08-03, after the first two seeds of `top-k-0.1` and `top-k-0.05`
disagreed with each other by more than the baseline's entire across-seed spread.
The cause was not noise. `"top_k"` maps to `WeightSparsificationPlugin`, which
zeroes the weights themselves; FedAvg then averages those zeros in, so the global
model is eroded rather than the update being compressed. Measured on the real
checkpoints of round 50:

    baseline      norm 16.99, 0.0% of weights exactly zero
    top-k-0.1     norm 13.13, 59.1% exactly zero
    top-k-0.05    norm 12.87, 75.4% exactly zero

That is a legitimate configuration to report, and the thesis reports it as a
documented negative result. What is not legitimate is confusing it with delta
sparsification, which compresses the update and leaves the global model dense.
These tests pin the distinction so neither can silently turn into the other.
"""

import unittest

import numpy as np

from federated_ueba.efficiency_plugins import (
    DeltaSparsificationPlugin,
    WeightSparsificationPlugin,
    _MODES,
    get_plugin,
)


def fake_global_model(seed=0):
    """A small stand-in for the real weights: several tensors, mixed scales."""
    rng = np.random.RandomState(seed)
    return [rng.randn(40, 10).astype(np.float32),
            rng.randn(40).astype(np.float32),
            rng.randn(12, 12).astype(np.float32)]


def federated_average(client_uploads):
    """Plain unweighted FedAvg, which is what the strategy does per coordinate."""
    return [np.mean(layer, axis=0) for layer in zip(*client_uploads)]


def zero_fraction(arrays):
    flat = np.concatenate([a.ravel() for a in arrays])
    return float((flat == 0.0).mean())


class TopKNameMapsToWeightSparsification(unittest.TestCase):
    """The experiment named `top-k-*` is the weight variant, not the delta one."""

    def test_top_k_alias_builds_the_weight_plugin(self):
        self.assertEqual(_MODES["top_k"], "weight")

    def test_delta_aliases_are_separate_modes(self):
        self.assertEqual(_MODES["delta_sparsification"], "delta")
        self.assertEqual(_MODES["delta_sparsification_ef"], "delta_ef")

    def test_the_three_modes_do_not_share_a_class(self):
        kinds = {_MODES[name] for name in
                 ("top_k", "delta_sparsification", "delta_sparsification_ef")}
        self.assertEqual(len(kinds), 3)


class WeightSparsificationErodesTheGlobalModel(unittest.TestCase):
    """The behaviour that made the top-k seeds disagree, pinned as a fact."""

    def setUp(self):
        self.reference = fake_global_model()
        self.clients = [fake_global_model(seed) for seed in range(1, 11)]

    def test_averaging_sparse_weights_leaves_the_global_model_sparse(self):
        plugin = WeightSparsificationPlugin(ratio=0.1)
        uploads = [plugin.apply_on_client(weights) for weights in self.clients]
        aggregate = plugin.apply_on_server(federated_average(uploads))

        # Ten clients each keeping a different 10% cannot cover the model, so a
        # large share of coordinates is zero in every upload and stays zero in
        # the average. On the real 450k-parameter model this settles near 60%.
        self.assertGreater(zero_fraction(aggregate), 0.3)

    def test_averaging_shrinks_the_surviving_weights(self):
        plugin = WeightSparsificationPlugin(ratio=0.1)
        uploads = [plugin.apply_on_client(weights) for weights in self.clients]
        eroded = federated_average(uploads)
        honest = federated_average(self.clients)

        eroded_norm = np.linalg.norm(np.concatenate([a.ravel() for a in eroded]))
        honest_norm = np.linalg.norm(np.concatenate([a.ravel() for a in honest]))
        # A coordinate kept by only some clients is averaged against the zeros
        # the others sent, so it arrives at a fraction of its true average.
        self.assertLess(eroded_norm, honest_norm)


class DeltaSparsificationKeepsTheGlobalModelDense(unittest.TestCase):
    """The contrast: the same ratio, without the erosion."""

    def setUp(self):
        self.reference = fake_global_model()
        self.clients = [fake_global_model(seed) for seed in range(1, 11)]

    def test_the_aggregate_has_no_zeroed_coordinates(self):
        uploads = []
        for weights in self.clients:
            plugin = DeltaSparsificationPlugin(ratio=0.1)
            uploads.append(plugin.apply_on_client(weights, reference=self.reference))
        aggregate = federated_average(uploads)

        self.assertEqual(zero_fraction(aggregate), 0.0)

    def test_an_untouched_coordinate_keeps_the_reference_value(self):
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        upload = plugin.apply_on_client(self.clients[0], reference=self.reference)

        delta = plugin.transport_payload(upload)
        untouched = delta[0] == 0.0
        np.testing.assert_allclose(upload[0][untouched],
                                   self.reference[0][untouched])

    def test_only_the_sparse_delta_is_charged_for_transport(self):
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        upload = plugin.apply_on_client(self.clients[0], reference=self.reference)

        payload = plugin.transport_payload(upload)
        self.assertGreater(zero_fraction(payload), 0.8)
        self.assertEqual(zero_fraction(upload), 0.0)


class MissingReferenceIsLoud(unittest.TestCase):
    """It used to fall back to weight sparsification, which is another experiment."""

    def test_no_reference_raises_rather_than_switching_variant(self):
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        with self.assertRaises(RuntimeError) as caught:
            plugin.apply_on_client(fake_global_model())
        self.assertIn("reference", str(caught.exception))

    def test_the_error_names_both_variants_so_the_log_is_actionable(self):
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        with self.assertRaises(RuntimeError) as caught:
            plugin.apply_on_client(fake_global_model())
        self.assertIn("different experiments", str(caught.exception))


class ConfiguredExperimentsBuildTheVariantTheyName(unittest.TestCase):
    """A config saying `delta_sparsification` must not produce the weight plugin."""

    class FakeConfig:
        def __init__(self, modes):
            self.modes = modes

        def get(self, section, key):
            values = {
                ("efficiency", "active_plugins"): self.modes,
                ("efficiency", "sparsification_ratio"): 0.1,
                ("efficiency", "payload_codec"): "bitmask",
                ("efficiency", "entropy_coder"): "zlib",
                ("data", "scaler_dir"): "scaler_data/test",
            }
            return values[(section, key)]

    def test_top_k_config_builds_weight_sparsification(self):
        manager = get_plugin(self.FakeConfig(["top_k"]))
        self.assertIsInstance(manager.plugins[0], WeightSparsificationPlugin)

    def test_delta_config_builds_delta_sparsification(self):
        manager = get_plugin(self.FakeConfig(["delta_sparsification"]))
        self.assertIsInstance(manager.plugins[0], DeltaSparsificationPlugin)
        self.assertFalse(manager.plugins[0].error_feedback)

    def test_delta_ef_config_turns_error_feedback_on(self):
        manager = get_plugin(self.FakeConfig(["delta_sparsification_ef"]))
        self.assertIsInstance(manager.plugins[0], DeltaSparsificationPlugin)
        self.assertTrue(manager.plugins[0].error_feedback)


if __name__ == "__main__":
    unittest.main()
