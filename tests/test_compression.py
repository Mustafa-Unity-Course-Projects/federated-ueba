"""Communication plugins: payload accounting and the delta/error-feedback maths.

Every assertion here corresponds to something that was either wrong or unverified
before. The payload numbers in particular are the thesis's headline result, so
they are pinned rather than trusted.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_ueba.efficiency_plugins import (  # noqa: E402
    DeltaSparsificationPlugin, PluginManager, QuantizationPlugin,
    WeightSparsificationPlugin, get_plugin)


def make_params(seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(100, 50).astype(np.float32),
            rng.randn(200).astype(np.float32)]


class TestWeightSparsification(unittest.TestCase):
    def test_keeps_expected_fraction_per_tensor(self):
        params = make_params()
        plugin = WeightSparsificationPlugin(ratio=0.1)
        out = plugin.apply_on_client(params)

        # Applied per tensor, not over a global ranking.
        self.assertEqual(np.count_nonzero(out[0]), 500)   # 5000 * 0.1
        self.assertEqual(np.count_nonzero(out[1]), 20)    # 200 * 0.1

    def test_keeps_the_largest_magnitudes(self):
        params = [np.array([0.1, -5.0, 0.2, 3.0], dtype=np.float32)]
        out = WeightSparsificationPlugin(ratio=0.5).apply_on_client(params)
        np.testing.assert_allclose(out[0], [0.0, -5.0, 0.0, 3.0])

    def test_small_tensor_keeps_at_least_one(self):
        params = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        out = WeightSparsificationPlugin(ratio=0.01).apply_on_client(params)
        self.assertEqual(np.count_nonzero(out[0]), 1)

    def test_payload_counts_index_plus_value(self):
        params = make_params()
        manager = PluginManager([WeightSparsificationPlugin(ratio=0.1)],
                                codec="index")
        out = manager.apply_on_client(params)
        nnz = sum(np.count_nonzero(p) for p in out)
        self.assertEqual(manager.measure_transport_size(out), nnz * (4 + 4))

    def test_bitmask_is_cheaper_than_indices_at_this_density(self):
        """The reason the default codec is not the four-byte index."""
        params = make_params()
        plugin = WeightSparsificationPlugin(ratio=0.1)
        out = plugin.apply_on_client(params)

        indexed = PluginManager([plugin], codec="index")
        masked = PluginManager([plugin], codec="bitmask")
        self.assertLess(masked.measure_transport_size(out),
                        indexed.measure_transport_size(out))

    def test_output_does_not_depend_on_reference(self):
        """It sparsifies the weights, so the round's starting point is irrelevant."""
        params = make_params()
        plugin = WeightSparsificationPlugin(ratio=0.1)
        a = plugin.apply_on_client(params, reference=make_params(seed=1))
        b = plugin.apply_on_client(params, reference=None)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


class TestDeltaSparsification(unittest.TestCase):
    def setUp(self):
        self.reference = make_params(seed=0)
        rng = np.random.RandomState(7)
        self.local = [r + rng.randn(*r.shape).astype(np.float32) * 0.01
                      for r in self.reference]

    def test_uploads_reference_plus_sparse_delta(self):
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        out = plugin.apply_on_client(self.local, reference=self.reference)

        for o, r in zip(out, self.reference):
            delta = o - r
            # Only the kept coordinates moved; everything else is unchanged.
            self.assertEqual(np.count_nonzero(delta), int(r.size * 0.1))
            np.testing.assert_allclose(o, r + delta, rtol=1e-6)

    def test_aggregation_still_receives_a_full_model(self):
        """FedAvg must keep working, so the upload has to be dense-shaped."""
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        out = plugin.apply_on_client(self.local, reference=self.reference)
        for o, r in zip(out, self.reference):
            self.assertEqual(o.shape, r.shape)
            self.assertGreater(np.count_nonzero(o), int(r.size * 0.5))

    def test_payload_counts_only_the_delta(self):
        """The upload is a dense model, but only the delta is charged for."""
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        manager = PluginManager([plugin], codec="index")
        out = manager.apply_on_client(self.local, reference=self.reference)
        expected_nnz = sum(int(r.size * 0.1) for r in self.reference)
        self.assertEqual(manager.measure_transport_size(out),
                         expected_nnz * (4 + 4))

    def test_wire_payload_is_the_sparse_delta_not_the_model(self):
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        manager = PluginManager([plugin])
        out = manager.apply_on_client(self.local, reference=self.reference)

        payload = manager.wire_payload(out)
        for sent, r in zip(payload, self.reference):
            self.assertEqual(np.count_nonzero(sent), int(r.size * 0.1))

    def test_raises_without_reference_instead_of_changing_variant(self):
        """A missing reference must stop the run, not quietly become top-k.

        This test used to assert the opposite, on the reasoning that a missing
        reference should "degrade, not crash mid-sweep". That reasoning was
        wrong, and it was wrong in a way that only became visible once both
        variants had been run: they do not differ in speed or in payload size,
        they differ in what the global model ends up being. Weight
        sparsification leaves 59% of the real model at exactly zero at ratio
        0.1; delta sparsification leaves none. Degrading from one to the other
        produces a run that is neither experiment, reports a plausible number,
        and cannot be told apart from noise afterwards.

        Crashing is cheap here: a pair can be re-run in half an hour, and the
        runner skips everything already finished. A silently mislabelled result
        is not cheap, because it is only caught by someone thinking to check.
        """
        plugin = DeltaSparsificationPlugin(ratio=0.1)
        with self.assertRaises(RuntimeError):
            plugin.apply_on_client(self.local, reference=None)


class TestErrorFeedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _plugin(self, ratio):
        return DeltaSparsificationPlugin(ratio=ratio, error_feedback=True,
                                         residual_dir=self.tmp, partition_id=0)

    def test_suppressed_update_is_eventually_sent(self):
        """A small but persistent update must not be silenced forever."""
        base = [np.zeros(4, dtype=np.float32)]
        update = np.array([1.0, 0.30, 0.0, 0.0], dtype=np.float32)
        plugin = self._plugin(ratio=0.25)          # one coordinate per round

        sent_small = []
        for round_idx in range(8):
            out = plugin.apply_on_client([base[0] + update], reference=base)
            if (out[0] - base[0])[1] != 0:
                sent_small.append(round_idx)

        self.assertTrue(sent_small,
                        "the suppressed coordinate was never transmitted")

    def test_nothing_is_lost_only_delayed(self):
        """Sum of what was sent, plus what is still held back, equals the truth.

        This is the invariant that separates error feedback from plain
        sparsification: without it the discarded part is gone for good.
        """
        base = [np.zeros(6, dtype=np.float32)]
        update = np.array([1.0, 0.4, 0.25, 0.1, 0.05, 0.02], dtype=np.float32)
        plugin = self._plugin(ratio=0.34)          # two coordinates per round

        rounds = 30
        total_sent = np.zeros(6, dtype=np.float64)
        for _ in range(rounds):
            out = plugin.apply_on_client([base[0] + update], reference=base)
            total_sent += (out[0] - base[0]).astype(np.float64)

        residual = np.load(os.path.join(self.tmp, "residual_client_0.npz"))["arr_0"]
        np.testing.assert_allclose(total_sent + residual, update * rounds,
                                   rtol=1e-4, atol=1e-4)

    def test_residual_survives_client_reconstruction(self):
        """Flower rebuilds clients every round, so the residual must be on disk."""
        base = [np.zeros(4, dtype=np.float32)]
        update = np.array([1.0, 0.30, 0.0, 0.0], dtype=np.float32)

        # A brand new plugin object each round, as happens in the simulation.
        for _ in range(3):
            self._plugin(ratio=0.25).apply_on_client([base[0] + update], reference=base)

        residual = np.load(os.path.join(self.tmp, "residual_client_0.npz"))["arr_0"]
        self.assertAlmostEqual(float(residual[1]), 0.90, places=4)

    def test_disabled_by_default(self):
        plugin = DeltaSparsificationPlugin(ratio=0.25)
        self.assertFalse(plugin.error_feedback)
        self.assertIsNone(plugin._residual_path())


class TestQuantization(unittest.TestCase):
    def test_halves_the_dense_payload(self):
        params = make_params()
        manager = PluginManager([QuantizationPlugin()])
        out = manager.apply_on_client(params)
        self.assertEqual(out[0].dtype, np.float16)
        dense = sum(p.nbytes for p in params)
        self.assertEqual(manager.measure_transport_size(out), dense // 2)

    def test_server_restores_float32(self):
        plugin = QuantizationPlugin()
        restored = plugin.apply_on_server(plugin.apply_on_client(make_params()))
        self.assertEqual(restored[0].dtype, np.float32)


class TestPluginChain(unittest.TestCase):
    def setUp(self):
        self.reference = make_params(seed=0)
        rng = np.random.RandomState(3)
        self.local = [r + rng.randn(*r.shape).astype(np.float32) * 0.01
                      for r in self.reference]

    def test_delta_plus_fp16_charges_four_plus_two_bytes(self):
        """Quantization runs after the delta, so the delta is charged at fp16."""
        manager = PluginManager([DeltaSparsificationPlugin(ratio=0.1),
                                 QuantizationPlugin()], codec="index")
        out = manager.apply_on_client(self.local, reference=self.reference)

        self.assertEqual(out[0].dtype, np.float16)
        expected_nnz = sum(int(r.size * 0.1) for r in self.reference)
        self.assertEqual(manager.measure_transport_size(out),
                         expected_nnz * (4 + 2))

    def test_codec_and_entropy_do_not_touch_the_weights(self):
        """The lossless layer changes the reported size and nothing else."""
        plain = PluginManager([DeltaSparsificationPlugin(ratio=0.1)])
        packed = PluginManager([DeltaSparsificationPlugin(ratio=0.1)],
                               codec="delta_varint", entropy="zlib")

        a = plain.apply_on_client(self.local, reference=self.reference)
        b = packed.apply_on_client(self.local, reference=self.reference)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

        self.assertLess(packed.measure_transport_size(b),
                        plain.measure_transport_size(a))


class TestBaselineFairness(unittest.TestCase):
    """The dense baseline must get the same entropy coder as everything else.

    Otherwise the reported saving would partly be zlib being applied to the
    compressed configurations only, which would inflate the headline number.
    """

    def test_entropy_coder_also_applies_to_the_dense_baseline(self):
        params = make_params()
        raw = PluginManager([])
        compressed = PluginManager([], entropy="zlib")
        self.assertLess(compressed.measure_transport_size(params),
                        raw.measure_transport_size(params))

    def test_dense_baseline_without_entropy_is_the_raw_byte_count(self):
        params = make_params()
        manager = PluginManager([])
        self.assertEqual(manager.measure_transport_size(params),
                         sum(p.nbytes for p in params))

    def test_empty_chain_is_the_dense_model(self):
        params = make_params()
        manager = PluginManager([])
        out = manager.apply_on_client(params)
        self.assertEqual(manager.measure_transport_size(out),
                         sum(p.nbytes for p in params))


class TestPluginSelection(unittest.TestCase):
    class FakeConfig:
        def __init__(self, modes, ratio=0.1, key="sparsification_ratio"):
            # Every key `get_plugin` reads. It has to be complete, because there
            # are no defaults any more: an omission raises rather than being
            # quietly filled in.
            self.data = {"efficiency": {"active_plugins": modes, key: ratio,
                                        "payload_codec": "bitmask",
                                        "entropy_coder": "none"},
                         "data": {"scaler_dir": "unused"}}

        def get(self, *keys):
            # Raises on a missing key, exactly as the real config does. A fake
            # that returned None instead would let a test pass against behaviour
            # production no longer has.
            node = self.data
            for key in keys:
                node = node[key]
            return node

    def test_legacy_top_k_mode_name_still_resolves(self):
        """Old experiment files name the mode `top_k`; that alias stays."""
        plugins = get_plugin(self.FakeConfig(["top_k"], ratio=0.05)).plugins
        self.assertIsInstance(plugins[0], WeightSparsificationPlugin)
        self.assertEqual(plugins[0].ratio, 0.05)

    def test_the_renamed_ratio_key_is_rejected_by_the_schema(self):
        """`top_k_ratio` became `sparsification_ratio` because "top-k" also names
        the scoring stage's feature count. A configuration left on the old name
        would read as configured and never be applied, so it is caught at startup
        rather than by a special case in the plugin code."""
        from config_manager import ConfigError, config

        config._active_config["efficiency"]["top_k_ratio"] = 0.05
        try:
            with self.assertRaises(ConfigError) as caught:
                config.validate()
            self.assertIn("top_k_ratio", str(caught.exception))
        finally:
            del config._active_config["efficiency"]["top_k_ratio"]

    def test_codec_and_entropy_come_from_the_config(self):
        cfg = self.FakeConfig(["top_k"])
        cfg.data["efficiency"]["payload_codec"] = "delta_varint"
        cfg.data["efficiency"]["entropy_coder"] = "lzma"
        manager = get_plugin(cfg)
        self.assertEqual(manager.codec, "delta_varint")
        self.assertEqual(manager.entropy, "lzma")

    def test_a_silent_config_raises_rather_than_choosing_for_you(self):
        """There are no defaults. A setting the file does not define stops the
        run, because a silent fallback would let it measure something other than
        what its own recorded configuration says it measured."""
        from config_manager import ConfigError

        cfg = self.FakeConfig(["top_k"])
        del cfg.data["efficiency"]["payload_codec"]
        with self.assertRaises((ConfigError, KeyError)):
            get_plugin(cfg)

    def test_delta_variants_resolve(self):
        plain = get_plugin(self.FakeConfig(["delta_sparsification"])).plugins[0]
        self.assertIsInstance(plain, DeltaSparsificationPlugin)
        self.assertFalse(plain.error_feedback)

        with_ef = get_plugin(self.FakeConfig(["delta_sparsification_ef"])).plugins[0]
        self.assertTrue(with_ef.error_feedback)

    def test_unknown_mode_raises_instead_of_being_ignored(self):
        """A typo used to silently produce an uncompressed run."""
        with self.assertRaises(ValueError):
            get_plugin(self.FakeConfig(["topk"]))


if __name__ == "__main__":
    unittest.main()
