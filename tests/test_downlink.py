"""Compression of the server's broadcast, which is most of the traffic.

The baseline run that prompted this measured 6,441 MB of download against
2,003 MB of upload over 50 rounds. Compressing only the uplink therefore caps
the total saving at about a quarter, however aggressive the uplink gets, and the
download was additionally being charged at raw dense fp32 while the upload was
charged through an entropy coder.

What is pinned here: the entropy coder is lossless so it may never change a
weight, quantization halves the payload and is charged at the precision actually
sent, and the codec refuses anything it cannot honestly do to a broadcast.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config_manager import config
from federated_ueba.efficiency_plugins import DownlinkCodec, get_downlink_codec

# Shapes are irrelevant to the codec, but a two-tensor model catches a codec that
# only ever handles the first array.
MODEL = [np.random.RandomState(0).randn(64, 50).astype(np.float32),
         np.random.RandomState(1).randn(128).astype(np.float32)]
DENSE_BYTES = sum(p.nbytes for p in MODEL)


def round_trip(codec, model=MODEL):
    """What the client ends up holding, and what the transfer was charged."""
    sent = codec.compress(model)
    return codec.decompress(sent), codec.measure(sent)


class TestLosslessDownlink(unittest.TestCase):
    """Entropy coding only. The client must receive exactly what was sent."""

    def setUp(self):
        self.codec = DownlinkCodec(quantize=False, entropy="zlib")

    def test_the_client_receives_the_identical_model(self):
        received, _ = round_trip(self.codec)
        for original, arrived in zip(MODEL, received):
            self.assertTrue(np.array_equal(original, arrived))

    def test_it_is_charged_less_than_the_raw_arrays(self):
        _, measured = round_trip(self.codec)
        self.assertLess(measured, DENSE_BYTES)

    def test_no_entropy_coder_costs_exactly_the_raw_bytes(self):
        """The floor. Anything below this without quantization would be a bug."""
        _, measured = round_trip(DownlinkCodec(quantize=False, entropy="none"))
        self.assertEqual(measured, DENSE_BYTES)


class TestQuantizedDownlink(unittest.TestCase):
    def setUp(self):
        self.codec = DownlinkCodec(quantize=True, entropy="none")

    def test_it_costs_half(self):
        """fp16 against fp32, and charged on what was sent rather than on what
        the client expands it back to."""
        _, measured = round_trip(self.codec)
        self.assertEqual(measured, DENSE_BYTES // 2)

    def test_the_client_trains_in_fp32_from_rounded_values(self):
        received, _ = round_trip(self.codec)
        for original, arrived in zip(MODEL, received):
            # Back to the precision the model trains in, so nothing downstream
            # has to know the transport quantized.
            self.assertEqual(arrived.dtype, np.float32)
            # But not the same numbers. This is the lossy step, and a test that
            # asserted equality here would be asserting the compression did
            # nothing.
            self.assertFalse(np.array_equal(original, arrived))
            self.assertTrue(np.allclose(original, arrived, atol=1e-2))

    def test_entropy_coding_stacks_on_top_of_quantization(self):
        _, quantized_only = round_trip(DownlinkCodec(True, "none"))
        _, both = round_trip(DownlinkCodec(True, "zlib"))
        self.assertLess(both, quantized_only)


class TestDownlinkConfiguration(unittest.TestCase):
    def test_the_configured_default_is_lossless(self):
        """The reported baseline must not silently be a quantized run."""
        codec = get_downlink_codec(config)
        self.assertFalse(codec.quantize)

    def test_an_unknown_downlink_plugin_is_refused(self):
        class Fake:
            def get(self, section, key):
                if key == "downlink_plugins":
                    return ["top_k"]
                return config.get(section, key)

        # Sparsification is the plausible mistake, because it is offered on the
        # uplink. The message has to say why rather than just listing valid names.
        with self.assertRaises(ValueError) as caught:
            get_downlink_codec(Fake())
        self.assertIn("broadcast", str(caught.exception))

    def test_every_declared_downlink_experiment_builds(self):
        """Catches an experiment that names the setting but never reaches the codec."""
        declared = []
        try:
            for name in config.experiment_names:
                config.set_experiment(name)
                if config.get("efficiency", "downlink_plugins"):
                    declared.append(name)
                    self.assertTrue(get_downlink_codec(config).quantize, name)
        finally:
            config.set_experiment("baseline")

        self.assertTrue(declared, "no downlink experiment is configured")


if __name__ == "__main__":
    unittest.main()
