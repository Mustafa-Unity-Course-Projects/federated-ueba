"""Payload encodings: exact round-trips and the byte counts they produce.

The whole argument for these codecs is that they cost nothing in accuracy. That
is only true if decode(encode(x)) reproduces x bit for bit, so that is asserted
first and for every combination, including the degenerate ones.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_ueba import payload_codecs as pc  # noqa: E402

CODECS = list(pc.CODECS)
ENTROPY = list(pc.ENTROPY_CODERS)


def sparse_tensor(shape, ratio, dtype=np.float32, seed=0):
    """A tensor with the largest `ratio` of its entries kept, rest zeroed."""
    rng = np.random.RandomState(seed)
    flat = rng.randn(int(np.prod(shape))).astype(dtype)
    k = max(1, int(len(flat) * ratio))
    keep = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.zeros_like(flat)
    out[keep] = flat[keep]
    return out.reshape(shape)


class TestRoundTrip(unittest.TestCase):
    """Lossless means exactly lossless."""

    def test_every_codec_and_entropy_combination(self):
        for dtype in (np.float32, np.float16):
            tensor = sparse_tensor((64, 32), 0.1, dtype=dtype)
            for codec in CODECS:
                for entropy in ENTROPY:
                    with self.subTest(codec=codec, entropy=entropy, dtype=dtype):
                        blob, count = pc.encode_tensor(tensor, codec, entropy)
                        restored = pc.decode_tensor(blob, tensor.shape, dtype,
                                                    count, codec, entropy)
                        np.testing.assert_array_equal(restored, tensor)

    def test_survives_a_dense_tensor(self):
        """Nothing zeroed: the encoding must still be exact."""
        rng = np.random.RandomState(1)
        tensor = rng.randn(100).astype(np.float32)
        for codec in CODECS:
            with self.subTest(codec=codec):
                blob, count = pc.encode_tensor(tensor, codec)
                restored = pc.decode_tensor(blob, tensor.shape, np.float32,
                                            count, codec)
                np.testing.assert_array_equal(restored, tensor)

    def test_survives_a_single_surviving_weight(self):
        tensor = np.zeros(1000, dtype=np.float32)
        tensor[737] = -3.5
        for codec in CODECS:
            with self.subTest(codec=codec):
                blob, count = pc.encode_tensor(tensor, codec)
                restored = pc.decode_tensor(blob, tensor.shape, np.float32,
                                            count, codec)
                np.testing.assert_array_equal(restored, tensor)

    def test_survives_first_and_last_positions(self):
        """Delta coding starts from a sentinel, so position 0 is a special case."""
        tensor = np.zeros(500, dtype=np.float32)
        tensor[0] = 1.25
        tensor[499] = -2.5
        for codec in CODECS:
            with self.subTest(codec=codec):
                blob, count = pc.encode_tensor(tensor, codec)
                restored = pc.decode_tensor(blob, tensor.shape, np.float32,
                                            count, codec)
                np.testing.assert_array_equal(restored, tensor)


class TestVarint(unittest.TestCase):
    def test_encodes_and_decodes_gap_values(self):
        gaps = np.array([0, 1, 127, 128, 300, 16383, 16384, 1_000_000])
        blob = pc._varint_encode(gaps)
        decoded, _ = pc._varint_decode(blob, len(gaps))
        np.testing.assert_array_equal(decoded, gaps)

    def test_small_gaps_cost_one_byte(self):
        self.assertEqual(len(pc._varint_encode([0] * 50)), 50)
        self.assertEqual(len(pc._varint_encode([127] * 50)), 50)
        self.assertEqual(len(pc._varint_encode([128] * 50)), 100)


class TestPayloadSizes(unittest.TestCase):
    """The byte counts the thesis will quote."""

    N = 450_258  # the model's parameter count

    def _payload(self, ratio, dtype, codec):
        tensor = sparse_tensor((self.N,), ratio, dtype=dtype)
        return pc.measure_payload([tensor], codec)

    def test_bitmask_beats_indices_at_both_working_ratios(self):
        for ratio in (0.05, 0.10):
            for dtype in (np.float32, np.float16):
                with self.subTest(ratio=ratio, dtype=dtype):
                    idx = self._payload(ratio, dtype, "index")
                    mask = self._payload(ratio, dtype, "bitmask")
                    self.assertLess(mask, idx)

    def test_index_overhead_is_the_larger_half_under_fp16(self):
        """The motivation for the whole module: indices dominate the payload."""
        tensor = sparse_tensor((self.N,), 0.10, dtype=np.float16)
        described = pc.describe_payload([tensor], "index")
        self.assertGreater(described["overhead_share"], 0.6)

    def test_bitmask_overhead_is_independent_of_density(self):
        """One bit per parameter, whatever survives."""
        sizes = []
        for ratio in (0.05, 0.10, 0.20):
            tensor = sparse_tensor((self.N,), ratio, dtype=np.float32)
            described = pc.describe_payload([tensor], "bitmask")
            sizes.append(described["overhead_bytes"])
        self.assertEqual(len(set(sizes)), 1)
        self.assertEqual(sizes[0], (self.N + 7) // 8)

    def test_crossover_density_is_one_in_thirtytwo(self):
        """Below 1/32 the index scheme is the cheaper one; above it, the mask."""
        below = 0.02
        above = 0.05
        self.assertLess(self._payload(below, np.float32, "index"),
                        self._payload(below, np.float32, "bitmask"))
        self.assertLess(self._payload(above, np.float32, "bitmask"),
                        self._payload(above, np.float32, "index"))

    def test_entropy_coding_never_inflates_the_reported_size(self):
        """Whatever the coder does internally, we must not report a worse number."""
        tensor = sparse_tensor((self.N,), 0.10, dtype=np.float32)
        raw = pc.measure_payload([tensor], "bitmask", "none")
        for entropy in ("zlib", "lzma"):
            with self.subTest(entropy=entropy):
                # Random weights are close to incompressible, so allow parity but
                # flag anything that balloons.
                self.assertLess(pc.measure_payload([tensor], "bitmask", entropy),
                                raw * 1.05)


class TestDensePayload(unittest.TestCase):
    """The uncompressed baseline every reported saving is measured against."""

    def test_without_entropy_coding_it_is_the_raw_byte_count(self):
        arrays = [np.zeros((10, 10), dtype=np.float32),
                  np.zeros(7, dtype=np.float16)]
        self.assertEqual(pc.measure_dense_payload(arrays),
                         sum(a.nbytes for a in arrays))

    def test_entropy_coding_shrinks_a_compressible_payload(self):
        arrays = [np.zeros(10_000, dtype=np.float32)]
        self.assertLess(pc.measure_dense_payload(arrays, "zlib"),
                        pc.measure_dense_payload(arrays))

    def test_unknown_entropy_coder_raises(self):
        with self.assertRaises(ValueError):
            pc.measure_dense_payload([np.zeros(4, dtype=np.float32)], "brotli")


class TestErrors(unittest.TestCase):
    def test_unknown_codec_raises(self):
        with self.assertRaises(ValueError):
            pc.encode_tensor(np.zeros(4, dtype=np.float32), codec="huffman")

    def test_unknown_entropy_coder_raises(self):
        with self.assertRaises(ValueError):
            pc.encode_tensor(np.zeros(4, dtype=np.float32), entropy="brotli")


if __name__ == "__main__":
    unittest.main()
