"""Lossless encodings for a sparsified parameter payload.

Sparsification and quantization are lossy: they trade accuracy for bytes. What
follows them is not. Once the client has decided *which* coordinates to send,
how those coordinates are written on the wire is a pure encoding question, and a
better encoding costs nothing in detection performance.

This matters here more than it usually would. At the ratios this work uses, the
indices are not a rounding error in the payload, they are most of it:

    K = 0.10, fp32   45,025 values     indices are 50% of the payload
    K = 0.10, fp16   45,025 values     indices are 67% of the payload

A four-byte index per surviving weight is the obvious encoding and the wrong one.
A bitmask spends one bit per *model* parameter instead of 32 bits per *surviving*
parameter, so it wins whenever the density exceeds 1/32 = 0.031. Both ratios used
here are above that threshold.

Every codec round-trips exactly. `test_payload_codecs.py` asserts it, because a
claim of losslessness is only worth as much as its verification.
"""

import lzma
import zlib

import numpy as np

from config_manager import config

# --- Index encoding --------------------------------------------------------


def _encode_index(flat, nonzero):
    """Four bytes of position per surviving value. The original scheme.

    Layout: all positions as uint32, then all values. Both blocks are in the same
    order, so position i in the first block belongs to value i in the second.
    """
    return nonzero.astype(np.uint32).tobytes() + flat[nonzero].tobytes()


def _decode_index(blob, size, dtype, count):
    """Inverse of `_encode_index`. `count` is how many values were kept."""
    # The position block is exactly count uint32s, that is 4 bytes each; the rest
    # of the blob is values.
    idx = np.frombuffer(blob[:count * 4], dtype=np.uint32)
    values = np.frombuffer(blob[count * 4:], dtype=dtype)

    # Everything not sent was zero, which is what made it droppable.
    out = np.zeros(size, dtype=dtype)
    out[idx] = values
    return out


# --- Bitmask encoding ------------------------------------------------------


def _encode_bitmask(flat, nonzero):
    """One bit per model parameter, then the surviving values in order.

    Costs ceil(N/8) bytes regardless of how many values survive, so it beats
    per-value indices at any density above 1/32.
    """
    mask = np.zeros(len(flat), dtype=bool)
    mask[nonzero] = True
    # packbits turns 8 booleans into 1 byte. Values follow in mask order, which
    # is ascending position, so the decoder can pair them up without an index.
    return np.packbits(mask).tobytes() + flat[mask].tobytes()


def _decode_bitmask(blob, size, dtype, count):
    """Inverse of `_encode_bitmask`."""
    # Ceiling division: 8 bits per byte, and a partial final byte still costs one.
    mask_bytes = (size + 7) // 8
    # `count=size` discards the padding bits packbits added to fill that last byte.
    mask = np.unpackbits(np.frombuffer(blob[:mask_bytes], dtype=np.uint8),
                         count=size).astype(bool)
    values = np.frombuffer(blob[mask_bytes:], dtype=dtype)
    out = np.zeros(size, dtype=dtype)
    out[mask] = values
    return out


# --- Delta + varint encoding -----------------------------------------------


def _varint_encode(gaps):
    """Write each gap as a variable number of bytes, small gaps costing one.

    Standard base-128: seven payload bits per byte, and the top bit says whether
    another byte follows. A gap under 128 fits in one byte, which is the whole
    point, since sparsified weights cluster and most gaps are small.
    """
    out = bytearray()
    for gap in gaps:
        gap = int(gap)
        while gap >= 0x80:                    # more than 7 bits left to write
            out.append((gap & 0x7F) | 0x80)   # low 7 bits, continuation bit set
            gap >>= 7
        out.append(gap)                       # final byte, continuation bit clear
    return bytes(out)


def _varint_decode(blob, count):
    """Read `count` varints. Returns the gaps and how many bytes they used.

    The byte count matters: the values follow immediately after, so a decoder
    that miscounts here slices the value buffer at the wrong offset.
    """
    gaps = np.empty(count, dtype=np.int64)
    written = shift = value = 0

    for pos, byte in enumerate(blob):
        # Accumulate the seven payload bits at their place in the number.
        value |= (byte & 0x7F) << shift
        if byte & 0x80:          # continuation bit: another byte belongs to this gap
            shift += 7
            continue

        gaps[written] = value
        written += 1
        shift = value = 0
        if written == count:
            # pos is the index of the last byte consumed, so pos + 1 is the
            # length. Returning pos would leave the values misaligned by a byte.
            return gaps, pos + 1

    # Reached only if the stream ran out mid-number, which means the payload was
    # truncated. Silently returning short would corrupt the weights instead.
    raise ValueError(f"varint stream ended after {written} of {count} values")


def _encode_delta_varint(flat, nonzero):
    """Gaps between sorted positions, each written as a variable-length integer.

    Dense runs produce small gaps and small gaps cost one byte, so this beats a
    fixed four-byte index whenever the surviving weights are not scattered
    uniformly.
    """
    ordered = np.sort(nonzero)
    # Prepending -1 makes the first gap measure from "before position 0", so the
    # first position is encoded the same way as every later one. The -1 then turns
    # a difference into a gap: adjacent positions differ by 1 and have gap 0.
    #   positions [0, 1, 5]  ->  differences [1, 1, 4]  ->  gaps [0, 0, 3]
    gaps = np.diff(np.concatenate(([-1], ordered))) - 1
    return _varint_encode(gaps) + flat[ordered].tobytes()


def _decode_delta_varint(blob, size, dtype, count):
    """Inverse of `_encode_delta_varint`."""
    gaps, consumed = _varint_decode(blob, count)
    # Undo the encoding: running total of (gap + 1), shifted back by the leading
    # -1 that the encoder prepended.
    idx = np.cumsum(gaps + 1) - 1
    # `consumed` is where the varints ended and the values begin. The gaps are
    # variable width, so this offset cannot be computed from `count` alone.
    values = np.frombuffer(blob[consumed:], dtype=dtype)
    out = np.zeros(size, dtype=dtype)
    out[idx] = values
    return out


CODECS = {
    "index": (_encode_index, _decode_index),
    "bitmask": (_encode_bitmask, _decode_bitmask),
    "delta_varint": (_encode_delta_varint, _decode_delta_varint),
}

# Effort settings for the two entropy coders. Set once at import from
# [tool.fueba.efficiency] rather than threaded through every call, because they
# change only the reported byte count and never a weight: both coders are
# lossless, so a different level compresses the same payload to a different size
# and decodes back to the same values.
_ZLIB_LEVEL = config.get("efficiency", "zlib_level")
_LZMA_PRESET = config.get("efficiency", "lzma_preset")

ENTROPY_CODERS = {
    "none": (lambda b: b, lambda b: b),
    "zlib": (lambda b: zlib.compress(b, _ZLIB_LEVEL), zlib.decompress),
    "lzma": (lambda b: lzma.compress(b, preset=_LZMA_PRESET), lzma.decompress),
}


def _codec(name):
    if name not in CODECS:
        raise ValueError(f"Unknown codec {name!r}. Expected one of {sorted(CODECS)}.")
    return CODECS[name]


def _entropy_coder(name):
    if name not in ENTROPY_CODERS:
        raise ValueError(f"Unknown entropy coder {name!r}. "
                         f"Expected one of {sorted(ENTROPY_CODERS)}.")
    return ENTROPY_CODERS[name]


def encode_tensor(array, codec="bitmask", entropy="none"):
    """Encode one sparsified tensor. Returns (blob, non-zero count)."""
    encode = _codec(codec)[0]
    compress = _entropy_coder(entropy)[0]

    flat = array.flatten()
    nonzero = np.flatnonzero(flat)
    return compress(encode(flat, nonzero)), len(nonzero)


def decode_tensor(blob, shape, dtype, count, codec="bitmask", entropy="none"):
    """Inverse of `encode_tensor`. Must reproduce the input exactly."""
    raw = _entropy_coder(entropy)[1](blob)
    size = int(np.prod(shape))
    flat = _codec(codec)[1](raw, size, np.dtype(dtype), count)
    return flat.reshape(shape)


def measure_payload(arrays, codec="bitmask", entropy="none"):
    """Bytes a sparse payload actually occupies under the given encoding.

    Measured by encoding, not estimated from a formula, so the figure reported in
    the thesis is the number of bytes that would cross the wire.
    """
    return sum(len(encode_tensor(a, codec, entropy)[0]) for a in arrays)


def measure_dense_payload(arrays, entropy="none"):
    """Bytes for a payload with no sparsity: the raw values, entropy-coded.

    The uncompressed baseline is measured with the same entropy coder as every
    sparse configuration. Otherwise part of the reported saving would only be
    zlib being applied to one side of the comparison.
    """
    compress = _entropy_coder(entropy)[0]
    return sum(len(compress(a.tobytes())) for a in arrays)


def describe_payload(arrays, codec="bitmask", entropy="none"):
    """Byte count plus the breakdown that explains where it went."""
    total = 0
    values = 0
    nonzeros = 0
    for array in arrays:
        blob, count = encode_tensor(array, codec, entropy)
        total += len(blob)
        nonzeros += count
        values += count * array.dtype.itemsize
    return {
        "codec": codec,
        "entropy": entropy,
        "bytes": total,
        "mb": total / 1024 / 1024,
        "non_zeros": nonzeros,
        "value_bytes": values,
        "overhead_bytes": total - values,
        "overhead_share": (total - values) / total if total else 0.0,
    }
