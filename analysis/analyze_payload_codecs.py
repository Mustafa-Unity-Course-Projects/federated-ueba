"""Measure what the sparsified payload costs under different lossless encodings.

Sparsification and quantization trade accuracy for bytes. The encoding of the
resulting payload does not: it is a pure representation choice, so a better
encoding is free. This script measures how much is being left on the table.

It runs offline, against saved checkpoints, and needs no retraining. That is not
a shortcut but a consequence of the encodings being lossless: the surviving
weights are identical whichever way they are written down, so detection
performance is unchanged by construction and only the byte count moves.

    python analyze_payload_codecs.py
    python analyze_payload_codecs.py --checkpoint model_pickle/baseline/parameters_round_20.pkl
"""

# This file lives in a subdirectory, so Python puts that subdirectory on
# sys.path rather than the project root and `import config_manager` fails.
# Adding the root explicitly keeps `python analysis/analyze_payload_codecs.py` working from the project
# root, which is how every path in the configuration is resolved anyway.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import glob
import re
import os
import pickle
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config_manager import config
from federated_ueba import payload_codecs as pc
from federated_ueba.efficiency_plugins import WeightSparsificationPlugin

RATIOS = (0.10, 0.05)
DTYPES = ((np.float32, "fp32"), (np.float16, "fp16"))
COMBINATIONS = [
    ("index", "none"),          # the original scheme: 4 bytes of position each
    ("bitmask", "none"),        # one bit per model parameter
    ("delta_varint", "none"),   # gaps between sorted positions
    ("bitmask", "zlib"),
    ("delta_varint", "zlib"),
    ("bitmask", "lzma"),
]


def find_checkpoint():
    """A *dense* saved global model *of the configured architecture*.

    Two things have to hold and only one of them is about density.

    The starting point has to be dense. A checkpoint from a sparsifying run
    already has most of its weights at zero, so asking for the top 10% of it
    yields far fewer than 10% non-zero values and every payload measured from it
    silently collapses onto the sparser configuration's numbers. Runs whose name
    implies sparsification are therefore skipped, and whatever is chosen is
    density-checked by the caller.

    The starting point also has to be the *proposed* model. This function used to
    take the middle element of a sorted glob, which was fine while `model_pickle`
    held a handful of runs and stopped being fine as the sweep grew: the middle
    drifted onto `no-bottleneck`, an ablation carrying 429,362 parameters in 26
    tensors instead of the proposed 450,258 in 36. Nothing failed. The payload
    table simply described a different model, and the only trace was the
    parameter count in the printed header. The preference order below is
    explicit, and `check_architecture` turns the remaining risk into an error.
    """
    sparse_markers = ("top-k", "delta-", "sparsif")
    ablation_markers = ("ablation", "no-bottleneck", "unidirectional",
                        "features-filtered", "nodes-")
    patterns = ("model_pickle/baseline__seed*/parameters_round_*.pkl",
                "model_pickle/baseline*/parameters_round_*.pkl",
                "model_pickle/*__seed*/parameters_round_*.pkl",
                "model_pickle/*/parameters_round_*.pkl")
    def round_number(path):
        match = re.search(r"parameters_round_(\d+)", path)
        return int(match.group(1)) if match else -1

    for pattern in patterns:
        found = [p for p in glob.glob(pattern)
                 if not any(m in p.replace("\\", "/").lower()
                            for m in sparse_markers + ablation_markers)]
        if found:
            # The last *round*, not the last string: sorted() puts round_9 after
            # round_50 and would hand back an early, under-trained checkpoint.
            return max(sorted(found), key=round_number)
    return None


def check_architecture(weights):
    """Refuse a checkpoint that is not the configured architecture.

    The payload sizes are a function of the parameter count and of how those
    parameters are split into tensors, because sparsification selects per tensor.
    Measuring the wrong model therefore produces a plausible table rather than an
    error, which is the failure this guard exists to prevent.
    """
    from federated_ueba.task import LSTMAutoencoder, Architecture

    arch = Architecture.from_config(config)
    n_features = len(config.get("data", "selected_features"))
    reference = LSTMAutoencoder(n_features, arch=arch)
    want_params = sum(p.numel() for p in reference.parameters()
                      if p.requires_grad)
    want_tensors = len([p for p in reference.parameters() if p.requires_grad])

    got_params = sum(a.size for a in weights)
    got_tensors = len(weights)
    if (got_params, got_tensors) != (want_params, want_tensors):
        raise SystemExit(
            f"Checkpoint carries {got_params:,} parameters in {got_tensors} "
            f"tensors; the configured architecture has {want_params:,} in "
            f"{want_tensors}. Payload sizes measured here would describe a "
            f"different model. Pass --checkpoint pointing at a run of the "
            f"configured architecture.")


def load_weights(path):
    """Load a checkpoint, refusing one that is already sparse.

    The guard is not paranoia. Measuring K = 0.1 on a checkpoint that was itself
    trained with K = 0.1 sparsifies an already sparse model, so the payload
    reported describes a far sparser configuration than the one asked for. That
    silently made two different ratios produce identical numbers.
    """
    with open(path, "rb") as f:
        weights = pickle.load(f)["global_parameters"]

    total = sum(a.size for a in weights)
    nonzero = sum(int(np.count_nonzero(a)) for a in weights)
    density = nonzero / total
    if density < 0.99:
        raise SystemExit(
            f"'{path}' is only {100 * density:.1f}% dense. Sparsification has "
            f"already been applied to it, so payload sizes measured here would "
            f"describe a sparser configuration than the one requested. Pass "
            f"--checkpoint pointing at a run without sparsification (baseline or "
            f"quantization-fp16).")
    return weights


def measure(weights):
    """Every ratio x precision x codec x entropy combination, as measured bytes.

    Measured by actually encoding rather than by formula, so the figures are the
    bytes that would cross the wire.
    """
    dense_bytes = sum(a.nbytes for a in weights)
    rows = []

    for ratio in RATIOS:
        sparse = WeightSparsificationPlugin(ratio=ratio).apply_on_client(weights)
        for dtype, dtype_name in DTYPES:
            arrays = [a.astype(dtype) for a in sparse]
            for codec, entropy in COMBINATIONS:
                described = pc.describe_payload(arrays, codec, entropy)
                rows.append({
                    "K": ratio,
                    "Deger": dtype_name,
                    "Kodlama": codec if entropy == "none" else f"{codec}+{entropy}",
                    "MB": round(described["mb"], 4),
                    "Yogun_gore_tasarruf_%": round(
                        100 * (1 - described["bytes"] / dense_bytes), 1),
                    "Ek_yuk_payi_%": round(100 * described["overhead_share"], 1),
                })
    return pd.DataFrame(rows), dense_bytes


def report(df, dense_bytes, n_params):
    """Print the comparison, then name the cheapest codec for each configuration."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)

    print(f"Model: {n_params:,} parametre, yogun yuk "
          f"{dense_bytes / 1024 / 1024:.4f} MB")
    print(f"Bit maskesi sabit maliyeti: {(n_params + 7) // 8:,} B "
          f"({(n_params + 7) / 8 / 1024 / 1024:.4f} MB)")
    print(f"Kritik yogunluk: 1/32 = {1/32:.4f}. Bu oranin uzerinde 4 baytlik "
          f"indeks her zaman kaybeder.\n")

    print("=" * 92)
    print("TUM KODLAMALAR")
    print("=" * 92)
    print(df.to_string(index=False))

    print("\n" + "=" * 92)
    print("MEVCUT KODLAMAYA (index) GORE KAZANC")
    print("=" * 92)
    gains = []
    for (k, dtype), group in df.groupby(["K", "Deger"], sort=False):
        baseline = group[group["Kodlama"] == "index"]["MB"].iloc[0]
        best_row = group.loc[group["MB"].idxmin()]
        gains.append({
            "Konfigurasyon": f"K={k} {dtype}",
            "index_MB": baseline,
            "En_iyi_kodlama": best_row["Kodlama"],
            "En_iyi_MB": best_row["MB"],
            "Kazanc_%": round(100 * (1 - best_row["MB"] / baseline), 1),
            "Yogun_gore_%": best_row["Yogun_gore_tasarruf_%"],
        })
    gains_df = pd.DataFrame(gains)
    print(gains_df.to_string(index=False))

    print("\nBu kazanclarin tamami kayipsizdir: gonderilen agirliklar birebir "
          "ayni, yalnizca yazim bicimi degisiyor. Tespit basarimi degismez.")
    return gains_df


def main():
    """Compare the lossless codecs on a real checkpoint. Needs no retraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="saved global model; one is picked automatically")
    parser.add_argument("--out", default="payload_codec_comparison.csv")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_checkpoint()
    if not checkpoint or not os.path.exists(checkpoint):
        print("No checkpoint found. Train an experiment first.")
        return 1

    print(f"Checkpoint: {checkpoint}\n")
    weights = load_weights(checkpoint)
    check_architecture(weights)
    df, dense_bytes = measure(weights)
    gains_df = report(df, dense_bytes, sum(a.size for a in weights))

    df.to_csv(args.out, index=False)
    gains_df.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nSaved '{args.out}' and "
          f"'{args.out.replace('.csv', '_summary.csv')}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
