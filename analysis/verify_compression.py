"""Is the compression applied, or only counted?

A fair question, and one the jury will ask: the configurations differ by only a
few hundredths of PR-AUC, which is what it would look like if the plugins never
touched the weights and the saving were an arithmetic exercise on the side. This
checks both halves of that suspicion against the artefacts on disk.

Two things have to hold, and they are different claims.

**The lossy steps must change the model.** Quantization and sparsification throw
information away, so a run using them cannot produce the same weights as the
baseline at the same seed. If it did, the plugin was never in the path. Checked
by comparing checkpoints directly.

**The reported bytes must be re-derivable.** The figure in the summary comes from
`payload_codecs` encoding the payload during the run. Here the same encoding is
applied from the outside, to a checkpoint this script loads itself, and the two
are compared. A number nobody can reproduce is not a measurement.

What is deliberately *not* applied to the weights is the lossless layer: the
bitmask, the index encoding and zlib. Those change how many bytes a payload
occupies, never what the client trained on, so applying them would be a no-op
that costs CPU. That is why the entropy coder appears in the byte count and not
in the model, and it is the one part of this that legitimately is "only
calculated".

Usage:

    python analysis/verify_compression.py
    python analysis/verify_compression.py --seed 2 --round 20
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_ueba import payload_codecs  # noqa: E402
from federated_ueba.efficiency_plugins import _topk_mask  # noqa: E402

BASELINE = "baseline"
CHECKPOINTS = "model_pickle"
REPORTS = "federated_evaluation_reports"

# What each configuration should have done to the weights, if its plugins ran.
# `sparse` is the fraction of the model expected to be exactly zero; `fp16` is
# whether every value should survive a float16 round trip unchanged.
#
# Note what decides the `fp16` column, because it is not "does this run quantize
# anything". A checkpoint is the server's model *before* it is broadcast, so a
# downlink codec has not run yet when it is written. Uplink quantization does
# show: clients return float16 arrays and FedAvg's weighted average keeps numpy's
# float16 all the way through, so the aggregate is fp16 representable. That is
# why `bidirectional-fp16`, which quantizes both directions, reads True while
# `downlink-fp16`, which quantizes only the broadcast, reads False. Downlink
# compression cannot be verified from checkpoints at all; `tests/test_downlink.py`
# covers it, and this script says so rather than pretending otherwise.
EXPECTED = {
    "quantization-fp16": {"sparse": 0.0, "fp16": True},
    "bidirectional-fp16": {"sparse": 0.0, "fp16": True},
    "downlink-fp16": {"sparse": 0.0, "fp16": False},
    "top-k-0.1": {"sparse": 0.4, "fp16": False},
    "top-k-0.05": {"sparse": 0.6, "fp16": False},
    "top-k-0.1-quant-fp16": {"sparse": 0.4, "fp16": True},
    "delta-0.1": {"sparse": 0.0, "fp16": False},
    "delta-0.05": {"sparse": 0.0, "fp16": False},
}


def load_weights(run_id, round_number):
    """One checkpoint, flattened, or None if that run did not reach it."""
    path = os.path.join(CHECKPOINTS, run_id,
                        f"parameters_round_{round_number}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        arrays = list(pickle.load(f).values())[0]
    return arrays, np.concatenate([a.ravel() for a in arrays])


def check_models_differ(seed, round_number):
    """No configuration may produce the baseline's weights."""
    print("\n" + "=" * 74)
    print("1. KAYIPLI ADIMLAR MODELI DEGISTIRIYOR MU")
    print("=" * 74)

    reference = load_weights(f"{BASELINE}__seed{seed}", round_number)
    if reference is None:
        print(f"  baseline__seed{seed} tur {round_number} yok; atlandi.")
        return []

    _, baseline = reference
    print(f"{'konfigurasyon':24s} {'ayni mi':>8} {'maks fark':>11} "
          f"{'sifir%':>8} {'fp16-tam':>9} {'beklenen':>10}")

    problems = []
    for name, expected in EXPECTED.items():
        loaded = load_weights(f"{name}__seed{seed}", round_number)
        if loaded is None:
            continue
        _, weights = loaded

        identical = np.array_equal(weights, baseline)
        zeros = float((weights == 0).mean())
        fp16 = bool(np.array_equal(
            weights, weights.astype(np.float16).astype(np.float32)))

        verdict = "TAMAM"
        if identical:
            verdict = "AYNI!"
            problems.append(f"{name}: baseline ile bit bit ayni")
        elif zeros < expected["sparse"]:
            verdict = "SEYREK DEGIL"
            problems.append(f"{name}: beklenen sifir orani "
                            f"{expected['sparse']:.0%}, olculen {zeros:.1%}")
        elif fp16 != expected["fp16"]:
            verdict = "FP16 UYUMSUZ"
            problems.append(f"{name}: fp16-tam beklendi {expected['fp16']}, "
                            f"olculen {fp16}")

        print(f"{name:24s} {str(identical):>8} "
              f"{np.abs(weights - baseline).max():11.6f} {zeros * 100:7.1f}% "
              f"{str(fp16):>9} {verdict:>10}")
    return problems


def logged_upload_per_client(run_id):
    """The mean bytes one client uploaded in one round.

    Read from the raw per-client transfer records rather than divided out of the
    summary total. `num_supernodes` is only declared by the experiments that
    override it, so deriving the upload count from the config silently produced
    nothing for every other run; the records carry one line per transfer and
    need no arithmetic to be trusted.
    """
    comm_dir = os.path.join(REPORTS, run_id, "comm")
    if not os.path.isdir(comm_dir):
        return None

    uploads = []
    for name in os.listdir(comm_dir):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(comm_dir, name)) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3 and parts[0] == "upload":
                    uploads.append(float(parts[2]))
    return sum(uploads) / len(uploads) if uploads else None


def check_bytes_reproduce(seed, round_number, tolerance):
    """Re-encode a checkpoint here and compare with what the run reported."""
    print("\n" + "=" * 74)
    print("2. RAPORLANAN BAYTLAR DISARIDAN YENIDEN URETILEBILIYOR MU")
    print("=" * 74)

    reference = load_weights(f"{BASELINE}__seed{seed}", round_number)
    if reference is None:
        print(f"  baseline__seed{seed} tur {round_number} yok; atlandi.")
        return []

    arrays, _ = reference
    # Encoded from the baseline's weights on purpose: the point is that the byte
    # count follows from the configuration, not from the particular run. Client
    # weights differ from the global model, so a few percent of disagreement is
    # expected and anything larger is not.
    candidates = {
        BASELINE: payload_codecs.measure_dense_payload(arrays, "zlib"),
        "quantization-fp16": payload_codecs.measure_dense_payload(
            [a.astype(np.float16) for a in arrays], "zlib"),
        "top-k-0.1": payload_codecs.measure_payload(
            [_topk_mask(a, 0.1)[0] for a in arrays], "bitmask", "zlib"),
        "top-k-0.05": payload_codecs.measure_payload(
            [_topk_mask(a, 0.05)[0] for a in arrays], "bitmask", "zlib"),
    }

    print(f"{'konfigurasyon':24s} {'bagimsiz':>12} {'kosumdan':>12} "
          f"{'fark':>8} {'karar':>8}")

    problems = []
    for name, size in candidates.items():
        logged = logged_upload_per_client(f"{name}__seed{seed}")
        if logged is None:
            continue
        independent = size / (1024 * 1024)
        gap = abs(independent - logged) / logged
        verdict = "TAMAM" if gap <= tolerance else "SAPMA"
        if gap > tolerance:
            problems.append(f"{name}: bagimsiz hesap {independent:.4f} MB, "
                            f"kosum {logged:.4f} MB ({gap:.1%} fark)")
        print(f"{name:24s} {independent:9.4f} MB {logged:9.4f} MB "
              f"{gap * 100:7.2f}% {verdict:>8}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--round", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="allowed gap between the two byte counts")
    args = parser.parse_args()

    problems = check_models_differ(args.seed, args.round)
    problems += check_bytes_reproduce(args.seed, args.round, args.tolerance)

    print("\n" + "=" * 74)
    if problems:
        print("SORUN VAR")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("Kayipli adimlar modele uygulanmis ve raporlanan baytlar disaridan")
    print("yeniden uretilebiliyor. Kayipsiz katman (bitmask, zlib) yalnizca")
    print("bayt sayisini degistirir, agirliklari degil; bu kasitlidir.")


if __name__ == "__main__":
    main()
