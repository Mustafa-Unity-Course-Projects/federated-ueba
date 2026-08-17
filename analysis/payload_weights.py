"""How many weights actually cross the wire, counted rather than assumed.

J12 asks for the parameters and bytes sent per round. The communication log
records bytes only, so the weight count has been quoted from the ratio: 10% of
450,258 is 45,025. That is not what the code sends. `_topk_mask` selects per
tensor with `max(1, int(size * ratio))`, so the total is a sum of 36 floors, not
one floor of the total, and a tensor smaller than 1/ratio contributes its
minimum of one. The difference is small and the point is that it is measured.

Three quantities are separated here, because conflating them is easy and the
first is the one the thesis needs:

  uplink count    weights one client sends in one round. Determined by the
                  plugin and the tensor shapes, so it is exact and does not
                  depend on the data.
  payload bytes   what those weights occupy after the codec and the entropy
                  coder. Measured by encoding, then cross-checked against the
                  run's own communication log.
  model sparsity  how much of the *published global model* is zero. Not the
                  same number: FedAvg averages 25 sparse clients whose masks
                  differ, so the aggregate is denser than any one payload.

Reads saved checkpoints and communication logs. Writes nothing.

    python analysis/payload_weights.py
    python analysis/payload_weights.py --experiment top-k-0.05 --seed 1
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import config  # noqa: E402
from federated_ueba import efficiency_plugins, payload_codecs  # noqa: E402

CHECKPOINTS = "model_pickle"
REPORTS = "federated_evaluation_reports"


def load_parameters(run_id, round_number):
    path = os.path.join(CHECKPOINTS, run_id,
                        "parameters_round_%d.pkl" % round_number)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["global_parameters"]


def kept_per_tensor(arrays, ratio):
    """What `_topk_mask` keeps, tensor by tensor. The exact uplink count."""
    return [min(max(1, int(a.size * ratio)), a.size) for a in arrays]


def logged_upload_mb(run_id):
    """Mean and total upload from the run's own per-client logs."""
    files = glob.glob(os.path.join(REPORTS, run_id, "comm", "client_*.csv"))
    values = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3 and parts[0] != "download":
                    values.append(float(parts[2]))
    if not values:
        return None, None, 0
    return float(np.mean(values)), float(np.sum(values)), len(values)


def report(experiment, seed, round_number):
    run_id = "%s__seed%d" % (experiment, seed)
    arrays = load_parameters(run_id, round_number)
    if arrays is None:
        print("%s: tur %d checkpoint'i yok" % (run_id, round_number))
        return

    arrays = [np.asarray(a) for a in arrays]
    total = sum(a.size for a in arrays)

    config.set_experiment(experiment)
    plugins = config.get("efficiency", "active_plugins") or []
    ratio = config.get("efficiency", "sparsification_ratio")
    codec = config.get("efficiency", "payload_codec")
    entropy = config.get("efficiency", "entropy_coder")

    print("=" * 74)
    print("%s  (tur %d)" % (run_id, round_number))
    print("=" * 74)
    print("  katman              %d" % len(arrays))
    print("  toplam parametre    %d" % total)
    print("  etkin eklenti       %s" % (", ".join(plugins) if plugins else "yok"))

    sparsifying = [p for p in plugins if "top_k" in p or "sparsif" in p]
    if sparsifying:
        kept = kept_per_tensor(arrays, ratio)
        naive = int(total * ratio)
        print("  seyreklestirme      oran %.2f, katman basina" % ratio)
        print("  GONDERILEN AGIRLIK  %d  (%.3f%% of %d)"
              % (sum(kept), 100.0 * sum(kept) / total, total))
        print("  naif oran x toplam  %d   fark %+d"
              % (naive, sum(kept) - naive))
        print("  atilan              %d" % (total - sum(kept)))
        sparse = [efficiency_plugins._topk_mask(a, ratio)[0] for a in arrays]
    else:
        print("  GONDERILEN AGIRLIK  %d  (yogun, seyreklestirme yok)" % total)
        sparse = arrays

    measured = payload_codecs.measure_payload(sparse, codec=codec,
                                              entropy=entropy)
    dense = payload_codecs.measure_dense_payload(arrays, entropy=entropy)
    raw = total * 4
    print("  ---")
    print("  analitik yogun      %.4f MB   (%d x 4 B, sikistirilmamis)"
          % (raw / 1024 / 1024, total))
    print("  yogun + %-6s      %.4f MB   (kayipsiz kazanc %%%.1f)"
          % (entropy, dense / 1024 / 1024, 100 * (1 - dense / raw)))
    print("  bu kolun yuku       %.4f MB   (codec %s + %s)"
          % (measured / 1024 / 1024, codec, entropy))

    mean_mb, total_mb, messages = logged_upload_mb(run_id)
    if messages:
        print("  ---")
        print("  LOGDAN olculen      %.4f MB ortalama, %d mesaj, %.1f MB toplam"
              % (mean_mb, messages, total_mb))
        gap = 100 * (measured / 1024 / 1024 - mean_mb) / mean_mb
        print("  yeniden uretim farki %+.1f%%" % gap)
        if abs(gap) > 15:
            print("  !!! fark buyuk: kayitli kuresel model, istemcinin gonderdigi")
            print("      yuk degil. Delta kollarinda bu beklenen bir fark.")

    zeros = sum(int(np.sum(a == 0)) for a in arrays)
    print("  ---")
    print("  YAYINLANAN MODEL    %d sifir / %d  (%%%.1f)"
          % (zeros, total, 100.0 * zeros / total))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--round", type=int, default=50)
    args = parser.parse_args()

    names = ([args.experiment] if args.experiment else
             ["baseline", "top-k-0.1", "top-k-0.05", "delta-0.1", "delta-0.05",
              "delta-0.05-downlink-fp16"])
    for name in names:
        report(name, args.seed, args.round)


if __name__ == "__main__":
    main()
