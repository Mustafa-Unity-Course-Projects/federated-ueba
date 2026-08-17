"""What is `top_k_features` worth, and is 5 the right number?

The scoring pipeline focuses the score on the m most deviant features. The
ablation already measured what the stage contributes when removed entirely
(+0.0377 at pipeline v8, five seeds), but m itself was never varied: the thesis
quotes 5 and the defence answer for "why 5" is currently "it seemed reasonable".

This closes that gap without retraining. The stage runs after the model, so a
different m is a rescoring pass over saved weights rather than a new experiment.

m = 50 is the degenerate case: every feature enters the mean, which is the same
thing the `ablation-no-topk` arm measures. It is included as a check that this
script agrees with that arm.

Usage:

    python analysis/topm_sweep.py --seed 1
    python analysis/topm_sweep.py --seed 1 --values 1 3 5 8 12 20 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle  # noqa: E402
from dataclasses import replace  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--values", nargs="*", type=int,
                        default=[1, 2, 3, 5, 8, 12, 20, 50])
    parser.add_argument("--out", default="generated_visuals/topm_sweep.csv")
    args = parser.parse_args()

    os.environ.setdefault("SEED", str(args.seed))

    import train_centralized as tc
    from federated_ueba import scaling

    df = pd.read_csv(tc.config.get("data", "processed_data_path"),
                     low_memory=False)
    features = tc.select_features(df)
    with open(tc.CENTRALIZED_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(tc.CENTRALIZED_ERROR_STATS_PATH, "rb") as f:
        stats = pickle.load(f)

    from federated_ueba import task
    model = task.LSTMAutoencoder(input_dim=len(features),
                                 hidden_dim=tc.HIDDEN_DIM).to(tc.DEVICE)
    model.load_state_dict(torch.load(tc.CENTRALIZED_MODEL_PATH,
                                     map_location=tc.DEVICE,
                                     weights_only=True))
    model.eval()

    base = tc.SCORING_CFG
    print(f"Merkezi model, seed {args.seed}. Ayni agirliklar, ayni veri, "
          f"yalniz m degisiyor.\n")
    print(f"{'m':>4s} {'dogrulama':>11s} {'test':>9s} {'tum':>9s} {'F1':>8s}")

    rows = []
    for m in args.values:
        # The scorer reads the module-level config, so the sweep is expressed by
        # swapping it rather than by threading a parameter through four layers.
        tc.SCORING_CFG = replace(base, top_k_features=m)
        metrics, _ = tc.evaluate_anomaly_detection(model, df, scaler, stats,
                                                   features)
        rows.append({"m": m, "pr_auc_val": metrics["pr_auc_val"],
                     "pr_auc_test": metrics["pr_auc_test"],
                     "pr_auc_all": metrics["pr_auc_all"],
                     "f1": metrics["best_f1"]})
        print(f"{m:4d} {metrics['pr_auc_val']:11.4f} "
              f"{metrics['pr_auc_test']:9.4f} {metrics['pr_auc_all']:9.4f} "
              f"{metrics['best_f1']:8.4f}")
    tc.SCORING_CFG = base

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out, index=False)

    # Chosen on the validation half and reported on the test half, which is the
    # same discipline the rest of the pipeline follows. Picking m by the number
    # it is going to be judged on would be tuning on the test set, and this
    # sweep exists to defend a choice rather than to manufacture one.
    best = table.loc[table["pr_auc_val"].idxmax()]
    configured = table[table["m"] == base.top_k_features]
    print(f"\nDogrulama yarisinda en iyi m: {int(best['m'])}  "
          f"(dogrulama {best['pr_auc_val']:.4f}, "
          f"o m'nin test degeri {best['pr_auc_test']:.4f})")
    if not configured.empty:
        row = configured.iloc[0]
        print(f"Yapilandirilmis m = {base.top_k_features}: "
              f"dogrulama {row['pr_auc_val']:.4f}, test {row['pr_auc_test']:.4f}")
        print(f"Secim test yarisinda {float(best['pr_auc_test']) - float(row['pr_auc_test']):+.4f} "
              f"getiriyor.")
    print(f"\nSaved '{args.out}'")


if __name__ == "__main__":
    main()
