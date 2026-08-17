"""How much of the detection comes from the model, and how much from the scoring?

A reported PR-AUC does not say which part of the system earned it. The scoring
pipeline calibrates, focuses, and aggregates before any number is produced, and
those stages work on whatever the network hands them. So the question a jury will
ask is fair: would a network that learned nothing do just as well?

Four probes, each removing one more thing, all sharing the same scoring:

  shuffled      the real scores permuted across users. Must land on the base
                rate, or the metric itself is broken and nothing else here means
                anything.
  zero output   a "model" returning zeros, so the squared error is the squared
                input. No weights at all, learned or random.
  random init   an untrained network, calibrated on its own error statistics.
  trained       the reported model.

One trap, and it is easy to fall into. A random model must be calibrated against
**its own** reference error distribution. Scoring it against the trained model's
reference flatters it badly, because the trained reference has small means and
standard deviations, and dividing a random model's large errors by them leaves
the input's own structure in charge. The borrowed-reference figure is reported
below as well, but only to show how misleading it is.

`--cross` additionally runs the scoring stages against both the trained model and
the zero model, which separates what each stage does on its own from what it does
to a learned signal. That table was quoted in the write-up for months with no
script behind it; it is produced here so it can be regenerated rather than
trusted.

    python analysis/untrained_control.py
    python analysis/untrained_control.py --cross
    python analysis/untrained_control.py --seed 1 --iterations 500
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


class ZeroModel(nn.Module):
    """Reconstructs everything as zero, so the squared error is the input squared.

    The floor of the design: nothing learned, nothing random. Whatever this
    scores is what the scoring pipeline extracts from the data alone.
    """

    def forward(self, x):
        return torch.zeros_like(x)


def average_precision(y_true, y_score):
    """PR-AUC, tolerant of the 2-D score shape this sklearn version wants."""
    return float(average_precision_score(
        np.asarray(y_true).ravel(),
        np.asarray(y_score, dtype=float).reshape(-1, 1)))


def shuffled_baseline(y, scores, iterations, rng):
    """What a score carrying no information scores. Must be the base rate."""
    values = []
    for _ in range(iterations):
        drawn = rng.permutation(scores)
        if len(np.unique(y)) < 2:
            continue
        values.append(average_precision(y, drawn))
    return float(np.mean(values)), float(np.std(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200,
                        help="permutations for the shuffled baseline")
    parser.add_argument("--random-seeds", nargs="*", type=int,
                        default=[777, 12345],
                        help="initialisations for the untrained control")
    parser.add_argument("--cross", action="store_true",
                        help="also cross the scoring stages with both models")
    args = parser.parse_args()

    os.environ.setdefault("SEED", str(args.seed))

    import train_centralized as tc
    from federated_ueba import scaling, task

    df = pd.read_csv(tc.config.get("data", "processed_data_path"),
                     low_memory=False)
    features = tc.select_features(df)
    with open(tc.CENTRALIZED_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(tc.CENTRALIZED_ERROR_STATS_PATH, "rb") as f:
        trained_stats = pickle.load(f)

    # The same windows training used, so a control model's own error statistics
    # are measured on exactly what the trained model's were.
    df_train = (df[df["insider"] == 0].copy()
                if "insider" in df.columns else df.copy())
    df_train[features] = scaler.transform(
        scaling.prepare_features(df_train, features))
    train_loader, _ = tc.build_loaders(tc.build_sequences(df_train, features))

    def fresh(seed):
        torch.manual_seed(seed)
        return task.LSTMAutoencoder(input_dim=len(features),
                                    hidden_dim=tc.HIDDEN_DIM).to(tc.DEVICE)

    def evaluate(model, label, stats=None):
        if stats is None:
            stats = tc.calculate_error_stats(model, train_loader)
        metrics, results = tc.evaluate_anomaly_detection(
            model, df, scaler, stats, features)
        print(f"  {label:44s} test {metrics['pr_auc_test']:.4f}   "
              f"tum {metrics['pr_auc_all']:.4f}   F1 {metrics['best_f1']:.4f}")
        return results

    print(f"Merkezi kol, seed {args.seed}. Ayni veri, ayni skorlama, "
          f"yalniz agirliklar degisiyor.\n")

    trained = fresh(0)
    trained.load_state_dict(torch.load(tc.CENTRALIZED_MODEL_PATH,
                                       map_location=tc.DEVICE,
                                       weights_only=True))
    trained.eval()
    reported = evaluate(trained, "egitilmis model", trained_stats)

    for seed in args.random_seeds:
        evaluate(fresh(seed), f"rastgele ilk deger {seed}, kendi referansi")

    # The same random model against the trained model's reference, to show the
    # size of the mistake rather than to claim anything from it.
    if args.random_seeds:
        evaluate(fresh(args.random_seeds[0]),
                 f"rastgele {args.random_seeds[0]}, ODUNC referans (yaniltici)",
                 trained_stats)

    evaluate(ZeroModel().to(tc.DEVICE), "sifir cikti, model yok")

    y = reported["is_actual_insider"].to_numpy()
    s = reported["max_z_score"].to_numpy()
    rng = np.random.RandomState(0)
    mean, sd = shuffled_baseline(y, s, args.iterations, rng)
    print(f"\n  {'skorlar karistirildi (bilgi tasimayan skor)':44s} "
          f"tum {mean:.4f} +- {sd:.4f}")
    print(f"  {'temel oran (pozitif orani)':44s} tum {y.mean():.4f}")

    print("\nOkunusu: karistirilmis skor temel orana dusuyorsa metrik saglam. "
          "Sifir cikti ile rastgele ag arasindaki fark agin agirliklarinin "
          "katkisi, rastgele ag ile egitilmis model arasindaki fark ise "
          "egitimin katkisidir.")

    if args.cross:
        report_stage_cross(tc, trained, trained_stats, train_loader,
                           df, scaler, features)


def report_stage_cross(tc, trained, trained_stats, train_loader,
                       df, scaler, features):
    """Each scoring stage against a learned signal and against no signal at all.

    `tc.SCORING_CFG` is read at call time inside `evaluate_anomaly_detection`,
    so rebinding it here is enough; `ScoringConfig.without` builds the ablation
    so that no stage list is written out by hand.

    The zero model's reference statistics are computed once, on the same windows
    training used. They do not depend on which scoring stages run: the reference
    is per-feature squared error, measured before any stage touches it.
    """
    full = tc.SCORING_CFG
    zero = ZeroModel().to(tc.DEVICE)
    zero_stats = tc.calculate_error_stats(zero, train_loader)

    rows = [("tam pipeline", full)]
    rows += [(f"{stage} cikarildi", full.without(stage))
             for stage in full.stages]
    rows.append(("hicbir asama yok", full.without(*full.stages)))

    print("\n" + "=" * 78)
    print("ASAMALAR x MODEL (tum 1000 kullanici)")
    print("=" * 78)
    print(f"{'yapilandirma':28s} {'egitilmis':>10s} {'sifir cikti':>12s} "
          f"{'fark':>8s}")

    try:
        for label, cfg in rows:
            tc.SCORING_CFG = cfg
            trained_metrics, _ = tc.evaluate_anomaly_detection(
                trained, df, scaler, trained_stats, features)
            zero_metrics, _ = tc.evaluate_anomaly_detection(
                zero, df, scaler, zero_stats, features)
            a = trained_metrics["pr_auc_all"]
            b = zero_metrics["pr_auc_all"]
            print(f"{label:28s} {a:10.4f} {b:12.4f} {a - b:+8.4f}")
    finally:
        # Restored even on failure: this module-level value is what every later
        # call in this process would score with.
        tc.SCORING_CFG = full

    print("\nOkunusu: sag sutun asamanin modelden bagimsiz olarak yaptigi is, "
          "fark sutunu ise ogrenilmis sinyalin o asamadan gecerken kattigi.")


if __name__ == "__main__":
    main()
