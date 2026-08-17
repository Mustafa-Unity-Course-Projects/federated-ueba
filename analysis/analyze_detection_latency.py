"""How long after an insider starts does the system flag them?

The jury asked this and it could not be answered. It is not a property of the
metrics: PR-AUC says how well the ranking separates insiders from everyone else,
and says nothing about when in an insider's campaign that separation appears.

The measurement here replays each user day by day. On day D the user is scored
using only the windows that ended on or before D, aggregated exactly as the
detector normally aggregates them, and compared against the operating threshold.
The first D at which the score crosses is the day the system would have raised
the alert. Latency is that day minus the user's first malicious day.

Scoring prospectively rather than once at the end is the point. A detector that
only recognises an insider after their campaign is over is not useful, and the
usual evaluation cannot tell the difference.

Two structural floors bound the answer, and both belong in the thesis:

  window_size    a window ending on day D covers D-13..D, so a single malicious
                 day sits among 13 normal ones and moves the window score only
                 slightly. Detection generally needs several malicious days in
                 one window.
  30-day history temporal.py expresses each day as a percentile of the user's own
                 preceding 30 days, so a user needs 30 days of history before any
                 feature value exists at all.

    python analyze_detection_latency.py
    python analyze_detection_latency.py --experiment baseline --seed 1
"""

# This file lives in a subdirectory, so Python puts that subdirectory on
# sys.path rather than the project root and `import config_manager` fails.
# Adding the root explicitly keeps `python analysis/analyze_detection_latency.py` working from the project
# root, which is how every path in the configuration is resolved anyway.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from config_manager import config
from federated_ueba import scaling, scoring

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def find_latest_checkpoint(save_path):
    """The highest-numbered saved round in a run's model directory."""
    if not os.path.isdir(save_path):
        return None
    rounds = []
    for filename in os.listdir(save_path):
        if filename.startswith("parameters_round_") and filename.endswith(".pkl"):
            rounds.append((int(filename.split("_")[-1].split(".")[0]), filename))
    if not rounds:
        return None
    return os.path.join(save_path, max(rounds)[1])


def window_scores_by_day(scorer, model, user_df):
    """Score every window of one user, paired with the day that window ends on.

    The end day is what matters for latency: a window is only available once its
    last day has happened, so that is the earliest a detector could have used it.

    The scoring itself is the scorer's, not a copy of it. This function only adds
    the day each window ends on; if it recomputed the scores its own way, the
    latency reported here would describe a detector that does not exist.
    """
    days = user_df["day"].to_numpy()
    if len(user_df) < scorer.window_size:
        return np.empty(0), np.empty(0)

    tensor = scorer.to_tensor(user_df)
    with torch.no_grad():
        scores = scorer.window_scores(model, tensor)

    # One end day per window, in the same order the scorer produced them.
    end_days = [days[start + scorer.window_size - 1]
                for start in range(0, len(days) - scorer.window_size + 1,
                                   scorer.cfg.scan_stride)]
    return np.asarray(scores), np.asarray(end_days)


def first_alert_day(scores, end_days, threshold, cfg):
    """The first day on which the running score crosses the threshold.

    Replays the aggregation the detector normally performs, but restricted at
    each step to the windows available by that day. Returns None if the user is
    never flagged.
    """
    for i in range(len(scores)):
        # Windows are in chronological order, so the first i + 1 of them are
        # exactly those available on day end_days[i].
        available = scores[:i + 1]
        if scoring.aggregate_windows(available, cfg) >= threshold:
            return float(end_days[i])
    return None


def measure_latency(scorer, model, df, insider_users, threshold):
    """One row per insider: when their campaign started and when it was caught."""
    rows = []
    for user in insider_users:
        user_df = df[df["user"] == user].sort_values("day")
        malicious_days = user_df.loc[user_df["insider"] != 0, "day"]
        if malicious_days.empty:
            continue

        scores, end_days = window_scores_by_day(scorer, model, user_df)
        if len(scores) == 0:
            continue

        first_malicious = float(malicious_days.min())
        alert_day = first_alert_day(scores, end_days, threshold, scorer.cfg)
        # The scenario label CERT assigns; the three campaigns look different
        # enough that a single average over them would hide the variation.
        scenario = int(user_df.loc[user_df["insider"] != 0, "insider"].iloc[0])

        rows.append({
            "user": user,
            "scenario": scenario,
            "first_malicious_day": first_malicious,
            "last_malicious_day": float(malicious_days.max()),
            "malicious_days": int(len(malicious_days)),
            "alert_day": alert_day,
            "latency_days": None if alert_day is None else alert_day - first_malicious,
            "detected": alert_day is not None,
        })
    return pd.DataFrame(rows)


def report(table):
    """Print the summary the thesis needs, overall and per scenario."""
    detected = table[table["detected"]]
    caught_after_start = detected[detected["latency_days"] >= 0]
    flagged_before = detected[detected["latency_days"] < 0]

    print(f"\nİçeriden saldırgan sayısı: {len(table)}")
    print(f"Eşiği geçen: {len(detected)} (%{100 * len(detected) / max(len(table), 1):.0f})")
    print(f"  kötücül faaliyet başladıktan sonra: {len(caught_after_start)}")
    print(f"  daha önce işaretlenmiş: {len(flagged_before)}")

    if not caught_after_start.empty:
        latency = caught_after_start["latency_days"]
        print(f"\nTespit gecikmesi (ilk kötücül günden itibaren, gün):")
        print(f"  medyan   {latency.median():.0f}")
        print(f"  ortalama {latency.mean():.1f}")
        print(f"  en hızlı {latency.min():.0f}")
        print(f"  en yavaş {latency.max():.0f}")

        # The operationally interesting question is not the average but how much
        # of the campaign runs before the alert.
        within = caught_after_start["alert_day"] <= caught_after_start["last_malicious_day"]
        print(f"\nKampanya bitmeden yakalanan: {int(within.sum())} / "
              f"{len(caught_after_start)}")

    if not table.empty:
        print("\nSenaryo bazında:")
        for scenario, group in table.groupby("scenario"):
            found = group[group["detected"] & (group["latency_days"] >= 0)]
            median = f"{found['latency_days'].median():.0f}" if not found.empty else "-"
            print(f"  senaryo {scenario}: {len(group)} saldırgan, "
                  f"{int(group['detected'].sum())} tespit, medyan gecikme {median} gün")

    print(f"\nYapısal alt sınır: bir pencere {int(config.get('model', 'window_size'))} "
          f"gün kapsıyor ve temporal.py her günü kendinden önceki 30 güne göre "
          f"ifade ediyor, yani ilk 30 gün skorlanamıyor.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="detection_latency.csv")
    args = parser.parse_args()

    if args.seed is not None:
        config.set_seed(args.seed)
    config.set_experiment(args.experiment)
    run_id = config.run_id

    checkpoint = find_latest_checkpoint(config.get("federation", "save_path"))
    if checkpoint is None:
        print(f"'{run_id}' için checkpoint bulunamadı. Önce deneyi çalıştır.")
        return 1

    scaler_dir = config.get("data", "scaler_dir")
    scaler_path = os.path.join(scaler_dir, "global_scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"Küresel ölçekleyici bulunamadı: {scaler_path}")
        return 1

    import federated_ueba.task as task

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    features = list(scaler.feature_names_in_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = task.LSTMAutoencoder(input_dim=len(features),
                                 hidden_dim=config.get("model", "hidden_dim")).to(device)
    with open(checkpoint, "rb") as f:
        weights = pickle.load(f)["global_parameters"]
    model.load_state_dict({k: torch.tensor(w)
                           for k, w in zip(model.state_dict().keys(), weights)})
    model.eval()

    from federated_insider_detection import load_error_reference
    ref_mean, ref_std = load_error_reference(scaler_dir, len(features))
    cfg = scoring.ScoringConfig.from_config(config)

    print(f"'{run_id}' için gecikme analizi ({os.path.basename(checkpoint)})...")
    df = pd.read_csv(config.get("data", "processed_data_path"), low_memory=False)
    all_users = sorted(df["user"].unique())

    # The operating threshold, obtained exactly as the reported metrics obtain
    # it: fitted on the validation half of the users. Using a threshold tuned on
    # the insiders being measured would make the latency look better than it is.
    scorer = scoring.Scorer(
        features=features, scaler=scaler, ref_mean=ref_mean, ref_std=ref_std,
        cfg=cfg, window_size=task.WINDOW_SIZE, device=device)

    results = scorer.scan(model, df, all_users)
    threshold = scoring.evaluate_scores(results, seed=config.seed)["threshold"]
    print(f"Eşik (doğrulama yarısından): {threshold:.4f}")

    insider_users = sorted(df.loc[df["insider"] != 0, "user"].unique())
    table = measure_latency(scorer, model, df, insider_users, threshold)

    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"CSV yazıldı: {args.out}")
    report(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
