# -*- coding: utf-8 -*-
"""The control the latency measurement never had: normal users, replayed the same way.

`analyze_detection_latency.py` replays only the 70 insiders, so it can say when an
insider is flagged and cannot say whether being flagged means anything. That gap
matters for one finding in particular. Every scenario-3 actor that crosses the
threshold crosses it on day 51 or 56, the first days the pipeline can score at
all, between 115 and 255 days before their first malicious day. Two readings fit:

  the flags are noise      the percentile baseline is thinnest in the first
                           scorable windows, everyone looks anomalous there, and
                           an alert on day 56 carries no information
  the flags are real       those users genuinely separate from the population,
                           just not because of the sabotage

Only a control group separates them. This replays the same day-by-day scan over
users CERT never labelled, against the same threshold, and asks what fraction of
them are flagged and when.

Nothing here writes to a results directory; the output is a CSV wherever --out
points.

    python analysis/erken_isaretleme_kontrolu.py --seed 1
    python analysis/erken_isaretleme_kontrolu.py --seed 1 --ornek 200
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch

from config_manager import config
from federated_ueba import scoring

import analyze_detection_latency as gecikme

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def olc(scorer, model, df, kullanicilar, esik):
    """Bir satır per kullanıcı: ilk uyarı günü, yoksa None."""
    satirlar = []
    baslangic = time.time()
    for i, u in enumerate(kullanicilar, 1):
        k = df[df["user"] == u].sort_values("day")
        skorlar, gunler = gecikme.window_scores_by_day(scorer, model, k)
        if len(skorlar) == 0:
            continue
        uyari = gecikme.first_alert_day(skorlar, gunler, esik, scorer.cfg)
        satirlar.append({"user": u, "alert_day": uyari,
                         "detected": uyari is not None})
        if i % 50 == 0:
            gecen = time.time() - baslangic
            print(f"  {i}/{len(kullanicilar)}  ({gecen:.0f} sn, "
                  f"kalan ~{gecen / i * (len(kullanicilar) - i):.0f} sn)")
    return pd.DataFrame(satirlar)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ornek", type=int, default=0,
                    help="Kaç normal kullanıcı örneklensin. 0 = hepsi.")
    ap.add_argument("--out", default="erken_isaretleme.csv")
    args = ap.parse_args()

    config.set_seed(args.seed)
    config.set_experiment(args.experiment)

    checkpoint = gecikme.find_latest_checkpoint(config.get("federation", "save_path"))
    if checkpoint is None:
        print(f"'{config.run_id}' için checkpoint yok.")
        return 1

    scaler_dir = config.get("data", "scaler_dir")
    with open(os.path.join(scaler_dir, "global_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    features = list(scaler.feature_names_in_)

    import federated_ueba.task as task
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

    df = pd.read_csv(config.get("data", "processed_data_path"), low_memory=False)
    scorer = scoring.Scorer(features=features, scaler=scaler, ref_mean=ref_mean,
                            ref_std=ref_std, cfg=scoring.ScoringConfig.from_config(config),
                            window_size=task.WINDOW_SIZE, device=device)

    # Aynı eşik: doğrulama yarısından, gecikme ölçümüyle birebir aynı yordam.
    esik = scoring.evaluate_scores(
        scorer.scan(model, df, sorted(df["user"].unique())),
        seed=config.seed)["threshold"]
    print(f"Eşik (doğrulama yarısından): {esik:.4f}")

    normaller = sorted(df.loc[df["insider"] == 0, "user"].unique())
    icerdekiler = set(df.loc[df["insider"] != 0, "user"].unique())
    normaller = [u for u in normaller if u not in icerdekiler]
    print(f"Etiketlenmemiş kullanıcı: {len(normaller)}")

    if args.ornek and args.ornek < len(normaller):
        rng = np.random.default_rng(config.seed)
        normaller = sorted(rng.choice(normaller, args.ornek, replace=False).tolist())
        print(f"Örneklendi: {len(normaller)}")

    tablo = olc(scorer, model, df, normaller, esik)
    tablo.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nCSV yazıldı: {args.out}")

    isaretli = tablo[tablo["detected"]]
    n = len(tablo)
    print(f"\nEşiği geçen: {len(isaretli)} / {n} (%{100 * len(isaretli) / max(n, 1):.1f})")
    if not isaretli.empty:
        g = isaretli["alert_day"]
        taban = (g <= 56).sum()
        print(f"  ilk skorlanabilir pencerede (gün <= 56): {taban} "
              f"(%{100 * taban / n:.1f} of {n})")
        print(f"  uyarı günü medyanı: {g.median():.0f}, "
              f"en küçük {g.min():.0f}, en büyük {g.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
