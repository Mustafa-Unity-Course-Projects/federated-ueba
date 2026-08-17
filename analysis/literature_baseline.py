"""Our results expressed in the metrics the CERT r4.2 literature reports.

Jury item J15. The thesis compared itself against a "temel model" of its own
making, which is not a baseline. The published work on this dataset is the
baseline, and the obstacle is that it does not report what we report.

Le and Zincir-Heywood (2021) evaluate on CERT R4.2 with **user-based ROC-AUC**
and with **detection rate at an investigation budget**: how many of the 70
malicious users appear if an analyst examines the top few percent of the ranking.
We report PR-AUC. On data this imbalanced the two are not interchangeable:
ROC-AUC counts true negatives, of which there are 930, so it flatters a ranking
that PR-AUC would judge harshly.

Arguing about which metric is better is the wrong move in a defence. Computing
theirs from our own per-user scores is the right one, and costs nothing: the
scores are already on disk.

    python analysis/literature_baseline.py
    python analysis/literature_baseline.py --experiment baseline --seed 1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FEDERATED_DIR = "federated_evaluation_reports"
CENTRALIZED_DIR = "centralized_evaluation_reports"

# Investigation budgets, as a share of the user population. Le and
# Zincir-Heywood express the budget over instances rather than users; ours is
# over users, because our scoring produces one score per user by construction.
# The difference is stated in the output rather than papered over.
BUDGETS = [0.01, 0.02, 0.05, 0.10]


def detection_rate_at_budget(y_true, y_score, budget):
    """Share of insiders inside the top `budget` of the ranking."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    examined = max(1, int(round(budget * len(y_true))))
    caught = int(y_true[order[:examined]].sum())
    total = int(y_true.sum())
    return caught, total, examined, caught / total if total else float("nan")


def report(label, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    print(f"\n{label}  ({len(y_true)} kullanici, {int(y_true.sum())} icerideki)")
    print(f"  kullanici bazli ROC-AUC   {roc_auc_score(y_true, y_score):.4f}")
    print(f"  PR-AUC                    "
          f"{average_precision_score(y_true, y_score.reshape(-1, 1)):.4f}")
    for budget in BUDGETS:
        caught, total, examined, rate = detection_rate_at_budget(
            y_true, y_score, budget)
        print(f"  IB {budget:>5.0%}: ilk {examined:>4d} kullanici incelenirse "
              f"{caught:>2d}/{total} yakalaniyor  ({rate:.1%})")


def reported_round(run):
    """The round this run is read at, round 0 excluded.

    Deliberately not `best_round` from the summary. That field selects over
    round 0, which is the model before any training scored against a reference
    the clients measured with their *trained* local models, so it comes out
    artificially high and wins the argmax. Three of the four baseline seeds
    record `best_round = 0`, so reading the summary field here compared the
    published literature against an untrained model. The other analysis
    consumers were fixed on 9-10 August; this one was missed.
    """
    path = os.path.join(FEDERATED_DIR, run, "federated_rounds_comparison.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["Round"] > 0]
    if frame.empty:
        return None
    return int(frame.loc[frame["PR-AUC_val"].idxmax(), "Round"])


def load_federated(experiment, seed):
    run = f"{experiment}__seed{seed}"
    summary_path = os.path.join(FEDERATED_DIR, run, "experiment_summary.json")
    if not os.path.exists(summary_path):
        return None, None
    with open(summary_path) as f:
        summary = json.load(f)
    round_number = reported_round(run)
    if round_number is None:
        return None, None
    path = os.path.join(FEDERATED_DIR, run, "round_by_round_results",
                        f"round_{round_number}_results.csv")
    if not os.path.exists(path):
        return None, None
    summary = dict(summary, reported_round=round_number)
    return pd.read_csv(path), summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print("=" * 74)
    print("YAYIMLANMIS SONUCLAR, CERT R4.2")
    print("=" * 74)
    print("""
Le, D. C., Zincir-Heywood, N. (2021). "Anomaly Detection for Insider Threats
Using Unsupervised Ensembles." IEEE Transactions on Network and Service
Management, 18(2), 1152-1164.

  Otokodlayici, hafta verisi, kullanici bazli ROC-AUC (Sekil 4b):
    P60 0,916   P30 0,899   C2 0,869   C3 0,860   Org 0,836   M60 0,739

  Metinden: en anormal orneklerin %1'i incelenerek kotucul kullanicilarin
  %77'si; %5 butce ile 70 saldirganin neredeyse tamami.

  Kurulum: denetimsiz (etiketsiz egitim), rastgele secilmis 200 kullanici,
  veri suresinin ilk %50'si, 10 tekrarin ortalamasi.
""")

    print("=" * 74)
    print("BIZIM SONUCLARIMIZ, AYNI METRIKLERDE")
    print("=" * 74)

    central = os.path.join(CENTRALIZED_DIR, f"seed{args.seed}",
                           "centralized_insider_results.csv")
    if os.path.exists(central):
        frame = pd.read_csv(central)
        report("Merkezi model, butun kullanicilar",
               frame["is_actual_insider"], frame["max_z_score"])

    frame, summary = load_federated(args.experiment, args.seed)
    if frame is not None:
        report(f"Federe {args.experiment}, butun kullanicilar",
               frame["is_actual_insider"], frame["max_z_score"])

        # The reported half, for readers who want the number the thesis quotes.
        from config_manager import config
        from federated_ueba import scoring
        _, test_users = scoring.split_users(
            frame["user"].tolist(), frame["is_actual_insider"].tolist(),
            seed=config.get("evaluation", "split_seed"),
            validation_fraction=config.get("evaluation", "validation_fraction"))
        half = frame[frame["user"].isin(test_users)]
        report(f"Federe {args.experiment}, raporlanan test yarisi",
               half["is_actual_insider"], half["max_z_score"])
    else:
        print(f"\n{args.experiment}__seed{args.seed} bulunamadi.")

    print("""
KARSILASTIRMA YAPILIRKEN SOYLENMESI GEREKENLER

  1. Onlarin butcesi ornekler (kullanici-gun/hafta) uzerinden, bizimki
     kullanicilar uzerinden. Bizim skorlama kullanici basina tek skor uretiyor,
     dolayisiyla ornek siralamasi yok. Sayilar ayni yonu gosterir ama birebir
     ayni sey degildir.

  2. ROC-AUC dengesiz veride PR-AUC'tan sistematik olarak yuksektir, cunku 930
     dogru negatifi sayar. Ikisini yan yana vermek, hangi metrigin secildigi
     sorusunu ortadan kaldirir.

  3. Onlar 200 kullanici ve surenin ilk yarisiyla egitiyor, biz butun normal
     veriyle. Bu bizim lehimize bir fark ve belirtilmelidir.

  4. Onlarin kurulumu merkezi; federe kolun onun altinda kalmasi beklenir ve
     aradaki fark mahremiyetin bedelidir.
""")


if __name__ == "__main__":
    main()
