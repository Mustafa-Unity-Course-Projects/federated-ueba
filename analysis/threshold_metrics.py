"""Eşiğe bağlı metrikler, PR-AUC ile aynı kural altında.

`experiment_summary.json` içindeki `best_metrics` bloğu bir tur seçer; tezin
federe kolu hiçbir tur seçmez. İki sayıyı yan yana koymak, biri seçilmiş biri
seçilmemiş iki kuralı karıştırmak olur.

Bu betik, kesinlik, duyarlılık, F1 ve dengeli doğruluğu PR-AUC ile aynı kuralla
raporlar: son `plateau_window_rounds` turun ortalaması, tur 0 hariç. Değerler
`federated_rounds_comparison.csv` içindeki tur satırlarından okunur; bu satırlar
eşiği doğrulama yarısında seçip test yarısında uygulayan hesabın çıktısıdır.

    python analysis/threshold_metrics.py
    python analysis/threshold_metrics.py --only baseline,delta-0.05-downlink-fp16
"""

import argparse
import io
import json
import os
import re
import statistics
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import PIPELINE_VERSION, config  # noqa: E402

RAPOR_KOKU = "federated_evaluation_reports"
SUTUNLAR = ["Precision", "Recall", "Max-F1", "Balanced_Accuracy",
            "TP", "FP", "FN", "Optimal-Threshold"]


def plato_satirlari(csv_yolu, pencere):
    """Raporlanan istatistiğin ortalama aldığı turlar."""
    frame = pd.read_csv(csv_yolu)
    frame = frame[frame["Round"] > 0]
    if frame.empty:
        return None
    esik = frame["Round"].max() - pencere
    return frame[frame["Round"] > esik]


def kol_adi(dizin):
    m = re.match(r"(.+)__seed(\d+)$", dizin)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def surum_uyuyor(dizin):
    """Koşum güncel pipeline sürümünden mi?

    `findings.py` ve `seed_variance.py` bu filtreyi uyguluyor; burada da
    uygulanmazsa eski bir sürümde puanlanmış bir kol sessizce çizelgeye girer.
    """
    yol = os.path.join(RAPOR_KOKU, dizin, "experiment_summary.json")
    if not os.path.exists(yol):
        return False
    with io.open(yol, encoding="utf-8") as f:
        return json.load(f).get("pipeline_version") == PIPELINE_VERSION


def topla(pencere, yalniz):
    """Kol -> metrik -> seed başına değer listesi."""
    sonuc, atlanan = {}, set()
    for dizin in sorted(os.listdir(RAPOR_KOKU)):
        ad, seed = kol_adi(dizin)
        if ad is None or (yalniz and ad not in yalniz):
            continue
        if not surum_uyuyor(dizin):
            atlanan.add(ad)
            continue
        csv_yolu = os.path.join(RAPOR_KOKU, dizin,
                                "federated_rounds_comparison.csv")
        if not os.path.exists(csv_yolu):
            continue
        satirlar = plato_satirlari(csv_yolu, pencere)
        if satirlar is None:
            continue
        kol = sonuc.setdefault(ad, {s: [] for s in SUTUNLAR})
        for sutun in SUTUNLAR:
            if sutun in satirlar:
                kol[sutun].append(float(satirlar[sutun].mean()))
    for ad in sorted(atlanan - set(sonuc)):
        print(f"!  Atlandi (pipeline surumu {PIPELINE_VERSION} degil): {ad}")
    return sonuc


def main():
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--only", default="")
    ayristirici.add_argument("--window", type=int, default=None)
    secenek = ayristirici.parse_args()

    pencere = secenek.window or config.get("evaluation", "plateau_window_rounds")
    yalniz = {a for a in secenek.only.split(",") if a}
    sonuc = topla(pencere, yalniz)

    print(f"Son {pencere} turun ortalaması, tur 0 hariç. Eşik doğrulama "
          f"yarısında seçilip test yarısında uygulanmıştır.\n")
    baslik = (f"{'deney':28s} {'n':>2} {'kesinlik':>9} {'duyarlılık':>11} "
              f"{'F1':>7} {'dengeli':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(baslik)
    print("-" * len(baslik))

    def anahtar(ad):
        return -statistics.mean(sonuc[ad]["Max-F1"]) if sonuc[ad]["Max-F1"] else 0

    for ad in sorted(sonuc, key=anahtar):
        k = sonuc[ad]
        if not k["Max-F1"]:
            continue
        ort = {s: statistics.mean(v) for s, v in k.items() if v}
        sapma = (statistics.stdev(k["Max-F1"]) if len(k["Max-F1"]) > 1 else 0.0)
        print(f"{ad:28s} {len(k['Max-F1']):>2} {ort['Precision']:>9.4f} "
              f"{ort['Recall']:>11.4f} {ort['Max-F1']:>7.4f} "
              f"{ort['Balanced_Accuracy']:>8.4f} {ort['TP']:>5.1f} "
              f"{ort['FP']:>5.1f} {ort['FN']:>5.1f}   sd(F1)={sapma:.4f}")


if __name__ == "__main__":
    main()
