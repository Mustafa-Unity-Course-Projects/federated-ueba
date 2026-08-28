# -*- coding: utf-8 -*-
"""Every reported configuration on one page.

The results chapter is organised by axis: each table shows a few arms with the
columns that axis needs. That is the right shape for an argument and the wrong
shape for a lookup. Three gaps follow from it and this table closes all three.

  rounds-25-epochs-10   appears in no table at all; its PR-AUC and its saving
                        live in prose only.
  no-bottleneck         is a 429,362 parameter model and encoder-unidirectional
  encoder-unidirectional  is a 515,794 parameter one, so their per-round payload
                        is not the baseline's. No table says so.
  the heterogeneity and architecture arms carry no communication column at all,
                        which is defensible only once it is stated that they run
                        the baseline's transport.

Parameter counts are read from each run's own final checkpoint rather than
rebuilt from configuration, so the number is the model that actually trained.
Everything else comes from the same helpers the axis tables use, so a row here
cannot disagree with the table it duplicates.

    python analysis/ana_cizelge.py            markdown
    python analysis/ana_cizelge.py --csv      dosyaya
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import numpy as np

import build_tables as bt

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


# (deney, eksen, etiket). Sıra Çizelge 5.1'i izler.
SIRA = [
    ("baseline",                 "Referans",     "baseline"),
    ("quantization-fp16",        "Sıkıştırma",   "quantization-fp16"),
    ("bidirectional-fp16",       "Sıkıştırma",   "bidirectional-fp16"),
    ("top-k-0.1",                "Sıkıştırma",   "top-k-0.1"),
    ("top-k-0.05",               "Sıkıştırma",   "top-k-0.05"),
    ("top-k-0.1-quant-fp16",     "Sıkıştırma",   "top-k-0.1-quant-fp16"),
    ("delta-0.1",                "Sıkıştırma",   "delta-0.1"),
    ("delta-0.05",               "Sıkıştırma",   "delta-0.05"),
    ("delta-0.05-downlink-fp16", "Sıkıştırma",   "delta-0.05-downlink-fp16"),
    ("no-bottleneck",            "Mimari",       "no-bottleneck"),
    ("encoder-unidirectional",   "Mimari",       "encoder-unidirectional"),
    ("encoder-unidirectional-64", "Mimari",      "encoder-unidirectional-64"),
    ("features-filtered",        "Mimari",       "features-filtered"),
    ("nodes-20",                 "Ölçek",        "nodes-20"),
    ("nodes-10",                 "Ölçek",        "nodes-10"),
    ("rounds-25-epochs-10",      "Eğitim",       "rounds-25-epochs-10"),
    ("non-iid-baseline",         "Heterojenlik", "non-iid-baseline"),
    ("role-non-iid-baseline",    "Heterojenlik", "role-non-iid-baseline"),
    ("fedprox-baseline",         "Heterojenlik", "fedprox-baseline"),
    ("fedprox-non-iid",          "Heterojenlik", "fedprox-non-iid"),
    ("role-fedprox-non-iid",     "Heterojenlik", "role-fedprox-non-iid"),
    ("ablation-full",            "Skorlama",     "ablation-full"),
    ("ablation-no-zscore",       "Skorlama",     "ablation-no-zscore"),
    ("ablation-no-topk",         "Skorlama",     "ablation-no-topk"),
    ("ablation-no-persistence",  "Skorlama",     "ablation-no-persistence"),
    ("ablation-with-diversity",  "Skorlama",     "ablation-with-diversity"),
]

# Bu kollar eğitim yapmaz; baseline'ın ağırlıklarını yeniden puanlarlar. Kendi
# checkpoint'leri ve kendi iletişim sayaçları yoktur, dolayısıyla parametre ve
# MB sütunları baseline'ınkidir ve tekrarlanmaz.
YENIDEN_PUANLAYAN = {"ablation-full", "ablation-no-zscore", "ablation-no-topk",
                     "ablation-no-persistence", "ablation-with-diversity"}


def parametre_sayisi(kol):
    """Koşumun kendi son checkpoint'inden okunan eğitilebilir parametre sayısı."""
    return int(sum(np.asarray(a).size for a in bt._checkpoint(kol, 1)))


def satirlar():
    d = bt._ozet()
    temel_mb = bt._ort(d, "baseline", "Total_Comm_MB")
    cikti = []
    for ad, eksen, etiket in SIRA:
        pr, sd = bt.tr(bt._pr(ad)), bt.tr(bt._prsd(ad))
        if ad in YENIDEN_PUANLAYAN:
            cikti.append([eksen, etiket, pr, sd, "baseline ile aynı",
                          "-", "-", "-", "-"])
            continue
        toplam = bt._ort(d, ad, "Total_Comm_MB")
        cikti.append([
            eksen, etiket, pr, sd,
            f"{parametre_sayisi(ad):,}".replace(",", "."),
            bt.tr(bt._ort(d, ad, "Upload_MB"), 1),
            bt.tr(bt._ort(d, ad, "Download_MB"), 1),
            bt.tr(toplam, 1),
            "referans" if ad == "baseline" else bt.yuzde(100 * (1 - toplam / temel_mb)),
        ])
    return cikti


BASLIKLAR = ["Eksen", "Konfigürasyon", "PR-AUC", "sd", "Parametre",
             "Yükleme (MB)", "İndirme (MB)", "Toplam (MB)", "Tasarruf"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="?", const="ana_cizelge.csv", default=None)
    args = ap.parse_args()

    veri = satirlar()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            y = csv.writer(f, delimiter=";")
            y.writerow(BASLIKLAR)
            y.writerows(veri)
        print(f"yazıldı: {args.csv}")
        return

    genis = [max(len(str(r[i])) for r in [BASLIKLAR] + veri)
             for i in range(len(BASLIKLAR))]
    def yaz(r):
        print("| " + " | ".join(str(c).ljust(genis[i])
                                for i, c in enumerate(r)) + " |")
    yaz(BASLIKLAR)
    print("|" + "|".join("-" * (g + 2) for g in genis) + "|")
    for r in veri:
        yaz(r)


if __name__ == "__main__":
    main()
