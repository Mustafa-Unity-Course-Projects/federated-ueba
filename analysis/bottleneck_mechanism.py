"""Darboğazın kaldırılması anomali sinyalini gerçekten çökertiyor mu?

Darboğaz ablasyonu tezin en güçlü mimari sonucu: tespit 0,8371'den 0,6934'e
düşüyor. Bunun standart açıklaması, darboğazsız bir otokodlayıcının girdiyi
kodlamak yerine taşımayı öğrendiği, yeniden kurma hatasının herkeste düştüğü ve
sinyalin bu yüzden düzleştiğidir. Açıklamanın ilk yarısı ölçülebilir, ikincisi
de öyle; bu betik ikisini birden ölçüyor.

Ölçülen üç büyüklük, kaydedilmiş 50. tur checkpoint'leri üzerinden:

    ortalama yeniden kurma hatası   düşüyor mu
    içerideki / diğeri oranı        sinyal sıkışıyor mu
    ham ortalama hatanın PR-AUC'si  skorlama hattı olmadan ayrım kalıyor mu

Üçüncüsü belirleyici olan: skorlama hattı devreye girmeden önce sinyalin
durumunu gösteriyor, dolayısıyla kaybın ağda mı yoksa hatta mı olduğunu ayırıyor.

Hiçbir sonuç dosyasına yazmaz, yalnız okur.

    python analysis/bottleneck_mechanism.py
    python analysis/bottleneck_mechanism.py --seeds 1,2,3 --others 200
"""

import argparse
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("FUEBA_EXPERIMENT", "baseline")

from config_manager import config  # noqa: E402
from federated_ueba import scaling  # noqa: E402
from federated_ueba.task import (  # noqa: E402
    Architecture, LSTMAutoencoder, build_windows, select_features)

KOLLAR = ("baseline", "no-bottleneck")


def checkpoint_yukle(kol, seed, tur, arch, girdi_boyutu):
    """Kaydedilmiş turun ağırlıklarını mimariye giydirir."""
    yol = os.path.join("model_pickle", f"{kol}__seed{seed}",
                       f"parameters_round_{tur}.pkl")
    with open(yol, "rb") as f:
        icerik = pickle.load(f)
    diziler = icerik["global_parameters"] if isinstance(icerik, dict) else icerik

    model = LSTMAutoencoder(girdi_boyutu, arch=arch)
    durum = model.state_dict()
    if len(diziler) != len(durum):
        sys.exit(f"{yol}: {len(diziler)} dizi, model {len(durum)} bekliyor. "
                 f"Mimari yapılandırması bu kolla eşleşmiyor.")
    for anahtar, dizi in zip(durum.keys(), diziler):
        durum[anahtar] = torch.tensor(
            np.asarray(dizi, dtype=np.float32)).reshape(durum[anahtar].shape)
    model.load_state_dict(durum)
    model.eval()
    return model


def kullanici_pencereleri(frame, ozellikler, olcekleyici):
    """Kullanıcı başına pencere tensörü. Ölçekleyici koşumun kendisininki."""
    cikti = {}
    for kullanici, grup in frame.groupby("user"):
        pencereler = build_windows(
            olcekleyici.transform(scaling.prepare_features(grup, ozellikler)))
        if len(pencereler):
            cikti[kullanici] = torch.tensor(np.array(pencereler),
                                            dtype=torch.float32)
    return cikti


def kol_olc(model, pencereler, icerideki):
    """Kullanıcı başına ortalama yeniden kurma hatası ve etiketi."""
    skorlar, etiketler = [], []
    for kullanici, x in pencereler.items():
        with torch.no_grad():
            hata = torch.mean((model(x) - x) ** 2, dim=(1, 2)).numpy()
        skorlar.append(float(np.mean(hata)))
        etiketler.append(int(icerideki[kullanici] > 0))
    return np.array(skorlar), np.array(etiketler)


def main():
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--seeds", default="1,2")
    ayristirici.add_argument("--round", type=int, default=50)
    ayristirici.add_argument("--others", type=int, default=200,
                             help="karşılaştırmaya alınan içeriden olmayan "
                                  "kullanıcı sayısı")
    args = ayristirici.parse_args()

    frame = pd.read_csv(config.get("data", "processed_data_path"))
    ozellikler = select_features(frame)
    icerideki = frame.groupby("user")["insider"].max()

    secilen = (list(icerideki[icerideki > 0].index)
               + list(icerideki[icerideki == 0].index[:args.others]))
    alt = frame[frame["user"].isin(secilen)]

    taban = Architecture.from_config(config)
    mimariler = {"baseline": taban,
                 "no-bottleneck": replace(taban, use_bottleneck=False)}

    print(f"{len(ozellikler)} öznitelik, {icerideki.gt(0).sum()} içeriden aktör, "
          f"{args.others} diğer kullanıcı, tur {args.round}")

    for seed in [int(s) for s in args.seeds.split(",")]:
        with open(os.path.join("scaler_data", f"baseline__seed{seed}",
                               "global_scaler.pkl"), "rb") as f:
            olcekleyici = pickle.load(f)
        pencereler = kullanici_pencereleri(alt, ozellikler, olcekleyici)

        print(f"\nseed {seed}, {len(pencereler)} kullanıcı")
        for kol in KOLLAR:
            model = checkpoint_yukle(kol, seed, args.round, mimariler[kol],
                                     len(ozellikler))
            skor, etiket = kol_olc(model, pencereler, icerideki)
            oran = skor[etiket == 1].mean() / skor[etiket == 0].mean()
            print(f"  {kol:14s} ortalama hata {skor.mean():.4f}   "
                  f"içerideki/diğeri {oran:.3f}   "
                  f"ham PR-AUC {average_precision_score(etiket, skor):.4f}")


if __name__ == "__main__":
    main()
