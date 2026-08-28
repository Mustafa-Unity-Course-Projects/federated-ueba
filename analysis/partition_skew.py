"""Bölümlemenin ürettiği çarpıklığı ölçer: miktar mı, bileşim mi?

Tez iki heterojenlik biçimi raporluyor ve ikisinin farkı ölçülmeden savunulamaz.
`quantity` istemcinin kaç kullanıcı tuttuğunu çarpıtır, `role` ise hangi
rollerden oluştuğunu. İkincisi rolleri istemcilere tek tek atamaz: her rol için
istemciler üzerinde ayrı bir Dirichlet çekimi yapılır, yani istemci tek bir role
inmez, farklı bir rol karışımı taşır. Bu ayrım metinde yanlış yazılabildiği için
buradan ölçülüyor.

Raporlanan ölçüt, istemcinin rol karışımı ile küresel karışım arasındaki toplam
değişim uzaklığının istemci boyutuyla ağırlıklandırılmış toplamıdır. Bölümleme
fonksiyonları `federated_ueba.task` içindekilerin aynısıdır; buraya
kopyalanmalarının sebebi, ölçümün 1,6 GB'lık işlenmiş tabloyu değil yalnız
kullanıcı ve rol sütunlarını gerektirmesidir.

    python analysis/partition_skew.py
    python analysis/partition_skew.py --seeds 1,2,3 --alpha 0.5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import config  # noqa: E402


def rol_etiketleri(yol):
    """Kullanıcı başına rol kodu, `task._role_labels` ile aynı kural."""
    frame = pd.read_csv(yol, usecols=["user", "role"])
    kullanicilar = sorted(frame["user"].unique())
    roller = frame.groupby("user")["role"].first().reindex(kullanicilar)
    return pd.factorize(roller.fillna(-1))[0]


def miktar_bolumleme(n, k, alpha, seed):
    """`task._dirichlet_partition`: yalnız istemci boyutları çarpıtılır."""
    rng = np.random.RandomState(seed)
    oranlar = rng.dirichlet(np.repeat(alpha, k))
    sayilar = np.maximum(1, np.round(oranlar * n).astype(int))
    fark = n - sayilar.sum()
    if fark > 0:
        sayilar[np.argmin(sayilar)] += fark
    elif fark < 0:
        sayilar[np.argmax(sayilar)] -= abs(fark)
    karisik = rng.permutation(np.arange(n))
    parcalar, bas = [], 0
    for sayi in sayilar:
        parcalar.append(karisik[bas:bas + sayi])
        bas += sayi
    return parcalar


def rol_bolumleme(etiketler, k, alpha, seed):
    """`task._cluster_dirichlet_partition`: her rol için ayrı Dirichlet."""
    rng = np.random.RandomState(seed)
    kullanicilar = np.arange(len(etiketler))
    parcalar = [[] for _ in range(k)]
    for rol in range(int(etiketler.max()) + 1):
        uyeler = rng.permutation(kullanicilar[etiketler == rol])
        if len(uyeler) == 0:
            continue
        oranlar = rng.dirichlet(np.repeat(alpha, k))
        kesikler = (np.cumsum(oranlar) * len(uyeler)).astype(int)[:-1]
        for istemci, parca in enumerate(np.split(uyeler, kesikler)):
            parcalar[istemci].extend(parca)
    # Boş istemci, eğitilmemiş bir modeli ortalamaya sokar; bu heterojenlik
    # değil hatadır. En büyük istemciden bir kullanıcı alınır.
    for istemci, uyeler in enumerate(parcalar):
        if uyeler:
            continue
        veren = max(range(k), key=lambda i: len(parcalar[i]))
        parcalar[istemci].append(parcalar[veren].pop())
    return [np.array(sorted(uyeler)) for uyeler in parcalar]


def olc(parcalar, etiketler, kuresel):
    """Bir bölümlemenin çarpıklığı: TV uzaklığı, rol sayısı, boyut."""
    n = len(etiketler)
    tv, rol_sayilari, boyutlar = 0.0, [], []
    for uyeler in parcalar:
        karisim = np.bincount(etiketler[uyeler], minlength=len(kuresel))
        karisim = karisim / len(uyeler)
        tv += (len(uyeler) / n) * 0.5 * np.abs(karisim - kuresel).sum()
        rol_sayilari.append(len(np.unique(etiketler[uyeler])))
        boyutlar.append(len(uyeler))
    return tv, rol_sayilari, boyutlar


def main():
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--seeds", default="1,2,3,4,5")
    ayristirici.add_argument("--alpha", type=float, default=None)
    ayristirici.add_argument("--clients", type=int, default=None)
    args = ayristirici.parse_args()

    seedler = [int(s) for s in args.seeds.split(",")]
    alpha = args.alpha if args.alpha is not None else config.get(
        "data", "non_iid_alpha")
    # Düğüm sayısı yalnız o ekseni değiştiren kollarda tanımlı; geri kalanı
    # Flower federasyon bloğundan devralıyor. `federated_insider_detection`
    # aynı sırayla okuyor.
    k = args.clients
    if k is None:
        k = (config.get("federation", "num_supernodes")
             or config.get_pyproject("tool", "flwr", "federations",
                                     "local-simulation", "options",
                                     "num-supernodes"))

    etiketler = rol_etiketleri(config.get("data", "processed_data_path"))
    n = len(etiketler)
    kuresel = np.bincount(etiketler) / n
    print(f"{n} kullanici, {int(etiketler.max()) + 1} rol, {k} istemci, "
          f"alpha {alpha}")

    for ad, uret in (("miktar", lambda s: miktar_bolumleme(n, k, alpha, s)),
                     ("role", lambda s: rol_bolumleme(etiketler, k, alpha, s))):
        tvler, roller, boyutlar = [], [], []
        for seed in seedler:
            tv, rs, bs = olc(uret(seed), etiketler, kuresel)
            tvler.append(tv)
            roller.extend(rs)
            boyutlar.extend(bs)
        tekil = sum(1 for r in roller if r == 1)
        print(f"\n{ad}")
        print(f"  TV uzakligi        {np.mean(tvler):.3f} "
              f"({min(tvler):.3f}-{max(tvler):.3f})")
        print(f"  istemci basina rol {np.mean(roller):.1f} "
              f"(en az {min(roller)}, en cok {max(roller)})")
        print(f"  tek rollu istemci  {tekil}/{len(roller)}")
        print(f"  istemci boyutu     {min(boyutlar)}-{max(boyutlar)}")


if __name__ == "__main__":
    main()
