"""Trafiği megabayt yerine parametre sayısıyla ölçer.

Tezin iletişim ekseni baştan sona MB raporluyor, çünkü iddia bant genişliği
üzerine. Parametre sayısı bunun yerini tutmaz ama yanına konduğunda **iki
kayıplı katmanı birbirinden ayırır**, ki MB tek başına ayıramaz:

    seyrekleştirme   daha az değer gönderir     -> parametre sayısı düşer
    nicemleme        aynı değerleri az bitle    -> parametre sayısı sabit kalır

Yani fp16 kolları burada %0 tasarruf gösterir. Bu bir eksiklik değil, ölçünün
tanımı; fp16'nın kazancı zaten bit genişliğinde ve onu MB gösteriyor.

İki tanım birden raporlanıyor, çünkü indirme yönünde ayrışıyorlar:

    yerleştirilen   telin taşıdığı dizinin uzunluğu. İndirme her zaman yoğun
                    olduğundan bu daima N = 450.258
    sıfır dışı      o dizideki gerçekten sıfır olmayan değer sayısı. Ağırlık
                    seyrekleştirmesi altında yayımlanan küresel model gerçekten
                    seyrekleşir, indirme MB'si de bu yüzden düşer

Yükleme sayıları formülden değil, seyrekleştiricinin kendi kuralı tensör tensör
uygulanarak çıkarılıyor; `_topk_mask` tensör başına
`k = min(max(1, int(n * oran)), n)` tutuyor. Bu yüzden gerçekleşen oran nominal
oranın biraz altında kalır: `int()` aşağı yuvarlıyor.

Mesaj sayıları da varsayılmıyor, koşumun kendi iletişim loglarından sayılıyor.

    python analysis/param_traffic.py
    python analysis/param_traffic.py --seed 2
"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Çizelge 5.3'ün sırası; okuyucu iki çizelgeyi yan yana koyabilsin diye aynı.
KOLLAR = ["baseline", "quantization-fp16", "bidirectional-fp16",
          "delta-0.1", "delta-0.05", "delta-0.05-downlink-fp16",
          "top-k-0.1", "top-k-0.1-quant-fp16", "top-k-0.05"]


def tr(x, basamak=1):
    """Türkçe ondalık ayracı; tez baştan sona virgül kullanıyor."""
    return f"{x:,.{basamak}f}".replace(",", " ").replace(".", ",")


def sayi(n):
    """Binlik ayraç olarak nokta, tezdeki gibi."""
    return f"{n:,}".replace(",", ".")


def yuzde(x, basamak=1):
    return "%" + tr(x, basamak)


def checkpoint(kol, seed):
    """Yayımlanan küresel modelin son turu, numpy dizileri olarak.

    Torch gerekmiyor: checkpoint'ler numpy dizisi listesi olarak saklanıyor.
    """
    kalip = os.path.join(KOK, "model_pickle", f"{kol}__seed{seed}",
                         "parameters_round_*.pkl")
    yollar = glob.glob(kalip)
    if not yollar:
        return None

    def no(p):
        m = re.search(r"parameters_round_(\d+)", p)
        return int(m.group(1)) if m else -1

    with open(max(yollar, key=no), "rb") as f:
        icerik = pickle.load(f)
    return icerik["global_parameters"] if isinstance(icerik, dict) else icerik


def sekiller(seed):
    """Modelin tensör tensör eleman sayıları, gerçek bir checkpoint'ten.

    baseline seçiliyor çünkü seyrekleştiren bir koşumun checkpoint'i de aynı
    şekle sahip olsa da, şekli yoğun bir koldan almak niyeti açık bırakıyor.
    """
    w = checkpoint("baseline", seed)
    if w is None:
        raise SystemExit("baseline checkpoint'i bulunamadı.")
    return [int(a.size) for a in w]


def seyreltme_orani(kol):
    """Kolun adından seyrekleştirme oranı. Nicemleme kolları sayı değiştirmez."""
    m = re.search(r"(?:top-k|delta)-(\d*\.?\d+)", kol)
    return float(m.group(1)) if m else None


def yuklenen(boyutlar, oran):
    """Bir yükleme mesajının taşıdığı sıfır dışı değer sayısı.

    Seyrekleştirme tensör tensör uygulanıyor, küresel bir sıralama üzerinden
    değil; küçük tensörlerin tamamen susmasını engelleyen taban da orada.
    """
    if oran is None:
        return sum(boyutlar)
    return sum(min(max(1, int(n * oran)), n) for n in boyutlar)


def mesaj_sayilari(kol, seed):
    """Koşumun kendi loglarından yön başına mesaj sayısı ve MB."""
    d = os.path.join(KOK, "federated_evaluation_reports",
                     f"{kol}__seed{seed}", "comm")
    dosyalar = glob.glob(os.path.join(d, "client_*.csv"))
    if not dosyalar:
        return None
    df = pd.concat(pd.read_csv(f, header=None, names=["yon", "faz", "mb"])
                   for f in dosyalar)
    return {
        "yukleme_mesaj": int((df.yon == "upload").sum()),
        "indirme_mesaj": int((df.yon == "download").sum()),
        "yukleme_mb": float(df.mb[df.yon == "upload"].sum()),
        "indirme_mb": float(df.mb[df.yon == "download"].sum()),
    }


def topla(seed):
    boyutlar = sekiller(seed)
    N = sum(boyutlar)
    satirlar = []
    for kol in KOLLAR:
        log = mesaj_sayilari(kol, seed)
        w = checkpoint(kol, seed)
        if log is None or w is None:
            print(f"  atlandı (veri yok): {kol}", file=sys.stderr)
            continue
        duz = np.concatenate([a.ravel() for a in w])
        indirme_sifirdisi = int((duz != 0).sum())

        yuk_bir = yuklenen(boyutlar, seyreltme_orani(kol))
        yuk = yuk_bir * log["yukleme_mesaj"]
        # İndirme her zaman yoğun bir dizi: yerleştirilen N, sıfır dışı daha az.
        ind_yerlesen = N * log["indirme_mesaj"]
        ind_sifirdisi = indirme_sifirdisi * log["indirme_mesaj"]

        satirlar.append({
            "kol": kol,
            "yuk_mesaj_basi": yuk_bir,
            "yuk_oran": yuk_bir / N,
            "ind_sifirdisi_basi": indirme_sifirdisi,
            "ind_oran": indirme_sifirdisi / N,
            "toplam_yerlesen": yuk + ind_yerlesen,
            "toplam_sifirdisi": yuk + ind_sifirdisi,
            "toplam_mb": log["yukleme_mb"] + log["indirme_mb"],
        })
    return N, satirlar


def yaz(N, tensor_sayisi, satirlar):
    if not satirlar:
        raise SystemExit("Hiçbir kol okunamadı.")
    temel = satirlar[0]
    print(f"Model: {sayi(N)} parametre, {tensor_sayisi} tensör\n")

    bas = ["Kol", "Yükleme (par/mesaj)", "İndirme sıfır dışı (par/mesaj)",
           "Toplam yerleşen", "Toplam sıfır dışı", "Par. tasarrufu",
           "MB tasarrufu"]
    print("\t".join(bas))
    for s in satirlar:
        ilk = s is temel
        print("\t".join([
            s["kol"],
            f'{sayi(s["yuk_mesaj_basi"])} ({yuzde(100*s["yuk_oran"])})',
            f'{sayi(s["ind_sifirdisi_basi"])} ({yuzde(100*s["ind_oran"])})',
            sayi(s["toplam_yerlesen"]),
            sayi(s["toplam_sifirdisi"]),
            "referans" if ilk else yuzde(
                100 * (1 - s["toplam_sifirdisi"] / temel["toplam_sifirdisi"])),
            "referans" if ilk else yuzde(
                100 * (1 - s["toplam_mb"] / temel["toplam_mb"])),
        ]))
    print("\nSon sütun tek seed'in ölçümüdür; Çizelge 5.3 beş seed ortalaması")
    print("raporladığı için ondalık hanede ayrışır. Karşılaştırılacak olan")
    print("büyüklük sırası, birebir değer değil.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    boyutlar = sekiller(a.seed)
    N, satirlar = topla(a.seed)
    yaz(N, len(boyutlar), satirlar)
