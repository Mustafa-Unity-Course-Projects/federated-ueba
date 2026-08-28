# -*- coding: utf-8 -*-
"""Karışıklık matrisi, birden çok kırılım ekseni için.

`rol_karisiklik_matrisi.py` tek bir soruyu soruyor: yanlış alarmlar hangi iş
rolüne düşüyor. Cevabı çarpıcı olduğu için doğal devamı, aynı sorunun başka
kırılımlarda da sorulmasıdır. İki sebeple:

  doğrulama    rol yanlılığının gerçekten role mi bağlı olduğu, yoksa rolle
               birlikte değişen başka bir şeye mi (etkinlik hacmi, veri
               miktarı) bağlı olduğu ancak o eksenler ayrı ayrı ölçülünce
               ayrılabilir
  kapsam       yanlılığın tek bir eksene özgü olup olmadığı, ancak organizasyon,
               kişilik ve hacim eksenleri birlikte bakılınca görülebilir

Eşik kuralı bütün eksenlerde aynıdır ve raporlanan sayıların kuralıyla birebir
örtüşür: eşik doğrulama yarısında seçilir, ölçüm test yarısında yapılır, son
`plateau_window_rounds` tur ve beş seed ortalanır. Her eksende satırlar
toplandığında Çizelge 5.10'un TP, FP ve FN değerlerini verir; bu, her tablonun
kendi içinde taşıdığı sağlamadır.

Kategorik eksenlerin sayısal kodları, `feature_extraction.get_u_features_dicts`
ile aynı kuralla (LDAP'taki benzersiz değerler sıralanıp indekslenerek) ada
çevrilir. Eşleme her eksende ayrıca sınanır: koda göre grup büyüklükleri ile
ada göre grup büyüklükleri sırayla eşleşmek zorundadır.

Hiçbir sonuç dizinine yazmaz; çıktı --out-dir altına gider.

    python analysis/eksen_karisiklik_matrisleri.py
    python analysis/eksen_karisiklik_matrisleri.py --kollar baseline,non-iid-baseline
"""

import argparse
import glob
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_manager import config  # noqa: E402
from federated_ueba import scoring  # noqa: E402

import rol_karisiklik_matrisi as rk  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# İşlenmiş veri sütunu -> LDAP sütunu. `b_unit` bilerek yok: bütün kurumda tek
# değer taşıyor, dolayısıyla bir kırılım ekseni değil.
KATEGORIK = {
    "rol": ("role", "role"),
    "departman": ("dept", "department"),
    "fonksiyonel_birim": ("f_unit", "functional_unit"),
    "takim": ("team", "team"),
}
KISILIK = {"O": "aciklik", "C": "sorumluluk", "E": "disadonukluk",
           "A": "uyumluluk", "N": "norotiklik"}
HACIM = {"etkinlik_ort": "etkinlik_hacmi", "gun": "kayit_gun_sayisi",
         "eposta": "eposta_hacmi", "web": "web_hacmi"}
KOLLAR = ["baseline", "non-iid-baseline", "role-non-iid-baseline",
          "fedprox-baseline", "quantization-fp16", "top-k-0.05", "nodes-10",
          "no-bottleneck"]


def ad_haritalari():
    """Her kategorik alan için sayısal koddan ada eşleme."""
    kayitlar = [pd.read_csv(y) for y in sorted(glob.glob(rk.LDAP_DESENI))]
    if not kayitlar:
        raise SystemExit("LDAP dizini bulunamadı; kategorik eksenler kurulamaz.")
    ldap = pd.concat(kayitlar, ignore_index=True)
    out = {}
    for eksen, (_, ldap_sutun) in KATEGORIK.items():
        # `get_u_features_dicts` boş değeri UNKNOWN'a çevirip sonra sıralıyor.
        degerler = ldap[ldap_sutun].fillna("UNKNOWN").astype(str)
        out[eksen] = {i: ad for i, ad in enumerate(sorted(set(degerler)))}
    return out, ldap


def dilim(seri, etiket):
    """Sürekli bir değişkeni dörtte birlik dilimlere ayırır.

    Kenarlar `qcut` ile veriden alınıyor; eşit genişlikli kutular bu
    değişkenlerin çoğunda bir dilimi boş bırakırdı. Bağlar yüzünden dört dilim
    çıkmazsa çıkan sayı kadar dilim adlandırılır.
    """
    try:
        kesim = pd.qcut(seri, 4, duplicates="drop")
    except ValueError:
        return pd.Series(f"{etiket}: tek dilim", index=seri.index)
    sinir = list(kesim.cat.categories)
    adlar = {}
    for i, aralik in enumerate(sinir, start=1):
        adlar[aralik] = (f"{etiket} Q{i} [{aralik.left:.4g}, "
                         f"{aralik.right:.4g}]")
    return kesim.map(adlar).astype(str)


def etkinlik_onbellegi(yol):
    """Kullanıcı başına ham etkinlik hacmi; yoksa bir kez üretilir.

    Yüzdelik dönüşümünden geçmiş veride hacim okunamaz, çünkü her değer zaten
    kullanıcının kendi geçmişine göre ifade edilmiştir. Ham günlük tablo 550 MB
    olduğundan sonuç küçük bir dosyada saklanıyor.
    """
    if os.path.exists(yol):
        return pd.read_csv(yol).set_index("user")
    ham_yol = os.path.join("ExtractedData", "dayr4.2.csv")
    if not os.path.exists(ham_yol):
        raise SystemExit(f"{ham_yol} yok; hacim eksenleri kurulamaz.")
    print(f"etkinlik önbelleği üretiliyor: {ham_yol}")
    ham = pd.read_csv(ham_yol, low_memory=False,
                      usecols=["user", "n_allact", "n_file", "n_email",
                               "n_http"])
    kisi = ham.groupby("user").agg(gun=("n_allact", "size"),
                                   etkinlik_ort=("n_allact", "mean"),
                                   eposta=("n_email", "mean"),
                                   web=("n_http", "mean")).reset_index()
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    kisi.to_csv(yol, index=False)
    return kisi.set_index("user")


def kimlikler(adlar, onbellek):
    """Kullanıcı başına bütün eksen değerleri, tek bir tabloda."""
    yol = config.get("data", "processed_data_path")
    sutunlar = (["user", "ITAdmin", "insider", "day"]
                + [p for p, _ in KATEGORIK.values()] + list(KISILIK))
    ham = pd.read_csv(yol, usecols=sutunlar, low_memory=False)

    kisi = ham.groupby("user").agg(
        **{p: (p, "first") for p, _ in KATEGORIK.values()},
        **{p: (p, "first") for p in KISILIK},
        itadmin=("ITAdmin", "max"), insider=("insider", "max"),
        puanlanabilir_gun=("day", "nunique"))

    for eksen, (islenmis, _) in KATEGORIK.items():
        degisken = ham.groupby("user")[islenmis].nunique()
        if int((degisken > 1).sum()):
            print(f"UYARI: {int((degisken > 1).sum())} kullanıcının {eksen} "
                  "değeri zaman içinde değişiyor; ilk değer alındı.")
        kisi[eksen] = kisi[islenmis].astype(int).map(adlar[eksen])
        # Sağlama: koda göre ve ada göre grup büyüklükleri sırayla eşleşmeli.
        koda_gore = kisi[islenmis].astype(int).value_counts().sort_index()
        ada_gore = kisi[eksen].value_counts()
        beklenen = [ada_gore.get(adlar[eksen][k], 0) for k in koda_gore.index]
        if list(koda_gore.values) != beklenen or kisi[eksen].isna().any():
            raise SystemExit(f"{eksen} kod-ad eşlemesi tutmuyor.")

    bayrakli = set(kisi.index[kisi["itadmin"] == 1])
    adli = set(kisi.index[kisi["rol"] == "ITAdmin"])
    if bayrakli != adli:
        raise SystemExit("Rol eşlemesi ITAdmin bayrağıyla doğrulanamadı.")

    kisi["etiket"] = (kisi["insider"] > 0).astype(int)
    kisi["senaryo"] = kisi["insider"].astype(int).map(
        {0: "normal", 1: "senaryo 1", 2: "senaryo 2", 3: "senaryo 3"})
    kisi["yonetici"] = np.where(kisi["itadmin"] == 1, "sistem yöneticisi",
                                "diğer roller")
    for kod, ad in KISILIK.items():
        kisi[ad] = dilim(kisi[kod], ad)

    hacim = etkinlik_onbellegi(onbellek)
    ortak = kisi.index.intersection(hacim.index)
    if len(ortak) != len(kisi):
        raise SystemExit("Etkinlik önbelleği kullanıcı kümesiyle örtüşmüyor.")
    for kaynak, ad in HACIM.items():
        kisi[ad] = dilim(hacim.loc[kisi.index, kaynak], ad)
    kisi["puanlanabilir_gun_dilimi"] = dilim(kisi["puanlanabilir_gun"],
                                             "puanlanabilir gun")

    # Ayırt edici kırılım. Rol ekseni ile hacim ekseni birlikte hareket ettiği
    # için ikisi tek başına bakıldığında ayrılamaz: yöneticiler hacim
    # bakımından da en üst dilimdedir. Aynı hacim diliminin içinde yönetici
    # olan ve olmayan kullanıcıları karşılaştırmak, yanlılığın hangisine
    # bağlandığını gösteren tek ölçümdür.
    for kaynak, ad in (("etkinlik_hacmi", "yonetici_x_etkinlik"),
                       ("web_hacmi", "yonetici_x_web")):
        dilim_kisa = kisi[kaynak].str.extract(r"(Q\d)")[0]
        kisi[ad] = kisi["yonetici"].str.replace("sistem yöneticisi", "yönetici",
                                                regex=False) + " / " + dilim_kisa
    return kisi


def eksen_listesi():
    return (list(KATEGORIK) + ["senaryo", "yonetici"] + list(KISILIK.values())
            + list(HACIM.values()) + ["puanlanabilir_gun_dilimi",
                                      "yonetici_x_etkinlik", "yonetici_x_web"])


def tek_gecis(experiment, seedler, kisi, eksenler, pencere, split_seed,
              val_frac):
    """Bütün eksenleri tek turda toplar.

    Her tur dosyası bir kez okunuyor; eksen başına ayrı geçiş yapmak aynı eşiği
    onlarca kez yeniden hesaplamak olurdu.
    """
    birikim = {(e, k): [] for e in eksenler for k in ("test", "tum")}
    esikler, sayac = [], 0
    for seed in seedler:
        yollar = rk.plato_turlari(f"{experiment}__seed{seed}", pencere)
        if not yollar:
            print(f"  atlandı: {experiment}__seed{seed}")
            continue
        for yol in yollar:
            tam, esik, _ = rk.bir_tur(yol, kisi, split_seed, val_frac,
                                      yarim=False)
            tam = tam.sort_values("user")
            # `evaluate_scores` bölmeyi tam olarak bu sırayla kuruyor; eşiği
            # üreten yarı ile burada ayrılan yarı aynı olmak zorunda.
            _, test_kul = scoring.split_users(
                tam["user"].tolist(), tam["etiket"].tolist(),
                seed=split_seed, validation_fraction=val_frac)
            yari = tam[tam["user"].isin(test_kul)]
            for e in eksenler:
                birikim[(e, "tum")].append(rk.hucreler(tam, e))
                birikim[(e, "test")].append(rk.hucreler(yari, e))
            esikler.append(esik)
            sayac += 1
        print(f"  {experiment}__seed{seed}: {len(yollar)} tur")
    if not sayac:
        return None, 0, []
    sonuc = {anahtar: rk.turet(pd.concat(y).groupby(level=0).sum() / sayac)
             for anahtar, y in birikim.items()}
    return sonuc, sayac, esikler


def yaz_eksen(eksen, sonuc, sayac, dizin):
    parcalar = {}
    for kume, etiket in (("test", "test_yarisi"), ("tum", "tum_populasyon")):
        parcalar[etiket] = sonuc[(eksen, kume)]
    frame = pd.concat(parcalar, names=["kume", "grup"])
    frame.insert(0, "olcum", sayac)
    yol = os.path.join(dizin, f"eksen_{eksen}.csv")
    frame.round(4).to_csv(yol, sep=";", encoding="utf-8-sig")
    return frame


def ozet_satirlari(eksen, frame):
    """Bir eksenin en yüksek ve en düşük yanlış alarm taşıyan grupları."""
    tum = frame.loc["tum_populasyon"]
    gecerli = tum[tum.negatif > 0].sort_values("FPR_%", ascending=False)
    if gecerli.empty:
        return []
    ust, alt = gecerli.iloc[0], gecerli.iloc[-1]
    yayilim = (ust["FPR_%"] / alt["FPR_%"]) if alt["FPR_%"] > 0 else np.inf
    return [{"eksen": eksen, "grup_sayisi": len(tum),
             "en_yuksek_grup": gecerli.index[0],
             "en_yuksek_n": ust.n, "en_yuksek_FPR_%": round(ust["FPR_%"], 3),
             "en_dusuk_grup": gecerli.index[-1],
             "en_dusuk_n": alt.n, "en_dusuk_FPR_%": round(alt["FPR_%"], 3),
             "yayilim_kat": round(yayilim, 1) if np.isfinite(yayilim) else "sonsuz",
             "toplam_FP": round(tum.FP.sum(), 2)}]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--kollar", default=",".join(KOLLAR),
                    help="Rol ekseninin ayrıca ölçüleceği yapılandırmalar.")
    ap.add_argument("--onbellek", default=os.path.join("ieee",
                                                       "kontrol_ciktilari",
                                                       "etkinlik_hacmi.csv"))
    ap.add_argument("--out-dir", default=os.path.join("ieee",
                                                      "kontrol_ciktilari",
                                                      "eksen_matrisleri"))
    args = ap.parse_args()

    seedler = [int(s) for s in args.seeds.split(",") if s.strip()]
    config.set_seed(seedler[0])
    config.set_experiment(args.experiment)
    cfg = scoring.ScoringConfig.from_config(config)
    pencere = config.get("evaluation", "plateau_window_rounds")
    os.makedirs(args.out_dir, exist_ok=True)

    adlar, _ = ad_haritalari()
    kisi = kimlikler(adlar, args.onbellek)
    eksenler = eksen_listesi()
    print(f"{len(kisi)} kullanıcı, {len(eksenler)} eksen, "
          f"split_seed={cfg.split_seed}, plato={pencere}\n")

    sonuc, sayac, esikler = tek_gecis(args.experiment, seedler, kisi, eksenler,
                                      pencere, cfg.split_seed,
                                      cfg.validation_fraction)
    print(f"\n{sayac} ölçüm; eşik ortalama {np.mean(esikler):.4f}, "
          f"aralık [{min(esikler):.4f}, {max(esikler):.4f}]")

    ozet, okuma = [], []
    for eksen in eksenler:
        frame = yaz_eksen(eksen, sonuc, sayac, args.out_dir)
        ozet += ozet_satirlari(eksen, frame)
        tum = frame.loc["tum_populasyon"].sort_values("FPR_%", ascending=False)
        kolon = ["n", "TP", "FN", "FP", "TN", "FPR_%", "Duyarlilik_%"]
        okuma.append(f"\n{'=' * 78}\nEKSEN: {eksen}   ({len(tum)} grup, "
                     f"bütün popülasyon, {sayac} ölçüm)\n{'=' * 78}\n"
                     + tum[kolon].round(2).to_string())
        genel = frame.loc["test_yarisi"].sum()
        if abs(genel.TP - 32.84) > 0.01 or abs(genel.FP - 2.80) > 0.01:
            raise SystemExit(f"{eksen}: satır toplamları Çizelge 5.10 ile "
                             f"uyuşmuyor (TP={genel.TP:.2f} FP={genel.FP:.2f}).")

    pd.DataFrame(ozet).to_csv(os.path.join(args.out_dir, "ozet.csv"),
                              sep=";", index=False, encoding="utf-8-sig")

    # Rol ekseni bütün kollarda: yanlılık temel koşuma mı özgü, yoksa her
    # yapılandırmada mı var.
    kol_satirlari = []
    for kol in [k for k in args.kollar.split(",") if k.strip()]:
        config.set_experiment(kol)
        kol_sonuc, kol_sayac, _ = tek_gecis(kol, seedler, kisi, ["yonetici"],
                                            pencere, cfg.split_seed,
                                            cfg.validation_fraction)
        if not kol_sayac:
            continue
        t = kol_sonuc[("yonetici", "tum")]
        for grup in t.index:
            s = t.loc[grup]
            kol_satirlari.append({"kol": kol, "olcum": kol_sayac, "grup": grup,
                                  "n": s.n, "TP": s.TP, "FN": s.FN, "FP": s.FP,
                                  "TN": s.TN,
                                  "FPR_%": round(s["FPR_%"], 3),
                                  "Duyarlilik_%": round(s["Duyarlilik_%"], 2)})
    kollar = pd.DataFrame(kol_satirlari)
    kollar.to_csv(os.path.join(args.out_dir, "kollar_yonetici.csv"), sep=";",
                  index=False, encoding="utf-8-sig")

    okuma.append(f"\n{'=' * 78}\nKOLLAR: yönetici ve diğer roller, bütün "
                 f"popülasyon\n{'=' * 78}\n" + kollar.to_string(index=False))
    okuma.append(f"\n{'=' * 78}\nEKSEN ÖZETİ\n{'=' * 78}\n"
                 + pd.DataFrame(ozet).to_string(index=False))
    with io.open(os.path.join(args.out_dir, "okuma.txt"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(okuma))

    print("\n".join(okuma[-2:]))
    print(f"\nyazıldı: {args.out_dir} ({len(eksenler)} eksen + özet + kollar)")


if __name__ == "__main__":
    main()
