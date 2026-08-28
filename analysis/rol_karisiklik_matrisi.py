# -*- coding: utf-8 -*-
"""Karışıklık matrisi, tek bir tane değil, her iş rolü için ayrı ayrı.

Tezde raporlanan karışıklık matrisi popülasyonun tamamı üzerinden alınıyor ve
tek bir sayı çifti veriyor: şu kadar yakalandı, şu kadar yanlış alarm. O matris
yanlış alarmların **kime** düştüğünü göremez. `erken_isaretleme_kontrolu.py`
işaretlenen normal kullanıcıların ezici çoğunluğunun sistem yöneticisi olduğunu
gösterdi; bu betik aynı soruyu bütün roller için ve tezin kendi eşik kuralıyla
soruyor.

Kural, raporlanan sayıların kuralıyla birebir aynıdır:

  eşik        doğrulama yarısında F1'i azamileştiren değer, `evaluate_scores`
              ile ve `split_seed` (42) ile seçilir
  ölçüm       eşiği görmemiş test yarısında yapılır
  turlar      son `plateau_window_rounds` tur, tur 0 hariç
  seed        beş seed'in her biri ayrı ölçülür, sonra ortalanır

Yani buradaki hücreler Çizelge 5.10'un TP/FP/FN sütunlarıyla toplandığında
uyuşur; tek fark satırların role göre ayrılmış olmasıdır.

Rol kodu işlenmiş veride sayısal geliyor. Ada çevirmek için LDAP dizinindeki
rol adları sıralanıp indeksleniyor, ki `feature_extraction.get_u_features_dicts`
kodları tam olarak böyle üretiyor. Eşleşme ayrıca sağlamadan geçiriliyor:
veride `ITAdmin == 1` olan satırların rol kodu, sıralı listede ITAdmin'in
indeksine eşit olmak zorunda.

Hiçbir sonuç dizinine yazmaz; çıktı --out ile verilen yere gider.

    python analysis/rol_karisiklik_matrisi.py
    python analysis/rol_karisiklik_matrisi.py --experiment baseline --min-users 20
"""

import argparse
import glob
import io
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import config  # noqa: E402
from federated_ueba import scoring  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RAPOR_KOKU = "federated_evaluation_reports"
LDAP_DESENI = os.path.join("dataset", "r4.2", "LDAP", "*.csv")


def rol_adlari():
    """Sayısal rol kodundan ada, çıkarım hattının kendi kuralıyla.

    `get_u_features_dicts` her kategorik alanın benzersiz değerlerini sıralayıp
    sıra numarasını kod olarak veriyor. Aynısı burada yeniden kuruluyor.
    """
    roller = set()
    for yol in sorted(glob.glob(LDAP_DESENI)):
        roller |= set(pd.read_csv(yol)["role"].astype(str))
    return {i: ad for i, ad in enumerate(sorted(roller))}


def kullanici_kimlikleri(adlar):
    """Kullanıcı başına tek satır: rol kodu, rol adı, içeriden mi.

    Rol zaman içinde değişmiyor (veride kullanıcı başına tek rol var), bu yüzden
    ilk değeri almak yeterli; yine de değişen bir kullanıcı çıkarsa uyarılır.
    """
    yol = config.get("data", "processed_data_path")
    ham = pd.read_csv(yol, usecols=["user", "role", "ITAdmin", "insider"],
                      low_memory=False)
    kisi = ham.groupby("user").agg(rol_kodu=("role", "first"),
                                   itadmin=("ITAdmin", "max"),
                                   insider=("insider", "max"),
                                   rol_sayisi=("role", "nunique"))
    # `insider` sütunu senaryo numarasını taşıyor: 0 normal, 1/2/3 senaryo.
    kisi["senaryo"] = kisi["insider"].astype(int).map(
        {0: "normal", 1: "senaryo 1", 2: "senaryo 2", 3: "senaryo 3"})
    degisen = int((kisi["rol_sayisi"] > 1).sum())
    if degisen:
        print(f"UYARI: {degisen} kullanıcının rolü zaman içinde değişiyor; "
              "ilk değer alındı.")
    kisi["rol"] = kisi["rol_kodu"].astype(int).map(adlar)
    kisi["etiket"] = (kisi["insider"] > 0).astype(int)

    # Sağlama: ITAdmin bayrağı ile ada çevrilen rol aynı kişileri göstermeli.
    bayrakli = set(kisi.index[kisi["itadmin"] == 1])
    adli = set(kisi.index[kisi["rol"] == "ITAdmin"])
    if bayrakli != adli:
        raise SystemExit(
            "Rol kodu eşlemesi tutmuyor: ITAdmin bayrağı taşıyan "
            f"{len(bayrakli)} kullanıcı ile ada göre ITAdmin çıkan "
            f"{len(adli)} kullanıcı aynı küme değil. LDAP dizini eksik olabilir.")
    return kisi[["rol", "senaryo", "etiket"]]


def plato_turlari(dizin, pencere):
    """Raporlanan istatistiğin ortaladığı turların dosya yolları."""
    kok = os.path.join(RAPOR_KOKU, dizin, "round_by_round_results")
    bulunan = {}
    for yol in glob.glob(os.path.join(kok, "round_*_results.csv")):
        m = re.search(r"round_(\d+)_results\.csv$", os.path.basename(yol))
        if m:
            bulunan[int(m.group(1))] = yol
    if not bulunan:
        return []
    enson = max(bulunan)
    esik = enson - pencere
    return [bulunan[t] for t in sorted(bulunan) if 0 < t and t > esik]


def bir_tur(yol, kisi, split_seed, val_frac, yarim=True):
    """Bir turun kullanıcı başına tahminleri, rolle birlikte.

    `yarim` doğruyken ölçüm eşiği görmemiş test yarısında yapılır; tezin
    raporladığı kural budur. Yanlışken bütün popülasyon döner ve sayılar
    betimleyicidir, çünkü doğrulama yarısı eşiğin seçildiği yarıdır.
    """
    skor = pd.read_csv(yol)
    skor = skor.sort_values("user").reset_index(drop=True)
    olcum = scoring.evaluate_scores(skor, seed=split_seed,
                                    validation_fraction=val_frac)
    esik = olcum["threshold"]

    if yarim:
        _, test_kul = scoring.split_users(
            skor["user"].tolist(), skor["is_actual_insider"].tolist(),
            seed=split_seed, validation_fraction=val_frac)
        skor = skor[skor["user"].isin(test_kul)]

    birlesik = skor.join(kisi, on="user", how="inner")
    if len(birlesik) != len(skor):
        raise SystemExit("Skor dosyasındaki bazı kullanıcılar işlenmiş veride "
                         "yok; rol eşlemesi eksik kalır.")
    birlesik = birlesik.copy()
    birlesik["tahmin"] = (birlesik["max_z_score"] >= esik).astype(int)
    # Etiket iki kaynaktan geliyor; uyuşmazlık sessiz bir hata olurdu.
    uyusmaz = int((birlesik["etiket"] !=
                   birlesik["is_actual_insider"].astype(int)).sum())
    if uyusmaz:
        raise SystemExit(f"{uyusmaz} kullanıcıda içeriden etiketi skor dosyası "
                         "ile işlenmiş veri arasında uyuşmuyor.")
    return birlesik, esik, olcum


def hucreler(frame, anahtar="rol"):
    """Bir gruplama anahtarı başına dört hücre."""
    g = frame.groupby(anahtar)
    out = pd.DataFrame({
        "n": g.size(),
        "TP": g.apply(lambda d: int(((d.etiket == 1) & (d.tahmin == 1)).sum()),
                      include_groups=False),
        "FN": g.apply(lambda d: int(((d.etiket == 1) & (d.tahmin == 0)).sum()),
                      include_groups=False),
        "FP": g.apply(lambda d: int(((d.etiket == 0) & (d.tahmin == 1)).sum()),
                      include_groups=False),
        "TN": g.apply(lambda d: int(((d.etiket == 0) & (d.tahmin == 0)).sum()),
                      include_groups=False),
    })
    return out


def topla(experiment, seedler, kisi, pencere, split_seed, val_frac, yarim,
          anahtar="rol"):
    """Bütün seed ve plato turları üzerinden anahtar başına hücre toplamları."""
    yiginlar, esikler, olcum_sayisi = [], [], 0
    for seed in seedler:
        dizin = f"{experiment}__seed{seed}"
        yollar = plato_turlari(dizin, pencere)
        if not yollar:
            print(f"  atlandı: {dizin} (tur dosyası yok)")
            continue
        for yol in yollar:
            birlesik, esik, _ = bir_tur(yol, kisi, split_seed, val_frac, yarim)
            yiginlar.append(hucreler(birlesik, anahtar))
            esikler.append(esik)
            olcum_sayisi += 1
        print(f"  {dizin}: {len(yollar)} tur")
    if not yiginlar:
        raise SystemExit("Hiç ölçüm toplanamadı.")

    toplam = pd.concat(yiginlar).groupby(level=0).sum()
    ortalama = toplam / olcum_sayisi
    return ortalama, olcum_sayisi, esikler


def yanlis_alarm_kararliligi(experiment, seedler, kisi, pencere, split_seed,
                             val_frac):
    """Her masum kullanıcı kaç ölçümde işaretlendi.

    Ortalama bir yanlış alarm sayısı, sayının aynı kişilerden mi yoksa her
    ölçümde başkalarından mı geldiğini gizler. İkisi çok farklı şeyler: dağınık
    bir küme örnekleme gürültüsüdür, her seferinde işaretlenen sabit bir küme
    ise modelin o kişileri sistematik olarak ayırdığı anlamına gelir.
    """
    sayac, olcum = {}, 0
    for seed in seedler:
        for yol in plato_turlari(f"{experiment}__seed{seed}", pencere):
            birlesik, _, _ = bir_tur(yol, kisi, split_seed, val_frac,
                                     yarim=False)
            yanlis = birlesik[(birlesik.etiket == 0) & (birlesik.tahmin == 1)]
            for kul, rol in zip(yanlis.user, yanlis.rol):
                sayac[(int(kul), rol)] = sayac.get((int(kul), rol), 0) + 1
            olcum += 1
    frame = pd.DataFrame(
        [{"kullanici": k, "rol": r, "isaretlenme": c, "olcum": olcum}
         for (k, r), c in sayac.items()])
    return frame.sort_values("isaretlenme", ascending=False), olcum


def turet(tablo):
    """Hücrelerden okunabilir oranlar."""
    t = tablo.copy()
    t["pozitif"] = t.TP + t.FN
    t["negatif"] = t.FP + t.TN
    # Yanlış alarm oranı: bu roldeki masum kişilerin yüzde kaçı işaretleniyor.
    t["FPR_%"] = np.where(t.negatif > 0, 100 * t.FP / t.negatif, np.nan)
    # Duyarlılık: bu roldeki içeridenlerin yüzde kaçı yakalanıyor.
    t["Duyarlilik_%"] = np.where(t.pozitif > 0, 100 * t.TP / t.pozitif, np.nan)
    # Kesinlik: bu rolde verilen alarmların yüzde kaçı gerçek.
    alarm = t.TP + t.FP
    t["Kesinlik_%"] = np.where(alarm > 0, 100 * t.TP / alarm, np.nan)
    return t


def yaz(baslik, tablo, min_users, akis):
    print(f"\n{baslik}", file=akis)
    print("=" * len(baslik), file=akis)
    buyuk = tablo[tablo.n >= min_users].sort_values("FPR_%", ascending=False)
    kucuk = tablo[tablo.n < min_users]
    kolonlar = ["n", "TP", "FN", "FP", "TN", "FPR_%", "Duyarlilik_%",
                "Kesinlik_%"]
    print(buyuk[kolonlar].round(2).to_string(), file=akis)
    if len(kucuk):
        s = kucuk[["n", "TP", "FN", "FP", "TN"]].sum()
        neg = s.FP + s.TN
        print(f"\n[{len(kucuk)} küçük rol toplandı] n={s.n:.1f} TP={s.TP:.2f} "
              f"FN={s.FN:.2f} FP={s.FP:.2f} TN={s.TN:.2f} "
              f"FPR={100 * s.FP / neg if neg else float('nan'):.2f}%", file=akis)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--min-users", type=int, default=15,
                    help="Bu kadar kullanıcısı olmayan roller tek satırda toplanır.")
    ap.add_argument("--out", default=os.path.join("ieee", "kontrol_ciktilari",
                                                  "rol_karisiklik.csv"))
    args = ap.parse_args()

    seedler = [int(s) for s in args.seeds.split(",") if s.strip()]
    config.set_seed(seedler[0])
    config.set_experiment(args.experiment)
    cfg = scoring.ScoringConfig.from_config(config)
    pencere = config.get("evaluation", "plateau_window_rounds")

    print(f"kol: {args.experiment}   seed: {seedler}   plato penceresi: {pencere}")
    print(f"eşik kuralı: doğrulama yarısı, split_seed={cfg.split_seed}, "
          f"val_frac={cfg.validation_fraction}")

    adlar = rol_adlari()
    kisi = kullanici_kimlikleri(adlar)
    print(f"{len(kisi)} kullanıcı, {kisi.rol.nunique()} rol, "
          f"{int(kisi.etiket.sum())} içeriden")

    print("\nTest yarısı (tezin raporladığı kural):")
    test_t, n_test, esikler = topla(args.experiment, seedler, kisi, pencere,
                                    cfg.split_seed, cfg.validation_fraction,
                                    yarim=True)
    print("\nBütün popülasyon (betimleyici):")
    tum_t, n_tum, _ = topla(args.experiment, seedler, kisi, pencere,
                            cfg.split_seed, cfg.validation_fraction,
                            yarim=False)

    test_t, tum_t = turet(test_t), turet(tum_t)
    print(f"\n{n_test} ölçüm ortalandı. Eşik: ortalama {np.mean(esikler):.4f}, "
          f"aralık [{min(esikler):.4f}, {max(esikler):.4f}]")

    yaz(f"ROL BAZLI KARIŞIKLIK MATRİSİ, test yarısı ({n_test} ölçümün ortalaması)",
        test_t, args.min_users, sys.stdout)
    yaz(f"ROL BAZLI KARIŞIKLIK MATRİSİ, bütün popülasyon ({n_tum} ölçüm)",
        tum_t, args.min_users, sys.stdout)

    # İkili özet: yönetici olan ve olmayan.
    for ad, tablo, n in (("test yarısı", test_t, n_test),
                         ("bütün popülasyon", tum_t, n_tum)):
        yon = tablo.loc[["ITAdmin"]] if "ITAdmin" in tablo.index else None
        digerleri = tablo.drop(index="ITAdmin", errors="ignore").sum()
        if yon is None:
            continue
        y = yon.iloc[0]
        print(f"\nİKİLİ ÖZET ({ad})")
        for etiket, satir in (("ITAdmin", y), ("diğer roller", digerleri)):
            neg = satir.FP + satir.TN
            poz = satir.TP + satir.FN
            print(f"  {etiket:<14} n={satir.n:6.1f}  TP={satir.TP:5.2f}  "
                  f"FN={satir.FN:5.2f}  FP={satir.FP:6.2f}  TN={satir.TN:7.2f}  "
                  f"FPR={100 * satir.FP / neg if neg else float('nan'):5.2f}%  "
                  f"duyarlılık="
                  f"{100 * satir.TP / poz if poz else float('nan'):5.1f}%")

    # Senaryo kırılımı. Rol tablosundaki duyarlılık farkının nereden geldiğini
    # ayırmak için gerekiyor: senaryo 3'ün on aktörünün onu da ITAdmin, ve o
    # senaryo yalnız iki kötücül gün taşıyor.
    sen_t, n_sen, _ = topla(args.experiment, seedler, kisi, pencere,
                            cfg.split_seed, cfg.validation_fraction,
                            yarim=False, anahtar="senaryo")
    sen_t = turet(sen_t)
    print(f"\nSENARYO BAZLI ({n_sen} ölçüm, bütün popülasyon)")
    print("=" * 44)
    print(sen_t[["n", "TP", "FN", "FP", "TN", "FPR_%", "Duyarlilik_%"]]
          .round(2).to_string())

    kararlilik, n_kararlilik = yanlis_alarm_kararliligi(
        args.experiment, seedler, kisi, pencere, cfg.split_seed,
        cfg.validation_fraction)
    hep = kararlilik[kararlilik.isaretlenme == n_kararlilik]
    print(f"\nYANLIŞ ALARM KARARLILIĞI ({n_kararlilik} ölçüm)")
    print("=" * 44)
    print(kararlilik.to_string(index=False))
    print(f"\n{len(hep)} kullanıcı {n_kararlilik} ölçümün hepsinde "
          f"işaretlendi; rolleri: {sorted(set(hep.rol))}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kararlilik.to_csv(args.out.replace(".csv", "_kararlilik.csv"), sep=";",
                      index=False, encoding="utf-8-sig")
    dis = pd.concat({"test_yarisi": test_t, "tum_populasyon": tum_t,
                     "senaryo": sen_t}, names=["kume", "rol"])
    dis.round(4).to_csv(args.out, sep=";", encoding="utf-8-sig")
    print(f"\nyazıldı: {args.out}")


if __name__ == "__main__":
    main()
