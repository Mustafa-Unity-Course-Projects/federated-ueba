"""Bütün analizleri koşar ve tek bir sonuç belgesi üretir.

Tezdeki her sayının bir üretici betiği var, fakat onları tek tek koşup çıktıları
elle bir araya getirmek hem yorucu hem de hataya açık: bir betiğin atlandığı ya
da farklı seed'lerle koşulduğu ancak sayılar çelişince fark ediliyor. Bu betik
hepsini aynı seed kümesiyle koşar, çıktıları olduğu gibi kaydeder ve hangisinin
ne kadar sürdüğünü, hangisinin başarısız olduğunu belgeye yazar.

**Sonuç dosyalarına yazmaz.** Alt betiklerin `--out` / `--output` seçenekleri
rapor klasörüne yönlendirilir; deposundaki `detection_latency.csv`,
`generated_visuals/` ve benzeri çıktılar olduğu gibi kalır. Yönlendirilemeyen tek
betik `compare_experiments.py`'dir: çıktı yolları sabit yazılmıştır, dolayısıyla
koşulmaz ve sebebi belgeye kaydedilir.

    python analysis/run_all_analyses.py
    python analysis/run_all_analyses.py --out rapor --seeds 1,2,3
    python analysis/run_all_analyses.py --only findings,seed_variance
    python analysis/run_all_analyses.py --quick      yalnız hızlı olanlar
"""

import argparse
import datetime
import io
import os
import subprocess
import sys
import time

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

# Sanal ortamın python'ı: alt betikler yapılandırmayı ve torch'u ondan okur.
PYTHON = os.path.join(KOK, ".venv", "Scripts", "python.exe")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


# Tekil `--seed` alan betikler. Bunlara seed listesi verilemez; listedeki ilk
# seed açıkça geçirilir. Varsayılana bırakmak yanlış olur, çünkü varsayılan
# yapılandırmadaki seed'e düşer ve o seed'in diskte koşumu olmayabilir:
# `analyze_detection_latency` tam olarak böyle `baseline__seed42` arayıp
# başarısız oldu.
TEK_SEEDLI = {
    "payload_weights.py", "verify_compression.py", "untrained_control.py",
    "literature_baseline.py", "scenario_breakdown.py",
    "analyze_detection_latency.py", "topm_sweep.py",
}


class Analiz:
    """Bir analiz betiği ve onun cevapladığı soru.

    `cikti_bayragi` doluysa betiğin yazma yolu rapor klasörüne çevrilir; boşsa
    betik zaten hiçbir şey yazmıyor demektir.
    """

    def __init__(self, ad, soru, argv, cikti_bayragi=None, cikti_adi=None,
                 seed_bayragi=None, tekil_seed=False, yavas=False,
                 calistirma=None):
        self.ad = ad
        self.soru = soru
        self.argv = argv
        self.cikti_bayragi = cikti_bayragi
        self.cikti_adi = cikti_adi
        self.seed_bayragi = seed_bayragi
        # Bazı betikler tek bir koşum üzerinde çalışır ve `--seed` alır. Bunu
        # varsayılana bırakmak yanlıştır: varsayılan, yapılandırmadaki seed'e
        # düşer ve o seed'in diskte bir koşumu olmayabilir. Listedeki ilk seed
        # açıkça verilir, ve belgeye tek seed olduğu yazılır.
        self.tekil_seed = tekil_seed
        self.yavas = yavas
        self.calistirma = calistirma      # dolu ise: koşulmama gerekçesi

    @property
    def tek_kosum(self):
        """Betik `--seeds` değil tekil `--seed` alıyor mu?"""
        return self.tekil_seed or os.path.basename(self.argv[0]) in TEK_SEEDLI

    def komut(self, rapor_dizini, seedler):
        argv = list(self.argv)
        if self.tek_kosum:
            argv += ["--seed", seedler.split(",")[0].strip()]
        elif self.seed_bayragi:
            argv += [self.seed_bayragi, seedler]
        if self.cikti_bayragi:
            argv += [self.cikti_bayragi,
                     os.path.join(rapor_dizini, self.cikti_adi)]
        return [PYTHON] + argv


ANALIZLER = [
    Analiz("findings",
           "Hangi iddialar eşleştirilmiş bootstrap altında ayakta kalıyor?",
           ["analysis/findings.py"], seed_bayragi="--seeds", yavas=True),

    Analiz("seed_variance",
           "Kolların bütün popülasyon üzerindeki plato ortalaması ve seed'ler "
           "arası sapması nedir?",
           ["analysis/seed_variance.py"], seed_bayragi="--seeds",
           cikti_bayragi="--output", cikti_adi="seed_variance.csv", yavas=True),

    Analiz("threshold_metrics",
           "Eşiğe bağlı metrikler PR-AUC ile aynı plato kuralı altında ne "
           "veriyor?",
           ["analysis/threshold_metrics.py"]),

    Analiz("partition_skew",
           "Miktar çarpıklığı ile role göre bölümleme ne kadar çarpıklık "
           "üretiyor?",
           ["analysis/partition_skew.py"], seed_bayragi="--seeds"),

    Analiz("bottleneck_mechanism",
           "Darboğaz kaldırılınca ham yeniden kurma sinyali gerçekten çöküyor "
           "mu?",
           ["analysis/bottleneck_mechanism.py"], seed_bayragi="--seeds",
           yavas=True),

    Analiz("payload_weights",
           "Bir turda kaç ağırlık gerçekten gönderiliyor?",
           ["analysis/payload_weights.py"]),

    Analiz("verify_compression",
           "Sıkıştırma eklentileri kayıpsız geri çözülüyor mu?",
           ["analysis/verify_compression.py"]),

    Analiz("analyze_payload_codecs",
           "Yük kodlayıcıları aynı yük üzerinde nasıl sıralanıyor?",
           ["analysis/analyze_payload_codecs.py"],
           cikti_bayragi="--out", cikti_adi="payload_codecs.csv"),

    Analiz("untrained_control",
           "Eğitilmemiş bir kontrol ne kadar PR-AUC üretiyor?",
           ["analysis/untrained_control.py"], yavas=True),

    Analiz("literature_baseline",
           "Literatür referansıyla aynı metrikte nasıl karşılaştırılıyoruz?",
           ["analysis/literature_baseline.py"], yavas=True),

    Analiz("scenario_breakdown",
           "Tespit senaryolara göre nasıl dağılıyor?",
           ["analysis/scenario_breakdown.py"], yavas=True),

    Analiz("analyze_detection_latency",
           "Bir içeriden aktör kampanyanın neresinde yakalanıyor?",
           ["analysis/analyze_detection_latency.py"],
           cikti_bayragi="--out", cikti_adi="detection_latency.csv", yavas=True),

    Analiz("topm_sweep",
           "Top-m parametresi taranınca ne oluyor?",
           ["analysis/topm_sweep.py"],
           cikti_bayragi="--out", cikti_adi="topm_sweep.csv", yavas=True),

    Analiz("confusion_matrix",
           "Karışıklık matrisi ve eşik davranışı nedir?",
           ["analysis/confusion_matrix.py", "--no-figure"],
           seed_bayragi="--seeds",
           cikti_bayragi="--output", cikti_adi="confusion_matrix.csv",
           yavas=True),

    Analiz("analyze_features",
           "Öznitelik kümesinin dağılım istatistikleri nedir, hangileri sabit?",
           ["analysis/analyze_features.py"],
           cikti_bayragi="--out", cikti_adi="feature_table.csv", yavas=True),

    Analiz("compare_experiments",
           "Kol karşılaştırma tablosu ve bootstrap aralıkları.",
           ["analysis/compare_experiments.py"],
           calistirma="Çıktı yolları betiğin içine sabit yazılmış "
                      "(`experiment_comparison_summary.csv`, "
                      "`experiment_comparison_by_seed.csv` ve bir şekil). "
                      "Yönlendirilemediği için koşulmadı; ürettiği bilgi "
                      "`seed_variance` ve `findings` çıktılarında zaten var."),
]


def calistir(analiz, rapor_dizini, seedler, zaman_asimi):
    """Bir analizi koşar, (donus_kodu, cikti, saniye) döndürür."""
    komut = analiz.komut(rapor_dizini, seedler)
    basla = time.time()
    try:
        sonuc = subprocess.run(komut, cwd=KOK, capture_output=True,
                               timeout=zaman_asimi)
        cikti = (sonuc.stdout + sonuc.stderr).decode("utf-8", errors="replace")
        return sonuc.returncode, cikti, time.time() - basla
    except subprocess.TimeoutExpired:
        return None, f"{zaman_asimi} saniyede tamamlanmadı.", time.time() - basla


def main():
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--out", default="analiz_raporu",
                             help="rapor klasörü; alt betiklerin çıktıları da "
                                  "buraya yazılır")
    ayristirici.add_argument("--seeds", default="1,2,3,4,5")
    ayristirici.add_argument("--only", default="",
                             help="virgülle ayrılmış analiz adları")
    ayristirici.add_argument("--quick", action="store_true",
                             help="yavaş olanları atla")
    ayristirici.add_argument("--timeout", type=int, default=1800,
                             help="analiz başına saniye")
    args = ayristirici.parse_args()

    rapor_dizini = os.path.join(KOK, args.out) if not os.path.isabs(args.out) \
        else args.out
    os.makedirs(rapor_dizini, exist_ok=True)

    secilen = [a.strip() for a in args.only.split(",") if a.strip()]
    sira = [a for a in ANALIZLER
            if (not secilen or a.ad in secilen)
            and not (args.quick and a.yavas)]

    belge = os.path.join(rapor_dizini, "analiz_sonuclari.md")
    out = io.open(belge, "w", encoding="utf-8")

    from config_manager import PIPELINE_VERSION
    print("# Analiz sonuçları\n", file=out)
    print(f"`analysis/run_all_analyses.py` tarafından üretildi, "
          f"{datetime.datetime.now():%d.%m.%Y %H:%M}. "
          f"Pipeline sürümü {PIPELINE_VERSION}, seed'ler {args.seeds}.\n",
          file=out)
    print("Her bölüm bir betiğin cevapladığı soruyu, çalıştırılan komutu ve "
          "çıktının tamamını taşır. Çıktılar kısaltılmadı.\n", file=out)

    ozet = []
    for analiz in sira:
        print(f"[{analiz.ad}] ...", flush=True)
        if analiz.calistirma:
            ozet.append((analiz.ad, "koşulmadı", 0.0))
            print(f"\n## {analiz.ad}\n", file=out)
            print(f"**Soru:** {analiz.soru}\n", file=out)
            print(f"**Koşulmadı.** {analiz.calistirma}\n", file=out)
            continue

        kod, cikti, sure = calistir(analiz, rapor_dizini, args.seeds,
                                    args.timeout)
        durum = "tamam" if kod == 0 else ("zaman aşımı" if kod is None
                                          else f"hata (çıkış {kod})")
        ozet.append((analiz.ad, durum, sure))

        print(f"\n## {analiz.ad}\n", file=out)
        print(f"**Soru:** {analiz.soru}\n", file=out)
        if analiz.tek_kosum:
            ilk = args.seeds.split(",")[0].strip()
            print(f"**Tek koşum:** bu analiz tek seed üzerinde çalışır "
                  f"(seed {ilk}); beş seed ortalaması değildir.\n", file=out)
        komut = " ".join(["python"] + analiz.komut(rapor_dizini, args.seeds)[1:])
        print(f"```\n{komut}\n```\n", file=out)
        print(f"**Durum:** {durum}, {sure:.1f} saniye\n", file=out)
        print("```", file=out)
        print(cikti.rstrip(), file=out)
        print("```", file=out)
        out.flush()

    print("\n## Koşum özeti\n", file=out)
    print("| analiz | durum | saniye |", file=out)
    print("|---|---|---|", file=out)
    for ad, durum, sure in ozet:
        print(f"| {ad} | {durum} | {sure:.1f} |", file=out)
    basarili = sum(1 for _, d, _ in ozet if d == "tamam")
    print(f"\n{basarili}/{len(ozet)} analiz tamamlandı.", file=out)
    out.close()

    print(f"\n{belge}")
    for ad, durum, sure in ozet:
        print(f"  {ad:26s} {durum:16s} {sure:7.1f}s")


if __name__ == "__main__":
    main()
