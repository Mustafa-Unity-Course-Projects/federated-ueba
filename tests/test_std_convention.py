"""Öznitelik sütunlarının standart sapması tek bir sözleşmeye bağlı kalmalı.

25 Ağustos 2026'da bulunan kusur, projenin tekrar tekrar düştüğü aileden: hiçbir
şey hata vermiyor, yalnız iki yer aynı büyüklüğü farklı hesaplıyor.

Çizelge 4.2'nin Std sütunu `analyze_features.py`'den geldi ve pandas'ın
varsayılanını, yani örneklem sapmasını (ddof=1) kullanıyor.
`build_tables.cizelge_4_2m` ise aynı sütunu yeniden üretirken anakütle sapmasını
(ddof=0) kullanıyordu. 308.023 satırda aradaki oran 1 + 1/(2n), yani dördüncü
ondalığı on bir satırda bir kaydırmaya yetiyor. Sonuç: `--check` belgeyi on bir
yerde yanlış gösteriyordu, oysa yanlış olan üreticiydi.

Aynı sözleşmeyi kullanması gereken üç yer daha var; dördü birlikte sınanıyor.

Referans hata sapması σⱼ (`task._reference_error_stats`, Denklem 4.8) bu kümenin
dışındadır: farklı bir nicelik, eğitim sırasında ölçülüyor ve orada ddof=0
bilinçli bir tercih. Bu test ona dokunmaz.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FeatureStdConvention(unittest.TestCase):
    """Dört okuyucu da örneklem sapmasını vermeli."""

    def setUp(self):
        rastgele = np.random.RandomState(0)
        self.frame = pd.DataFrame({
            "a": rastgele.normal(3.0, 7.0, 5000),
            "b": rastgele.gamma(2.0, 4.0, 5000),
            "sabit": np.zeros(5000),
        })
        # Küçük bir örnekte iki sözleşme belirgin biçimde ayrılmalı, aksi hâlde
        # test geçmesi gerektiği için değil ayırt edemediği için geçer.
        self.assertNotEqual(
            round(float(self.frame["a"].std(ddof=0)), 4),
            round(float(self.frame["a"].std(ddof=1)), 4),
            "örnek boyutu iki sözleşmeyi ayıramayacak kadar büyük")

    def test_cizelge_4_2m_ornek_sapmasi_veriyor(self):
        """Üretici ile belgenin sütunu aynı büyüklüğü göstermeli."""
        from analysis.build_tables import cizelge_4_2m

        import tempfile
        with tempfile.TemporaryDirectory() as klasor:
            yol = os.path.join(klasor, "veri.csv")
            self.frame.to_csv(yol, index=False)

            import config_manager
            gercek = config_manager.config.get
            config_manager.config.get = (
                lambda b, a, *k, **kw: list(self.frame.columns)
                if (b, a) == ("data", "selected_features") else gercek(b, a, *k, **kw))
            try:
                _, satirlar = cizelge_4_2m(veri=yol)
            finally:
                config_manager.config.get = gercek

        uretilen = {ad: std for ad, std, _ in satirlar}
        for sutun in self.frame.columns:
            beklenen = f"{self.frame[sutun].std(ddof=1):.4f}".replace(".", ",")
            self.assertEqual(uretilen[sutun], beklenen,
                             f"{sutun}: üretici ddof=1 vermiyor")

    def test_sabit_oznitelik_esigi_ayni_sozlesmeyi_kullaniyor(self):
        """`task` sabit özniteliği ayıklarken aynı sapmaya bakmalı."""
        from federated_ueba import task

        kaynak = open(task.__file__, encoding="utf-8").read()
        self.assertIn("column.std()", kaynak,
                      "sabit öznitelik eşiği pandas varsayılanını kullanmıyor")
        self.assertNotIn("column.std(ddof=0)", kaynak)

    def test_secim_kurali_ayni_sozlesmeyi_kullaniyor(self):
        """`feature_selection_rule` de aynı sapmayı raporlamalı."""
        from analysis import feature_selection_rule

        kaynak = open(feature_selection_rule.__file__, encoding="utf-8").read()
        self.assertNotIn("std(ddof=0)", kaynak)


if __name__ == "__main__":
    unittest.main()
