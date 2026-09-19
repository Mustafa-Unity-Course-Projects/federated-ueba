"""Produce the feature table the thesis needs, from the data rather than by hand.

Answers two open questions with measurements instead of assertions:

  - no feature table existed at all: which feature came from which raw CSV, what
    it measures, and what its distribution looks like
  - the 50 features were described as chosen by intuition. The extraction script
    emits 510 columns, so the question "why these 50" is a real one

Written as a script so the table in the thesis and the features in the code
cannot drift apart: rerun it and the numbers are current by construction.

That guarantee only holds if the file the write-up cites is the file this script
writes. A hand-made copy once sat beside the generated one; the two happened to
agree, but nothing kept them agreeing, which is the exact hazard this script
exists to remove. Cite the generated file, or pass `--markdown` to write where
the prose already points.

    python analysis/analyze_features.py
    python analysis/analyze_features.py --out x.csv --markdown x.md
"""

# This file lives in a subdirectory, so Python puts that subdirectory on
# sys.path rather than the project root and `import config_manager` fails.
# Adding the root explicitly keeps `python analysis/analyze_features.py` working from the project
# root, which is how every path in the configuration is resolved anyway.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import os
import sys

import pandas as pd

from config_manager import config
from federated_ueba import scaling

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Which raw CERT file each feature family is extracted from. The prefix before
# the first underscore identifies the family; `n_<family>` counts its events.
SOURCE_BY_FAMILY = {
    "logon": "logon.csv",
    "usb": "device.csv",
    "file": "file.csv",
    "email": "email.csv",
    "http": "http.csv",
}

# Features that are not tied to one source file.
CALENDAR_FEATURES = {"isweekday", "isweekend"}
ALL_ACTIVITY_FEATURES = {"n_workhourallact", "n_afterhourallact"}

# The naming convention carries the meaning; these describe the irregular parts.
PC_ROLE_MEANING = {
    "pc0": "kendi bilgisayarı",
    "pc1": "paylaşımlı bilgisayar",
    "pc2": "başka bir kullanıcının bilgisayarı",
    "pc3": "amirinin bilgisayarı",
}
FILE_TYPE_MEANING = {
    "otherf": "diğer", "compf": "sıkıştırılmış", "phof": "görüntü",
    "docf": "belge", "txtf": "metin", "exef": "yürütülebilir",
}
HTTP_CATEGORY_MEANING = {
    "socnetf": "sosyal ağ", "cloudf": "bulut depolama", "jobf": "iş arama",
}

# Written out for the features the thesis uses, because the naming convention
# alone produces raw field names ("n des", "http c len") in the description
# column. Each line follows the extraction code: `disk` is 1 for a path starting
# with C: and 0 for any path starting with neither C: nor R:; `Xemail` and
# `exbccmail` flag an external recipient and an external blind-copy recipient;
# `*_len` is the length of the content and `*_nwords` its word count.
FEATURE_DESCRIPTIONS = {
    "isweekday": "Gün hafta içi mi",
    "isweekend": "Gün hafta sonu mu",
    "n_workhourallact": "Mesai içi toplam olay sayısı",
    "n_afterhourallact": "Mesai dışı toplam olay sayısı",
    "n_logon": "Günlük oturum açma sayısı",
    "n_workhourlogon": "Mesai içi oturum açma sayısı",
    "n_afterhourlogon": "Mesai dışı oturum açma sayısı",
    "logon_n-pc0": "Kendi bilgisayarında oturum açma sayısı",
    "logon_n-pc1": "Paylaşımlı bilgisayarda oturum açma sayısı",
    "logon_n-pc2": "Başka bir kullanıcının bilgisayarında oturum açma sayısı",
    "n_usb": "Günlük USB bağlantı sayısı",
    "usb_mean_usb_dur": "USB bağlantı süresinin gün içi ortalaması",
    "usb_n-pc0": "Kendi bilgisayarında USB bağlantı sayısı",
    "usb_n-pc1": "Paylaşımlı bilgisayarda USB bağlantı sayısı",
    "usb_n-pc2": "Başka bir kullanıcının bilgisayarında USB bağlantı sayısı",
    "n_workhourusb": "Mesai içi USB bağlantı sayısı",
    "n_afterhourusb": "Mesai dışı USB bağlantı sayısı",
    "n_file": "Günlük dosya olayı sayısı",
    "file_mean_file_len": "Dosya içeriği uzunluğunun gün içi ortalaması",
    "file_mean_file_depth": "Dosya yolu derinliğinin gün içi ortalaması",
    "file_mean_file_nwords": "Dosya içeriği kelime sayısının gün içi ortalaması",
    "file_n-disk0": "Yolu C: ya da R: ile başlamayan dosya olayı sayısı",
    "file_n-disk1": "Yolu C: ile başlayan dosya olayı sayısı",
    "file_n_otherf": "Diğer türdeki dosya sayısı",
    "file_n_compf": "Sıkıştırılmış dosya sayısı",
    "file_n_phof": "Görüntü dosyası sayısı",
    "file_n_docf": "Belge dosyası sayısı",
    "file_n_txtf": "Metin dosyası sayısı",
    "file_n_exef": "Yürütülebilir dosya sayısı",
    "n_email": "Günlük e-posta sayısı",
    "email_mean_n_des": "E-posta başına alıcı sayısının gün içi ortalaması",
    "email_mean_n_atts": "E-posta başına ek sayısının gün içi ortalaması",
    "email_n-Xemail1": "Kurum dışı alıcı içeren e-posta sayısı",
    "email_mean_n_exdes": "Kurum dışı alıcı sayısının gün içi ortalaması",
    "email_mean_n_bccdes": "Gizli alıcı (BCC) sayısının gün içi ortalaması",
    "email_n-exbccmail1": "Kurum dışı gizli alıcı içeren e-posta sayısı",
    "email_mean_email_size": "E-posta boyutunun gün içi ortalaması",
    "email_mean_email_text_slen": "E-posta metni uzunluğunun gün içi ortalaması",
    "email_mean_email_text_nwords": "E-posta metni kelime sayısının gün içi ortalaması",
    "email_n-pc0": "Kendi bilgisayarındaki e-posta olayı sayısı",
    "email_n-pc1": "Paylaşımlı bilgisayardaki e-posta olayı sayısı",
    "email_n-pc2": "Başka bir kullanıcının bilgisayarındaki e-posta olayı sayısı",
    "n_http": "Günlük web ziyareti sayısı",
    "http_mean_url_len": "Adres uzunluğunun gün içi ortalaması",
    "http_mean_url_depth": "Adres yolu derinliğinin gün içi ortalaması",
    "http_mean_http_c_len": "Sayfa içeriği uzunluğunun gün içi ortalaması",
    "http_mean_http_c_nwords": "Sayfa içeriği kelime sayısının gün içi ortalaması",
    "http_n_socnetf": "Sosyal ağ sitesi ziyareti sayısı",
    "http_n_cloudf": "Bulut depolama sitesi ziyareti sayısı",
    "http_n_jobf": "İş arama sitesi ziyareti sayısı",
}

# A cell counts as zero below this magnitude. The percentile transform leaves
# values that should be zero as float residue around 7.1e-15 in every column, so
# an exact `== 0` test undercounts zeros everywhere, not only in the four
# constant features.
ZERO_TOLERANCE = 1e-9


def feature_source(name):
    """The raw CERT file a feature is extracted from."""
    if name in CALENDAR_FEATURES:
        return "takvim (tarihten türetilmiş)"
    if name in ALL_ACTIVITY_FEATURES:
        return "tüm kaynaklar"

    for family, source in SOURCE_BY_FAMILY.items():
        # Matches both `usb_mean_usb_dur` and `n_usb` / `n_afterhourusb`.
        if name.startswith(f"{family}_") or name.endswith(family):
            return source
    return "bilinmiyor"


def feature_description(name):
    """A one-line meaning: written out for the thesis features, derived from the
    naming convention for any other column."""
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name == "isweekday":
        return "Gün hafta içi mi"
    if name == "isweekend":
        return "Gün hafta sonu mu"

    # `logon_n-pc2` style: activity counted per PC role.
    if "_n-" in name:
        family, _, suffix = name.partition("_n-")
        if suffix in PC_ROLE_MEANING:
            return f"{family} olaylarının {PC_ROLE_MEANING[suffix]} üzerindeki sayısı"
        if suffix.startswith("disk"):
            return f"{family} olaylarının disk {suffix[-1]} üzerindeki sayısı"
        return f"{family} olaylarının {suffix} kırılımındaki sayısı"

    # `file_n_exef` style: counted per category.
    for category, meaning in {**FILE_TYPE_MEANING, **HTTP_CATEGORY_MEANING}.items():
        if name.endswith(f"_n_{category}"):
            return f"{meaning} türündeki dosya/sayfa sayısı"

    # `usb_mean_usb_dur` style: a per-event quantity averaged over the day.
    if "_mean_" in name:
        _, _, quantity = name.partition("_mean_")
        return f"O günkü olayların ortalama {quantity.replace('_', ' ')} değeri"

    if name.startswith("n_workhour"):
        return f"Mesai içi {name[len('n_workhour'):]} olay sayısı"
    if name.startswith("n_afterhour"):
        return f"Mesai dışı {name[len('n_afterhour'):]} olay sayısı"
    if name.startswith("n_"):
        return f"Günlük {name[2:]} olay sayısı"
    return name


def describe_features(df, features):
    """Per-feature statistics over the whole dataset.

    "Sabit" uses the same test the scaler uses, a standard deviation below
    `data.constant_feature_tolerance`, not by counting distinct values. Counting
    values gets this wrong here: four of the features are zero everywhere except
    for float64 noise around 7.1e-15, which `nunique` faithfully reports as two
    distinct values. That noise is what the scaler's variance floor exists to
    catch, so the table and the code have to agree on the definition.
    """
    rows = []
    for name in features:
        column = pd.to_numeric(df[name], errors="coerce").fillna(0)
        std = float(column.std())
        rows.append({
            "Öznitelik": name,
            "Kaynak": feature_source(name),
            "Açıklama": feature_description(name),
            "Min": float(column.min()),
            "Maks": float(column.max()),
            "Ortalama": round(float(column.mean()), 4),
            "Std": round(std, 4),
            "Sıfır_oranı_%": round(100 * float((column.abs() < ZERO_TOLERANCE).mean()), 1),
            "Farklı_değer": int(column.nunique()),
            "Sabit": bool(std < scaling.constant_feature_tolerance()),
        })
    return pd.DataFrame(rows)


def write_markdown(table, path, total_columns):
    """A table that can be pasted into the thesis as is."""
    constant = table[table["Sabit"]]["Öznitelik"].tolist()

    # Derived from this file's own location rather than typed, because the
    # script moved into `analysis/` once already and the hard-coded name went on
    # citing the old path. A generated table that misreports its own generator is
    # the drift this script exists to prevent.
    generator = os.path.relpath(os.path.abspath(__file__),
                                os.path.dirname(os.path.dirname(
                                    os.path.abspath(__file__)))).replace(os.sep, "/")

    lines = [
        "# Öznitelik Tablosu",
        "",
        f"Kaynak: `{generator}`, {len(table)} öznitelik "
        f"(çıkarım betiğinin ürettiği {total_columns} sütun arasından seçilmiş).",
        "",
        "| # | Öznitelik | Kaynak | Açıklama | Ortalama | Std | Sıfır % |",
        "|---|---|---|---|---|---|---|",
    ]
    # Iterated as dictionaries rather than named tuples: "Sıfır_oranı_%" is not a
    # valid Python identifier, so itertuples silently renames it to a position.
    for i, row in enumerate(table.to_dict("records"), start=1):
        lines.append(
            f"| {i} | `{row['Öznitelik']}` | {row['Kaynak']} | {row['Açıklama']} | "
            f"{row['Ortalama']} | {row['Std']} | {row['Sıfır_oranı_%']} |")

    lines += ["", "## Bilgi taşımayan öznitelikler", ""]
    if constant:
        lines.append(
            f"{len(constant)} öznitelik tüm veri setinde etkin olarak sabit, yani "
            f"hiçbir bilgi taşımıyor:")
        lines.append("")
        for name in constant:
            column = table[table["Öznitelik"] == name].iloc[0]
            lines.append(f"- `{name}` (std = {column['Std']:.3g}, "
                         f"maks = {column['Maks']:.3g})")
        lines.append("")
        lines.append(
            "Bunlar her satırda sıfır. `Farklı_değer` sütunu 2 gösteriyorsa, "
            "ikinci değer 7,1e-15 civarında bir kayan nokta artığıdır (2⁻⁴⁷), "
            "gerçek bir ölçüm değil. Bu yüzden sabitlik testi farklı değer "
            "saymaya değil, standart sapmanın "
            f"{scaling.constant_feature_tolerance():.0e} altında kalmasına bakıyor; "
            "ölçekleyici de "
            "aynı ölçütü kullanıyor (`federated_ueba/scaling.py`).")
        lines.append("")
        lines.append(
            "Ölçekleyici bunları sabit işaretleyip ölçeklemeden bırakıyor, yani "
            "modele sıfır olarak giriyorlar ve sonucu etkilemiyorlar. Bu koruma "
            "olmadan sapma 3,3e-15'e bölünüyor ve öznitelik 1e15 mertebesinde "
            "z-skoru üretiyordu.")
    else:
        lines.append("Yok.")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown tablosu yazıldı: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="generated_visuals/feature_table.csv")
    parser.add_argument("--markdown", default="generated_visuals/feature_table.md")
    args = parser.parse_args()

    data_path = config.get("data", "processed_data_path")
    if not os.path.exists(data_path):
        print(f"Veri bulunamadı: {data_path}")
        return 1

    features = list(config.get("data", "selected_features"))

    # The header alone tells us how many columns the extraction produces, which
    # is the denominator for "why these 50". Reading it costs nothing.
    total_columns = len(pd.read_csv(data_path, nrows=0).columns)

    # Only the selected columns: the file is 1.6 GB and 510 columns wide.
    print(f"{len(features)} öznitelik okunuyor ({total_columns} sütun arasından)...")
    df = pd.read_csv(data_path, usecols=features, low_memory=False)

    table = describe_features(df, features)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"CSV yazıldı: {args.out}")

    write_markdown(table, args.markdown, total_columns)

    constant = table[table["Sabit"]]["Öznitelik"].tolist()
    print(f"\nSeçilen: {len(features)} / {total_columns} sütun")
    print(f"Etkin olarak sabit (bilgi taşımayan): {len(constant)}")
    for name in constant:
        print(f"  - {name}")

    # The other end of the same question: a feature that is almost always zero
    # carries little, even though a variance filter would keep it.
    mostly_zero = table[table["Sıfır_oranı_%"] >= 99]["Öznitelik"].tolist()
    informative = [f for f in mostly_zero if f not in constant]
    if informative:
        print(f"\n%99'dan fazla sıfır olan ama sabit olmayan: {len(informative)}")
        for name in informative:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
