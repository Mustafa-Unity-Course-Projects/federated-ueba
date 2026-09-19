"""The untrained control, run on the federated arm instead of the centralized one.

`untrained_control.py` answers "would a network that learned nothing do just as
well?" for the centralized model. Its answer, 0.7471 against a trained 0.9570,
is quoted often. The federated arm reports 0.8371, and a reader who puts those
two side by side will subtract them. That subtraction is wrong: the two numbers
come from different arms, different scalers, different reference statistics and
different seed counts. The thesis says so, but saying so does not stop the
question. This script produces the number that does.

Everything here is held to the federated arm's own conditions:

  scaler        the run's `global_scaler.pkl`, built from per-client sufficient
                statistics, not the centralized `StandardScaler`
  reference     for a control model, computed from that model's own errors on
                the same client partitions the trained clients used, and
                averaged over clients exactly the way `load_error_reference`
                averages the trained ones. Scoring a control against the
                *trained* reference flatters it badly; that figure is printed
                too, but only to show the size of the mistake
  population    all 1000 users, the `PR-AUC_all` column the thesis reports
  estimator     `scoring.evaluate_scores` with the fixed split seed

Three probes, each removing one more thing:

  trained       a saved checkpoint, rescored here rather than trusted. Round 50
                is compared against the round 50 the run itself recorded; if
                those two disagree, nothing below this line means anything
  random init   an untrained network, calibrated on its own error statistics
  zero output   a "model" returning zeros, so the squared error is the squared
                input. No weights at all, learned or random

The zero probe does not depend on the run seed: under the IID partition the
user chunks are fixed by user id, so the global scaler and the zero model's own
reference are identical in all five runs. It is reported once, and that
constancy is checked rather than assumed.

Nothing is written into the result directories. Output goes to stdout and,
optionally, to a CSV named by `--out`.

    python analysis/federated_untrained_control.py
    python analysis/federated_untrained_control.py --seeds 1,2 --out kontrol.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class ZeroModel(nn.Module):
    """Reconstructs everything as zero, so the squared error is the input squared.

    The floor of the design: nothing learned, nothing random. Whatever this
    scores is what the scoring pipeline extracts from the data alone.
    """

    def forward(self, x):
        return torch.zeros_like(x)


def error_reference(model, sequences_by_client, device, batch_size):
    """Per-feature error mean and std, averaged over clients.

    Mirrors `task.get_error_distribution` (mean over the window's timesteps)
    and then `federated_insider_detection.load_error_reference` (plain mean
    over clients). Both steps are reproduced rather than imported because the
    trained pipeline computes the first inside the client process and reads the
    second from pickles on disk; a control model has neither.
    """
    means, stds = [], []
    model.eval()
    with torch.no_grad():
        for windows in sequences_by_client:
            if windows is None or len(windows) == 0:
                continue
            hatalar = []
            for bas in range(0, len(windows), batch_size):
                yigin = torch.as_tensor(windows[bas:bas + batch_size],
                                        dtype=torch.float32, device=device)
                kare = torch.mean((model(yigin) - yigin) ** 2, dim=1)
                hatalar.append(kare.cpu().numpy())
            hepsi = np.concatenate(hatalar, axis=0)
            means.append(hepsi.mean(axis=0))
            stds.append(hepsi.std(axis=0))
    if not means:
        raise SystemExit("Hicbir istemcide pencere bulunamadi.")
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def client_windows(df, features, scaler, user_chunks, window_size):
    """The training windows each client saw, in client order.

    Same three rules the client loader applies: labelled insider days are
    dropped, scaling uses the federation's one global scaler, and windows are
    built per user so none spans two people's days.
    """
    from federated_ueba import scaling
    parcalar = []
    for kullanicilar in user_chunks:
        istemci = df[df["user"].isin(kullanicilar)]
        if "insider" in istemci.columns:
            istemci = istemci[istemci["insider"] == 0]
        diziler = []
        for _, grup in istemci.groupby("user"):
            olcekli = scaler.transform(scaling.prepare_features(grup, features))
            if len(olcekli) < window_size:
                continue
            for i in range(len(olcekli) - window_size + 1):
                diziler.append(olcekli[i:i + window_size])
        parcalar.append(np.asarray(diziler, dtype=np.float32)
                        if diziler else None)
    return parcalar


def main():
    ayristirici = argparse.ArgumentParser()
    ayristirici.add_argument("--experiment", default="baseline")
    ayristirici.add_argument("--seeds", default="1,2,3,4,5")
    ayristirici.add_argument("--round", type=int, default=50,
                             help="Egitilmis kontrol icin kullanilacak tur.")
    ayristirici.add_argument("--out", default="")
    args = ayristirici.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    os.environ.setdefault("EXPERIMENT_NAME", args.experiment)
    os.environ["SEED"] = str(seeds[0])

    import federated_insider_detection as fid
    from config_manager import config
    from federated_ueba import scaling, scoring, task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    veri_yolu = config.get("data", "processed_data_path")
    df = pd.read_csv(veri_yolu, low_memory=False)
    all_users = sorted(df["user"].unique())
    scoring_cfg = scoring.ScoringConfig.from_config(config)
    # Kolun kendi degeri, yoksa pyproject varsayilani. Ayni dusme sirasi
    # `resolve_run_layout` icinde de var; ikisi ayrisirsa bolutleme
    # egitimdekiyle ayni olmaz.
    num_partitions = (config.get("federation", "num_supernodes")
                      or config.get_pyproject("tool", "flwr", "federations",
                                              "local-simulation", "options",
                                              "num-supernodes")
                      or 10)
    batch_size = config.get("data", "batch_size")

    print(f"Federe kol '{args.experiment}'. Ayni olcekleyici, ayni skorlama "
          f"hatti, ayni 1000 kullanici; yalniz agirliklar degisiyor.")
    print(f"Cihaz: {device}. Seed: {seeds}. Egitilmis kontrol turu: "
          f"{args.round}.\n")

    kayit = []
    for seed in seeds:
        kosum = f"{args.experiment}__seed{seed}"
        # Kaydedilmis agirliklar, egitildikleri girdi donusumuyle puanlanmali.
        scaling.assert_transform_matches_run(os.path.join(
            "federated_evaluation_reports", kosum, "experiment_summary.json"))
        scaler_dir = os.path.join("./scaler_data", kosum)
        scaler, features = fid.load_reference_scaler(scaler_dir)
        if scaler is None:
            print(f"  {kosum}: olcekleyici yok, atlaniyor.")
            continue

        egitilmis_ref = fid.load_error_reference(scaler_dir, len(features))
        user_chunks = task.partition_users(df, num_partitions)
        pencereler = client_windows(df, features, scaler, user_chunks,
                                    task.WINDOW_SIZE)

        def puanla(model, ref, etiket):
            scorer = scoring.Scorer(
                features=features, scaler=scaler, ref_mean=ref[0],
                ref_std=ref[1], cfg=scoring_cfg,
                window_size=task.WINDOW_SIZE, device=device)
            sonuc = scorer.scan(model, df, all_users)
            olcum = scoring.evaluate_scores(
                sonuc, seed=scoring_cfg.split_seed,
                validation_fraction=scoring_cfg.validation_fraction)
            print(f"  seed {seed}  {etiket:46s} "
                  f"tum {olcum['pr_auc_all']:.4f}   "
                  f"test {olcum['pr_auc']:.4f}   F1 {olcum['f1']:.4f}")
            kayit.append({"seed": seed, "kontrol": etiket,
                          "PR-AUC_all": olcum["pr_auc_all"],
                          "PR-AUC_test": olcum["pr_auc"],
                          "F1": olcum["f1"]})
            return olcum

        # Egitilmis kontrol. Kosumun kendi kaydettigi degerle karsilastirilir;
        # tutmazsa bu betigin urettigi hicbir sayiya guvenilmemeli.
        ag = task.LSTMAutoencoder(input_dim=len(features)).to(device)
        yol = os.path.join("./model_pickle", kosum,
                           f"parameters_round_{args.round}.pkl")
        fid.load_checkpoint_into(ag, yol)
        olcum = puanla(ag, egitilmis_ref, f"egitilmis, tur {args.round}")
        beklenen = kayitli_deger(kosum, args.round)
        if beklenen is not None:
            fark = abs(beklenen - olcum["pr_auc_all"])
            durum = "TUTUYOR" if fark < 5e-4 else "TUTMUYOR"
            print(f"  seed {seed}  kosumun kendi kaydi {beklenen:.4f}, "
                  f"fark {fark:.6f}  [{durum}]")
            kayit[-1]["kayitli"] = beklenen

        # Rastgele ilk deger, kendi referansiyla.
        torch.manual_seed(seed)
        rastgele = task.LSTMAutoencoder(input_dim=len(features)).to(device)
        ref_r = error_reference(rastgele, pencereler, device, batch_size)
        puanla(rastgele, ref_r, "rastgele ilk deger, kendi referansi")

        # Sifir cikti, kendi referansiyla. Agirlik yok, dolayisiyla seed'den
        # bagimsiz olmasi beklenir; asagida sabitligi denetleniyor.
        sifir = ZeroModel().to(device)
        ref_s = error_reference(sifir, pencereler, device, batch_size)
        puanla(sifir, ref_s, "sifir cikti, model yok, kendi referansi")

        # Ayni sifir model, egitilmis kolun referansiyla. Yaniltici oldugu icin
        # bir iddiaya dayanak degil, hatanin buyuklugunu gostermek icin.
        puanla(sifir, egitilmis_ref, "sifir cikti, ODUNC referans (yaniltici)")
        print()

    cerceve = pd.DataFrame(kayit)
    print("\nOzet, kontrol basina bes seed:")
    ozet = cerceve.groupby("kontrol")["PR-AUC_all"].agg(["mean", "std", "min",
                                                         "max", "count"])
    for ad, satir in ozet.iterrows():
        sd = 0.0 if pd.isna(satir["std"]) else satir["std"]
        print(f"  {ad:46s} {satir['mean']:.4f} +- {sd:.4f}   "
              f"[{satir['min']:.4f}, {satir['max']:.4f}]  n={int(satir['count'])}")

    sabit = ozet.loc["sifir cikti, model yok, kendi referansi"]
    yayilim = sabit["max"] - sabit["min"]
    print(f"\nSifir kontrolunun seed'ler arasi yayilimi: {yayilim:.6f}. "
          f"Turdes bolumleme seed'den bagimsiz oldugundan sifir beklenir.")
    print("Okunusu: sifir cikti ile rastgele ag arasindaki fark agin "
          "agirliklarinin, rastgele ag ile egitilmis model arasindaki fark ise "
          "egitimin katkisidir. Dordu de ayni kolda, ayni olcekleyiciyle ve "
          "ayni populasyon uzerinde olculmustur.")

    if args.out:
        cerceve.to_csv(args.out, index=False)
        print("yazildi:", args.out)


def kayitli_deger(kosum, tur):
    """Kosumun kendi kaydettigi PR-AUC_all, dogrulama icin."""
    yol = os.path.join("./federated_evaluation_reports", kosum,
                       "federated_rounds_comparison.csv")
    if not os.path.exists(yol):
        return None
    cerceve = pd.read_csv(yol)
    satir = cerceve[cerceve["Round"] == tur]
    return None if satir.empty else float(satir["PR-AUC_all"].iloc[0])


if __name__ == "__main__":
    main()
