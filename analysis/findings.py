"""What the runs on disk actually support, claim by claim.

`compare_experiments.py` produces the results table and `seed_variance.py`
measures where the uncertainty comes from. Neither answers the question the
thesis has to answer in a sentence: *is this claim supported by the data, and
how strongly?* That is what this does.

Every claim the results chapter makes is written down here as a comparison, in
one place, so that a claim cannot drift away from the number underneath it.

Claims come in two kinds and are judged differently, because most of what this
thesis says about compression is "it costs nothing" rather than "it is better",
and those need opposite evidence. An equivalence claim wants a narrow interval
around zero; a superiority claim wants an interval clear of it.

  esdeger      equivalence: worst case in the interval is a loss under the margin
  ustun        superiority: the whole interval is above zero
  kotu         the interval rules out the claim in the other direction
  belirsiz     equivalence claim whose interval is too wide to decide either way
  ayrilmamis   superiority claim whose interval spans zero
  gecici       fewer than three seeds on one side, so the interval understates
               the training noise; the direction is reported, the width is not
               to be trusted

The comparison is paired on users and unpaired on seeds, for the reason spelled
out in `seed_variance.paired_difference`: both arms score the same 1000 people,
so a resample that draws the hard insiders drags both down together and that
part of the noise cancels; two training runs, by contrast, are genuinely
independent and pairing them would hide the variance this exists to expose.

Communication figures are not bootstrapped. Across five baseline seeds the total
varied by 2.9 MB out of 8021, so the point estimate is the measurement.

Usage:

    python analysis/findings.py                 every claim the data can reach
    python analysis/findings.py --min-seeds 3   ignore claims that are still thin
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.seed_variance import (  # noqa: E402
    load_runs, paired_difference, plateau_frames, plateau_pr_auc,
)
from config_manager import config  # noqa: E402

BASELINE = "baseline"

# Most of what the thesis claims about compression is not "this is better", it
# is "this costs nothing". Those need different evidence, and treating them the
# same hides the difference between a tight interval around zero and an interval
# too wide to say anything. fp16 came out at [-0.0075, +0.0398] and weight
# sparsification at 0.05 at [-0.1377, +0.0234]; both exclude zero from neither
# side, but the first rules out any loss worth caring about and the second rules
# out nothing.
EQUIVALENCE = "esdeger"    # the claim is "no meaningful loss"
SUPERIORITY = "ustun"      # the claim is "measurably better"

# A scale for "how large a loss would still be negligible", not a pass/fail gate.
# 0.02 PR-AUC is about three times the baseline's own across-seed spread (sd
# 0.0069 over five seeds) and a little above the range those five seeds covered
# (0.0194), so a difference under it cannot be separated from rerunning the same
# configuration.
#
# Do not read the verdict it produces as a decision when the interval's endpoint
# lands near the margin. Endpoints carry their own Monte Carlo uncertainty of
# roughly 0.001, and three reported arms sit within 0.002 of the boundary: their
# verdict flips with the bootstrap seed even though the interval itself
# reproduces exactly at the default one. The thesis reports the interval and what
# it fails to rule out; `esdeger` here is a summary label, not the claim.
DEFAULT_MARGIN = 0.02

# The results chapter, as comparisons. A is the configuration being argued for.
CLAIMS = [
    ("quantization-fp16", BASELINE, EQUIVALENCE,
     "fp16 nicemleme tespite mal olmuyor"),
    ("top-k-0.1", BASELINE, EQUIVALENCE,
     "agirlik seyreklestirme 0.1'de tespite mal olmuyor"),
    ("top-k-0.05", BASELINE, EQUIVALENCE,
     "agirlik seyreklestirme 0.05'te tespite mal olmuyor"),
    ("delta-0.1", BASELINE, EQUIVALENCE,
     "delta seyreklestirme 0.1'de tespite mal olmuyor"),
    ("delta-0.05", BASELINE, EQUIVALENCE,
     "delta seyreklestirme 0.05'te tespite mal olmuyor"),
    ("delta-0.1", "top-k-0.1", SUPERIORITY,
     "ayni oranda delta, agirlik varyantindan iyi"),
    ("delta-0.05", "top-k-0.05", SUPERIORITY,
     "ayni oranda delta, agirlik varyantindan iyi (agresif)"),
    ("bidirectional-fp16", BASELINE, EQUIVALENCE,
     "cift yonlu fp16 tespite mal olmuyor"),
    ("top-k-0.1-quant-fp16", BASELINE, EQUIVALENCE,
     "birlesik konfigurasyon tespite mal olmuyor"),
    ("non-iid-baseline", BASELINE, EQUIVALENCE,
     "non-IID bolunme tespiti dusurmuyor"),
    ("fedprox-non-iid", "non-iid-baseline", SUPERIORITY,
     "FedProx non-IID altinda FedAvg'den iyi"),
    # The second form of heterogeneity. Dirichlet skews how much data a client
    # holds; this skews what is in it. Note what the partition does and does not
    # do: it draws a separate Dirichlet per job role and splits that role's
    # members across clients, the way label-skewed benchmarks draw one per class.
    # No client ends up holding a single role. Measured over five seeds, a client
    # holds 9.9 roles on average and none holds one, while the size-weighted
    # total variation between a client's role mixture and the global one reaches
    # 0.537 against quantity skew's 0.311. See analysis/partition_skew.py.
    ("role-non-iid-baseline", BASELINE, EQUIVALENCE,
     "role gore bolumleme tespiti dusurmuyor"),
    ("role-non-iid-baseline", "non-iid-baseline", EQUIVALENCE,
     "role gore bolumleme, miktar carpikligindan kotu degil"),
    ("role-fedprox-non-iid", "role-non-iid-baseline", SUPERIORITY,
     "FedProx role gore bolumlemede FedAvg'den iyi"),
    ("features-filtered", BASELINE, EQUIVALENCE,
     "sabit oznitelikleri atmak zarar vermiyor"),
    ("rounds-25-epochs-10", BASELINE, EQUIVALENCE,
     "25 tur x 10 epoch, 50 tur x 5 epoch kadar iyi"),
    # The configuration the thesis recommends: delta sparsification on the
    # uplink, fp16 on the downlink. Registered as its own claim because it is
    # the one the conclusion chapter rests on.
    ("delta-0.05-downlink-fp16", BASELINE, EQUIVALENCE,
     "onerilen konfigurasyon tespite mal olmuyor"),
    ("delta-0.05-downlink-fp16", "top-k-0.1-quant-fp16", SUPERIORITY,
     "onerilen konfigurasyon agirlik seyreklestirmeli birlesikten iyi"),
    # Two architecture choices the thesis asserts but never measured. A is the
    # choice as built, so a null result means the assertion is unsupported.
    (BASELINE, "encoder-unidirectional", SUPERIORITY,
     "cift yonlu kodlayici tek yonluden iyi"),
    (BASELINE, "no-bottleneck", SUPERIORITY,
     "darbogaz katmani tespite katki sagliyor"),
    # The IID control for FedProx: if the proximal term helps here too, then it
    # is a regularizer and not a heterogeneity remedy.
    ("fedprox-baseline", BASELINE, SUPERIORITY,
     "FedProx IID altinda da FedAvg'den iyi"),
]

# Each scoring stage as (stage key, label, arm that has it, arm that does not),
# so the reported figure always means "what this stage contributes". The stage
# key is carried so a test can check the pair against the configuration instead
# of trusting the arm names to still mean what they say.
DEFAULT_PIPELINE = "ablation-full"
ABLATIONS = [
    ("zscore", "z-skor kalibrasyonu", DEFAULT_PIPELINE, "ablation-no-zscore"),
    ("topk", "top-m oznitelik odagi", DEFAULT_PIPELINE, "ablation-no-topk"),
    # Inverted, and this is the point of writing the pairs out. The diversity
    # multiplier is the one stage the default pipeline leaves off, so the arm
    # that *has* it is `ablation-with-diversity`. The old
    # `ablation-full` vs `ablation-no-diversity` pair survived the v8 flip as
    # working code that measured nothing: both arms are the same three stages,
    # so the row reported -0.0003 and called a stage we removed on evidence
    # indistinguishable.
    ("diversity", "cesitlilik carpani", "ablation-with-diversity", DEFAULT_PIPELINE),
    ("persistence", "zamansal kalicilik", DEFAULT_PIPELINE,
     "ablation-no-persistence"),
]


def scored_runs(runs, name):
    """Each seed's plateau frames: the rounds the reported metric averages over.

    Not the argmax round, and not `summary['best_round']`. Before the v8
    rescore that field selected over round 0 and round 0 won it for a third of
    the finished runs, baseline seeds among them; reading it is what sent the
    literature comparison against an untrained model. Every claim below is
    measured on the same statistic the results chapter quotes.
    """
    if name not in runs:
        return []
    collected = []
    for _, exp_dir in runs[name]:
        frames = plateau_frames(exp_dir)
        if frames:
            collected.append(frames)
    return collected


def detection_summary(runs, name):
    """Mean PR-AUC across seeds, and the spread if there is more than one."""
    points = [plateau_pr_auc(frames) for frames in scored_runs(runs, name)]
    if not points:
        return None
    return {
        "n": len(points),
        "mean": statistics.mean(points),
        "sd": statistics.stdev(points) if len(points) > 1 else float("nan"),
    }


def communication_summary(runs, name):
    """Mean traffic over the seeds that finished, in MB."""
    if name not in runs:
        return None
    totals, uploads, downloads = [], [], []
    for summary, _ in runs[name]:
        totals.append(summary.get("total_communication_mb"))
        uploads.append(summary.get("total_upload_mb"))
        downloads.append(summary.get("total_download_mb"))
    if not any(totals):
        return None
    return {
        "total": statistics.mean(totals),
        "upload": statistics.mean(uploads),
        "download": statistics.mean(downloads),
    }


def verdict(difference, kind, seeds_a, seeds_b, min_seeds, margin):
    """How far the evidence goes, in one word rather than a p-value.

    An equivalence claim is supported when the interval's worst case is a loss
    smaller than the margin. It is refused when the interval's best case is
    still a loss larger than the margin. In between the evidence is too wide to
    decide, which is a different state from either and has to be reported as
    such rather than folded into "no significant difference".
    """
    if min(seeds_a, seeds_b) < min_seeds:
        return "gecici"

    if kind == EQUIVALENCE:
        if difference["CI_Lo"] > -margin:
            return "esdeger"
        if difference["CI_Hi"] < -margin:
            return "kotu"
        return "belirsiz"

    if difference["CI_Lo"] > 0:
        return "ustun"
    if difference["CI_Hi"] < 0:
        return "kotu"
    return "ayrilmamis"


def report_configurations(runs):
    """One row per configuration: what it detects and what it costs."""
    reference = communication_summary(runs, BASELINE)
    print("\n" + "=" * 78)
    print("KONFIGURASYONLAR")
    print("=" * 78)
    print(f"{'deney':28s} {'n':>2} {'PR-AUC':>8} {'sd':>7} "
          f"{'toplam MB':>10} {'tasarruf':>9}")

    for name in sorted(runs):
        detection = detection_summary(runs, name)
        if detection is None:
            continue
        traffic = communication_summary(runs, name)
        if traffic and reference and traffic["total"] > 0:
            saving = f"{(1 - traffic['total'] / reference['total']) * 100:8.1f}%"
            total = f"{traffic['total']:10.1f}"
        else:
            # Ablations rescore another run's weights, so they send nothing.
            saving, total = "        -", "         -"
        sd = ("      -" if detection["n"] < 2
              else f"{detection['sd']:7.4f}")
        print(f"{name:28s} {detection['n']:2d} {detection['mean']:8.4f} {sd} "
              f"{total} {saving}")


def report_claims(runs, iterations, confidence, min_seeds, margin):
    """Each claim, with the interval that decides it."""
    print("\n" + "=" * 78)
    print("IDDIALAR")
    print("=" * 78)
    print(f"{'iddia':52s} {'fark':>8} {'%95 aralik':>18} {'karar':>11}")

    rows = []
    for name_a, name_b, kind, claim in CLAIMS:
        arm_a, arm_b = scored_runs(runs, name_a), scored_runs(runs, name_b)
        if not arm_a or not arm_b:
            print(f"{claim:52s} {'veri yok':>8}")
            continue
        if len({len(labels) for arm in arm_a + arm_b
                for labels, _ in arm}) != 1:
            print(f"{claim:52s}  kollar farkli kullanici sayisi, atlandi")
            continue

        difference = paired_difference(arm_a, arm_b, iterations, confidence)
        decision = verdict(difference, kind, len(arm_a), len(arm_b),
                           min_seeds, margin)
        interval = f"[{difference['CI_Lo']:+.4f},{difference['CI_Hi']:+.4f}]"
        print(f"{claim:52s} {difference['Mean_Difference']:+8.4f} "
              f"{interval:>18} {decision:>11}")
        rows.append((claim, kind, difference, decision))
    return rows


def report_ablations(runs, iterations, confidence, min_seeds, margin):
    """What each scoring stage is worth: the arm holding it minus the arm without."""
    print("\n" + "=" * 78)
    print("SKORLAMA ASAMALARI (asamanin katkisi: olan kol eksi olmayan kol)")
    print("=" * 78)
    print(f"{'asama':28s} {'n':>2} {'katki':>8} {'%95 aralik':>18} {'karar':>11}")

    for _, label, with_name, without_name in ABLATIONS:
        with_stage = scored_runs(runs, with_name)
        without = scored_runs(runs, without_name)
        if not with_stage or not without:
            missing = with_name if not with_stage else without_name
            print(f"{label:28s} {'-':>2} veri yok ({missing})")
            continue
        # Signed so that a stage which helps shows a positive figure.
        difference = paired_difference(with_stage, without, iterations, confidence)
        decision = verdict(difference, SUPERIORITY, len(with_stage), len(without), min_seeds, margin)
        interval = f"[{difference['CI_Lo']:+.4f},{difference['CI_Hi']:+.4f}]"
        print(f"{label:28s} {len(without):2d} "
              f"{difference['Mean_Difference']:+8.4f} {interval:>18} "
              f"{decision:>11}")


GROUPS = [
    ("esdeger", "Esdeger sayilabilir (kayip marjin altinda kaliyor)"),
    ("ustun", "Olculebilir sekilde iyi"),
    ("kotu", "Olculebilir sekilde kotu; tezde bu sekilde yazilmali"),
    ("belirsiz", "Aralik cok genis; ne esdeger ne kotu denebilir"),
    ("ayrilmamis", "Ayirt edilemiyor"),
    ("gecici", "Tohum sayisi yetersiz; yon var, aralik dar degil"),
]


def report_conclusions(rows, margin):
    """The sentences the results chapter is allowed to write."""
    print("\n" + "=" * 78)
    print(f"YAZILABILIR OLANLAR (esdegerlik marjini {margin})")
    print("=" * 78)

    for decision, heading in GROUPS:
        group = [r for r in rows if r[3] == decision]
        if not group:
            continue
        print(f"\n{heading}:")
        for claim, _, difference, _ in group:
            print(f"  - {claim}: {difference['Mean_Difference']:+.4f} "
                  f"[{difference['CI_Lo']:+.4f}, {difference['CI_Hi']:+.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-seeds", type=int, default=3,
                        help="below this a claim is reported as provisional")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="PR-AUC loss the thesis calls negligible")
    parser.add_argument("--seeds", default="",
                        help="comma separated run seeds, e.g. 1,2,3. Restrict "
                             "to the seeds every experiment has finished, or "
                             "the arms are averaged over different numbers of "
                             "runs and the comparison stops being paired.")
    args = parser.parse_args()

    iterations = config.get("evaluation", "bootstrap_iterations")
    confidence = config.get("evaluation", "bootstrap_confidence")

    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
    runs = load_runs(seeds=seeds or None)
    if not runs:
        raise SystemExit("No finished runs found.")

    report_configurations(runs)
    rows = report_claims(runs, iterations, confidence, args.min_seeds,
                         args.margin)
    report_ablations(runs, iterations, confidence, args.min_seeds, args.margin)
    report_conclusions(rows, args.margin)


if __name__ == "__main__":
    main()
