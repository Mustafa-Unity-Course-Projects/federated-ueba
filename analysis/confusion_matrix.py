"""The confusion matrix at the reported operating point, averaged the same way.

`thesis_paper.md` section 4.1 lists the confusion matrix among the reported
metrics, and the four counts have always been written to
`federated_rounds_comparison.csv`, but nothing ever read them back out. This does.

Two things have to match how PR-AUC is reported, or the matrix would describe a
different run than the number beside it:

  the rounds     the mean over the plateau window, not one selected round. Counts
                 are integers per round, so the mean is fractional; rounding it
                 to report would break TP + FN = insiders.
  the users      the test half, which is what the per-round counts already hold
                 (500 users, 35 insiders). The reported precision and recall come
                 from these same counts.

Each round carries its own threshold, chosen on the validation half, so the
matrix is the average behaviour at the reported operating point rather than at
one arbitrary cut.

Usage:

    python analysis/confusion_matrix.py
    python analysis/confusion_matrix.py --runs baseline top-k-0.05 --seeds 1,2,3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from analysis.compare_experiments import plateau_rounds  # noqa: E402

FEDERATED_DIR = "federated_evaluation_reports"
CENTRALIZED_DIR = "centralized_evaluation_reports"
COUNTS = ["TP", "FP", "TN", "FN"]

FIGURE_PATH = "generated_visuals/confusion_matrix.png"
CENTRAL_LABEL = "Merkezi model"

# One hue, light to dark, for a magnitude scale. A rainbow would imply the
# middle of the range is a different kind of thing from the ends.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"


def plateau_counts(exp_dir):
    """Mean TP/FP/TN/FN over the rounds the reported metric averages over."""
    path = os.path.join(exp_dir, "federated_rounds_comparison.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    # Round 0 is the model before training, scored against a reference the
    # clients measured with their trained local models. Excluded here for the
    # same reason it is excluded from every other comparison.
    frame = frame[frame["Round"] > 0]
    if frame.empty or not set(COUNTS) <= set(frame.columns):
        return None
    rounds = plateau_rounds(frame["Round"].tolist())
    window = frame[frame["Round"].isin(rounds)]
    if window.empty:
        return None
    return {name: float(window[name].mean()) for name in COUNTS}


def derived(counts):
    """Precision, recall and specificity, from the counts rather than beside them."""
    tp, fp, tn, fn = (counts[name] for name in COUNTS)
    return {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "fpr": fp / (tn + fp) if tn + fp else float("nan"),
    }


def across_seeds(experiment, seeds):
    """One experiment's mean matrix over seeds, and the spread of each count."""
    per_seed = []
    for seed in seeds:
        counts = plateau_counts(os.path.join(FEDERATED_DIR,
                                             f"{experiment}__seed{seed}"))
        if counts:
            per_seed.append(counts)
    if not per_seed:
        return None

    mean = {name: float(np.mean([c[name] for c in per_seed])) for name in COUNTS}
    sd = {name: (float(np.std([c[name] for c in per_seed], ddof=1))
                 if len(per_seed) > 1 else float("nan"))
          for name in COUNTS}
    return {"n": len(per_seed), "mean": mean, "sd": sd, "rates": derived(mean)}


def centralized(seed):
    """The single-machine arm, read from its own summary."""
    path = os.path.join(CENTRALIZED_DIR, f"seed{seed}",
                        "centralized_experiment_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        metrics = json.load(f).get("anomaly_detection_metrics", {})
    # The centralized summary records its counts at the max-F1 operating point,
    # which is the same rule the federated rounds use for theirs.
    counts = {name: metrics.get(f"{name.lower()}_at_best_f1") for name in COUNTS}
    if any(value is None for value in counts.values()):
        return None
    counts = {name: float(value) for name, value in counts.items()}
    return {"n": 1, "mean": counts,
            "sd": {name: float("nan") for name in COUNTS},
            "rates": derived(counts)}


def print_matrix(label, result):
    """The 2x2 laid out, then the rates it implies."""
    mean, sd, rates = result["mean"], result["sd"], result["rates"]
    spread = (lambda name: "" if np.isnan(sd[name]) else f" ±{sd[name]:.1f}")

    print(f"\n{label}  (n={result['n']} seed)")
    print(f"{'':22s}{'tahmin: icerideki':>20s}{'tahmin: normal':>18s}")
    print(f"{'gercek: icerideki':22s}"
          f"{f'TP {mean['TP']:.1f}' + spread('TP'):>20s}"
          f"{f'FN {mean['FN']:.1f}' + spread('FN'):>18s}")
    print(f"{'gercek: normal':22s}"
          f"{f'FP {mean['FP']:.1f}' + spread('FP'):>20s}"
          f"{f'TN {mean['TN']:.1f}' + spread('TN'):>18s}")
    print(f"  kesinlik {rates['precision']:.3f}   duyarlilik {rates['recall']:.3f}   "
          f"ozgulluk {rates['specificity']:.3f}   YP orani {rates['fpr']:.4f}")


def _cell_grid(counts):
    """The 2x2 laid out as it is read, and each cell as a share of its row.

    Normalised per row rather than over the whole matrix, which is the usual
    convention and the only one that works here: 465 of the 500 users are not
    insiders, so on raw counts TN alone would carry the entire colour range and
    the other three cells would all be the lightest step. Row shares put the two
    rows on their own scales, so the top row reads as recall and the bottom row
    as the false alarm rate.
    """
    grid = np.array([[counts["TP"], counts["FN"]],
                     [counts["FP"], counts["TN"]]], dtype=float)
    totals = grid.sum(axis=1, keepdims=True)
    shares = np.divide(grid, totals, out=np.zeros_like(grid),
                       where=totals > 0)
    return grid, shares


def _draw_panel(ax, label, result, cmap):
    """One configuration's matrix. Counts are the data; colour is the ordering."""
    grid, shares = _cell_grid(result["mean"])

    ax.imshow(shares, cmap=cmap, vmin=0.0, vmax=1.0)

    # A 2px gap in the surface colour rather than a border around each cell.
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)

    names = [["TP", "FN"], ["FP", "TN"]]
    for row in range(2):
        for column in range(2):
            share = shares[row, column]
            # White ink once the fill is dark enough to swallow black text.
            colour = "#ffffff" if share > 0.55 else INK
            count = f"{grid[row, column]:.1f}".replace(".", ",")
            ax.text(column, row - 0.15, count, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=colour)
            ax.text(column, row + 0.18,
                    f"{names[row][column]}  %{share * 100:.1f}".replace(".", ","),
                    ha="center", va="center", fontsize=7.5, color=colour)

    # Predicted classes on top, so a panel title never lands on the row of
    # labels belonging to the panel above it.
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["tahmin\niçeriden", "tahmin\nnormal"], fontsize=7.5,
                       color=MUTED)
    ax.set_yticklabels(["gerçek\niçeriden", "gerçek\nnormal"], fontsize=7.5,
                       color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(label, fontsize=10, color=INK, pad=26)


def plot_matrices(results, output_path=FIGURE_PATH, columns=3):
    """One 2x2 per configuration, on a shared scale so panels are comparable.

    Small multiples rather than one grouped chart: a confusion matrix is read as
    a shape, diagonal against off-diagonal, and four configurations interleaved
    on one pair of axes would destroy that shape. The colour scale is shared and
    fixed to [0, 1] so a cell that looks darker in one panel really is a larger
    share of its row than the same cell elsewhere.
    """
    if not results:
        print("  Cizilecek matris yok; sekil atlandi.")
        return None

    cmap = LinearSegmentedColormap.from_list("fueba_blue", BLUE_RAMP)
    rows = int(np.ceil(len(results) / columns))

    fig, axes = plt.subplots(rows, columns,
                             figsize=(3.2 * columns, 3.15 * rows + 1.0),
                             dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.13,
                        hspace=0.48, wspace=0.42)
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, result, _) in zip(axes, results):
        ax.set_facecolor(SURFACE)
        _draw_panel(ax, label, result, cmap)
    for ax in axes[len(results):]:
        ax.set_visible(False)

    # Its own axes rather than one stolen from the grid, so the panel spacing
    # above is not rearranged to make room for it.
    bar_ax = fig.add_axes([0.28, 0.055, 0.44, 0.014])
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=bar_ax,
                       orientation="horizontal")
    bar.set_label("hücrenin kendi satırındaki payı", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    bar.outline.set_visible(False)

    federated = max((result["n"] for label, result, _ in results
                     if label != CENTRAL_LABEL), default=0)
    fig.text(0.5, 0.017,
             f"Test yarısı: 500 kullanıcı, 35 içeriden. Federe kollar plato "
             f"penceresindeki turların ortalaması, {federated} seed; merkezi "
             f"kol tek seed. Eşik her turda doğrulama yarısında max-F1'e göre "
             f"seçildi.",
             ha="center", va="center", fontsize=7.5, color=MUTED)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stem = os.path.splitext(output_path)[0]
    for extension in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{extension}", bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"\nSaved '{output_path}' (+ pdf, svg)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*",
                        default=["baseline", "quantization-fp16",
                                 "bidirectional-fp16", "delta-0.05",
                                 "top-k-0.05"])
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--output", default="generated_visuals/confusion_matrix.csv")
    parser.add_argument("--figure", default=FIGURE_PATH)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows, panels = [], []
    central = centralized(seeds[0])
    if central:
        print_matrix(f"{CENTRAL_LABEL} (seed {seeds[0]})", central)
        rows.append({"Experiment": "Centralized", "Seeds": 1,
                     **central["mean"], **central["rates"]})
        panels.append((CENTRAL_LABEL, central, None))

    for experiment in args.runs:
        result = across_seeds(experiment, seeds)
        if result is None:
            print(f"\n{experiment}: tur tablosu yok, atlandi.")
            continue
        print_matrix(f"Federe: {experiment}", result)
        rows.append({"Experiment": experiment, "Seeds": result["n"],
                     **result["mean"], **result["rates"]})
        panels.append((experiment, result, None))

    if rows:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nSaved '{args.output}'")

    if panels and not args.no_figure:
        plot_matrices(panels, args.figure)


if __name__ == "__main__":
    main()
