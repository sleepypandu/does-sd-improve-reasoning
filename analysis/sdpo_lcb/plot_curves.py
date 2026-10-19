"""LCB validation curves, GRPO vs SDPO: one figure per model, one panel per split.

Mean ± std (ddof=1) over seeds of val-aux/<split>/score/mean@4, at the steps every seed
validated at (test_freq=10).

    python analysis/sdpo_lcb/plot_curves.py --logs outputs/sdpo_lcb --out figures/sdpo_lcb
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from lcb_metrics import METHOD_LABELS, METHODS, MODELS, SPLIT_LABELS, load_scores

PANELS = ["livecodebench_test_unseen", "livecodebench_train_seen", "livecodebench_train_unseen"]
COLORS = {"grpo": "#1f77b4", "sdpo": "#2ca02c"}
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def curve(seeds, split, max_step):
    # Only steps every seed validated at; a resumed run also validates at its resume step.
    steps = sorted(set.intersection(*({s for s in d if s <= max_step and split in d[s]} for d in seeds.values())))
    vals = np.array([[d[s][split] for d in seeds.values()] for s in steps])
    std = vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(steps))
    return np.array(steps), vals.mean(axis=1), std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="run directory root, or a directory of LSF logs")
    ap.add_argument("--out", default="figures/sdpo_lcb")
    ap.add_argument("--max_step", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scores = load_scores(args.logs)
    with plt.rc_context(RC):
        for model in MODELS:
            if not any((model, m) in scores for m in METHODS):
                continue
            fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.1))
            fig.subplots_adjust(wspace=0.38)
            for ax, split in zip(axes, PANELS):
                for method in METHODS:
                    seeds = scores.get((model, method))
                    if not seeds:
                        continue
                    x, mean, std = curve(seeds, split, args.max_step)
                    ax.plot(x, mean, color=COLORS[method], label=f"{METHOD_LABELS[method]} (n={len(seeds)})")
                    ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.15, lw=0)
                ax.set_title(f"LCB {SPLIT_LABELS[split]}")
                ax.set_xlabel("Training step")
                ax.set_xlim(0, args.max_step)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            axes[0].set_ylabel("mean@4 score")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                       bbox_to_anchor=(0.5, 1.08))
            fig.suptitle(model, y=1.14)
            path = os.path.join(args.out, f"lcb_curves_{model}")
            fig.savefig(path + ".pdf")
            fig.savefig(path + ".png")
            plt.close(fig)
            print(f"wrote {path}.pdf")


if __name__ == "__main__":
    main()
