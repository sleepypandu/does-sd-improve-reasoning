"""Figure 1: unique solutions per training problem among 8 samples (Base / GRPO / SDPO).

Reads generation dumps from scripts/sdpo_lcb/dump_generations.py. Each sample's program is
the grader's own extraction (the LONGEST fenced code block); two programs are the same
solution when their ASTs are equal, so comments, whitespace and formatting are ignored.
Bars are the mean over runs of the per-problem unique count, error bars the std (ddof=1):
over training seeds for GRPO/SDPO, over sampling seeds for the base model.

    python analysis/sdpo_lcb/plot_unique_solutions.py --dumps dumps/ --out figures/sdpo_lcb/unique_solutions
"""
import argparse
import ast
import glob
import gzip
import importlib.util
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["Base", "GRPO", "SDPO"]
COLORS = {"Base": "0.55", "GRPO": "#1f77b4", "SDPO": "#2ca02c"}
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def grader_extract_code():
    path = os.path.join(REPO_ROOT, "sdpo", "verl", "utils", "reward_score", "feedback", "code.py")
    spec = importlib.util.spec_from_file_location("lcb_code_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_code


extract_code = grader_extract_code()


def solution_key(text):
    code = extract_code(text or "") or ""
    try:
        return ast.dump(ast.parse(code), annotate_fields=False)
    except Exception:
        return "unparsed:" + code


def arm_of(meta):
    ckpt = meta.get("ckpt") or meta.get("run_dir")
    if not ckpt:
        return "Base"
    return "GRPO" if "-grpo-" in ckpt else "SDPO"


def mean_unique(rows):
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["problem_index"]].append(r)
    counts = [len({solution_key(r["text"]) for r in rs}) for rs in by_problem.values()]
    n_samples = min(len(rs) for rs in by_problem.values())
    return float(np.mean(counts)), n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="directory of *.json.gz generation dumps")
    ap.add_argument("--out", default="figures/sdpo_lcb/unique_solutions")
    args = ap.parse_args()

    per_arm, n_samples = defaultdict(list), 0
    for path in sorted(glob.glob(os.path.join(args.dumps, "*.json.gz"))):
        if path.endswith(".graded.json.gz"):
            continue
        with gzip.open(path, "rt") as f:
            dump = json.load(f)
        mean, n_samples = mean_unique(dump["rows"])
        per_arm[arm_of(dump["meta"])].append(mean)
    if not per_arm:
        raise SystemExit(f"no dumps in {args.dumps}")

    arms = [a for a in ARMS if a in per_arm]
    mean = [np.mean(per_arm[a]) for a in arms]
    std = [np.std(per_arm[a], ddof=1) if len(per_arm[a]) > 1 else 0.0 for a in arms]
    for a, m, s in zip(arms, mean, std):
        print(f"{a:<5} {m:.2f} ± {s:.2f}  (n={len(per_arm[a])} runs)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(2.31, 2.30))
        x = np.arange(len(arms))
        ax.bar(x, mean, 0.66, yerr=std, capsize=2.2, color=[COLORS[a] for a in arms],
               edgecolor="white", linewidth=0.6, error_kw=dict(lw=0.8, ecolor="0.25"))
        for xi, m, s in zip(x, mean, std):
            ax.text(xi, m + s + 0.15, f"{m:.2f}", ha="center", va="bottom", fontsize=7, color="0.15")
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_ylabel("Unique solutions per problem")
        ax.set_ylim(0, n_samples)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.2)
        fig.savefig(args.out + ".pdf")
        fig.savefig(args.out + ".png")
    print(f"wrote {args.out}.pdf")


if __name__ == "__main__":
    main()
