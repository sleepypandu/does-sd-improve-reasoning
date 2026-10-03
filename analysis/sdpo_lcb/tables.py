"""LCB strict-score tables at steps 0 / 80 / 200, mean ± std (ddof=1) over seeds, in %.

    python analysis/sdpo_lcb/tables.py --logs outputs/sdpo_lcb [--csv scores.csv]
"""
import argparse
import csv

import numpy as np

from lcb_metrics import METHOD_LABELS, METHODS, MODELS, SPLIT_LABELS, SPLITS, load_scores

STEPS = [0, 80, 200]


def cell(values):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{100 * values[0]:.1f}"
    return f"{100 * np.mean(values):.1f} ± {100 * np.std(values, ddof=1):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="run directory root, or a directory of LSF logs")
    ap.add_argument("--steps", type=int, nargs="+", default=STEPS)
    ap.add_argument("--csv", help="also write per-seed values here")
    args = ap.parse_args()

    scores = load_scores(args.logs)
    rows = []
    for split in SPLITS:
        print(f"\n## {SPLIT_LABELS[split]}  (val-aux/{split}/score/mean@4, %)\n")
        print("| model | method | " + " | ".join(f"step {s}" for s in args.steps) + " |")
        print("|---|---|" + "---|" * len(args.steps))
        for model in MODELS:
            for method in METHODS:
                seeds = scores.get((model, method))
                if not seeds:
                    continue
                cells = []
                for step in args.steps:
                    vals = {seed: d[step][split] for seed, d in sorted(seeds.items())
                            if split in d.get(step, {})}
                    cells.append(cell(list(vals.values())) + (f" (n={len(vals)})" if len(vals) != len(seeds) else ""))
                    rows += [dict(split=split, model=model, method=method, step=step, seed=seed, score=v)
                             for seed, v in vals.items()]
                print(f"| {model} | {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["split", "model", "method", "step", "seed", "score"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
