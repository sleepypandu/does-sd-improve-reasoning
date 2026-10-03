"""Validation accuracy of the SciKnowEval / ToolAlpaca runs: Figure 4 and Table 7.

Reads val-core/<data_source>/acc/mean@<n> from the local wandb datastores (run-*/*.wandb and
offline-run-*/*.wandb; both modes write them) of runs named <method>-<dataset>-lr<lr>-s<seed>, as
produced by scripts/sdpo_sci. The RL trainer logs validation accuracy only to wandb.

  python analysis/sdpo_sci/val_acc.py table  --wandb-dir outputs/sdpo_sci/wandb
  python analysis/sdpo_sci/val_acc.py figure --wandb-dir outputs/sdpo_sci/wandb --out figure4.pdf
  python analysis/sdpo_sci/val_acc.py csv    --wandb-dir outputs/sdpo_sci/wandb --out val_acc.csv

Curves and the table use, per (method, lr, dataset), only the validation steps present in every seed.
Table 7 reports the step whose seed-mean accuracy is highest, with the seed std at that step.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

DATASETS = ["biology", "chemistry", "material", "physics", "tooluse"]
TITLES = {"biology": "Biology", "chemistry": "Chemistry", "material": "Material", "physics": "Physics", "tooluse": "Tool Use"}
METHODS = ["grpo", "sdpo", "sft"]
LABELS = {"grpo": "GRPO", "sdpo": "SDPO", "sft": "SFT"}
COLORS = {"grpo": "#1f77b4", "sdpo": "#2ca02c", "sft": "#d62728"}
PAPER_LRS = {"grpo": "1e-6", "sdpo": "1e-5", "sft": "1e-5"}

RUN_RE = re.compile(r"^(sft|sdpo|grpo)-(\w+)-lr(\S+?)-s(\d+)$")
METRIC_RE = re.compile(r"^val-core/[^/]+/acc/mean@\d+$")


def read_run(path):
    """Return (run name, {step: accuracy}) from one .wandb datastore."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(path)
    name, points = None, {}
    while True:
        try:
            data = store.scan_data()
        except Exception:
            break
        if data is None:
            break
        record = pb.Record()
        try:
            record.ParseFromString(data)
        except Exception:
            continue
        kind = record.WhichOneof("record_type")
        if kind == "run" and record.run.display_name:
            name = record.run.display_name
        elif kind == "history":
            # verl logs under nested_key and leaves key empty
            row = {item.key or "/".join(item.nested_key): item.value_json for item in record.history.item}
            metric = next((k for k in row if METRIC_RE.match(k)), None)
            if metric is not None and "_step" in row:
                points[int(json.loads(row["_step"]))] = float(json.loads(row[metric]))
    return name, points


def collect(wandb_dirs):
    """Return {run name: {step: accuracy}}; for a name logged more than once, the longest attempt."""
    runs = {}
    for wandb_dir in wandb_dirs:
        for path in sorted(glob.glob(os.path.join(wandb_dir, "*run-*", "*.wandb"))):
            name, points = read_run(path)
            if name and RUN_RE.match(name) and points and len(points) > len(runs.get(name, {})):
                runs[name] = points
    return runs


def seed_matrix(runs, method, lr, dataset):
    """Return (steps, accuracy[seed, step]) over the steps every seed logged, or None."""
    seeds = {}
    for name, points in runs.items():
        m, ds, run_lr, seed = RUN_RE.match(name).groups()
        if (m, ds, run_lr) == (method, dataset, lr):
            seeds[int(seed)] = points
    if not seeds:
        return None
    steps = sorted(set.intersection(*(set(p) for p in seeds.values())))
    if not steps:
        return None
    return np.array(steps), np.array([[seeds[s][t] for t in steps] for s in sorted(seeds)])


def mean_std(acc):
    std = acc.std(0, ddof=1) if len(acc) > 1 else np.zeros(acc.shape[1])
    return acc.mean(0), std


def table(runs, lrs):
    print("| method | " + " | ".join(TITLES[d] for d in DATASETS) + " |")
    print("|---|" + "---|" * len(DATASETS))
    for method in METHODS:
        cells = []
        for dataset in DATASETS:
            found = seed_matrix(runs, method, lrs[method], dataset)
            if found is None:
                cells.append("--")
                continue
            steps, acc = found
            mean, std = mean_std(acc)
            i = int(mean.argmax())
            cells.append(f"{100 * mean[i]:.2f} ± {100 * std[i]:.2f} (step {steps[i]}, n={len(acc)})")
        print(f"| {LABELS[method]} (lr={lrs[method]}) | " + " | ".join(cells) + " |")


def figure(runs, lrs, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # drawn at about twice the printed width, so text is scaled by 1.9
    scale = 1.9
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9 * scale,
            "axes.titlesize": 11 * scale,
            "axes.labelsize": 10 * scale,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "legend.fontsize": 10 * scale,
            "axes.grid": True,
            "grid.color": "#dadada",
            "grid.linewidth": 0.7 * scale,
            "axes.axisbelow": True,
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.9 * scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(15, 4.3))
    grid = fig.add_gridspec(1, len(DATASETS), left=0.075, right=0.985, top=0.92, bottom=0.355, wspace=0.30)
    handles = {}
    for j, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(grid[0, j])
        for method in METHODS:
            found = seed_matrix(runs, method, lrs[method], dataset)
            if found is None:
                continue
            steps, acc = found
            mean, std = mean_std(acc)
            (handles[method],) = ax.plot(steps, mean, "-", lw=1.6 * scale, color=COLORS[method])
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[method], alpha=0.18, linewidth=0)
        ax.set_title(TITLES[dataset])
        ax.set_xlabel("Training step")
        ax.set_xticks([0, 100, 200])
        ax.set_xlim(0, 200)
        if j == 0:
            ax.set_ylabel("mean@16 accuracy")
    order = [m for m in METHODS if m in handles]
    legend = fig.legend(
        [handles[m] for m in order],
        [LABELS[m] for m in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(order),
        handlelength=2.2,
        columnspacing=2.6,
        borderpad=0.5,
        handletextpad=0.7,
    )
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.8)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def write_csv(runs, out):
    with open(out, "w") as f:
        f.write("run,method,dataset,lr,seed,step,acc\n")
        for name, points in sorted(runs.items()):
            method, dataset, lr, seed = RUN_RE.match(name).groups()
            for step, acc in sorted(points.items()):
                f.write(f"{name},{method},{dataset},{lr},{seed},{step},{acc}\n")
    print(f"wrote {out} ({len(runs)} runs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["table", "figure", "csv"])
    parser.add_argument("--wandb-dir", nargs="+", required=True, help="directories containing (offline-)run-*")
    parser.add_argument("--lr", nargs="*", default=[], metavar="METHOD=LR", help="arm learning rates (default: paper)")
    parser.add_argument("--out", help="output file for figure/csv")
    args = parser.parse_args()

    lrs = dict(PAPER_LRS, **dict(item.split("=", 1) for item in args.lr))
    runs = collect(args.wandb_dir)
    print(f"{len(runs)} runs with validation accuracy")
    if args.command == "table":
        table(runs, lrs)
    elif args.command == "figure":
        figure(runs, lrs, args.out or "figure4.pdf")
    else:
        write_csv(runs, args.out or "val_acc.csv")


if __name__ == "__main__":
    main()
