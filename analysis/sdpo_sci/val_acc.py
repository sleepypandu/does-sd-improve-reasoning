"""Cross-domain validation accuracy of the SciKnowEval / ToolAlpaca runs: Table 1 and Figure 6.

Every run validates on the test splits of all five domains (sdpo/datasets/allval/test.parquet) and
logs val-core/<domain>/acc/mean@<n> to wandb, one key per evaluated domain. This reads them from the
local wandb datastores (run-*/*.wandb and offline-run-*/*.wandb; both modes write them) of runs
named <method>-<trained domain>-lr<lr>[-<tag>]-s<seed>, as produced by scripts/sdpo_sci.

  python analysis/sdpo_sci/val_acc.py table  --wandb-dir outputs/sdpo_sci/wandb
  python analysis/sdpo_sci/val_acc.py figure --wandb-dir outputs/sdpo_sci/wandb --out figure6.pdf
  python analysis/sdpo_sci/val_acc.py csv    --wandb-dir outputs/sdpo_sci/wandb --out val_acc.csv

Only runs that logged every domain at step 0 and at the final step count. Table 1 reports the seed
mean ± std (ddof=1) at the final step per evaluated domain. The Base row is step 0 of every counted
run, pooled over methods and trained domains. OOD change is, per seed, the mean over the four domains
not trained on of (final − Base mean), then mean ± std over seeds. Figure 6 uses, per (method,
trained domain), only the validation steps present in every seed.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

DOMAINS = ["biology", "chemistry", "material", "physics", "tooluse"]
TITLES = {"biology": "Biology", "chemistry": "Chemistry", "material": "Material", "physics": "Physics", "tooluse": "Tool use"}
METHODS = ["sdpo", "grpo"]
LABELS = {"grpo": "GRPO", "sdpo": "SDPO"}
COLORS = {"grpo": "#1f77b4", "sdpo": "#2ca02c"}
PAPER_LRS = {"grpo": "1e-6", "sdpo": "1e-5"}

RUN_RE = re.compile(r"^(sdpo|grpo)-(" + "|".join(DOMAINS) + r")-lr(\d+(?:\.\d+)?e-?\d+)(?:-.+?)?-s(\d+)$")
# SciKnowEval domains are logged as sciknoweval_<domain>, ToolAlpaca as tooluse.
METRIC_RE = re.compile(r"^val-core/(?:sciknoweval_)?(" + "|".join(DOMAINS) + r")/acc/mean@\d+$")


def read_run(path):
    """Return (run name, {step: {evaluated domain: accuracy}}) from one .wandb datastore."""
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
            if "_step" not in row:
                continue
            step = int(json.loads(row["_step"]))
            for key, value in row.items():
                m = METRIC_RE.match(key)
                if m:
                    points.setdefault(step, {})[m.group(1)] = float(json.loads(value))
    return name, points


def collect(wandb_dirs, steps):
    """Return {run name: {step: {domain: acc}}} for runs with every domain at step 0 and `steps`.

    For a name logged more than once (a killed attempt and its rerun), keep the longest attempt.
    """
    runs = {}
    for wandb_dir in wandb_dirs:
        for path in sorted(glob.glob(os.path.join(wandb_dir, "*run-*", "*.wandb"))):
            name, points = read_run(path)
            if not (name and RUN_RE.match(name)):
                continue
            if not all(set(DOMAINS) <= set(points.get(t, {})) for t in (0, steps)):
                continue
            if len(points) > len(runs.get(name, {})):
                runs[name] = points
    return runs


def arm(runs, method, lr, trained):
    """Return {seed: {step: {domain: acc}}} for one (method, lr, trained domain)."""
    found = {}
    for name, points in runs.items():
        m, ds, run_lr, seed = RUN_RE.match(name).groups()
        if (m, ds, run_lr) == (method, trained, lr):
            if int(seed) in found:
                raise SystemExit(f"two runs for {method} {trained} lr={lr} seed {seed}; pass a narrower --wandb-dir")
            found[int(seed)] = points
    return found


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return x.mean(0), (x.std(0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1:]))


def fmt(mean, std, signed=False):
    sign = ("+" if mean >= 0 else "−") if signed else ""
    return f"{sign}{100 * abs(mean) if signed else 100 * mean:.1f} ± {100 * std:.1f}"


def table(runs, lrs, steps):
    print("| Trained on | " + " | ".join(TITLES[d] for d in DOMAINS) + " | OOD change | seeds |")
    print("|---|" + "---|" * (len(DOMAINS) + 2))
    base = []
    for method in METHODS:
        for trained in DOMAINS:
            base += [[p[0][d] for d in DOMAINS] for p in arm(runs, method, lrs[method], trained).values()]
    if not base:
        raise SystemExit("no runs")
    base_mean, base_std = mean_std(base)
    base_mean = dict(zip(DOMAINS, base_mean))
    print("| Base | " + " | ".join(fmt(base_mean[d], b) for d, b in zip(DOMAINS, base_std)) + f" | | {len(base)} runs |")
    for method in METHODS:
        print(f"| **{LABELS[method]}** (lr={lrs[method]}) |" + " |" * (len(DOMAINS) + 2))
        for trained in DOMAINS:
            seeds = arm(runs, method, lrs[method], trained)
            if not seeds:
                print(f"| {TITLES[trained]} |" + " -- |" * (len(DOMAINS) + 2))
                continue
            cells = []
            for d in DOMAINS:
                text = fmt(*mean_std([p[steps][d] for p in seeds.values()]))
                cells.append(f"**{text}**" if d == trained else text)
            others = [d for d in DOMAINS if d != trained]
            ood = [np.mean([p[steps][d] - base_mean[d] for d in others]) for p in seeds.values()]
            print(f"| {TITLES[trained]} | " + " | ".join(cells) + f" | {fmt(*mean_std(ood), signed=True)} | {len(seeds)} |")


def curve(seeds, evaluated):
    """Return (steps, mean, std) over the steps every seed logged."""
    steps = sorted(set.intersection(*(set(p) for p in seeds.values())))
    mean, std = mean_std([[p[t][evaluated] for t in steps] for p in seeds.values()])
    return np.array(steps), mean, std


def figure(runs, lrs, steps, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 12,
            "axes.grid": True,
            "grid.color": "#dadada",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    n = len(DOMAINS)
    order = ["grpo", "sdpo"]
    fig, axes = plt.subplots(n, n, figsize=(14, 13), sharex=True, sharey="col")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.085, wspace=0.22, hspace=0.18)
    handles = {}
    for i, trained in enumerate(DOMAINS):
        for j, evaluated in enumerate(DOMAINS):
            ax = axes[i, j]
            for method in order:
                seeds = arm(runs, method, lrs[method], trained)
                if not seeds:
                    continue
                x, mean, std = curve(seeds, evaluated)
                (handles[method],) = ax.plot(x, mean, "-", lw=1.6, color=COLORS[method])
                ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.18, linewidth=0)
            if i == j:  # the trained domain
                ax.set_facecolor("#f4f4f4")
            ax.set_xlim(0, steps)
            ax.set_xticks(np.linspace(0, steps, 5))
            if i == 0:
                ax.set_title(f"Eval: {TITLES[evaluated]}")
            if i == n - 1:
                ax.set_xlabel("Training step")
            if j == 0:
                ax.set_ylabel(f"Trained: {TITLES[trained]}\nmean@16 accuracy")
    present = [m for m in order if m in handles]
    legend = fig.legend(
        [handles[m] for m in present],
        [LABELS[m] for m in present],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(present),
        handlelength=2.2,
        columnspacing=2.6,
        borderpad=0.5,
        handletextpad=0.7,
    )
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.8)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def write_csv(runs, out):
    with open(out, "w") as f:
        f.write("run,method,trained,lr,seed,step,evaluated,acc\n")
        for name, points in sorted(runs.items()):
            method, trained, lr, seed = RUN_RE.match(name).groups()
            for step, accs in sorted(points.items()):
                for evaluated, acc in sorted(accs.items()):
                    f.write(f"{name},{method},{trained},{lr},{seed},{step},{evaluated},{acc}\n")
    print(f"wrote {out} ({len(runs)} runs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["table", "figure", "csv"])
    parser.add_argument("--wandb-dir", nargs="+", required=True, help="directories containing (offline-)run-*")
    parser.add_argument("--lr", nargs="*", default=[], metavar="METHOD=LR", help="arm learning rates (default: paper)")
    parser.add_argument("--steps", type=int, default=200, help="final training step (default: 200)")
    parser.add_argument("--out", help="output file for figure/csv")
    args = parser.parse_args()

    lrs = dict(PAPER_LRS, **dict(item.split("=", 1) for item in args.lr))
    runs = collect(args.wandb_dir, args.steps)
    print(f"{len(runs)} runs validated on every domain at step 0 and step {args.steps}")
    if args.command == "table":
        table(runs, lrs, args.steps)
    elif args.command == "figure":
        figure(runs, lrs, args.steps, args.out or "figure6.pdf")
    else:
        write_csv(runs, args.out or "val_acc.csv")


if __name__ == "__main__":
    main()
