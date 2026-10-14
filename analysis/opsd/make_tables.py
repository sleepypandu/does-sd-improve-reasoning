"""Build the paper's OPSD tables from evaluate_math.py result JSONs.

    python analysis/opsd/make_tables.py --results outputs/opsd/eval_results --out tables/opsd

- accuracy (Tables 2, 8-11): per model and eval mode, the checkpoint with the best seed-averaged
  accuracy (mean over AIME24/AIME25/HMMT25) among checkpoints every seed has evaluated on every
  benchmark; mean +- std over seeds per benchmark at that checkpoint.
- length (Tables 12-16): mean response tokens at checkpoint 200 and at the accuracy-selected
  checkpoint, and % of thinking-off rollouts with an orphan </think>; benchmarks are pooled
  within a seed, then mean +- std over seeds.
- masking (Table 17): loss-mask arms (Qwen3-8B, teacher thinking on), selected like the accuracy
  tables.

Length statistics tokenize every generation of the selected files; they are cached in --cache.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from corpus import BENCHMARKS, generation_stats, scan

BENCH_LABEL = {"aime24": "AIME24", "aime25": "AIME25", "hmmt25": "HMMT25"}
MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Olmo-3-7B-Instruct"]
GROUPS = [
    ("Baselines", [("base", "Base"), ("grpo_think", r"GRPO$_{\text{think}}$"),
                   ("grpo_nothink", r"GRPO$_{\text{nothink}}$"), ("sft", "SFT")]),
    ("OPSD (teacher thinking-on)", [("opsd/tt=T/z=normal", "Ref.\\ $z$"), ("opsd/tt=T/z=none", "No $z$"),
                                    ("opsd/tt=T/z=swap", "Shuffled $z$")]),
    ("OPSD (teacher thinking-off)", [("opsd/tt=F/z=normal", "Ref.\\ $z$"), ("opsd/tt=F/z=none", "No $z$"),
                                     ("opsd/tt=F/z=swap", "Shuffled $z$")]),
]
MASK_GROUPS = [
    (f"Loss on {label}", [(f"opsd/tt=T/z={z}/mask={mask}", name) for z, name in
                          [("normal", "Ref.\\ $z$"), ("none", "No $z$"), ("swap", "Shuffled $z$")]])
    for mask, label in [("last_10pct", "last 10\\%"), ("all_but_last_10pct", "all but last 10\\%")]
]
MODES = [(True, "on"), (False, "off")]


def select_step(df, model, arm, mode):
    """Accuracy-selected checkpoint (NaN for the untrained base), or None if the arm has no evals."""
    a = df[(df.model == model) & (df.arm == arm) & (df.eval_thinking == mode)]
    if a.empty:
        return None
    if arm == "base":
        return np.nan
    n_seeds = a.seed.nunique()
    per_step = a.groupby(["step", "benchmark"]).seed.nunique().unstack()
    complete = per_step.index[(per_step.reindex(columns=BENCHMARKS) >= n_seeds).all(axis=1)]
    a = a[a.step.isin(complete)]
    if a.empty:
        return None
    objective = a.groupby(["step", "benchmark"]).acc.mean().groupby("step").mean()
    return float(objective.round(9).idxmax())  # exact ties go to the earliest checkpoint


def at_step(df, model, arm, mode, step):
    a = df[(df.model == model) & (df.arm == arm) & (df.eval_thinking == mode)]
    return a[a.step.isna()] if step != step else a[a.step == step]


def mean_std(values, fmt):
    values = np.asarray(values, dtype=float)
    s = fmt.format(values.mean())
    return s + r"$\pm$" + fmt.format(values.std(ddof=1)) if len(values) > 1 else s


def latex_table(header, groups, rows, caption, label):
    n = len(header)
    out = [r"\begin{table}[h]", r"  \centering", r"  \caption{" + caption + "}", r"  \label{" + label + "}",
           r"  \begin{tabular}{l" + "c" * n + "}", r"    \toprule"]
    out += ["    " + line for line in header_lines(header)]
    out.append(r"    \midrule")
    for gi, (group, arms) in enumerate(groups):
        if gi:
            out.append(r"    \midrule")
        out.append(r"    \multicolumn{%d}{l}{\textit{%s}} \\" % (n + 1, group))
        for arm, name in arms:
            out.append(r"    \quad " + name + " & " + " & ".join(rows[arm]) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def header_lines(header):
    """header: list of (mode label, column label)."""
    spans, start = [], 2
    for mode in dict.fromkeys(m for m, _ in header):
        width = sum(1 for m, _ in header if m == mode)
        spans.append((mode, start, width))
        start += width
    top = " & " + " & ".join(r"\multicolumn{%d}{c}{\textbf{Thinking-%s eval}}" % (w, m) for m, _, w in spans) + r" \\"
    rules = " ".join(r"\cmidrule(lr){%d-%d}" % (s, s + w - 1) for _, s, w in spans)
    cols = r"\textbf{Method} & " + " & ".join(r"\textbf{%s}" % c for _, c in header) + r" \\"
    return [top, rules, cols]


def accuracy_rows(df, model, groups):
    """Per arm: AIME24/AIME25/HMMT25/Avg cells for each eval mode, plus the column maxima for bolding."""
    rows, steps, values = {}, {}, {}
    for _, arms in groups:
        for arm, _ in arms:
            cells = []
            for mode, _ in MODES:
                step = select_step(df, model, arm, mode)
                steps[(arm, mode)] = step
                a = None if step is None else at_step(df, model, arm, mode, step)
                if a is None or a.empty:
                    cells += ["--"] * 4
                    continue
                means = []
                for i, bench in enumerate(BENCHMARKS):
                    acc = a[a.benchmark == bench].acc
                    cells.append(mean_std(acc, "{:.1f}"))
                    means.append(acc.mean())
                    values[(arm, mode, i)] = acc.mean()
                cells.append("{:.1f}".format(np.mean(means)))
                values[(arm, mode, 3)] = np.mean(means)
            rows[arm] = cells
    for mode_index, (mode, _) in enumerate(MODES):
        for i in range(4):
            col = {arm: v for (arm, m, j), v in values.items() if m == mode and j == i}
            if col:
                best = max(col, key=col.get)
                rows[best][4 * mode_index + i] = r"\textbf{" + rows[best][4 * mode_index + i] + "}"
    return rows, steps


def accuracy_table(df, model, groups=GROUPS, label_suffix="acc"):
    rows, steps = accuracy_rows(df, model, groups)
    header = [(m, BENCH_LABEL.get(b, b)) for _, m in MODES for b in list(BENCHMARKS) + ["Avg."]]
    caption = (f"{model}: avg@12 accuracy (\\%) on held-out math, mean$\\pm$std over seeds at the checkpoint "
               "with the best seed-averaged accuracy per evaluation mode.")
    return latex_table(header, groups, rows, caption, f"tab:{slug(model)}_{label_suffix}"), steps


def length_table(df, stats, model, steps):
    """Tok@200 / Tok@best per eval mode, and orphan </think> % (thinking-off only) at the best checkpoint."""
    rows = {}
    for _, arms in GROUPS:
        for arm, _ in arms:
            cells = []
            for mode, _ in MODES:
                best = steps.get((arm, mode))
                cols = [("orphan", best)] if not mode else []
                cols += [("tokens", 200.0 if best == best else best), ("tokens", best)]
                for metric, step in cols:
                    a = None if best is None else at_step(df, model, arm, mode, step)
                    if a is None or a.empty:
                        cells.append("--")
                        continue
                    a = a.merge(stats, on="path")
                    per_seed = a.groupby(a.seed.fillna(-1))[metric].mean()
                    cells.append(mean_std(per_seed, "{:.1f}" if metric == "orphan" else "{:.0f}"))
            rows[arm] = cells
    header = [("on", "Tok@200"), ("on", "Tok@best"), ("off", r"Orph.\,\%"), ("off", "Tok@200"), ("off", "Tok@best")]
    caption = (f"{model}: mean response length in tokens at checkpoint 200 and at the accuracy-selected checkpoint, "
               "and the share of thinking-off rollouts emitting a \\texttt{</think>} with no opener of their own, "
               "pooled over AIME24/AIME25/HMMT25; mean$\\pm$std over seeds.")
    return latex_table(header, GROUPS, rows, caption, f"tab:{slug(model)}_length")


def slug(model):
    return model.lower().replace(".", "").replace("-", "")


def needed_files(df, model, steps):
    keep = []
    for (arm, mode), best in steps.items():
        if best is None:
            continue
        for step in {best, 200.0 if best == best else best}:
            keep += at_step(df, model, arm, mode, step).path.tolist()
    return keep


def _stats(job):
    path, model = job
    return {"path": path, **generation_stats(path, model)}


def load_stats(df, paths, cache, workers):
    cached = pd.read_parquet(cache) if cache and Path(cache).exists() else pd.DataFrame(columns=["path"])
    todo = sorted(set(paths) - set(cached.path))
    if todo:
        model_of = dict(zip(df.path, df.model))
        print(f"tokenizing {len(todo)} eval files with {workers} workers")
        with Pool(workers) as pool:
            new = pool.map(_stats, [(p, model_of[p]) for p in todo], chunksize=1)
        cached = pd.concat([cached, pd.DataFrame(new)], ignore_index=True)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(cache, index=False)
    return cached[cached.path.isin(paths)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="directories holding eval_results_*.json")
    p.add_argument("--out", default="tables/opsd")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--tables", nargs="+", default=["accuracy", "length", "masking"])
    p.add_argument("--cache", default=None, help="parquet cache of per-file length statistics")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--records", default=None, help="also write the per-file accuracy records (csv)")
    args = p.parse_args()

    df = pd.DataFrame(scan(args.results))
    if df.empty:
        raise SystemExit("no eval JSONs found")
    if args.records:
        df.to_csv(args.records, index=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = {}
    for model in args.models:
        if df[df.model == model].empty:
            continue
        tex, steps = accuracy_table(df, model)
        selected[model] = steps
        if "accuracy" in args.tables:
            (out / f"{model}_accuracy.tex").write_text(tex)
        if "masking" in args.tables and ((df.model == model) & df.arm.str.contains("/mask=")).any():
            tex, _ = accuracy_table(df, model, MASK_GROUPS, label_suffix="masking")
            (out / f"{model}_masking.tex").write_text(tex)
    if "length" in args.tables:
        paths = [p for model, steps in selected.items() for p in needed_files(df, model, steps)]
        stats = load_stats(df, paths, args.cache, args.workers)
        for model, steps in selected.items():
            (out / f"{model}_length.tex").write_text(length_table(df, stats, model, steps))
    (out / "selected_checkpoints.json").write_text(json.dumps(
        {m: {f"{arm}|{'on' if mode else 'off'}": s for (arm, mode), s in st.items() if s is not None}
         for m, st in selected.items()}, indent=1))
    print(f"wrote tables for {', '.join(selected)} to {out}")


if __name__ == "__main__":
    main()
