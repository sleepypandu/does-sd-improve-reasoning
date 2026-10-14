"""Pooled cosines, jackknife SEs and gradient noise scales from recorded gradients.

    python -m gradprobe.analyze gram --reference 'outputs/.../reference/*-seed-*' \
        --method 'GRPO=outputs/.../train/grpo-*-seed-*' --method 'SFT=...' --out gram.npz
    python -m gradprobe.analyze report gram.npz

Each glob matches one dump per seed (the seed is read from `seed-<n>` / `seed<n>` in the path).
`gram --part k --nparts N` reads only every N-th tensor; `report` sums the partial Gram matrices.
"""
import argparse
import glob
import json
import re

import numpy as np

from .stats import gram, noise_scale, pooled_cosine

REFERENCE = "reference"


def _seed(path):
    found = re.findall(r"seed-?(\d+)", path)
    if not found:
        raise ValueError(f"no seed in {path}")
    return int(found[-1])


def _expand(label, pattern, seeds):
    dumps = {}
    for path in sorted(glob.glob(pattern)):
        s = _seed(path)
        if seeds is None or s in seeds:
            if s in dumps:
                raise ValueError(f"{label}: seed {s} matches both {dumps[s]} and {path}")
            dumps[s] = path
    if not dumps:
        raise ValueError(f"{label}: nothing matches {pattern}")
    return [(label, s, dumps[s]) for s in sorted(dumps)]


def cmd_gram(args):
    entries = _expand(REFERENCE, args.reference, args.seeds)
    for spec in args.method:
        label, pattern = spec.split("=", 1)
        entries += _expand(label, pattern, args.seeds)
    for label, s, path in entries:
        print(f"{label:24s} seed {s}  {path}")
    G = gram([p for *_, p in entries], chunk=args.chunk, part=args.part, nparts=args.nparts)
    np.savez(args.out, gram=G, labels=[e[0] for e in entries], seeds=[e[1] for e in entries],
             paths=[e[2] for e in entries])
    print(f"wrote {args.out}")


def load(files):
    G, meta = None, None
    for f in files:
        z = np.load(f)
        m = (list(z["labels"]), list(z["seeds"]), list(z["paths"]))
        if meta is not None and m != meta:
            raise ValueError(f"{f} covers different dumps than {files[0]}")
        G, meta = (z["gram"] if G is None else G + z["gram"]), m
    return G, meta[0], [int(s) for s in meta[1]]


def report(G, labels, seeds):
    index = {}
    for i, (label, s) in enumerate(zip(labels, seeds)):
        index.setdefault(label, {})[s] = i
    ref = index[REFERENCE]
    rows = {}
    for label, by_seed in index.items():
        row = {"seeds": sorted(by_seed)}
        if len(by_seed) > 1:
            row["noise_scale"] = float(noise_scale(G, list(by_seed.values())))
        if label != REFERENCE:
            common = sorted(set(ref) & set(by_seed))
            if common != sorted(by_seed) or common != sorted(ref):
                raise ValueError(f"{label} has seeds {sorted(by_seed)}, the reference {sorted(ref)}")
            c, se, reps = pooled_cosine(G, [ref[s] for s in common], [by_seed[s] for s in common])
            row.update(cosine=float(c), jackknife_se=float(se), leave_one_out=[float(r) for r in reps])
        rows[label] = row
    return rows


def cmd_report(args):
    rows = report(*load(args.grams))
    w = max(map(len, rows))
    print(f"{'':{w}s} {'cosine':>8s} {'SE':>7s} {'noise scale':>12s}")
    for label, r in rows.items():
        cos = f"{r['cosine']:+8.3f} {r['jackknife_se']:7.3f}" if "cosine" in r else f"{'':8s} {'':7s}"
        print(f"{label:{w}s} {cos} {r.get('noise_scale', float('nan')):12.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gram", help="accumulate the Gram matrix of the reference and method dumps")
    g.add_argument("--reference", required=True, help="glob of the reference dumps, one per seed")
    g.add_argument("--method", action="append", default=[], help="NAME=GLOB, one per method")
    g.add_argument("--seeds", type=int, nargs="+", help="only these seeds (default: all found)")
    g.add_argument("--out", required=True)
    g.add_argument("--part", type=int, default=0)
    g.add_argument("--nparts", type=int, default=1)
    g.add_argument("--chunk", type=int, default=1 << 22, help="elements per read slice")
    g.set_defaults(func=cmd_gram)
    r = sub.add_parser("report", help="cosines, jackknife SEs and noise scales from (partial) Gram files")
    r.add_argument("grams", nargs="+")
    r.add_argument("--json")
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
