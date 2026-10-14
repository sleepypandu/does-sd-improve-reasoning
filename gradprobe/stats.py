"""Gradient-similarity statistics over seed-replicated gradient dumps (Section 4.4, Appendix E.3).

A dump is one recorded gradient: a safetensors file, or a directory holding either
`gradient.safetensors` (the TRL/DeepSpeed probe) or one shard per rank, `gradient_rank<k>.safetensors`
(the verl/FSDP probe). Every statistic is a function of the Gram matrix of the dumps,
G[i, j] = <g_i, g_j>, which `gram` accumulates in float64 one slice of one tensor at a time, so a
full gradient is never held in memory.

With g_s the dump of seed s and S seeds, the mean gradient is g = (1/S) sum_s g_s and
  pooled cosine         cos(g_ref, g_method)
  jackknife SE          sqrt((S-1)/S * sum_s (c_-s - c_bar)^2), where c_-s is the pooled cosine
                        with seed s dropped from both the reference and the method
  gradient noise scale  tr(Sigma) / ||g||^2, tr(Sigma) = 1/(S-1) sum_s ||g_s - g||^2.
"""
import contextlib
import glob
import os
import re

import numpy as np
from safetensors import safe_open


def dump_files(path):
    """The safetensors files of one dump, shards in rank order."""
    if os.path.isfile(path):
        return [path]
    single = os.path.join(path, "gradient.safetensors")
    if os.path.isfile(single):
        return [single]
    shards = glob.glob(os.path.join(path, "gradient_rank*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no gradient.safetensors or gradient_rank*.safetensors in {path}")
    return sorted(shards, key=lambda f: int(re.search(r"rank(\d+)\.safetensors$", f).group(1)))


def _layout(handles):
    return [(i, name, tuple(h.get_slice(name).get_shape())) for i, h in enumerate(handles) for name in sorted(h.keys())]


def _split(layout, nparts):
    """Assign tensors to nparts parts of near-equal size (largest first, to the smallest part)."""
    sizes, owner = [0] * nparts, {}
    for k in sorted(range(len(layout)), key=lambda k: -int(np.prod(layout[k][2]))):
        p = sizes.index(min(sizes))
        owner[k] = p
        sizes[p] += int(np.prod(layout[k][2]))
    return [[t for k, t in enumerate(layout) if owner[k] == p] for p in range(nparts)]


def gram(paths, chunk=1 << 22, part=0, nparts=1):
    """Float64 Gram matrix of the dumps at `paths`.

    Tensors are read in slices of at most ~`chunk` elements. With nparts > 1 only part `part` of
    a size-balanced split of the tensors is read, and the partial Gram matrices sum to the full one.
    """
    n = len(paths)
    G = np.zeros((n, n))
    with contextlib.ExitStack() as stack:
        handles = [[stack.enter_context(safe_open(f, "numpy")) for f in dump_files(p)] for p in paths]
        layout = _layout(handles[0])
        for p, h in zip(paths[1:], handles[1:]):
            if _layout(h) != layout:
                raise ValueError(f"{p} does not have the tensor layout of {paths[0]}")
        for i, name, shape in _split(layout, nparts)[part]:
            slices = [h[i].get_slice(name) for h in handles]
            if not shape:
                X = np.array([[float(h[i].get_tensor(name))] for h in handles])
                G += X @ X.T
                continue
            row = int(np.prod(shape[1:]))
            step = max(1, chunk // max(row, 1))
            for r in range(0, shape[0], step):
                X = np.stack([s[r : min(r + step, shape[0])].reshape(-1) for s in slices]).astype(np.float64)
                G += X @ X.T
    return G


def _mean_dot(G, a, b):
    return G[np.ix_(a, b)].mean()


def cosine(G, a, b):
    """Cosine between the mean of dumps `a` and the mean of dumps `b` (index lists into G)."""
    return _mean_dot(G, a, b) / np.sqrt(_mean_dot(G, a, a) * _mean_dot(G, b, b))


def pooled_cosine(G, ref, method):
    """(pooled cosine, jackknife SE, leave-one-seed-out cosines) for seed-paired index lists."""
    if len(ref) != len(method) or len(ref) < 2:
        raise ValueError("the reference and the method need the same seeds, at least two")
    S = len(ref)
    reps = np.array([cosine(G, ref[:s] + ref[s + 1 :], method[:s] + method[s + 1 :]) for s in range(S)])
    se = np.sqrt((S - 1) / S * ((reps - reps.mean()) ** 2).sum())
    return cosine(G, ref, method), se, reps


def noise_scale(G, idx):
    """tr(Sigma) / ||mean||^2 over the dumps `idx`, with the unbiased estimate of tr(Sigma)."""
    S = len(idx)
    sub = G[np.ix_(idx, idx)]
    mean_sq = sub.mean()
    return (np.trace(sub) - S * mean_sq) / (S - 1) / mean_sq
