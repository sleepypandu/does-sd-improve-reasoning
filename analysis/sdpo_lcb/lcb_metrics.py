"""Validation scores of the LCB runs, parsed from verl's console logs.

verl's console logger prints one line per step, `step:N - key:value - key:value ...`. The
metric used throughout is the strict score (fraction of problems whose sample passes every
test case), averaged over the 4 validation samples:

    val-aux/livecodebench_<split>/score/mean@4

A run that was resumed has several logs; for a step present in more than one, the later log
wins. Two layouts are read:

  * scripts/sdpo_lcb/train.sh: <root>/lcb-<method>-<model>-seed<k>/logs/train-<timestamp>.log
  * the original LSF logs:      <root>/lcb_exp-<method>-<org>-<model>-<arm>-seed<k>_<jobid>.out
"""
import os
import re
from collections import defaultdict

SPLITS = ["livecodebench_train_seen", "livecodebench_train_unseen", "livecodebench_test_unseen"]
SPLIT_LABELS = {
    "livecodebench_train_seen": "Train (seen)",
    "livecodebench_train_unseen": "Train (unseen)",
    "livecodebench_test_unseen": "Test (unseen)",
}
MODELS = ["Qwen3-8B", "Qwen3-14B", "Olmo-3-7B-Instruct"]
METHODS = ["grpo", "sdpo"]
METHOD_LABELS = {"grpo": "GRPO", "sdpo": "SDPO"}

STEP_RE = re.compile(r"step:(\d+) - ")
SCORE_RE = re.compile(r"val-aux/(livecodebench_[a-z_]+)/score/mean@4:([-+0-9.eE]+)")
RUN_RE = re.compile(r"^lcb-(?P<method>sdpo|grpo)-(?P<model>.+)-seed(?P<seed>\d+)$")
LSF_RE = re.compile(r"^lcb_exp-(?P<method>sdpo|grpo)-(?:Qwen|allenai)-(?P<model>.+?)-(?P<arm>privnormal-fb|sparse)"
                    r"-seed(?P<seed>\d+)_(?P<jobid>\d+)\.out$")


def parse_log(path):
    """-> {step: {split: score}} for one console log."""
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "val-aux/" not in line:
                continue
            m = STEP_RE.search(line)
            scores = SCORE_RE.findall(line)
            if m and scores:
                out.setdefault(int(m.group(1)), {}).update({k: float(v) for k, v in scores})
    return out


def _segments(root):
    """-> {(model, method, seed): [log paths, oldest first]}"""
    segs = defaultdict(list)
    for name in sorted(os.listdir(root)):
        m = LSF_RE.match(name)
        if m:
            key = (m["model"], m["method"], int(m["seed"]))
            segs[key].append((int(m["jobid"]), os.path.join(root, name)))
            continue
        m = RUN_RE.match(name)
        logdir = os.path.join(root, name, "logs")
        if m and os.path.isdir(logdir):
            key = (m["model"], m["method"], int(m["seed"]))
            segs[key] += [(log, os.path.join(logdir, log)) for log in sorted(os.listdir(logdir))]
    return {k: [p for _, p in sorted(v)] for k, v in segs.items()}


def load_scores(root):
    """-> scores[(model, method)][seed] = {step: {split: score}}, stitched across resumes."""
    scores = defaultdict(dict)
    for (model, method, seed), paths in _segments(root).items():
        merged = {}
        for path in paths:
            merged.update(parse_log(path))
        if merged:
            scores[(model, method)][seed] = merged
    return scores
