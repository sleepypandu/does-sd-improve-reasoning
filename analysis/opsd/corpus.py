"""Index the eval JSONs written by opsd/eval/evaluate_math.py.

File names follow evaluate_math.py's auto-naming:

    eval_results_<dataset>_<model>[_<run>_checkpoint-<step>]_<thinking|nonthinking>_temp<T>_valn<N>.json

where <run> is a training run directory named ``<method>-model-<model>[-<key>-<value>]*`` by
scripts/opsd/train_*.sh (the paper runs carry an extra ``opsd-train-`` prefix, a
``-dataset_name-`` field and per-model method names, all handled here).
"""

import json
import re
from pathlib import Path

BENCHMARKS = ("aime24", "aime25", "hmmt25")

FILE_RE = re.compile(
    r"^eval_results_(?P<benchmark>aime24|aime25|hmmt25)_(?P<model>[A-Za-z0-9.\-]+)"
    r"(?:_(?P<run>.+)_checkpoint-(?P<step>\d+))?"
    r"_(?P<mode>thinking|nonthinking)_temp(?P<temp>[\d.]+)_valn(?P<valn>\d+)\.json$"
)
KEYS = ("dataset_name", "student_thinking", "teacher_thinking", "teacher", "privileged_mode", "loss_mask",
        "enable_thinking", "seed")
KV_RE = re.compile(r"-(?P<key>" + "|".join(sorted(KEYS, key=len, reverse=True)) + r")-(?P<value>[A-Za-z0-9._]+)$")
RUN_RE = re.compile(r"^(?:opsd-train-)?(?P<method>[a-z0-9_]+)-model-(?P<model>[A-Za-z0-9.\-]+)$")

# Method names of the paper runs -> (method, training thinking mode for GRPO).
METHODS = {
    "opsd": "opsd", "opsd_14b_train": "opsd", "opsd_olmo": "opsd",
    "sft": "sft", "sft_14b_train": "sft", "sft_olmo": "sft",
    "grpo": "grpo", "grpo_14b_think": "grpo", "grpo_olmo": "grpo", "grpo_nothink": "grpo",
}
TRAIN_DATASET = "Openthoughts_math_30k_opsd"


def parse_run(run):
    """Run directory name -> dict(method, model, seed, teacher_thinking, privileged_mode, ...), or None."""
    fields = {}
    rest = run
    while (m := KV_RE.search(rest)) is not None:
        fields.setdefault(m["key"], m["value"])
        rest = rest[: m.start()]
    m = RUN_RE.match(rest)
    if m is None or m["method"] not in METHODS or "seed" not in fields:
        return None
    if fields.get("dataset_name", TRAIN_DATASET) != TRAIN_DATASET:
        return None
    method = METHODS[m["method"]]
    out = {"method": method, "train_model": m["model"], "seed": int(fields["seed"])}
    if method == "opsd":
        out["teacher_thinking"] = fields.get("teacher_thinking", fields.get("teacher")) == "True"
        out["privileged_mode"] = fields["privileged_mode"]
        out["loss_mask"] = fields.get("loss_mask", "none")
    elif method == "grpo":
        out["enable_thinking"] = m["method"] != "grpo_nothink" and fields.get("enable_thinking", "True") == "True"
    return out


def arm_of(run_fields):
    """Row label key of a run: base, grpo_think, grpo_nothink, sft or opsd/tt=<T|F>/z=<mode>[/mask=<m>]."""
    if run_fields is None:
        return "base"
    method = run_fields["method"]
    if method == "grpo":
        return "grpo_think" if run_fields["enable_thinking"] else "grpo_nothink"
    if method == "opsd":
        arm = f"opsd/tt={'T' if run_fields['teacher_thinking'] else 'F'}/z={run_fields['privileged_mode']}"
        return arm if run_fields["loss_mask"] == "none" else f"{arm}/mask={run_fields['loss_mask']}"
    return method


def parse_file(name):
    """Eval JSON file name -> record dict, or None for files that are not paper evals."""
    m = FILE_RE.match(name)
    if m is None or float(m["temp"]) != 1.0:
        return None
    run = parse_run(m["run"]) if m["run"] else None
    if m["run"] and run is None:
        return None
    if run is not None and run["train_model"] != m["model"]:
        return None
    return {
        "model": m["model"],
        "benchmark": m["benchmark"],
        "eval_thinking": m["mode"] == "thinking",
        "arm": arm_of(run),
        "seed": run["seed"] if run else None,
        "step": int(m["step"]) if m["step"] else None,
        "val_n": int(m["valn"]),
    }


SUMMARY_RE = re.compile(r'"average_at_n_pct":\s*([0-9.eE+-]+)')


def read_accuracy(path, head_bytes=16384):
    """avg@n in percent. evaluate_math.py writes every summary scalar before the `results` list."""
    with open(path, "rb") as fh:
        head = fh.read(head_bytes).decode("utf-8", "replace")
    m = SUMMARY_RE.search(head.split('"results"', 1)[0])
    if m is None:
        return float(json.loads(Path(path).read_text())["average_at_n_pct"])
    return float(m.group(1))


def scan(dirs):
    """All paper eval JSONs under `dirs` as a list of records with their accuracy."""
    records = []
    for d in dirs:
        for path in sorted(Path(d).rglob("eval_results_*.json")):
            rec = parse_file(path.name)
            if rec is None or path.stat().st_size == 0:
                continue
            rec["path"] = str(path)
            rec["acc"] = read_accuracy(path)
            records.append(rec)
    return records


TOKENIZERS = {}


def tokenizer_for(model):
    if model not in TOKENIZERS:
        from transformers import AutoTokenizer

        org = "allenai" if model.startswith("Olmo") else "Qwen"
        TOKENIZERS[model] = AutoTokenizer.from_pretrained(f"{org}/{model}")
    return TOKENIZERS[model]


def is_orphan_close(text):
    """A `</think>` with no `<think>` opener of the generation's own.

    In thinking-off evaluation the chat template prefills a closed think block outside the stored
    generation, so any closer in the text was emitted by the model.
    """
    return "</think>" in text and "<think>" not in text


def generation_stats(path, model):
    """Mean response length in tokens (the model's own tokenizer) and % of orphan `</think>`."""
    tok = tokenizer_for(model)
    texts = [g["full_generation"] or "" for r in json.loads(Path(path).read_text())["results"] for g in r["generations"]]
    lengths = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    return {
        "tokens": sum(lengths) / len(lengths),
        "orphan": 100.0 * sum(map(is_orphan_close, texts)) / len(texts),
    }
