"""Grade a generation dump from dump_generations.py with the training reward function (CPU only).

Samples are graded one at a time: the grader already runs each test case in its own process,
and test cases share one deadline, so oversubscribing the CPUs turns slow tests into timeouts.

    python scripts/sdpo_lcb/grade_dump.py --dump dumps/lcb-sdpo-Qwen3-14B-seed0-step200.json.gz
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", help="default: <dump>.graded.json.gz")
    args = ap.parse_args()
    out = args.out or args.dump.removesuffix(".json.gz") + ".graded.json.gz"

    import pandas as pd
    from verl.utils.reward_score.feedback import compute_score

    with gzip.open(args.dump, "rt") as f:
        dump = json.load(f)
    df = pd.read_parquet(dump["meta"]["data"])
    problems = {(r["data_source"], int(r["extra_info"]["index"])): r for _, r in df.iterrows()}

    per_problem = defaultdict(list)
    for i, row in enumerate(dump["rows"], 1):
        problem = problems[(row["data_source"], row["problem_index"])]
        extra_info = dict(problem["extra_info"], truncated=row["finish_reason"] != "stop")
        result = compute_score(row["data_source"], row["text"], problem["reward_model"]["ground_truth"], extra_info)
        row.update(score=float(result["score"]), acc=float(result["acc"]), feedback=result.get("feedback", ""))
        per_problem[(row["data_source"], row["problem_index"])].append(row["score"])
        if i % 50 == 0:
            print(f"graded {i}/{len(dump['rows'])}", flush=True)

    summary = defaultdict(list)
    for (source, _), scores in per_problem.items():
        summary[source].append(np.mean(scores))
    dump["meta"]["summary"] = {k: {"score": float(np.mean(v)), "n_problems": len(v)} for k, v in summary.items()}
    with gzip.open(out + ".tmp", "wt") as f:
        json.dump(dump, f)
    os.replace(out + ".tmp", out)
    for source, s in dump["meta"]["summary"].items():
        print(f"{source}: score {s['score']:.4f} over {s['n_problems']} problems")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
