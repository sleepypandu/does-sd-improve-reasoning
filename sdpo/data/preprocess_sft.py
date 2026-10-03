"""Build SFT parquet files from the same train.json/test.json used by `preprocess.py`.

Each row keeps the RL prompt (system message, if any, followed by the user question) and adds a
supervised target:

  - mcq:     "<answer>{letter}</answer>", the format the mcq scorer extracts.
  - tooluse: one "Action: {name}\\nAction Input: {json}" block per ground-truth action.

`ground_truth` and `data_source` are kept so the SFT trainer can score sampled generations with
`verl.utils.reward_score.feedback.compute_score`, exactly as the RL validation does.

Usage: python data/preprocess_sft.py --data_source datasets/sciknoweval/biology
"""

import argparse
import json
import os

from preprocess import write_rowgrouped_large

import datasets


def render_target(kind, answer):
    if kind == "mcq":
        return f"<answer>{str(answer).strip()}</answer>"
    if kind == "tooluse":
        actions = json.loads(answer) if isinstance(answer, str) else answer
        parts = []
        for item in actions:
            action_input = item["Action_Input"]
            if not isinstance(action_input, str):
                action_input = json.dumps(action_input)
            parts.append(f"Action: {item['Action']}\nAction Input: {action_input}")
        return "\n".join(parts)
    raise ValueError(f"Unsupported kind for SFT: {kind!r}")


def build_records(path, split):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = []
            if ex.get("system") is not None:
                messages.append({"role": "system", "content": ex["system"]})
            messages.append({"role": "user", "content": ex["prompt"]})
            answer = ex["answer"]
            records.append(
                {
                    "data_source": ex["dataset"],
                    "prompt": messages,
                    "response": render_target(ex["kind"], answer),
                    "ground_truth": answer if isinstance(answer, str) else json.dumps(answer),
                    "extra_info": {"split": split, "index": str(ex.get("idx", i)), "kind": ex["kind"]},
                }
            )
    return records


def run(data_source):
    for split in ["train", "test"]:
        records = build_records(os.path.join(data_source, f"{split}.json"), split)
        out_path = os.path.join(data_source, f"{split}.sft.parquet")
        write_rowgrouped_large(datasets.Dataset.from_list(records), out_path)
        print(f"{split}: {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT parquet (prompt/response) from train.json/test.json.")
    parser.add_argument("--data_source", type=str, required=True, help="directory containing train.json and test.json")
    args = parser.parse_args()
    run(args.data_source)
