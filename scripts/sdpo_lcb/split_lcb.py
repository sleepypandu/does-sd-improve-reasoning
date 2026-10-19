"""Split LiveCodeBench v6 into train / held-out problems, and train tests into seen / unseen.

Half of the problems are used for training. Each training problem's test cases are split
50/50: the seen half is the training reward, the unseen half is only used for evaluation.
The other half of the problems is held out entirely.

Writes to --output_dir:
  train.json  training problems with their seen tests
  test.json   evaluation rows, with the split encoded in `dataset` (-> data_source):
                livecodebench_train_seen    training problems, seen tests
                livecodebench_train_unseen  training problems, unseen tests
                livecodebench_test_unseen   held-out problems, all tests
"""
import argparse
import copy
import json
import os

import datasets
import numpy as np

TRAIN_FRACTION = 0.5
SEEN_FRACTION = 0.5

TRAIN_SEEN = "livecodebench_train_seen"
TRAIN_UNSEEN = "livecodebench_train_unseen"
TEST_UNSEEN = "livecodebench_test_unseen"


def split_tests(example):
    tests = json.loads(example["tests"])
    inputs, outputs = tests["inputs"], tests["outputs"]
    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * SEEN_FRACTION))
    seen_idx = set(np.random.choice(num_tests, size=keep_count, replace=False).tolist())
    unseen_idx = [i for i in range(num_tests) if i not in seen_idx] or list(seen_idx)[:1]

    def with_tests(indices):
        t = copy.deepcopy(tests)
        t["inputs"] = [inputs[i] for i in indices]
        t["outputs"] = [outputs[i] for i in indices]
        return dict(example, tests=json.dumps(t))

    return with_tests(sorted(seen_idx)), with_tests(unseen_idx)


def with_label(label):
    return lambda example: dict(example, dataset=label, split_label=label)


def main(json_path, output_dir, seed):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    ds = datasets.load_dataset("json", data_files=json_path, split="train").shuffle(seed=seed)
    num_train = int(len(ds) * TRAIN_FRACTION)
    train_problems = ds.select(range(num_train))
    test_problems = ds.select(range(num_train, len(ds)))
    assert not set(train_problems["idx"]) & set(test_problems["idx"])

    seen_rows, unseen_rows = zip(*(split_tests(ex) for ex in train_problems))
    train_seen = datasets.Dataset.from_list(list(seen_rows))
    train_unseen = datasets.Dataset.from_list(list(unseen_rows))

    train_seen.to_json(os.path.join(output_dir, "train.json"))
    test = datasets.concatenate_datasets([
        train_seen.map(with_label(TRAIN_SEEN)),
        train_unseen.map(with_label(TRAIN_UNSEEN)),
        test_problems.map(with_label(TEST_UNSEEN)),
    ])
    test.to_json(os.path.join(output_dir, "test.json"))

    def num_tests(rows):
        return np.mean([len(json.loads(r["tests"])["inputs"]) for r in rows])

    print(f"train: {len(train_seen)} problems (seen tests)")
    print(f"test:  {len(test)} rows = {TRAIN_SEEN} {len(train_seen)} + {TRAIN_UNSEEN} "
          f"{len(train_unseen)} + {TEST_UNSEEN} {len(test_problems)}")
    print(f"mean tests per problem: seen {num_tests(seen_rows):.2f}, unseen {num_tests(unseen_rows):.2f}, "
          f"held-out {num_tests(test_problems):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="output of sdpo/data/load_dataset.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.json_path, args.output_dir, args.seed)
