import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collator import SelfDistillationDataCollator, add_swapped_solutions, derangement  # noqa: E402


class FakeTokenizer:
    padding_side = "left"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return messages[0]["content"]

    def __call__(self, texts, padding=False, truncation=True, max_length=None, return_tensors=None):
        import torch

        ids = [[ord(c) for c in t] for t in texts]
        if padding == "max_length":
            ids = [x[:max_length] + [0] * (max_length - len(x[:max_length])) for x in ids]
            return {"input_ids": torch.tensor(ids), "attention_mask": (torch.tensor(ids) != 0).long()}
        return {"input_ids": ids}


def make_dataset(n):
    return Dataset.from_dict({"problem": [f"p{i}" for i in range(n)], "solution": [f"s{i}" for i in range(n)]})


@pytest.mark.parametrize("n", [2, 3, 10, 1000])
def test_derangement_has_no_fixed_point(n):
    perm = derangement(n, seed=0)
    assert sorted(perm.tolist()) == list(range(n))
    assert not np.any(perm == np.arange(n))


def test_derangement_is_deterministic_per_seed():
    assert derangement(500, 1).tolist() == derangement(500, 1).tolist()
    assert derangement(500, 1).tolist() != derangement(500, 2).tolist()


def teacher_prompts(collator, rows):
    out = collator(rows)["teacher_prompts"]
    return ["".join(chr(c) for c in row.tolist() if c) for row in out]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_swap_never_shows_own_solution_at_any_batch_size(batch_size):
    ds = add_swapped_solutions(make_dataset(64), seed=0)
    collator = SelfDistillationDataCollator(FakeTokenizer(), max_length=4096, reason_first=False, privileged_mode="swap")
    rows = list(ds)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        for row, prompt in zip(batch, teacher_prompts(collator, batch)):
            assert f"Begin ===\n{row['solution']}\n===" not in prompt
            assert f"Begin ===\n{row['swapped_solution']}\n===" in prompt


def test_normal_and_none():
    rows = list(make_dataset(4))
    normal = SelfDistillationDataCollator(FakeTokenizer(), reason_first=False, privileged_mode="normal")
    none = SelfDistillationDataCollator(FakeTokenizer(), reason_first=False, privileged_mode="none")
    for row, p in zip(rows, teacher_prompts(normal, rows)):
        assert f"Begin ===\n{row['solution']}\n===" in p
    for p in teacher_prompts(none, rows):
        assert "Begin ===\n\n=== Reference Solution End" in p
