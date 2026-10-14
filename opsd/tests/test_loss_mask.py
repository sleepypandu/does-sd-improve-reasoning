import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opsd_trainer import apply_loss_mask  # noqa: E402


def labels_for(prompt_len, n_completion, pad, width):
    row = [-100] * prompt_len + list(range(1, n_completion + 1)) + [-100] * pad
    return row + [-100] * (width - len(row))


@pytest.mark.parametrize("n", [1, 7, 10, 23, 100])
def test_masks_split_active_tokens_at_ninety_percent(n):
    labels = torch.tensor([labels_for(5, n, 3, 5 + n + 3), labels_for(2, n // 2 + 1, 0, 5 + n + 3)])
    for row in range(2):
        active = labels[row][labels[row] != -100].tolist()
        k = int(len(active) * 0.9)
        last = apply_loss_mask(labels.clone(), "last_10pct")[row]
        head = apply_loss_mask(labels.clone(), "all_but_last_10pct")[row]
        assert last[last != -100].tolist() == active[k:]
        assert head[head != -100].tolist() == active[:k]
    assert torch.equal(apply_loss_mask(labels.clone(), "none"), labels)
