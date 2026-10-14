#!/bin/bash
# The gradient-similarity reference set: AIME24 + AIME25 + HMMT25 (the evaluation benchmarks, as in
# opsd/eval/evaluate_math.py) in the Question/Answer columns grpo_train.py reads, in a fixed order.
#   bash scripts/gradprobe/build_reference_set.sh [out.parquet]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${1:-$REPO/data/gradprobe/aime24_aime25_hmmt25.parquet}
mkdir -p "$(dirname "$OUT")"

python - "$OUT" <<'PY'
import sys

from datasets import Dataset, load_dataset

SOURCES = [
    ("aime24", "HuggingFaceH4/aime_2024", "id"),
    ("aime25", "yentinglin/aime_2025", "problem_idx"),
    ("hmmt25", "MathArena/hmmt_feb_2025", "problem_idx"),
]
rows = []
for name, hub_id, id_col in SOURCES:
    for i, ex in enumerate(load_dataset(hub_id, split="train")):
        rows.append({"Question": ex["problem"], "Answer": str(ex["answer"]), "source": name,
                     "question_id": str(ex.get(id_col, i))})
assert all(r["Answer"].strip() for r in rows)
Dataset.from_list(rows).to_parquet(sys.argv[1])
print(f"wrote {sys.argv[1]}: {len(rows)} problems")
PY
