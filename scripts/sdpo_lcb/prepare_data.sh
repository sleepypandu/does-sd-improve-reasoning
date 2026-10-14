#!/bin/bash
# Build the LiveCodeBench v6 split used by the SDPO-vs-GRPO experiments.
#
#   bash scripts/sdpo_lcb/prepare_data.sh
#
# DATA_DIR/lcb_v6.json        131 LCB v6 problems (sdpo/data/load_dataset.py)
# DATA_DIR/lcb_exp/*.parquet  65 train problems, and the three validation splits
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/sdpo_lcb}
SEED=${SEED:-0}
# A fresh datasets cache, so a stale json build cannot leak into the split.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$(mktemp -d)}

mkdir -p "$DATA_DIR"
(cd "$REPO_ROOT/sdpo" && PYTHONPATH=. python data/load_dataset.py \
    --dataset_name livecodebench/code_generation_lite-v6 \
    --output_path "$DATA_DIR/lcb_v6.json")
python "$REPO_ROOT/scripts/sdpo_lcb/split_lcb.py" \
    --json_path "$DATA_DIR/lcb_v6.json" --output_dir "$DATA_DIR/lcb_exp" --seed "$SEED"
python "$REPO_ROOT/sdpo/data/preprocess.py" --data_source "$DATA_DIR/lcb_exp"
ls -la "$DATA_DIR/lcb_exp"
