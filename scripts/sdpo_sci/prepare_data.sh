#!/usr/bin/env bash
# Build RL ({train,test}.parquet) and SFT ({train,test}.sft.parquet) data for the five datasets.
# The SciKnowEval train/test splits are the committed ones, i.e. data/split_tasks.py with
# --test_ratio 0.1 --seed 42; set RESPLIT=1 to regenerate them from datasets/sciknoweval/*.json.
#   bash scripts/sdpo_sci/prepare_data.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$SDPO_DIR"

for dataset in "${DATASETS[@]}"; do
  dir="$(dataset_dir "$dataset")"
  if [ "${RESPLIT:-0}" = 1 ] && [ "$dataset" != tooluse ]; then
    python data/split_tasks.py --json_path "datasets/sciknoweval/$dataset.json" --output_dir "$dir" --test_ratio 0.1 --seed 42
  fi
  python data/preprocess.py --data_source "$dir"
  python data/preprocess_sft.py --data_source "$dir"
done
