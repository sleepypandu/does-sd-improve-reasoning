#!/usr/bin/env bash
# The paper grid: {GRPO, SDPO} x 5 datasets x seeds 1-4, each run on one 8-GPU node.
# Runs sequentially; DRY_RUN=1 prints one command per run instead (e.g. to feed a scheduler).
#   DRY_RUN=1 bash scripts/sdpo_sci/paper_grid.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

METHODS=(${METHODS:-grpo sdpo})
SEEDS=(${SEEDS:-1 2 3 4})
read -r -a GRID_DATASETS <<< "${GRID_DATASETS:-${DATASETS[*]}}"

for dataset in "${GRID_DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cmd=(env METHOD="$method" DATASET="$dataset" SEED="$seed" bash "$HERE/train_rl.sh")
      if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "${cmd[@]}"
        echo
      else
        "${cmd[@]}"
      fi
    done
  done
done
