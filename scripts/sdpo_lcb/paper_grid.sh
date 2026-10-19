#!/bin/bash
# Every LCB training run in the paper: 3 models x {GRPO, SDPO} x 4 seeds, 200 steps each.
# Runs them one after another on this node; DRY_RUN=1 prints the commands instead, e.g. to
# hand each line to a cluster scheduler.
#
#   DRY_RUN=1 bash scripts/sdpo_lcb/paper_grid.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODELS=${MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-14B allenai/Olmo-3-7B-Instruct"}
METHODS=${METHODS:-"grpo sdpo"}
SEEDS=${SEEDS:-"0 1 2 3"}

for model in $MODELS; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            cmd="METHOD=$method MODEL=$model SEED=$seed bash $REPO_ROOT/scripts/sdpo_lcb/train.sh"
            if [[ "${DRY_RUN:-0}" == 1 ]]; then
                echo "$cmd"
            else
                echo "=== $cmd"
                env METHOD="$method" MODEL="$model" SEED="$seed" bash "$REPO_ROOT/scripts/sdpo_lcb/train.sh"
            fi
        done
    done
done
