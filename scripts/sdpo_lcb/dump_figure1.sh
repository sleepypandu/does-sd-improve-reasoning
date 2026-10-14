#!/bin/bash
# Generations for Figure 1: Qwen3-14B base (sampling seeds 0-3) and the step-200 GRPO and SDPO
# checkpoints of training seeds 0-3, all on the 65 training problems, one GPU per dump.
#
#   bash scripts/sdpo_lcb/dump_figure1.sh
#   python analysis/sdpo_lcb/plot_unique_solutions.py --dumps $DUMP_DIR
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3-14B}
STEP=${STEP:-200}
SEEDS=${SEEDS:-"0 1 2 3"}
NGPUS=${NGPUS:-8}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/sdpo_lcb/lcb_exp}
OUT_ROOT=${OUT_ROOT:-$REPO_ROOT/outputs/sdpo_lcb}
DUMP_DIR=${DUMP_DIR:-$OUT_ROOT/dumps}
MODEL_TAG=$(basename "$MODEL")
DUMP="python $REPO_ROOT/scripts/sdpo_lcb/dump_generations.py --data $DATA_DIR/train.parquet"

jobs=()
for seed in $SEEDS; do
    jobs+=("$DUMP --model $MODEL --seed $seed --out $DUMP_DIR/base-$MODEL_TAG-seed$seed.json.gz")
done
for method in grpo sdpo; do
    for seed in $SEEDS; do
        run=lcb-$method-$MODEL_TAG-seed$seed
        jobs+=("$DUMP --ckpt $OUT_ROOT/$run/global_step_$STEP --out $DUMP_DIR/$run-step$STEP.json.gz")
    done
done

mkdir -p "$DUMP_DIR"
failed=0
pids=()
for i in "${!jobs[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % NGPUS)) bash -c "${jobs[$i]}" > "$DUMP_DIR/dump-$i.log" 2>&1 &
    pids+=($!)
    if (( ${#pids[@]} == NGPUS || i + 1 == ${#jobs[@]} )); then
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        pids=()
    fi
done
ls -la "$DUMP_DIR"
exit $failed
