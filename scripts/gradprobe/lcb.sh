#!/bin/bash
# Record the gradients of the LCB gradient-similarity analysis for one MODEL: the reference (GRPO on
# the held-out test_unseen problems) and GRPO and SDPO on the training problems, each one run of
# scripts/sdpo_lcb/train.sh at the frozen base model with learning rate 0 for ITERATIONS generation
# batches, for every seed. Each rank writes its FSDP shard of the mean gradient to
# $OUT_ROOT/<model>/{reference,train}/<run>/gradient_rank<k>.safetensors.
#   MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" bash scripts/gradprobe/lcb.sh
# ARMS selects a subset of "reference grpo sdpo"; DRY_RUN=1 prints the commands. Extra arguments are
# forwarded to train.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL=${MODEL:-Qwen/Qwen3-8B}
SEEDS=${SEEDS:-0 1 2 3}
ITERATIONS=${ITERATIONS:-8}
ARMS=${ARMS:-reference grpo sdpo}
DATA_DIR=${DATA_DIR:-$REPO/data/sdpo_lcb/lcb_exp}
ROOT=${OUT_ROOT:-$REPO/outputs/gradprobe/lcb}/$(basename "$MODEL")

# The reference set: the held-out problems' rows of the validation file.
REFERENCE_SET=$DATA_DIR/test_unseen.parquet
if [[ " $ARMS " == *" reference "* && ! -f $REFERENCE_SET && ${DRY_RUN:-0} != 1 ]]; then
    python - "$DATA_DIR/test.parquet" "$REFERENCE_SET" <<'PY'
import os
import sys

import pandas as pd

test = pd.read_parquet(sys.argv[1])
test[test.data_source == "livecodebench_test_unseen"].to_parquet(sys.argv[2] + ".tmp", index=False)
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
PY
fi

# No checkpoints and no validation; the run ends after ITERATIONS training steps.
export STEPS=$ITERATIONS SAVE_FREQ=-1 TEST_FREQ=-1 GRADPROBE_ITERS=$ITERATIONS
PROBE=(actor_rollout_ref.actor.optim.lr=0 trainer.val_before_train=False)

for arm in $ARMS; do
    for seed in $SEEDS; do
        case "$arm" in
            reference) method=grpo dir=reference args=("data.train_files=['$REFERENCE_SET']") ;;
            grpo | sdpo) method=$arm dir=train args=() ;;
            *) echo "unknown arm $arm (reference, grpo, sdpo)" >&2; exit 1 ;;
        esac
        run=lcb-$method-$(basename "$MODEL")-seed$seed
        export GRADPROBE_OUT=$ROOT/$dir/$run
        cmd=(env METHOD=$method MODEL=$MODEL SEED=$seed OUT_ROOT=$ROOT/$dir RUN_NAME=$run
            bash "$REPO/scripts/sdpo_lcb/train.sh" "${PROBE[@]}" "${args[@]}" "$@")
        if [ "${DRY_RUN:-0}" = 1 ]; then
            echo "GRADPROBE_OUT=$GRADPROBE_OUT GRADPROBE_ITERS=$ITERATIONS STEPS=$STEPS ${cmd[*]}"
            continue
        fi
        "${cmd[@]}"
        [ -f "$GRADPROBE_OUT/gradient_rank0.safetensors" ] || {
            echo "no gradient written to $GRADPROBE_OUT" >&2
            exit 1
        }
    done
done
