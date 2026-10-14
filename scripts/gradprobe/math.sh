#!/bin/bash
# Record the gradients of the math gradient-similarity analysis for one MODEL: the reference (GRPO on
# the AIME24+AIME25+HMMT25 set, thinking off) and every objective on OpenThoughts (GRPO, SFT, OPSD for
# teacher thinking x z), each one run of the paper's training launcher at the frozen base model with
# dense gradients (no LoRA), learning rate 0 and ITERATIONS optimizer steps, for every seed.
#   MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" bash scripts/gradprobe/math.sh
# Gradients land in $OUT_ROOT/<model>/{reference,train}/<run>/gradient.safetensors.
# ARMS selects a subset of "reference grpo sft opsd", GRPO_THINKING, TEACHER_THINKING and Z a subset of
# those arms' cells (e.g. to spread the grid over nodes); DRY_RUN=1 prints the commands.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN="$REPO/scripts/opsd"

export MODEL=${MODEL:-Qwen/Qwen3-8B}
SEEDS=${SEEDS:-0 1 2 3}
ITERATIONS=${ITERATIONS:-64}
ARMS=${ARMS:-reference grpo sft opsd}
REFERENCE_SET=${REFERENCE_SET:-$REPO/data/gradprobe/aime24_aime25_hmmt25.parquet}
ROOT=${OUT_ROOT:-$REPO/outputs/gradprobe/math}/$(basename "$MODEL")
export DRY_RUN=${DRY_RUN:-0}
# The probe stops after ITERATIONS steps, so neither the last step nor a checkpoint is reached.
export MAX_STEPS=$((ITERATIONS + 1)) SAVE_STEPS=$((ITERATIONS + 1))

[ -f "$REFERENCE_SET" ] || [ "$DRY_RUN" = 1 ] || {
    echo "missing $REFERENCE_SET: run scripts/gradprobe/build_reference_set.sh" >&2
    exit 1
}
REFERENCE_SET="$(realpath -m "$REFERENCE_SET")"

PROBE=(--learning_rate 0 --use_peft False --gradprobe_out auto --gradprobe_iterations "$ITERATIONS" --logging_steps 1)
# GRPO trains with 2 updates per generation batch; at learning rate 0 the second is a duplicate gradient.
GRPO=("${PROBE[@]}" --num_iterations 1)
case "$(basename "$MODEL")" in
    Olmo*) REFERENCE_THINKING=True GRPO_THINKING=${GRPO_THINKING:-True} ;;   # its templates ignore enable_thinking
    *) REFERENCE_THINKING=False GRPO_THINKING=${GRPO_THINKING:-True False} ;;
esac
TEACHER_THINKING=${TEACHER_THINKING:-True False}
Z=${Z:-normal none swap}

for arm in $ARMS; do
    for seed in $SEEDS; do
        case "$arm" in
            reference)
                OUT_ROOT=$ROOT/reference ENABLE_THINKING=$REFERENCE_THINKING SEED=$seed \
                    bash "$TRAIN/train_grpo.sh" "${GRPO[@]}" --dataset_name "$REFERENCE_SET" ;;
            grpo)
                for thinking in $GRPO_THINKING; do
                    OUT_ROOT=$ROOT/train ENABLE_THINKING=$thinking SEED=$seed bash "$TRAIN/train_grpo.sh" "${GRPO[@]}"
                done ;;
            sft)
                OUT_ROOT=$ROOT/train SEED=$seed bash "$TRAIN/train_sft.sh" "${PROBE[@]}" ;;
            opsd)
                for tt in $TEACHER_THINKING; do
                    for z in $Z; do
                        OUT_ROOT=$ROOT/train TEACHER_THINKING=$tt PRIVILEGED_MODE=$z SEED=$seed \
                            bash "$TRAIN/train_opsd.sh" "${PROBE[@]}"
                    done
                done ;;
            *) echo "unknown arm $arm (reference, grpo, sft, opsd)" >&2; exit 1 ;;
        esac
    done
done
