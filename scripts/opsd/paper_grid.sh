#!/bin/bash
# Every paper run for one MODEL: OPSD (teacher thinking x z), GRPO, SFT, the Qwen3-8B masking
# ablation, and the avg@12 evals of the base model and of every saved checkpoint.
#   MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" DRY_RUN=1 bash scripts/opsd/paper_grid.sh
# STAGE=train|eval|all (default all). Evals run concurrently, one per GPU in EVAL_GPUS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

MODEL=${MODEL:-Qwen/Qwen3-8B}
SEEDS=${SEEDS:-0 1 2 3}
STAGE=${STAGE:-all}
MASKING=${MASKING:-auto}   # auto: only Qwen3-8B, as in the paper
EVAL_GPUS=${EVAL_GPUS:-0 1 2 3 4 5 6 7}
export DRY_RUN=${DRY_RUN:-0}
export OUT_ROOT=${OUT_ROOT:-$REPO/outputs/opsd}
NAME=$(basename "$MODEL")

[ "$MASKING" = auto ] && { [ "$NAME" = Qwen3-8B ] && MASKING=1 || MASKING=0; }
case "$NAME" in Olmo*) GRPO_THINKING="True" ;; *) GRPO_THINKING="True False" ;; esac

# One line per training run: "<script> <run name> <save_steps> <max_steps> <VAR=value ...>".
# Run names must match the ones the train_*.sh scripts derive.
runs() {
    for seed in $SEEDS; do
        for tt in True False; do
            for z in normal none swap; do
                echo "train_opsd.sh opsd-model-$NAME-teacher_thinking-$tt-privileged_mode-$z-seed-$seed 20 200" \
                    "TEACHER_THINKING=$tt PRIVILEGED_MODE=$z SEED=$seed"
            done
        done
        if [ "$MASKING" = 1 ]; then
            for mask in last_10pct all_but_last_10pct; do
                for z in normal none swap; do
                    echo "train_opsd.sh opsd-model-$NAME-teacher_thinking-True-privileged_mode-$z-loss_mask-$mask-seed-$seed" \
                        "25 200 TEACHER_THINKING=True PRIVILEGED_MODE=$z LOSS_MASK=$mask SAVE_STEPS=25 SEED=$seed"
                done
            done
        fi
        for think in $GRPO_THINKING; do
            echo "train_grpo.sh grpo-model-$NAME-enable_thinking-$think-seed-$seed 50 500 ENABLE_THINKING=$think SEED=$seed"
        done
        echo "train_sft.sh sft-model-$NAME-seed-$seed 20 200 SEED=$seed"
    done
}

train() {
    while read -r script _run _save _max vars; do
        echo "# MODEL=$MODEL $vars $script"
        env MODEL="$MODEL" $vars bash "$HERE/$script" </dev/null
    done < <(runs)
}

evaluate() {
    local gpus=($EVAL_GPUS) i=0 ckpts=("")
    while read -r _script run save max _vars; do
        for ((step = save; step <= max; step += save)); do
            ckpts+=("$OUT_ROOT/$run/checkpoint-$step")
        done
    done < <(runs)
    for ckpt in "${ckpts[@]}"; do
        if [ -n "$ckpt" ] && [ ! -d "$ckpt" ] && [ "$DRY_RUN" != 1 ]; then
            echo "skipping missing $ckpt" >&2
            continue
        fi
        local gpu=${gpus[$((i % ${#gpus[@]}))]}
        i=$((i + 1))
        echo "# CUDA_VISIBLE_DEVICES=$gpu MODEL=$MODEL CKPT=$ckpt"
        if [ "$DRY_RUN" = 1 ]; then
            MODEL="$MODEL" CKPT="$ckpt" bash "$HERE/eval.sh"
            continue
        fi
        while [ "$(jobs -rp | wc -l)" -ge "${#gpus[@]}" ]; do wait -n; done
        CUDA_VISIBLE_DEVICES=$gpu MODEL="$MODEL" CKPT="$ckpt" bash "$HERE/eval.sh" &
    done
    wait
}

case "$STAGE" in
    train) train ;;
    eval) evaluate ;;
    all) train; evaluate ;;
    *) echo "STAGE must be train, eval or all" >&2; exit 1 ;;
esac
