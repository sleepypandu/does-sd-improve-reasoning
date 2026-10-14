#!/bin/bash
# avg@12 evaluation of a base model or a LoRA checkpoint on AIME24/AIME25/HMMT25, as in the paper.
# Writes eval_results/eval_results_<dataset>_<model>[_<run>_checkpoint-N]_<mode>_temp1.0_valn12.json
# under $OUT_ROOT. Extra arguments are forwarded to opsd/eval/evaluate_math.py
# (e.g. --num_samples 2 to evaluate only the first problems).
#   MODEL=Qwen/Qwen3-8B CKPT=outputs/opsd/<run>/checkpoint-200 bash scripts/opsd/eval.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/opsd/presets.sh"

MODEL=${MODEL:-Qwen/Qwen3-8B}
CKPT=${CKPT:-}
DATASETS=${DATASETS:-aime24 aime25 hmmt25}
if [ -z "${THINKING:-}" ]; then
    # Olmo-3-7B-Instruct has no thinking mode, so the paper evaluates it thinking-off only.
    case "$MODEL" in *Olmo*) THINKING=False ;; *) THINKING="True False" ;; esac
fi
VAL_N=${VAL_N:-12}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/opsd}
mkdir -p "$OUT_ROOT" && OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"

CKPT_ARGS=()
[ -z "$CKPT" ] || CKPT_ARGS=(--checkpoint_dir "$(realpath -m "$CKPT")")

cd "$OUT_ROOT"
for thinking in $THINKING; do
    MODE_ARGS=()
    [ "$thinking" = True ] || MODE_ARGS=(--no_thinking)
    for dataset in $DATASETS; do
        launch python "$REPO/opsd/eval/evaluate_math.py" \
            --base_model "$MODEL" \
            "${CKPT_ARGS[@]}" \
            --dataset "$dataset" \
            "${MODE_ARGS[@]}" \
            --val_n "$VAL_N" \
            --temperature 1.0 \
            --tensor_parallel_size 1 \
            "$@"
    done
done
