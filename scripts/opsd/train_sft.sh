#!/bin/bash
# SFT baseline run as in the paper. Extra arguments are forwarded to opsd/sft_train.py.
#   MODEL=Qwen/Qwen3-8B SEED=0 bash scripts/opsd/train_sft.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/opsd/presets.sh"

MODEL=${MODEL:-Qwen/Qwen3-8B}
SEED=${SEED:-0}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-20}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/opsd}
mkdir -p "$OUT_ROOT" && OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"

sft_preset "$(basename "$MODEL")"
NUM_PROCESSES=${GPUS:-$NUM_PROCESSES}
RUN="sft-model-$(basename "$MODEL")-seed-$SEED"

[ -z "$ALLOC_CONF" ] || export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"

cd "$REPO/opsd"
launch accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$(free_port)" \
    --gradient_accumulation_steps "$GA" \
    sft_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate 5e-6 \
    --per_device_train_batch_size "$PD" \
    --gradient_accumulation_steps "$GA" \
    --gradient_checkpointing True \
    --num_train_epochs 4 \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 20 \
    --max_length 16000 \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --seed "$SEED" \
    --output_dir "$OUT_ROOT/$RUN" \
    --wandb_project "${WANDB_PROJECT:-OPSD}" \
    --report_to "${REPORT_TO:-wandb}" \
    "$@"
