#!/bin/bash
# OPSD training run as in the paper. Extra arguments are forwarded to opsd/opsd_train.py.
#   MODEL=Qwen/Qwen3-8B TEACHER_THINKING=True PRIVILEGED_MODE=normal SEED=0 bash scripts/opsd/train_opsd.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/opsd/presets.sh"

MODEL=${MODEL:-Qwen/Qwen3-8B}
TEACHER_THINKING=${TEACHER_THINKING:-True}   # True | False
PRIVILEGED_MODE=${PRIVILEGED_MODE:-normal}   # normal | none | swap
LOSS_MASK=${LOSS_MASK:-none}                 # none | last_10pct | all_but_last_10pct
SEED=${SEED:-0}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-20}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/opsd}
mkdir -p "$OUT_ROOT" && OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"

opsd_preset "$(basename "$MODEL")" "$TEACHER_THINKING"
NUM_PROCESSES=${GPUS:-$NUM_PROCESSES}

RUN="opsd-model-$(basename "$MODEL")-teacher_thinking-$TEACHER_THINKING-privileged_mode-$PRIVILEGED_MODE"
[ "$LOSS_MASK" = none ] || RUN="$RUN-loss_mask-$LOSS_MASK"
RUN="$RUN-seed-$SEED"

[ -z "$ALLOC_CONF" ] || export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"
# The paper runs sampled with vLLM's PyTorch top-k/top-p path (no FlashInfer JIT).
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

cd "$REPO/opsd"
launch accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$(free_port)" \
    --gradient_accumulation_steps "$GA" \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PD" \
    --gradient_accumulation_steps "$GA" \
    --gradient_checkpointing True \
    --num_train_epochs 30 \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 20 \
    --max_completion_length 1024 \
    --max_length 20000 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --vllm_tensor_parallel_size "$VLLM_TP" \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --beta 0 \
    --lmbda 1 \
    --jsd_token_clip "$TOKEN_CLIP" \
    "${TEACHER_ARGS[@]}" \
    --student_thinking False \
    --teacher_thinking "$TEACHER_THINKING" \
    --privileged_mode "$PRIVILEGED_MODE" \
    --loss_mask "$LOSS_MASK" \
    --seed "$SEED" \
    --output_dir "$OUT_ROOT" \
    --run_config "$RUN" \
    --wandb_project "${WANDB_PROJECT:-OPSD}" \
    --report_to "${REPORT_TO:-wandb}" \
    "$@"
