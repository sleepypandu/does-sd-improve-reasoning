#!/usr/bin/env bash
# Final-answer SFT on one dataset on a single node; extra arguments are passed to hydra.
#   DATASET=biology SEED=1 bash scripts/sdpo_sci/train_sft.sh
# Validation samples with the RL decoding parameters every TEST_FREQ steps (no step-0 point).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET="${DATASET:?set DATASET to one of: ${DATASETS[*]}}"
SEED="${SEED:-1}"
LR="${LR:-1e-5}"
DATA_DIR="$(dataset_dir "$DATASET")"
RUN_NAME="${RUN_NAME:-sft-$DATASET-lr$LR-s$SEED}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
MODEL_PATH="$(resolve_model)"
mkdir -p "$RUN_DIR" "$WANDB_DIR"
export HF_DATASETS_CACHE="$RUN_DIR/hf_datasets_cache"

cd "$SDPO_DIR"
torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/train.sft.parquet" \
  data.val_files="$DATA_DIR/test.sft.parquet" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=8192 \
  data.truncation=right \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  model.partial_pretrain="$MODEL_PATH" \
  model.strategy=fsdp \
  model.trust_remote_code=True \
  optim.lr="$LR" \
  optim.lr_scheduler=constant \
  optim.lr_warmup_steps_ratio=0.0 \
  trainer.seed="$SEED" \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=1000 \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq=-1 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.logger="[console,wandb]" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.val_generation.enable=True \
  trainer.val_generation.n="$VAL_N" \
  trainer.val_generation.temperature="$VAL_TEMPERATURE" \
  trainer.val_generation.top_p="$VAL_TOP_P" \
  trainer.val_generation.do_sample=True \
  trainer.val_generation.max_new_tokens="$MAX_RESPONSE_LENGTH" \
  trainer.val_generation.max_prompt_length="$MAX_PROMPT_LENGTH" \
  "$@"
