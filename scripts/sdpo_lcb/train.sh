#!/bin/bash
# Train one SDPO or GRPO run on the LiveCodeBench v6 split (one 8xH100 node).
#
#   METHOD=sdpo MODEL=Qwen/Qwen3-8B SEED=0 bash scripts/sdpo_lcb/train.sh [hydra overrides...]
#
# Re-running the same command resumes from the latest checkpoint in the run directory.
# Every invocation appends a console log to $OUT_ROOT/$RUN_NAME/logs/, which analysis/sdpo_lcb reads.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

METHOD=${METHOD:-sdpo}                    # sdpo | grpo
MODEL=${MODEL:-Qwen/Qwen3-8B}             # Qwen/Qwen3-8B | Qwen/Qwen3-14B | allenai/Olmo-3-7B-Instruct
SEED=${SEED:-0}
STEPS=${STEPS:-200}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/sdpo_lcb/lcb_exp}
OUT_ROOT=${OUT_ROOT:-$REPO_ROOT/outputs/sdpo_lcb}
NGPUS=${NGPUS:-8}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-25}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-1}   # the last checkpoint (step $STEPS) is always kept
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}    # >0 validates on a random subset (smoke tests)
LOGGER=${LOGGER:-"['console','wandb']"}
WANDB_PROJECT=${WANDB_PROJECT:-sdpo-lcb}

MODEL_TAG=$(basename "$MODEL")
RUN_NAME=${RUN_NAME:-lcb-${METHOD}-${MODEL_TAG}-seed${SEED}}
RUN_DIR=$OUT_ROOT/$RUN_NAME

# Model presets. Qwen3-14B does not fit next to the colocated vLLM engine without CPU offload,
# and SDPO additionally needs the chunked top-k reduction.
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
TOPK_CHUNK_SIZE=null
case "$MODEL_TAG" in
    Qwen3-14B)
        PARAM_OFFLOAD=True
        OPTIMIZER_OFFLOAD=True
        TOPK_CHUNK_SIZE=1024
        ;;
esac

case "$METHOD" in
    sdpo)
        CONFIG_NAME=sdpo
        METHOD_ARGS=(
            actor_rollout_ref.actor.ppo_mini_batch_size=1
            actor_rollout_ref.actor.self_distillation.distillation_topk=20
            actor_rollout_ref.actor.self_distillation.topk_chunk_size=$TOPK_CHUNK_SIZE
            actor_rollout_ref.actor.self_distillation.alpha=1.0
            actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01
            actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
            actor_rollout_ref.actor.self_distillation.include_environment_feedback=True
        )
        ;;
    grpo)
        CONFIG_NAME=baseline_grpo
        METHOD_ARGS=(
            actor_rollout_ref.actor.ppo_mini_batch_size=8
            # GRPO never uses the distillation top-k; skip computing it.
            actor_rollout_ref.actor.self_distillation.full_logit_distillation=False
        )
        ;;
    *)
        echo "METHOD must be sdpo or grpo, got '$METHOD'" >&2
        exit 2
        ;;
esac

# Ray workers do not reliably inherit HF offline settings, so pass a local snapshot path
# whenever the model is already in the HF cache.
MODEL_PATH=$(python - "$MODEL" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
if os.path.isdir(model):
    print(model)
else:
    try:
        print(snapshot_download(model, local_files_only=True))
    except Exception:
        print(model)
PY
)

# A fresh local Ray cluster per run, so a stale cluster left on the host is never joined.
export RAY_ADDRESS=local
export RAY_TMPDIR=${RAY_TMPDIR:-$(mktemp -d /tmp/ray.XXXXXX)}
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# sdpo/verl/trainer/config/user.yaml interpolates these.
export TASK=$DATA_DIR
export EXPERIMENT=$RUN_NAME
ulimit -c 0

mkdir -p "$RUN_DIR/logs"
LOG=$RUN_DIR/logs/train-$(date +%Y%m%d-%H%M%S).log
echo "run $RUN_NAME -> $RUN_DIR (model $MODEL_PATH)" | tee "$LOG"

python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    "data.train_files=['$DATA_DIR/train.parquet']" \
    "data.val_files=['$DATA_DIR/test.parquet']" \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.train_batch_size=32 \
    data.seed=$SEED \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.data_loader_seed=$SEED \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    custom_reward_function.path=$REPO_ROOT/sdpo/verl/utils/reward_score/feedback/__init__.py \
    trainer.logger="$LOGGER" \
    trainer.project_name=$WANDB_PROJECT \
    ++trainer.group_name=lcb-${METHOD}-${MODEL_TAG} \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$RUN_DIR \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=200 \
    trainer.total_training_steps=$STEPS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_TO_KEEP \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=True \
    hydra.run.dir=$RUN_DIR \
    "${METHOD_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$LOG"
