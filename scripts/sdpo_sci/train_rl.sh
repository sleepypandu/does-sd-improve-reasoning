#!/usr/bin/env bash
# Train SDPO or GRPO on one dataset on a single node and validate on all five; extra arguments
# are passed to hydra.
#   METHOD=sdpo DATASET=chemistry SEED=1 bash scripts/sdpo_sci/train_rl.sh
# LR defaults to 1e-5 for SDPO and 1e-6 for GRPO (GRPO is unstable at 1e-5).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

METHOD="${METHOD:?set METHOD=sdpo or grpo}"
DATASET="${DATASET:?set DATASET to one of: ${DATASETS[*]}}"
SEED="${SEED:-1}"
case "$METHOD" in
  # The teacher sees only successful sibling rollouts, no environment feedback, as upstream's
  # generalization runs do. On ToolAlpaca the feedback would contain the ground-truth call.
  sdpo) CONFIG=sdpo LR="${LR:-1e-5}" METHOD_ARGS=(actor_rollout_ref.actor.self_distillation.include_environment_feedback=False) ;;
  grpo) CONFIG=baseline_grpo LR="${LR:-1e-6}" METHOD_ARGS=() ;;
  *) echo "METHOD must be sdpo or grpo, got '$METHOD'" >&2; exit 1 ;;
esac
DATA_DIR="$(dataset_dir "$DATASET")"
# verl resumes from an existing default_local_dir, so every run needs its own name.
RUN_NAME="${RUN_NAME:-$METHOD-$DATASET-lr$LR-s$SEED}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
MODEL_PATH="$(resolve_model)"
[ -f "$ALLVAL" ] || { echo "missing $ALLVAL; run scripts/sdpo_sci/prepare_data.sh" >&2; exit 1; }
mkdir -p "$RUN_DIR" "$WANDB_DIR"

# A fresh local Ray cluster in a private temp dir, instead of attaching to a stale session left
# on the node by another job.
export RAY_ADDRESS=local
RAY_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/ray.XXXXXX")"
export RAY_TMPDIR
trap 'rm -rf "$RAY_TMPDIR"' EXIT
# Concurrent runs on the same parquet race on a shared datasets cache.
export HF_DATASETS_CACHE="$RUN_DIR/hf_datasets_cache"
# Read by verl/trainer/config/user.yaml.
export TASK="$DATASET" EXPERIMENT="$RUN_NAME"

cd "$SDPO_DIR"
python -m verl.trainer.main_ppo --config-name "$CONFIG" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  data.train_files="[$DATA_DIR/train.parquet]" \
  data.val_files="[$ALLVAL]" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.seed="$SEED" \
  custom_reward_function.path="$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train=True \
  trainer.save_freq=-1 \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR" \
  "${METHOD_ARGS[@]}" \
  "$@"
