# Shared settings for scripts/sdpo_sci/*.sh (sourced, not executed).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDPO_DIR="$REPO_ROOT/sdpo"
export PYTHONPATH="$SDPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASETS=(biology chemistry material physics tooluse)

MODEL="${MODEL:-Qwen/Qwen3-8B}"
STEPS="${STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-5}"
N_GPUS="${N_GPUS:-8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/sdpo_sci}"
export WANDB_PROJECT="${WANDB_PROJECT:-sdpo-sci}"
export WANDB_DIR="${WANDB_DIR:-$OUT_ROOT}"

# Validation decoding, identical for RL (rollout.val_kwargs) and SFT (trainer.val_generation).
VAL_N=16
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192

dataset_dir() {
  case "$1" in
    biology | chemistry | material | physics) echo "$SDPO_DIR/datasets/sciknoweval/$1" ;;
    tooluse) echo "$SDPO_DIR/datasets/tooluse" ;;
    *) echo "unknown DATASET '$1' (expected one of: ${DATASETS[*]})" >&2; return 1 ;;
  esac
}

# Ray workers and torchrun ranks resolve the model by local path, so a hub id never triggers
# a network lookup on an offline node.
resolve_model() {
  if [ -d "$MODEL" ]; then
    echo "$MODEL"
  else
    python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$MODEL"
  fi
}
