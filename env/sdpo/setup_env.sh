#!/bin/bash
# Build the SDPO training env: python 3.12, torch 2.9.1, vllm 0.14.0,
# transformers 4.57.1, ray 2.53.0, flash-attn 2.8.3, flashinfer 0.5.3.
# Every package is pinned in requirements.txt and installed with --no-deps.
#
#   bash env/sdpo/setup_env.sh            # venv at env/sdpo/.venv
#   VENV=/path/to/venv bash env/sdpo/setup_env.sh
set -euo pipefail

ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$ENV_DIR/../.." && pwd)
VENV=${VENV:-$ENV_DIR/.venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.12.8}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}

uv venv --clear --python "$PYTHON_VERSION" "$VENV"
source "$VENV/bin/activate"
uv pip install --no-deps -r "$ENV_DIR/requirements.txt"
uv pip install --no-deps -r "$ENV_DIR/requirements-extra.txt"
uv pip install --no-deps -e "$REPO_ROOT/sdpo"

python -c "import torch, transformers, ray; print('torch', torch.__version__, '| transformers', transformers.__version__, '| ray', ray.__version__)"
echo "env ready: source $VENV/bin/activate (GPU check: bash env/sdpo/verify_env.sh on a GPU node)"
