#!/bin/bash
# Build the pinned OPSD environment with uv and print how to activate it.
#   bash env/opsd/setup_env.sh            # creates env/opsd/.venv
#   VENV=/path/to/venv bash env/opsd/setup_env.sh
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="${VENV:-$ENV_DIR/.venv}"

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --project "$ENV_DIR" --frozen --python 3.12

echo "Activate with: source $UV_PROJECT_ENVIRONMENT/bin/activate"
