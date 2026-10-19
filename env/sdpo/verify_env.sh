#!/bin/bash
# GPU-side check of the SDPO env. Run on a GPU node; prints ALL_CHECKS_PASSED.
set -uo pipefail

ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$ENV_DIR/.venv}
PY="$VENV/bin/python"

echo "host=$(hostname) python=$PY"
nvidia-smi -L || echo "nvidia-smi failed"

"$PY" - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| devices", torch.cuda.device_count())
x = torch.randn(2048, 2048, device="cuda")
print("gpu matmul ok, trace =", float((x @ x).diagonal().sum()))
import flash_attn
print("flash_attn", flash_attn.__version__)
import flashinfer
print("flashinfer", flashinfer.__version__)
import transformers
print("transformers", transformers.__version__)
import vllm
print("vllm", vllm.__version__)
import ray
print("ray", ray.__version__)
import verl
from verl.trainer import main_ppo  # noqa: F401
print("verl", verl.__file__)
print("ALL_CHECKS_PASSED")
PY
rc=$?
echo "verify_env.sh rc=$rc"
exit $rc
