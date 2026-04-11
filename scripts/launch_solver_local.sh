#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RZERO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RZERO_DIR}/../.." && pwd)"
cd "${RZERO_DIR}"

SOLVER_MODEL_PATH="${1:-Qwen/Qwen3-8B-Base}"
QUESTIONER_MODEL_PATH="${2:-/nemo-workspace/inf-evolve/home/project/self-evolution-explore/baselines/r-zero/checkpoints/qwen3_8b/models/qwen3_8b_questioner_v1/global_step_5/actor/huggingface}"
EXPERIMENT_NAME="${3:-qwen3_8b_solver_v1}"

export PYTHONPATH="${RZERO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_tmp}"
mkdir -p "${RAY_TMPDIR}"

export STORAGE_PATH="${STORAGE_PATH:-/nemo-workspace/inf-evolve/home/project/self-evolution-explore/baselines/r-zero/checkpoints/qwen3_8b}"
mkdir -p "${STORAGE_PATH}/models"

export HUGGINGFACENAME="${HUGGINGFACENAME:-self-evolution-exploration}"
export WANDB_ENTITY="${WANDB_ENTITY:-self-evolution-exploration}"
export RZERO_WANDB_PROJECT="${RZERO_WANDB_PROJECT:-rzero}"

ensure_rzero_env() {
  if python3 - <<'PY'
import sys

expected = {
    "torch": "2.7.0",
    "transformers": "4.52.4",
    "flash_attn": "2.7.4.post1",
}

try:
    import torch
    import transformers
    import flash_attn
except Exception:
    sys.exit(1)

versions = {
    "torch": torch.__version__.split("+", 1)[0],
    "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
}

sys.exit(0 if versions == expected else 1)
PY
  then
    echo "=== R-Zero environment already matches pinned versions ==="
    return
  fi

  echo "=== Repairing R-Zero Python dependencies ==="
  pip install torch==2.7.0 --no-cache-dir
  rm -rf /usr/lib/python3/dist-packages/blinker* 2>/dev/null || true
  grep -v '^av==' requirements.txt | grep -v '^torch==' | grep -v '^flash_attn==' | \
    pip install -r /dev/stdin --no-cache-dir
  pip install flash_attn==2.7.4.post1 --no-build-isolation --no-cache-dir
  pip install stopit --no-cache-dir

  echo "=== Verified R-Zero package versions ==="
  python3 - <<'PY'
import flash_attn
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
PY
}

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HF_TOKEN_POOL_JSON:-}" ]; then
  HF_TOKEN="$(python - <<'PY'
import json
import os

raw = os.environ.get("HF_TOKEN_POOL_JSON", "[]")
pool = json.loads(raw)
token = ""
if isinstance(pool, list) and pool:
    first = pool[0]
    token = first.get("token", "") if isinstance(first, dict) else str(first)
print(token)
PY
)"
  export HF_TOKEN
fi

cat > tokens.json <<EOF
{
  "huggingface": "${HF_TOKEN:-placeholder}",
  "wandb": "${WANDB_API_KEY:-placeholder}"
}
EOF

if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
fi

ensure_rzero_env

if ! grep -q 'trainer.project_name=' scripts/solver_train.sh; then
  sed -i "s|trainer.experiment_name=|trainer.project_name=${RZERO_WANDB_PROJECT} trainer.experiment_name=|g" \
    scripts/solver_train.sh scripts/questioner_train_penalty.sh
fi

mkdir -p .output
bash scripts/solver_train.sh \
  "${SOLVER_MODEL_PATH}" \
  "${QUESTIONER_MODEL_PATH}" \
  "${EXPERIMENT_NAME}" \
  2>&1 | tee -a ".output/${EXPERIMENT_NAME}_resume.log" /proc/1/fd/1
