#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
export PATH="/venv/main/bin:${PATH}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET=1

python run.py sample \
  --checkpoint preflight/gpu-overfit/checkpoint-last.pt \
  --prompt-file scripts/cloud/preflight-prompts.txt \
  --negative-prompt "blurry, distorted, text" \
  --steps 30 \
  --guidance 3 \
  --seed 31415 \
  --device cuda \
  --dtype bfloat16 \
  --output preflight/samples-real
