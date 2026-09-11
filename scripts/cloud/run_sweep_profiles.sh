#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
export PATH=/venv/main/bin:"$PATH"
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1

trap 'status=$?; printf "%s\n" "$status" > reports/sweep-profiles-exit-code' EXIT
rm -f reports/sweep-profiles-exit-code

for study in objectives conditioning backbones; do
  python run.py profile-sweep-batches \
    --sweep "configs/sweeps/${study}.yaml" \
    --cache-dir data/representation-cache/dc-clip \
    --output "reports/batch-profiles/${study}" \
    --batch-sizes 64 96 128 192 256 384 768 \
    --warmup 2 --iterations 5
done
