#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
export PATH=/venv/main/bin:"$PATH"
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1

trap 'status=$?; printf "%s\n" "$status" > reports/post-representation-exit-code' EXIT
rm -f reports/post-representation-exit-code

# The first two representation arms predated latent-distribution diagnostics.
python run.py probe-representation \
  --config configs/representations/sd-clip.yaml \
  --output runs/exploratory/representations/sd-clip/representation-probe.json \
  --samples 256 --batch-size 16
python run.py probe-representation \
  --config configs/representations/dc-clip.yaml \
  --output runs/exploratory/representations/dc-clip/representation-probe.json \
  --samples 256 --batch-size 16

# Exercise complete forward/backward/optimizer steps on the selected real cache.
python run.py profile-batch \
  --config configs/representations/dc-clip.yaml \
  --output reports/dc-clip-batch-profile.json \
  --batch-sizes 64 128 256 384 512 768 \
  --warmup 2 --iterations 5
