#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
export PATH=/venv/main/bin:$PATH
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}"
export HEMERA_GPU_DATA_CACHE="${HEMERA_GPU_DATA_CACHE:-1}"
mkdir -p logs reports
exec > >(tee -a logs/full-exploratory-pipeline.log) 2>&1
trap 'status=$?; printf "%s\n" "$status" > reports/full-pipeline-exit-code' EXIT
rm -f reports/full-pipeline-exit-code

while [[ ! -f reports/exploratory-cache-exit-code ]]; do
  sleep 15
done
if [[ "$(cat reports/exploratory-cache-exit-code)" != "0" ]]; then
  echo "Exploratory cache failed; refusing to train." >&2
  exit 1
fi
python - <<'PY'
import json

index = json.load(open("data/cache/index.json", encoding="utf-8"))
progress = json.load(open("data/cache/precompute-progress.json", encoding="utf-8"))
representation = index["representation"]
if progress.get("stage") != "complete":
    raise SystemExit("Cache progress is not complete")
if index.get("statistics", {}).get("total_examples") != 20_000:
    raise SystemExit("Full pipeline requires the frozen 20k cache")
if representation.get("selection_count") != 20_000:
    raise SystemExit("Cache selection is not the frozen 20k subset")
if not representation.get("repa_encoder"):
    raise SystemExit("Cache lacks required REPA targets")
PY

if [[ ! -f reports/objective-ranking.json ]]; then
  bash scripts/cloud/run_exploratory.sh objectives
fi
if [[ ! -f reports/frozen-objective.json ]]; then
  python scripts/cloud/freeze_stage.py freeze \
    --ranking reports/objective-ranking.json \
    --base configs/experiment-base.yaml --stage objective \
    --record reports/frozen-objective.json
fi

if [[ ! -f reports/conditioning-ranking.json ]]; then
  bash scripts/cloud/run_exploratory.sh conditioning
fi
if [[ ! -f reports/frozen-conditioning.json ]]; then
  python scripts/cloud/freeze_stage.py freeze \
    --ranking reports/conditioning-ranking.json \
    --base configs/experiment-base.yaml --stage conditioning \
    --record reports/frozen-conditioning.json
fi

if [[ ! -f reports/backbone-ranking.json ]]; then
  bash scripts/cloud/run_exploratory.sh backbones
fi
if [[ ! -f reports/frozen-backbone-transformer.json ]]; then
  python scripts/cloud/freeze_stage.py freeze \
    --ranking reports/backbone-ranking.json \
    --base configs/experiment-base.yaml --stage backbone-transformer \
    --record reports/frozen-backbone-transformer.json
fi

if [[ ! -f reports/selected-backbone-attention-complete ]]; then
  bash scripts/cloud/run_exploratory.sh attention
fi

if [[ ! -f reports/efficiency-ranking.json || ! -f reports/data-sampling-ranking.json ]]; then
  python run.py profile-sweep-batches \
    --sweep configs/sweeps/efficiency.yaml --cache-dir data/cache \
    --output reports/batch-profiles/efficiency \
    --batch-sizes 64 96 128 192 256 384 768 --warmup 2 --iterations 5
  python scripts/cloud/freeze_stage.py apply-batches \
    --sweep configs/sweeps/efficiency.yaml \
    --profiles reports/batch-profiles/efficiency/summary.json
  bash scripts/cloud/run_exploratory.sh efficiency
fi
if [[ ! -f reports/frozen-efficiency.json ]]; then
  python scripts/cloud/freeze_stage.py freeze \
    --ranking reports/efficiency-ranking.json \
    --base configs/experiment-base.yaml --stage efficiency \
    --record reports/frozen-efficiency.json
fi
if [[ ! -f reports/frozen-data-sampling.json ]]; then
  python scripts/cloud/freeze_stage.py freeze \
    --ranking reports/data-sampling-ranking.json \
    --base configs/experiment-base.yaml --stage data \
    --record reports/frozen-data-sampling.json
fi

if [[ ! -f reports/confirmation-comparison.json ]]; then
  if ! python scripts/cloud/validate_completed_confirmation.py \
    --output reports/confirmation-checkpoints.json; then
    # Always materialize this from the immutable ranking. Reusing a stale
    # placeholder can silently turn confirmation into a same-recipe comparison.
    python scripts/cloud/freeze_stage.py prepare-confirmation \
      --ranking reports/efficiency-ranking.json \
      --output configs/sweeps/confirmation.yaml \
      --max-eur "${HEMERA_CONFIRMATION_BUDGET_EUR:-0.9}"
    python run.py profile-sweep-batches \
      --sweep configs/sweeps/confirmation.yaml --cache-dir data/cache \
      --output reports/batch-profiles/confirmation \
      --batch-sizes 64 96 128 192 256 384 768 --warmup 2 --iterations 5
    python scripts/cloud/freeze_stage.py apply-batches \
      --sweep configs/sweeps/confirmation.yaml \
      --profiles reports/batch-profiles/confirmation/summary.json
  fi
  bash scripts/cloud/run_confirmation.sh
fi

python run.py report --runs runs/exploratory --output reports/EXPLORATORY.md
