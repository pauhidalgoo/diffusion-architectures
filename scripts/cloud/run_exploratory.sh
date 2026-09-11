#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.348576158}" exploratory
trap hemera_budget_report EXIT

test -f data/cache/index.json || {
  echo "Missing data/cache/index.json; audit and precompute Photonyx first." >&2
  exit 1
}

STAGE="${1:-objectives}"
case "$STAGE" in
  objectives)
    hemera_run python run.py sweep --config configs/sweeps/objectives.yaml
    bash scripts/cloud/evaluate_screening.sh \
      data/manifests/photonyx.jsonl runs/exploratory/objectives \
      reports/objective-ranking.json
    echo "Freeze the winning objective in experiment-base.yaml."
    echo "Then run: bash scripts/cloud/run_exploratory.sh conditioning"
    ;;
  conditioning)
    hemera_run python run.py sweep --config configs/sweeps/conditioning.yaml
    bash scripts/cloud/evaluate_screening.sh \
      data/manifests/photonyx.jsonl runs/exploratory/conditioning \
      reports/conditioning-ranking.json
    echo "Freeze the winning conditioning in experiment-base.yaml."
    echo "Then run: bash scripts/cloud/run_exploratory.sh backbones"
    ;;
  backbones)
    hemera_run python run.py sweep --config configs/sweeps/backbones.yaml
    bash scripts/cloud/evaluate_screening.sh \
      data/manifests/photonyx.jsonl runs/exploratory/backbones \
      reports/backbone-ranking.json
    echo "Freeze the winning efficiency-compatible transformer."
    echo "Then run: bash scripts/cloud/run_exploratory.sh attention"
    ;;
  attention)
    if [[ -f reports/selected-backbone-attention-complete ]]; then
      echo "Selected-backbone attention gate is already complete."
      exit 0
    fi
    hemera_run python run.py profile-attention --config configs/experiment-base.yaml \
      --output reports/attention-profile.json
    if python -c "import json,sys; sys.exit(0 if json.load(open('reports/attention-profile.json'))['train_window_candidate'] else 1)"; then
      hemera_run python run.py sweep --config configs/sweeps/window-attention.yaml
      bash scripts/cloud/evaluate_screening.sh \
        data/manifests/photonyx.jsonl runs/exploratory/window-attention \
        reports/window-attention-ranking.json
    else
      echo "Sliding-window attention missed the 15% speed threshold; preserving the negative profile."
    fi
    touch reports/selected-backbone-attention-complete
    echo "Then run: bash scripts/cloud/run_exploratory.sh efficiency"
    ;;
  efficiency)
    if [[ ! -f reports/selected-backbone-attention-complete ]]; then
      bash scripts/cloud/run_exploratory.sh attention
    fi
    hemera_run python run.py sweep --config configs/sweeps/efficiency.yaml
    hemera_run python run.py sweep --config configs/sweeps/data-sampling.yaml
    bash scripts/cloud/evaluate_screening.sh \
      data/manifests/photonyx.jsonl runs/exploratory/efficiency \
      reports/efficiency-ranking.json
    bash scripts/cloud/evaluate_screening.sh \
      data/manifests/photonyx.jsonl runs/exploratory/data-sampling \
      reports/data-sampling-ranking.json
    hemera_run python run.py report --runs runs/exploratory \
      --output reports/EXPLORATORY.md
    echo "Freeze the top two complete recipes in configs/sweeps/confirmation.yaml."
    echo "Then run: bash scripts/cloud/run_confirmation.sh"
    ;;
  *)
    echo "Usage: $0 {objectives|conditioning|backbones|attention|efficiency}" >&2
    exit 2
    ;;
esac
