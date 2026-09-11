#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh
mkdir -p reports

while [[ ! -f reports/final-train-exit-code ]]; do
  if ! pgrep -f \
      'python run.py train --config configs/final-hemera-nano(-resume)?\.yaml' \
      >/dev/null; then
    echo "Training process disappeared before writing an exit status." \
      > reports/final-postprocess.log
    printf '%s\n' 67 > reports/final-postprocess-exit-code
    exit 67
  fi
  sleep 30
done

exec bash scripts/cloud/run_final_postprocess.sh \
  configs/final-hemera-nano.yaml
