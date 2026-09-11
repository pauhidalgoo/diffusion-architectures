#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh
mkdir -p reports

while [[ ! -f reports/final-train-exit-code ]]; do
  sleep 60
done

OUTPUT="reports/final-postprocess-telemetry.jsonl"
while [[ ! -f reports/final-postprocess-exit-code ]]; do
  timestamp="$(date +%s)"
  gpu="$(
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  active_command="$(
    pgrep -af \
      'run.py (evaluate|generate-eval|benchmark-images|generate-prompt-suite|export)|tune_and_evaluate|package_final_delivery' \
      | head -n 1 || true
  )"
  disk_available_bytes="$(
    df --output=avail -B1 /workspace | tail -n 1 | tr -d ' '
  )"

  python - "$timestamp" "$gpu" "$active_command" \
    "$disk_available_bytes" "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

timestamp, gpu, active_command, disk_available, output = sys.argv[1:]
parts = [part.strip() for part in gpu.split(",")]
payload = {
    "time": int(timestamp),
    "active_command": active_command,
    "disk_available_bytes": int(disk_available),
}
if len(parts) == 5:
    payload["gpu"] = {
        "utilization_percent": int(parts[0]),
        "memory_used_mib": int(parts[1]),
        "memory_total_mib": int(parts[2]),
        "temperature_c": int(parts[3]),
        "power_w": float(parts[4]),
    }
with Path(output).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  sleep 60
done
