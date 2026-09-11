#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh
mkdir -p reports

METRICS="runs/final/hemera-nano/metrics.jsonl"
OUTPUT="reports/final-telemetry.jsonl"

while [[ ! -f reports/final-train-exit-code ]]; do
  timestamp="$(date +%s)"
  gpu="$(
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  metric="$(tail -n 1 "$METRICS" 2>/dev/null || printf '{}')"
  process_alive=false
  if pgrep -f \
      'python run.py train --config configs/final-hemera-nano(-resume)?\.yaml' \
      >/dev/null; then
    process_alive=true
  fi
  disk_available_bytes="$(df --output=avail -B1 /workspace | tail -n 1 | tr -d ' ')"

  python - "$timestamp" "$gpu" "$metric" "$process_alive" \
    "$disk_available_bytes" "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

timestamp, gpu, raw_metric, alive, disk_available, output = sys.argv[1:]
parts = [part.strip() for part in gpu.split(",")]
payload = {
    "time": int(timestamp),
    "process_alive": alive == "true",
    "disk_available_bytes": int(disk_available),
    "latest_metric": json.loads(raw_metric),
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
