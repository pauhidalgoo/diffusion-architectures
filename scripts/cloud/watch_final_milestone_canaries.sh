#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

PROMPTS="evaluation/prompts/final-canary-v1.txt"
mkdir -p reports
rm -f reports/final-milestone-canaries-exit-code

resume_pid=""
resume_training() {
  if [[ -n "$resume_pid" ]]; then
    kill -CONT "$resume_pid" 2>/dev/null || true
    resume_pid=""
  fi
}
on_exit() {
  status=$?
  resume_training
  printf '%s\n' "$status" > reports/final-milestone-canaries-exit-code
  exit "$status"
}
trap on_exit EXIT

for percentage in 25 50 75 95; do
  checkpoint="runs/final/hemera-nano/checkpoint-${percentage}pct.pt"
  output="runs/final/canary-${percentage}pct"
  report="reports/final-canary-${percentage}pct-checkpoint.json"

  while [[ ! -f "$checkpoint" ]]; do
    if [[ -f reports/final-train-exit-code ]]; then
      echo "Training ended before ${percentage}% canary checkpoint" >&2
      exit 71
    fi
    sleep 30
  done

  resume_pid="$(
    pgrep -f \
      '^(/venv/main/bin/)?python run.py train --config configs/final-hemera-nano(-resume)?\.yaml$' \
      | head -n 1
  )"
  if [[ -z "$resume_pid" ]]; then
    echo "Could not identify trainer for ${percentage}% canary" >&2
    exit 72
  fi
  kill -STOP "$resume_pid"

  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  /venv/main/bin/python scripts/cloud/inspect_checkpoint.py "$checkpoint" \
    --output "$report"
  /venv/main/bin/python run.py sample \
    --checkpoint "$checkpoint" \
    --prompt-file "$PROMPTS" \
    --steps 30 \
    --guidance 3 \
    --seed 2026 \
    --dtype bfloat16 \
    --output "$output"

  /venv/main/bin/python - "$output" "$percentage" "$report" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
percentage = int(sys.argv[2])
checkpoint = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
paths = sorted(root.glob("sample-*.png"))
if len(paths) != 8:
    raise SystemExit(
        f"Expected eight images for {percentage}% canary, found {len(paths)}"
    )
rows = []
for path in paths:
    with Image.open(path) as image:
        image.load()
        if image.size != (256, 256) or image.mode != "RGB":
            raise SystemExit(
                f"Unexpected canary image contract: {path} "
                f"{image.size} {image.mode}"
            )
    rows.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
target = Path(f"reports/final-canary-{percentage}pct-images.json")
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(
    json.dumps(
        {
            "budget_percentage": percentage,
            "checkpoint": checkpoint,
            "images": rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)
temporary.replace(target)
PY

  resume_training
  echo "Completed ${percentage}% milestone canary." \
    >> reports/final-milestone-canaries.log
done

trap - EXIT
printf '%s\n' 0 > reports/final-milestone-canaries-exit-code
