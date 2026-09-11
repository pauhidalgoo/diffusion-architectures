#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f run.py ]]; then
  echo "Run this script from the Hemera repository root." >&2
  exit 1
fi

source scripts/cloud/budget_guard.sh
PHASE="${HEMERA_PHASE:-exploratory}"
CAP="${HEMERA_PHASE_CAP_EUR:-10}"
hemera_budget_init "$CAP" "${HEMERA_HOURLY_EUR:-0.5}" "$PHASE"
trap hemera_budget_report EXIT

hemera_run python -m pip install --upgrade pip
hemera_run python -m pip install -r requirements.txt
# HPSv2 pins an obsolete pytest version in its package metadata. Install its
# runtime dependencies explicitly so it cannot replace the tested environment.
hemera_run python -m pip install \
  ftfy einops braceexpand clint sentencepiece "protobuf<4" timm webdataset
hemera_run python -m pip install hpsv2==1.2.0 --no-deps
# `pip freeze` preserves non-portable conda build paths for a few base
# packages; `pip list --format=freeze` records equivalent exact versions.
python -m pip list --format=freeze > requirements-lock.txt

mkdir -p data/cache data/manifests runs reports

hemera_run python - <<'PY'
import os
import shutil
import torch

free = shutil.disk_usage(".").free / 1024**3
minimum = float(os.environ.get("HEMERA_MIN_DISK_GIB", "250"))
print(f"Free disk: {free:.1f} GiB")
if free < minimum:
    raise SystemExit(
        f"Hemera requires at least {minimum:.1f} GiB free for this configured workflow"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
    raise SystemExit("Hemera requires at least 24 GiB VRAM")
PY

hemera_run python -m pytest
echo "Hemera cloud bootstrap passed."
