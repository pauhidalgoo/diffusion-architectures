#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
mkdir -p delivery
rm -f delivery/hemera-final-source.tgz \
  delivery/hemera-final-checkpoints.tar \
  delivery/hemera-final-results.tar \
  delivery/delivery-manifest.json \
  delivery/SHA256SUMS

source_paths=(
  README.md
  FINAL_REPORT.md
  RESEARCH_PLAN.md
  MODEL_CARD.md
  LICENSE_AUDIT.md
  HUMAN_EVALUATION_GUIDE.md
  LOCAL_USE.md
  LICENSE
  pytest.ini
  requirements.txt
  requirements-lock.txt
  run.py
  hemera
  configs
  scripts
  tests
  evaluation
)
existing_source_paths=()
for path in "${source_paths[@]}"; do
  if [[ -e "$path" ]]; then
    existing_source_paths+=("$path")
  fi
done
tar \
  --sort=name \
  --mtime='@0' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -czf delivery/hemera-final-source.tgz \
  "${existing_source_paths[@]}"

mapfile -t checkpoints < <(
  find runs/final/hemera-nano -maxdepth 1 -type f \
    -name 'checkpoint-*.pt' -print | sort
)
if (( ${#checkpoints[@]} < 2 )); then
  echo "Expected at least best and last final checkpoints" >&2
  exit 73
fi
tar \
  --sort=name \
  --mtime='@0' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -cf delivery/hemera-final-checkpoints.tar \
  "${checkpoints[@]}"

result_paths=(
  reports
  data/manifests
  runs/final
  preflight-final
)
existing_result_paths=()
for path in "${result_paths[@]}"; do
  if [[ -e "$path" ]]; then
    existing_result_paths+=("$path")
  fi
done
tar \
  --sort=name \
  --mtime='@0' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --exclude='runs/final/hemera-nano/checkpoint-*.pt' \
  -cf delivery/hemera-final-results.tar \
  "${existing_result_paths[@]}"

(
  cd delivery
  sha256sum \
    hemera-final-source.tgz \
    hemera-final-checkpoints.tar \
    hemera-final-results.tar \
    > SHA256SUMS
  sha256sum --check SHA256SUMS
)

/venv/main/bin/python - <<'PY'
import hashlib
import json
import tarfile
from pathlib import Path

delivery = Path("delivery")
archives = []
for path in (
    delivery / "hemera-final-source.tgz",
    delivery / "hemera-final-checkpoints.tar",
    delivery / "hemera-final-results.tar",
):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with tarfile.open(path) as archive:
        members = [member.name for member in archive.getmembers()]
    if not members:
        raise SystemExit(f"Delivery archive is empty: {path}")
    archives.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "members": len(members),
            "first_member": members[0],
            "last_member": members[-1],
        }
    )
payload = {"schema_version": 1, "archives": archives}
target = delivery / "delivery-manifest.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
temporary.replace(target)
print(json.dumps(payload, indent=2))
PY

echo "Final delivery archives created and verified."
