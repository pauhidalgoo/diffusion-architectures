from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.cloud.verify_final_delivery import (
    EXPECTED_ARCHIVES,
    verify_delivery,
)


CONTENTS = {
    "hemera-final-source.tgz": {
        "README.md",
        "RESEARCH_PLAN.md",
        "MODEL_CARD.md",
        "run.py",
        "hemera",
        "configs/final-hemera-nano.yaml",
        "scripts/cloud/tune_and_evaluate.sh",
    },
    "hemera-final-checkpoints.tar": {
        "runs/final/hemera-nano/checkpoint-best.pt",
        "runs/final/hemera-nano/checkpoint-last.pt",
    },
    "hemera-final-results.tar": {
        "runs/final/locked-inference.json",
        "runs/final/test/metrics.json",
        "runs/final/hemera-nano/export/model.safetensors",
        "runs/final/memorization/neighbors-clip.json",
        "runs/final/memorization/neighbors-dinov2.json",
        "reports/cost-final.json",
    },
}


def build_delivery(root: Path) -> None:
    manifest_rows = []
    checksums = []
    for name in sorted(EXPECTED_ARCHIVES):
        path = root / name
        mode = "w:gz" if name.endswith(".tgz") else "w"
        with tarfile.open(path, mode) as archive:
            for member_name in sorted(CONTENTS[name]):
                payload = member_name.encode()
                info = tarfile.TarInfo(member_name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {name}\n")
        manifest_rows.append(
            {
                "path": f"delivery/{name}",
                "bytes": path.stat().st_size,
                "sha256": digest,
                "members": len(CONTENTS[name]),
            }
        )
    (root / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    (root / "delivery-manifest.json").write_text(
        json.dumps({"schema_version": 1, "archives": manifest_rows}),
        encoding="utf-8",
    )


def test_verify_delivery_accepts_complete_archives(tmp_path: Path) -> None:
    build_delivery(tmp_path)
    result = verify_delivery(tmp_path)
    assert result["required_artifacts_verified"] is True
    assert len(result["archives"]) == 3


def test_verify_delivery_rejects_tampered_archive(tmp_path: Path) -> None:
    build_delivery(tmp_path)
    with (tmp_path / "hemera-final-results.tar").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_delivery(tmp_path)
