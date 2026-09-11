from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.assemble_publication_candidate import assemble


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_assemble_publication_candidate_is_atomic_and_checksummed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    export = tmp_path / "export"
    output = tmp_path / "candidate"
    model = b"synthetic-safetensors"

    for name in ("MODEL_CARD.md", "LICENSE_AUDIT.md", "LICENSE", "FINAL_REPORT.md"):
        _write(source_root / name, name.encode())
    _write(export / "model.safetensors", model)
    _write(export / "config.json", b"{}")
    _write(export / "model_index.json", b"{}")
    locked = tmp_path / "locked.json"
    metrics = tmp_path / "metrics.json"
    _write(locked, b"{}")
    _write(metrics, b"{}")

    monkeypatch.setattr(
        "scripts.assemble_publication_candidate.ROOT", source_root
    )
    model_hash = hashlib.sha256(model).hexdigest()
    manifest = assemble(
        export_dir=export,
        output_dir=output,
        locked_inference=locked,
        metrics=metrics,
        expected_model_sha256=model_hash,
    )

    assert output.is_dir()
    assert manifest["model_sha256"] == model_hash
    assert (output / "README.md").read_bytes() == b"MODEL_CARD.md"
    assert (output / "evaluation" / "test-metrics.json").is_file()
    disk_manifest = json.loads(
        (output / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert disk_manifest == manifest
    assert "model.safetensors" in (output / "SHA256SUMS").read_text()

    with pytest.raises(FileExistsError):
        assemble(
            export_dir=export,
            output_dir=output,
            locked_inference=locked,
            metrics=metrics,
        )


def test_assemble_rejects_wrong_model_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    export = tmp_path / "export"
    output = tmp_path / "candidate"
    for name in ("MODEL_CARD.md", "LICENSE_AUDIT.md", "LICENSE", "FINAL_REPORT.md"):
        _write(source_root / name, b"x")
    for name in ("model.safetensors", "config.json", "model_index.json"):
        _write(export / name, b"x")
    locked = tmp_path / "locked.json"
    metrics = tmp_path / "metrics.json"
    _write(locked, b"{}")
    _write(metrics, b"{}")
    monkeypatch.setattr(
        "scripts.assemble_publication_candidate.ROOT", source_root
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        assemble(
            export_dir=export,
            output_dir=output,
            locked_inference=locked,
            metrics=metrics,
            expected_model_sha256="0" * 64,
        )
    assert not output.exists()
