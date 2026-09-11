"""Assemble a minimal, checksum-verifiable Hemera-Nano publication directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = (
    ROOT
    / "artifacts"
    / "final-model"
    / "runs"
    / "final"
    / "hemera-nano"
    / "export"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "publication-candidate"
DEFAULT_LOCKED = (
    ROOT
    / "artifacts"
    / "final-model"
    / "runs"
    / "final"
    / "locked-inference.json"
)
DEFAULT_METRICS = (
    ROOT
    / "artifacts"
    / "final-model"
    / "runs"
    / "final"
    / "test"
    / "metrics.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def assemble(
    *,
    export_dir: Path,
    output_dir: Path,
    locked_inference: Path,
    metrics: Path,
    expected_model_sha256: str | None = None,
) -> dict[str, object]:
    """Build an atomic publication directory and return its manifest."""
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing publication directory: {output_dir}"
        )

    model_path = export_dir / "model.safetensors"
    actual_model_sha256 = sha256(model_path)
    if (
        expected_model_sha256 is not None
        and actual_model_sha256.lower() != expected_model_sha256.lower()
    ):
        raise ValueError(
            "Exported model hash mismatch: "
            f"expected {expected_model_sha256}, got {actual_model_sha256}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for filename in ("model.safetensors", "config.json", "model_index.json"):
            _copy_file(export_dir / filename, staging / filename)

        _copy_file(ROOT / "MODEL_CARD.md", staging / "README.md")
        _copy_file(ROOT / "MODEL_CARD.md", staging / "MODEL_CARD.md")
        _copy_file(ROOT / "LICENSE_AUDIT.md", staging / "LICENSE_AUDIT.md")
        _copy_file(ROOT / "LICENSE", staging / "LICENSE")
        _copy_file(ROOT / "FINAL_REPORT.md", staging / "FINAL_REPORT.md")
        _copy_file(locked_inference, staging / "evaluation" / "locked-inference.json")
        _copy_file(metrics, staging / "evaluation" / "test-metrics.json")

        records = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                records.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

        manifest: dict[str, object] = {
            "schema_version": 1,
            "model_sha256": actual_model_sha256,
            "files": records,
        }
        manifest_path = staging / "PUBLICATION_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksums = [
            f"{record['sha256']}  {record['path']}" for record in records
        ]
        checksums.append(f"{sha256(manifest_path)}  PUBLICATION_MANIFEST.json")
        (staging / "SHA256SUMS").write_text(
            "\n".join(checksums) + "\n", encoding="utf-8"
        )

        staging.rename(output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--locked-inference", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--expected-model-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = assemble(
        export_dir=args.export_dir.resolve(),
        output_dir=args.output.resolve(),
        locked_inference=args.locked_inference.resolve(),
        metrics=args.metrics.resolve(),
        expected_model_sha256=args.expected_model_sha256,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "files": len(manifest["files"]),
                "model_sha256": manifest["model_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
