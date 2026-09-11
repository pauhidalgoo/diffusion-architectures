"""Verify downloaded Hemera final-delivery archives before instance shutdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_ARCHIVES = {
    "hemera-final-source.tgz",
    "hemera-final-checkpoints.tar",
    "hemera-final-results.tar",
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_sha256s(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Invalid SHA256SUMS row {line_number}")
        name = parts[1].lstrip("*")
        if Path(name).name != name or name in rows:
            raise ValueError(f"Unsafe or duplicate checksum target: {name!r}")
        rows[name] = parts[0].lower()
    if set(rows) != EXPECTED_ARCHIVES:
        raise ValueError(
            "SHA256SUMS archive set differs: "
            f"expected {sorted(EXPECTED_ARCHIVES)}, got {sorted(rows)}"
        )
    return rows


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def verify_delivery(root: str | Path) -> dict[str, Any]:
    delivery = Path(root)
    checksums = read_sha256s(delivery / "SHA256SUMS")
    manifest = json.loads(
        (delivery / "delivery-manifest.json").read_text(encoding="utf-8")
    )
    manifest_rows = {
        Path(row["path"]).name: row for row in manifest.get("archives", [])
    }
    if set(manifest_rows) != EXPECTED_ARCHIVES:
        raise ValueError("delivery-manifest.json has the wrong archive set")

    verified: list[dict[str, Any]] = []
    member_sets: dict[str, set[str]] = {}
    for name in sorted(EXPECTED_ARCHIVES):
        path = delivery / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty archive: {path}")
        digest = sha256_file(path)
        row = manifest_rows[name]
        if digest != checksums[name] or digest != str(row["sha256"]).lower():
            raise RuntimeError(f"Checksum mismatch: {name}")
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"Byte-size mismatch: {name}")
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
        if len(members) != int(row["members"]) or not members:
            raise RuntimeError(f"Tar member-count mismatch: {name}")
        unsafe = [member.name for member in members if not safe_member_name(member.name)]
        if unsafe:
            raise RuntimeError(f"Unsafe tar members in {name}: {unsafe[:3]}")
        names = {member.name for member in members}
        member_sets[name] = names
        verified.append(
            {
                "archive": name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "members": len(members),
            }
        )

    source = member_sets["hemera-final-source.tgz"]
    for required in {
        "README.md",
        "RESEARCH_PLAN.md",
        "MODEL_CARD.md",
        "run.py",
        "hemera",
        "configs/final-hemera-nano.yaml",
        "scripts/cloud/tune_and_evaluate.sh",
    }:
        if required not in source:
            raise RuntimeError(f"Source archive is missing {required}")

    checkpoints = member_sets["hemera-final-checkpoints.tar"]
    for suffix in {"checkpoint-best.pt", "checkpoint-last.pt"}:
        if not any(name.endswith(suffix) for name in checkpoints):
            raise RuntimeError(f"Checkpoint archive is missing {suffix}")

    results = member_sets["hemera-final-results.tar"]
    required_results = {
        "runs/final/locked-inference.json",
        "runs/final/test/metrics.json",
        "runs/final/hemera-nano/export/model.safetensors",
        "runs/final/memorization/neighbors-clip.json",
        "runs/final/memorization/neighbors-dinov2.json",
        "reports/cost-final.json",
    }
    for required in required_results:
        if required not in results:
            raise RuntimeError(f"Results archive is missing {required}")

    return {
        "schema_version": 1,
        "delivery": str(delivery.resolve()),
        "archives": verified,
        "required_artifacts_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("delivery")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify_delivery(args.delivery)
    rendered = json.dumps(result, indent=2)
    if args.output:
        target = Path(args.output)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(target)
    print(rendered)


if __name__ == "__main__":
    main()
