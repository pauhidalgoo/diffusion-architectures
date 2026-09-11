from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        return True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path = ".") -> str:
    """Hash the runnable source/config snapshot even when no Git metadata exists."""
    base = Path(root).resolve()
    candidates: list[Path] = []
    for pattern in (
        "run.py",
        "requirements*.txt",
        "hemera/**/*.py",
        "configs/**/*.yaml",
        "scripts/cloud/*.sh",
        "scripts/cloud/*.py",
        "scripts/cloud/*.txt",
    ):
        candidates.extend(path for path in base.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "torch",
        "numpy",
        "diffusers",
        "transformers",
        "datasets",
        "safetensors",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def hardware_details(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {"device": str(device), "platform": platform.platform()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        details.update(
            {
                "gpu_name": properties.name,
                "gpu_vram_bytes": properties.total_memory,
                "cuda": torch.version.cuda,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return details


@dataclass
class RunManifest:
    name: str
    config: dict[str, Any]
    seed: int
    git_commit: str = field(default_factory=git_commit)
    git_dirty: bool = field(default_factory=git_dirty)
    source_tree_sha256: str = field(default_factory=source_tree_sha256)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    torch: str = field(default_factory=lambda: torch.__version__)
    platform: str = field(default_factory=platform.platform)
    device: str = ""
    parameter_count: int = 0
    active_parameter_count: int = 0
    estimated_flops: float | None = None
    flops_definition: str = "model forward FLOPs per sample"
    mean_throughput: float = 0.0
    peak_vram_bytes: int = 0
    wall_seconds: float = 0.0
    dependencies: dict[str, str] = field(default_factory=dependency_versions)
    hardware: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    gpu_seconds: float = 0.0
    estimated_eur: float = 0.0
    samples_seen: int = 0
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)
    data_artifacts: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, values: dict[str, Any]) -> None:
        payload = dict(values)
        payload.setdefault("time", time.time())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class BudgetWatchdog:
    def __init__(
        self,
        max_eur: float,
        hourly_eur: float,
        reserve_minutes: float = 10.0,
        already_spent_eur: float = 0.0,
    ):
        self.max_eur = max_eur
        self.hourly_eur = hourly_eur
        self.reserve_seconds = reserve_minutes * 60.0
        self.already_spent_eur = already_spent_eur
        self.started = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def spent_eur(self) -> float:
        return self.already_spent_eur + self.elapsed_seconds * self.hourly_eur / 3600.0

    @property
    def usable_eur(self) -> float:
        """Spend available to the run before the reserved shutdown window."""
        reserve_eur = self.reserve_seconds * self.hourly_eur / 3600.0
        return max(self.max_eur - reserve_eur, 0.0)

    @property
    def progress(self) -> float | None:
        """Normalized usable-budget progress, or None for step-limited runs."""
        if self.max_eur <= 0:
            return None
        if self.usable_eur <= 0:
            return 1.0
        return min(max(self.spent_eur / self.usable_eur, 0.0), 1.0)

    def should_stop(self) -> bool:
        if self.max_eur <= 0:
            return False
        remaining_seconds = (self.max_eur - self.spent_eur) * 3600.0 / self.hourly_eur
        return remaining_seconds <= self.reserve_seconds
