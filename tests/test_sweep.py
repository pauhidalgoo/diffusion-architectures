from __future__ import annotations

import json
from pathlib import Path

import yaml

from hemera.runtime import sha256_file
from hemera.sweep import run_sweep


def _write_sweep(tmp_path: Path) -> Path:
    base = {
        "name": "base",
        "train": {
            "output_dir": str(tmp_path / "unused"),
            "steps": 1,
            "dataset_mode": "synthetic",
        },
        "budget": {"max_eur": 0, "hourly_eur": 0.5},
    }
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(base), encoding="utf-8"
    )
    sweep = {
        "base": "base.yaml",
        "output_dir": str(tmp_path / "runs"),
        "runs": [{"name": "candidate", "overrides": {}}],
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(sweep), encoding="utf-8")
    return path


def test_sweep_reuses_checksum_validated_complete_run(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_sweep(tmp_path)
    calls = {"train": 0, "evaluate": 0}

    def train(config):
        calls["train"] += 1
        output = Path(config.train.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "checkpoint-last.pt"
        checkpoint.write_bytes(b"checkpoint")
        manifest = {
            "config": config.to_dict(),
            "checkpoint_hashes": {
                checkpoint.name: sha256_file(checkpoint)
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return checkpoint

    def evaluate(config, checkpoint):
        calls["evaluate"] += 1
        result = {"validation_loss": 1.0}
        (Path(config.train.output_dir) / "evaluation.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    monkeypatch.setattr("hemera.sweep.train_from_config", train)
    monkeypatch.setattr("hemera.sweep.evaluate_checkpoint", evaluate)
    first = run_sweep(path)
    second = run_sweep(path)
    assert first == second
    assert calls == {"train": 1, "evaluate": 1}


def test_sweep_preserves_partial_attempt_before_restart(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_sweep(tmp_path)
    partial = tmp_path / "runs" / "candidate"
    partial.mkdir(parents=True)
    (partial / "metrics.jsonl").write_text(
        '{"step": 1}\n', encoding="utf-8"
    )

    def train(config):
        output = Path(config.train.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "checkpoint-last.pt"
        checkpoint.write_bytes(b"new")
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "checkpoint_hashes": {
                        checkpoint.name: sha256_file(checkpoint)
                    },
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    monkeypatch.setattr("hemera.sweep.train_from_config", train)
    monkeypatch.setattr(
        "hemera.sweep.evaluate_checkpoint",
        lambda config, checkpoint: {"validation_loss": 2.0},
    )
    run_sweep(path)
    preserved = (
        tmp_path
        / "runs"
        / "_failed_attempts"
        / "candidate-attempt-001"
        / "metrics.jsonl"
    )
    assert preserved.read_text(encoding="utf-8") == '{"step": 1}\n'
