from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.local_playground import Settings, apply_command, status


def test_playground_commands_update_settings() -> None:
    settings = Settings()
    assert apply_command(settings, "/steps 30") == "/steps"
    apply_command(settings, "/cfg 3.5")
    apply_command(settings, "/seed 42")
    apply_command(settings, "/count 4")
    apply_command(settings, '/negative "low quality, text"')
    apply_command(settings, "/show on")

    assert settings.steps == 30
    assert settings.guidance == 3.5
    assert settings.seed == 42
    assert settings.count == 4
    assert settings.negative == "low quality, text"
    assert settings.show is True
    assert "next_seed=42" in status(settings)


@pytest.mark.parametrize(
    "command",
    (
        "/steps 0",
        "/cfg 31",
        "/seed -1",
        "/count 9",
        "/show maybe",
        "/unknown",
    ),
)
def test_playground_rejects_invalid_commands(command: str) -> None:
    with pytest.raises(ValueError):
        apply_command(Settings(), command)


def test_playground_direct_script_help_works() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/local_playground.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Persistent interactive Hemera-Nano" in result.stdout
