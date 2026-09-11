from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from hemera.checkpoint import load_checkpoint
from hemera.config import load_config
from hemera.data import SyntheticLatentDataset, collate_examples
from hemera.models import build_model
from hemera.objectives import build_objective
from hemera.pipeline import HemeraPipeline
from hemera.runtime import set_seed
from hemera.trainer import (
    _learning_rate,
    _microdit_mask_ratio,
    train_from_config,
)


def _state(path: Path, config):
    model = build_model(config.model)
    load_checkpoint(path, model, restore_rng=False)
    return model.state_dict()


def test_interrupted_resume_matches_uninterrupted(tmp_path: Path) -> None:
    uninterrupted = load_config("configs/smoke.yaml")
    uninterrupted.train.steps = 4
    uninterrupted.train.save_every = 2
    uninterrupted.train.output_dir = str(tmp_path / "uninterrupted")
    full_checkpoint = train_from_config(uninterrupted)

    first_leg = load_config("configs/smoke.yaml")
    first_leg.train.steps = 2
    first_leg.train.save_every = 2
    first_leg.train.output_dir = str(tmp_path / "resumed")
    first_checkpoint = train_from_config(first_leg)

    second_leg = load_config("configs/smoke.yaml")
    second_leg.train.steps = 4
    second_leg.train.save_every = 2
    second_leg.train.output_dir = str(tmp_path / "resumed")
    second_leg.train.resume = str(first_checkpoint)
    resumed_checkpoint = train_from_config(second_leg)

    full = _state(full_checkpoint, uninterrupted)
    resumed = _state(resumed_checkpoint, second_leg)
    assert full.keys() == resumed.keys()
    for name in full:
        torch.testing.assert_close(full[name], resumed[name], rtol=0, atol=0)


def test_smoke_loss_decreases(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.train.steps = 40
    config.train.output_dir = str(tmp_path / "overfit")
    set_seed(config.train.seed)
    initial_model = build_model(config.model).eval()
    dataset = SyntheticLatentDataset(
        config.data.synthetic_items,
        config.data.latent_channels,
        config.data.latent_size,
        config.data.text_length,
        config.data.text_dim,
        config.train.seed,
    )
    batch = collate_examples([dataset[index] for index in range(config.train.batch_size)], sample_posterior=False)
    objective = build_objective(config.objective)
    target = objective.make_training_batch(
        batch["latents"], torch.Generator().manual_seed(123)
    )
    with torch.no_grad():
        initial_loss = objective.loss(
            initial_model(target.noisy, target.time, batch["condition"]), target
        )

    checkpoint = train_from_config(config)
    trained_model = build_model(config.model).eval()
    load_checkpoint(checkpoint, trained_model, restore_rng=False)
    with torch.no_grad():
        trained_loss = objective.loss(
            trained_model(target.noisy, target.time, batch["condition"]), target
        )
    assert trained_loss < initial_loss


def test_safetensors_pipeline_export_reloads_reproducibly(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.train.steps = 1
    config.train.output_dir = str(tmp_path / "run")
    checkpoint = train_from_config(config)
    pipeline = HemeraPipeline.from_pretrained(
        checkpoint, device="cpu", dtype="float32", load_vae=False
    )
    exported = pipeline.save_pretrained(tmp_path / "export")
    assert exported.name == "model.safetensors"
    reloaded = HemeraPipeline.from_pretrained(
        exported.parent, device="cpu", dtype="float32", load_vae=False
    )
    first = pipeline(
        ["a gold circle"],
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(99),
        output_type="latent",
    )
    second = reloaded(
        ["a gold circle"],
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(99),
        output_type="latent",
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_pipeline_resolves_hugging_face_repo_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    downloaded = tmp_path / "hub-snapshot"
    downloaded.mkdir()
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    resolved = HemeraPipeline._resolve_pretrained_root(
        "pauhidalgoo/hemera-nano",
        revision="release-v1",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        token=False,
    )
    assert resolved == downloaded
    assert captured["repo_id"] == "pauhidalgoo/hemera-nano"
    assert captured["revision"] == "release-v1"
    assert captured["local_files_only"] is True
    assert captured["allow_patterns"] == [
        "model.safetensors",
        "config.json",
        "model_index.json",
    ]


def test_pipeline_seed_is_deterministic_and_exclusive(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.train.steps = 1
    config.train.output_dir = str(tmp_path / "seed-run")
    pipeline = HemeraPipeline.from_pretrained(
        train_from_config(config), device="cpu", dtype="float32", load_vae=False
    )
    first = pipeline(
        "a deterministic fixture",
        seed=123,
        num_inference_steps=2,
        guidance_scale=1.0,
        output_type="latent",
    )
    second = pipeline(
        "a deterministic fixture",
        seed=123,
        num_inference_steps=2,
        guidance_scale=1.0,
        output_type="latent",
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="either generator or seed"):
        pipeline(
            "invalid",
            seed=1,
            generator=torch.Generator(),
            output_type="latent",
        )


def test_budget_progress_controls_lr_and_microdit_finish() -> None:
    config = load_config("configs/smoke.yaml")
    config.train.steps = 1_000_000
    config.train.warmup_ratio = 0.1
    config.train.learning_rate = 1e-3
    config.train.min_learning_rate_ratio = 0.1
    assert _learning_rate(config, 5, budget_progress=0.05) == pytest.approx(
        5e-4
    )
    assert _learning_rate(config, 5, budget_progress=0.1) == pytest.approx(1e-3)
    assert _learning_rate(config, 5, budget_progress=1.0) == pytest.approx(
        1e-4
    )
    config.model.efficiency = "microdit"
    config.model.mask_ratio = 0.5
    config.model.mask_finish_ratio = 0.1
    assert _microdit_mask_ratio(config, 1, 0.89) == 0.5
    assert _microdit_mask_ratio(config, 1, 0.9) == 0.0
