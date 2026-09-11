from __future__ import annotations

import pytest
import torch
import yaml

from hemera.config import ModelConfig, config_from_dict
from hemera.interfaces import TextCondition
from hemera.models import (
    BACKBONES,
    build_model,
    count_active_parameters,
    count_parameters,
)
from hemera.sweep import deep_update
from hemera.models.components import ExpertChoiceMoE


def tiny_config(backbone: str, efficiency: str = "dense") -> ModelConfig:
    return ModelConfig(
        backbone=backbone,
        in_channels=4,
        hidden_size=32,
        depth=4,
        heads=4,
        patch_size=2,
        mlp_ratio=2.0,
        text_dim=32,
        text_length=6,
        conditioning={"dit": "pooled", "pixart": "cross", "uvit": "joint", "mmdit": "joint", "sana": "cross", "unet": "cross"}[backbone],
        ffn="mix" if backbone == "sana" else "swiglu",
        attention="linear" if backbone == "sana" else "full",
        efficiency=efficiency,
        max_parameters=2_000_000,
    )


def fixture() -> tuple[torch.Tensor, torch.Tensor, TextCondition]:
    latent = torch.randn(2, 4, 8, 8)
    time = torch.rand(2)
    hidden = torch.randn(2, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    return latent, time, TextCondition(hidden, mask, pooled)


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
def test_all_backbones_forward_backward(backbone: str) -> None:
    model = build_model(tiny_config(backbone))
    latent, time, condition = fixture()
    output = model(latent, time, condition)
    assert output.shape == latent.shape
    output.square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert count_parameters(model) < 2_000_000


@pytest.mark.parametrize("backbone", ["dit", "pixart"])
def test_transformer_sampling_supports_bfloat16_weights(backbone: str) -> None:
    model = build_model(tiny_config(backbone)).to(torch.bfloat16).eval()
    latent, time, condition = fixture()
    condition = condition.to(torch.device("cpu"), torch.bfloat16)
    with torch.no_grad():
        output = model(latent.to(torch.bfloat16), time, condition)
    assert output.dtype == torch.bfloat16
    assert output.shape == latent.shape


@pytest.mark.parametrize("efficiency", ["microdit", "tread", "moe"])
def test_efficiency_variants_preserve_shape(efficiency: str) -> None:
    config = tiny_config("pixart", efficiency)
    if efficiency == "moe":
        config.ffn = "moe"
    model = build_model(config)
    model.train()
    latent, time, condition = fixture()
    output = model(latent, time, condition)
    assert output.shape == latent.shape


def test_microdit_can_enter_unmasked_finishing_phase() -> None:
    model = build_model(tiny_config("pixart", "microdit"))
    model.set_patch_mask_ratio(0.0)
    assert model.current_mask_ratio == 0.0
    latent, time, condition = fixture()
    assert model(latent, time, condition).shape == latent.shape


def test_tread_evaluation_disables_random_training_routing() -> None:
    model = build_model(tiny_config("pixart", "tread")).eval()
    latent, time, condition = fixture()
    with torch.no_grad():
        torch.manual_seed(1)
        first = model(latent, time, condition)
        torch.manual_seed(999)
        second = model(latent, time, condition)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_parameter_limit_is_enforced() -> None:
    config = tiny_config("dit")
    config.max_parameters = 10
    with pytest.raises(ValueError, match="exceeding"):
        build_model(config)


def test_unsupported_efficiency_backbone_cannot_silently_run_dense() -> None:
    config = tiny_config("mmdit")
    config.efficiency = "tread"
    with pytest.raises(ValueError, match="not implemented"):
        build_model(config)


def test_backbone_sweep_models_respect_final_parameter_cap() -> None:
    base = yaml.safe_load(open("configs/experiment-base.yaml", encoding="utf-8"))
    sweep = yaml.safe_load(open("configs/sweeps/backbones.yaml", encoding="utf-8"))
    counts = []
    for run in sweep["runs"]:
        raw = __import__("copy").deepcopy(base)
        deep_update(raw, run["overrides"])
        raw["name"] = run["name"]
        config = config_from_dict(raw)
        model = build_model(config.model)
        counts.append(count_parameters(model))
    assert max(counts) <= 50_000_000
    assert (max(counts) - min(counts)) / max(counts) < 0.05


def test_conditioning_sweep_is_parameter_matched_and_structurally_distinct() -> None:
    base = yaml.safe_load(open("configs/experiment-base.yaml", encoding="utf-8"))
    sweep = yaml.safe_load(open("configs/sweeps/conditioning.yaml", encoding="utf-8"))
    models = {}
    counts = []
    for run in sweep["runs"]:
        raw = __import__("copy").deepcopy(base)
        deep_update(raw, run["overrides"])
        raw["name"] = run["name"]
        model = build_model(config_from_dict(raw).model)
        models[run["name"]] = model
        counts.append(count_parameters(model))
    assert (max(counts) - min(counts)) / max(counts) < 0.05
    assert not models["pooled-adaln"].joint_tokens
    assert all(block.cross_attention is None for block in models["pooled-adaln"].blocks)
    assert all(
        block.cross_attention is not None
        for block in models["token-cross-attention"].blocks
    )
    assert models["joint-attention"].joint_tokens


def test_conditioning_ablation_uses_pooled_text_only_for_pooled_adaln() -> None:
    _, time, condition = fixture()
    changed = TextCondition(
        condition.hidden_states,
        condition.attention_mask,
        condition.pooled + 10,
    )
    pooled_model = build_model(tiny_config("dit"))
    cross_model = build_model(tiny_config("pixart"))
    assert not torch.equal(
        pooled_model._global_condition(time, condition),
        pooled_model._global_condition(time, changed),
    )
    assert torch.equal(
        cross_model._global_condition(time, condition),
        cross_model._global_condition(time, changed),
    )


def test_efficiency_sweep_respects_parameter_ceiling_and_matching() -> None:
    base = yaml.safe_load(open("configs/experiment-base.yaml", encoding="utf-8"))
    sweep = yaml.safe_load(open("configs/sweeps/efficiency.yaml", encoding="utf-8"))
    counts = []
    for run in sweep["runs"]:
        raw = __import__("copy").deepcopy(base)
        deep_update(raw, run["overrides"])
        raw["name"] = run["name"]
        counts.append(count_parameters(build_model(config_from_dict(raw).model)))
    assert max(counts) <= 50_000_000
    assert (max(counts) - min(counts)) / max(counts) < 0.05


def test_decision_sprint_sizes_and_matched_arms() -> None:
    base = yaml.safe_load(open("configs/experiment-base.yaml", encoding="utf-8"))
    sweep = yaml.safe_load(
        open("configs/sweeps/decision-sprint.yaml", encoding="utf-8")
    )
    counts: dict[str, int] = {}
    for run in sweep["runs"]:
        raw = __import__("copy").deepcopy(base)
        deep_update(raw, run["overrides"])
        raw["name"] = run["name"]
        counts[run["name"]] = count_parameters(
            build_model(config_from_dict(raw).model)
        )

    assert counts["size-15m"] == 15_137_052
    assert counts["dense-30m-control"] == 29_756_564
    assert counts["size-48m"] == 47_585_012
    matched = [
        counts["dense-30m-control"],
        counts["microdit-30m-matched"],
        counts["moe-30m-matched"],
        counts["token-cross-attention-30m"],
    ]
    assert (max(matched) - min(matched)) / max(matched) < 0.05


def test_final_recipe_matches_frozen_dense_control() -> None:
    final = yaml.safe_load(
        open("configs/final-hemera-nano.yaml", encoding="utf-8")
    )
    config = config_from_dict(final)
    assert config.model.backbone == "dit"
    assert config.model.conditioning == "pooled"
    assert config.model.efficiency == "dense"
    assert not config.representation.precompute_repa
    assert config.train.batch_size == 384
    assert config.train.effective_batch_size == 768
    assert count_parameters(build_model(config.model)) == 29_756_564


def test_active_parameter_count_distinguishes_sparse_moe() -> None:
    dense = build_model(tiny_config("pixart", "dense"))
    moe_config = tiny_config("pixart", "moe")
    moe_config.ffn = "moe"
    moe = build_model(moe_config)
    assert count_active_parameters(dense) == count_parameters(dense)
    assert count_active_parameters(moe) < count_parameters(moe)


def test_expert_choice_routing_is_independent_across_batch_items() -> None:
    module = ExpertChoiceMoE(dim=16, ratio=2.0, num_experts=4).eval()
    inputs = torch.randn(2, 12, 16)
    with torch.no_grad():
        alone = module(inputs[:1])
        together = module(inputs)
    torch.testing.assert_close(alone, together[:1], rtol=0, atol=1e-6)
    assert module.last_balance_loss is not None
    assert torch.isfinite(module.last_balance_loss)
