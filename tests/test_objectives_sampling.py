from __future__ import annotations

import pytest
import torch
from torch import nn

from hemera.config import ObjectiveConfig
from hemera.interfaces import TextCondition
from hemera.objectives import build_objective
from hemera.sampling import sample_latents


@pytest.mark.parametrize("name", ["epsilon", "v", "flow"])
def test_objectives_create_finite_targets(name: str) -> None:
    clean = torch.randn(4, 4, 8, 8)
    objective = build_objective(ObjectiveConfig(name=name))
    target = objective.make_training_batch(clean, torch.Generator().manual_seed(4))
    assert target.noisy.shape == clean.shape
    assert target.target.shape == clean.shape
    assert torch.isfinite(objective.loss(torch.zeros_like(clean), target))


@pytest.mark.parametrize("name", ["epsilon", "v", "flow"])
def test_objectives_preserve_latent_dtype(name: str) -> None:
    clean = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
    objective = build_objective(ObjectiveConfig(name=name))
    target = objective.make_training_batch(clean, torch.Generator().manual_seed(4))
    assert target.noisy.dtype == clean.dtype
    assert target.target.dtype == clean.dtype
    assert target.time.dtype == torch.float32


def test_epsilon_sampling_avoids_singular_cosine_endpoint() -> None:
    epsilon = build_objective(ObjectiveConfig(name="epsilon", train_timesteps=1000))
    schedule = epsilon.sampling_times(10, torch.device("cpu"))
    assert float(schedule[0]) == pytest.approx(0.999)
    assert schedule[-1] == 0
    assert torch.all(schedule[:-1] > schedule[1:])


@pytest.mark.parametrize("name", ["v", "flow"])
def test_non_epsilon_sampling_starts_at_noise_endpoint(name: str) -> None:
    objective = build_objective(ObjectiveConfig(name=name))
    assert objective.sampling_times(10, torch.device("cpu"))[0] == 1


class ZeroDenoiser(nn.Module):
    def forward(self, x, time, condition):
        return torch.zeros_like(x)


class ConditioningDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x, time, condition):
        self.batch_sizes.append(x.shape[0])
        values = condition.pooled[:, :1].view(-1, 1, 1, 1)
        return values.expand_as(x)


def test_sampling_is_deterministic() -> None:
    condition = TextCondition(
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        torch.randn(2, 8),
    )
    objective = build_objective(ObjectiveConfig(name="flow"))
    first = sample_latents(
        ZeroDenoiser(),
        objective,
        (2, 4, 4, 4),
        condition,
        steps=4,
        guidance_scale=1,
        generator=torch.Generator().manual_seed(12),
    )
    second = sample_latents(
        ZeroDenoiser(),
        objective,
        (2, 4, 4, 4),
        condition,
        steps=4,
        guidance_scale=1,
        generator=torch.Generator().manual_seed(12),
    )
    torch.testing.assert_close(first, second)


def test_sampling_accepts_one_generator_per_sample() -> None:
    condition = TextCondition(
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        torch.randn(2, 8),
    )
    objective = build_objective(ObjectiveConfig(name="flow"))
    seeds = [17, 29]
    batched = sample_latents(
        ZeroDenoiser(),
        objective,
        (2, 4, 4, 4),
        condition,
        steps=2,
        guidance_scale=1,
        generator=[
            torch.Generator().manual_seed(seed) for seed in seeds
        ],
    )
    expected = torch.cat(
        [
            torch.randn(
                (1, 4, 4, 4),
                generator=torch.Generator().manual_seed(seed),
            )
            for seed in seeds
        ]
    )
    torch.testing.assert_close(batched, expected)
    with pytest.raises(ValueError, match="one generator per sample"):
        sample_latents(
            ZeroDenoiser(),
            objective,
            (2, 4, 4, 4),
            condition,
            steps=2,
            guidance_scale=1,
            generator=[torch.Generator().manual_seed(17)],
        )


def test_cfg_uses_one_batched_forward_per_step() -> None:
    condition = TextCondition(
        torch.ones(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        torch.ones(2, 8),
    )
    model = ConditioningDenoiser()
    result = sample_latents(
        model,
        build_objective(ObjectiveConfig(name="flow")),
        (2, 4, 4, 4),
        condition,
        steps=3,
        guidance_scale=2,
        generator=torch.Generator().manual_seed(12),
    )
    assert result.shape == (2, 4, 4, 4)
    assert model.batch_sizes == [4, 4, 4]
    initial = torch.randn(
        (2, 4, 4, 4), generator=torch.Generator().manual_seed(12)
    )
    torch.testing.assert_close(result, initial - 2)
