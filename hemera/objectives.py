from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ObjectiveConfig


@dataclass
class TrainingTarget:
    noisy: Tensor
    time: Tensor
    target: Tensor
    weight: Tensor


def _expand(value: Tensor, target: Tensor) -> Tensor:
    # Timesteps remain FP32 for schedule accuracy, but schedule coefficients
    # must follow the latent dtype. Otherwise every BF16 training batch is
    # silently promoted to FP32 before entering the denoiser.
    return value.to(dtype=target.dtype).view(value.shape[0], *((1,) * (target.ndim - 1)))


class GenerativeObjective:
    name = "base"

    def make_training_batch(self, clean: Tensor, generator: torch.Generator | None = None) -> TrainingTarget:
        raise NotImplementedError

    def loss(self, prediction: Tensor, batch: TrainingTarget) -> Tensor:
        per_example = (prediction.float() - batch.target.float()).square().flatten(1).mean(1)
        return (per_example * batch.weight.float()).mean()

    def velocity(self, model_output: Tensor, x: Tensor, time: Tensor) -> Tensor:
        raise NotImplementedError

    def sampling_times(self, steps: int, device: torch.device) -> Tensor:
        return torch.linspace(1.0, 0.0, steps + 1, device=device)


class EpsilonObjective(GenerativeObjective):
    name = "epsilon"

    def __init__(self, timesteps: int = 1000, min_snr_gamma: float = 5.0):
        self.timesteps = timesteps
        self.min_snr_gamma = min_snr_gamma

    @staticmethod
    def alpha_sigma(time: Tensor) -> tuple[Tensor, Tensor]:
        angle = time * math.pi / 2
        return torch.cos(angle), torch.sin(angle)

    def make_training_batch(self, clean: Tensor, generator: torch.Generator | None = None) -> TrainingTarget:
        batch = clean.shape[0]
        time = torch.rand(batch, device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        alpha, sigma = self.alpha_sigma(time)
        noisy = _expand(alpha, clean) * clean + _expand(sigma, clean) * noise
        snr = alpha.square() / sigma.square().clamp_min(1e-5)
        weight = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr.clamp_min(1e-5)
        return TrainingTarget(noisy, time, noise, weight)

    def velocity(self, model_output: Tensor, x: Tensor, time: Tensor) -> Tensor:
        alpha, sigma = self.alpha_sigma(time)
        predicted_clean = (x - _expand(sigma, x) * model_output) / _expand(alpha, x).clamp_min(1e-4)
        angular_velocity = _expand(alpha, x) * model_output - _expand(sigma, x) * predicted_clean
        return (math.pi / 2) * angular_velocity

    def sampling_times(self, steps: int, device: torch.device) -> Tensor:
        # At t=1 the cosine schedule has alpha=0, so x_0 cannot be recovered
        # from an epsilon prediction. Start at the last finite training-time
        # point; pure Gaussian noise is an accurate initialization there.
        start = 1.0 - 1.0 / max(self.timesteps, 2)
        return torch.linspace(start, 0.0, steps + 1, device=device)


class VObjective(EpsilonObjective):
    name = "v"

    def make_training_batch(self, clean: Tensor, generator: torch.Generator | None = None) -> TrainingTarget:
        batch = clean.shape[0]
        time = torch.rand(batch, device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        alpha, sigma = self.alpha_sigma(time)
        noisy = _expand(alpha, clean) * clean + _expand(sigma, clean) * noise
        target = _expand(alpha, clean) * noise - _expand(sigma, clean) * clean
        return TrainingTarget(noisy, time, target, torch.ones_like(time))

    def velocity(self, model_output: Tensor, x: Tensor, time: Tensor) -> Tensor:
        return (math.pi / 2) * model_output

    def sampling_times(self, steps: int, device: torch.device) -> Tensor:
        # Unlike epsilon prediction, v-prediction is well-defined at alpha=0.
        return GenerativeObjective.sampling_times(self, steps, device)


class FlowObjective(GenerativeObjective):
    name = "flow"

    def __init__(self, logit_mean: float = 0.0, logit_std: float = 1.0):
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def make_training_batch(self, clean: Tensor, generator: torch.Generator | None = None) -> TrainingTarget:
        logits = torch.randn(clean.shape[0], device=clean.device, generator=generator)
        time = torch.sigmoid(logits * self.logit_std + self.logit_mean)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        noisy = (1 - _expand(time, clean)) * clean + _expand(time, clean) * noise
        return TrainingTarget(noisy, time, noise - clean, torch.ones_like(time))

    def velocity(self, model_output: Tensor, x: Tensor, time: Tensor) -> Tensor:
        return model_output


def build_objective(config: ObjectiveConfig) -> GenerativeObjective:
    if config.name == "epsilon":
        return EpsilonObjective(config.train_timesteps, config.min_snr_gamma)
    if config.name == "v":
        return VObjective(config.train_timesteps, config.min_snr_gamma)
    if config.name == "flow":
        return FlowObjective(config.flow_logit_mean, config.flow_logit_std)
    raise KeyError(f"Unknown objective: {config.name}")
