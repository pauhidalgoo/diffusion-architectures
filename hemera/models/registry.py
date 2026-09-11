from __future__ import annotations

from collections.abc import Callable

from torch import nn

from ..config import ModelConfig
from .components import ExpertChoiceMoE
from .transformers import MMDiTDenoiser, TransformerDenoiser
from .unet import UNetDenoiser

BACKBONES: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "unet": UNetDenoiser,
    "dit": lambda cfg: TransformerDenoiser(cfg, "dit"),
    "pixart": lambda cfg: TransformerDenoiser(cfg, "pixart"),
    "uvit": lambda cfg: TransformerDenoiser(cfg, "uvit"),
    "mmdit": MMDiTDenoiser,
    "sana": lambda cfg: TransformerDenoiser(cfg, "sana"),
}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def count_active_parameters(model: nn.Module) -> int:
    """Count shared parameters plus one expert per expert-choice MoE layer.

    Expert-choice routing executes every expert across a batch, but its
    conventional per-token active-parameter count includes one expert in each
    sparse layer. Routers and all non-expert weights remain active.
    """
    total = count_parameters(model)
    inactive = 0
    for module in model.modules():
        if not isinstance(module, ExpertChoiceMoE):
            continue
        expert_counts = [count_parameters(expert) for expert in module.experts]
        if expert_counts:
            inactive += sum(expert_counts) - max(expert_counts)
    return total - inactive


def build_model(config: ModelConfig, enforce_limit: bool = True) -> nn.Module:
    if (
        config.efficiency != "dense"
        and config.backbone in {"unet", "mmdit"}
    ):
        raise ValueError(
            f"Efficiency intervention {config.efficiency!r} is not implemented "
            f"for backbone {config.backbone!r}; do not run a silent dense control"
        )
    try:
        model = BACKBONES[config.backbone](config)
    except KeyError as error:
        raise KeyError(f"Unknown backbone {config.backbone!r}; choose from {sorted(BACKBONES)}") from error
    parameters = count_parameters(model)
    if enforce_limit and parameters > config.max_parameters:
        raise ValueError(
            f"{config.backbone} has {parameters:,} trainable parameters, exceeding "
            f"the configured {config.max_parameters:,} limit"
        )
    return model
