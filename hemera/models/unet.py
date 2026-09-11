from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from ..interfaces import HemeraModule, TextCondition
from .components import MultiHeadAttention, timestep_embedding
from .transformers import masked_mean


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(torch.nn.functional.silu(hidden))
        return self.skip(x) + hidden


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels: int, heads: int, text_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = MultiHeadAttention(channels, heads, context_dim=text_dim)

    def forward(self, x: Tensor, text: Tensor, mask: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attention(self.norm(tokens), text, mask)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class UNetDenoiser(HemeraModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        base = config.hidden_size
        condition_dim = base * 4
        self.config = config
        self.input = nn.Conv2d(config.in_channels, base, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(base, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.text = (
            nn.Linear(config.text_dim, condition_dim)
            if config.conditioning == "pooled"
            else None
        )
        self.down1 = ResidualBlock(base, base, condition_dim)
        self.downsample1 = nn.Conv2d(base, 2 * base, 3, stride=2, padding=1)
        self.down2 = ResidualBlock(2 * base, 2 * base, condition_dim)
        self.downsample2 = nn.Conv2d(2 * base, 4 * base, 3, stride=2, padding=1)
        self.middle1 = ResidualBlock(4 * base, 4 * base, condition_dim)
        middle_heads = min(config.heads, 4 * base)
        while (4 * base) % middle_heads:
            middle_heads -= 1
        self.cross = SpatialCrossAttention(4 * base, middle_heads, config.text_dim)
        self.middle2 = ResidualBlock(4 * base, 4 * base, condition_dim)
        self.upsample2 = nn.ConvTranspose2d(4 * base, 2 * base, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(4 * base, 2 * base, condition_dim)
        self.upsample1 = nn.ConvTranspose2d(2 * base, base, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(2 * base, base, condition_dim)
        self.output_norm = nn.GroupNorm(_groups(base), base)
        self.output = nn.Conv2d(base, config.in_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_latents: Tensor, time: Tensor, condition: TextCondition) -> Tensor:
        global_condition = self.time(
            timestep_embedding(time, self.config.hidden_size).to(noisy_latents.dtype)
        )
        if self.text is not None:
            pooled = condition.pooled
            if pooled is None:
                pooled = masked_mean(
                    condition.hidden_states, condition.attention_mask
                )
            global_condition = global_condition + self.text(pooled)
        first = self.down1(self.input(noisy_latents), global_condition)
        second = self.down2(self.downsample1(first), global_condition)
        middle = self.middle1(self.downsample2(second), global_condition)
        middle = self.cross(middle, condition.hidden_states, condition.attention_mask.bool())
        middle = self.middle2(middle, global_condition)
        up_second = self.upsample2(middle)
        up_second = self.up2(torch.cat([up_second, second], dim=1), global_condition)
        up_first = self.upsample1(up_second)
        up_first = self.up1(torch.cat([up_first, first], dim=1), global_condition)
        return self.output(torch.nn.functional.silu(self.output_norm(up_first)))
