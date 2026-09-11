from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass
class TextCondition:
    hidden_states: Tensor
    attention_mask: Tensor
    pooled: Tensor | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "TextCondition":
        hidden = self.hidden_states.to(device=device, dtype=dtype or self.hidden_states.dtype)
        mask = self.attention_mask.to(device=device)
        pooled = None if self.pooled is None else self.pooled.to(device=device, dtype=dtype or self.pooled.dtype)
        return TextCondition(hidden, mask, pooled)

    def dropped(self, drop_mask: Tensor) -> "TextCondition":
        keep = (~drop_mask).to(self.hidden_states.dtype)
        hidden = self.hidden_states * keep[:, None, None]
        mask = self.attention_mask & (~drop_mask[:, None])
        pooled = None if self.pooled is None else self.pooled * keep[:, None]
        return TextCondition(hidden, mask, pooled)

    def index_select(self, indices: Tensor) -> "TextCondition":
        """Select or reorder complete prompt records without breaking alignment."""
        pooled = (
            None
            if self.pooled is None
            else self.pooled.index_select(0, indices)
        )
        return TextCondition(
            hidden_states=self.hidden_states.index_select(0, indices),
            attention_mask=self.attention_mask.index_select(0, indices),
            pooled=pooled,
        )

    @classmethod
    def unconditional_like(cls, other: "TextCondition") -> "TextCondition":
        pooled = None if other.pooled is None else torch.zeros_like(other.pooled)
        return cls(
            hidden_states=torch.zeros_like(other.hidden_states),
            attention_mask=torch.zeros_like(other.attention_mask, dtype=torch.bool),
            pooled=pooled,
        )


@runtime_checkable
class Denoiser(Protocol):
    def __call__(self, noisy_latents: Tensor, time: Tensor, condition: TextCondition) -> Tensor: ...


class HemeraModule(nn.Module):
    """Base class for denoisers with optional auxiliary representation loss."""

    last_repa_prediction: Tensor | None = None

    def take_repa_prediction(self) -> Tensor | None:
        value = self.last_repa_prediction
        self.last_repa_prediction = None
        return value
