from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(time: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=time.device, dtype=torch.float32) / max(half, 1)
    )
    values = time.float()[:, None] * frequencies[None] * max_period
    embedding = torch.cat([torch.cos(values), torch.sin(values)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def sincos_2d_position(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if dim % 4:
        raise ValueError("Position embedding dimension must be divisible by four")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10_000 ** (omega / max(dim // 4, 1)))
    x_values = x.reshape(-1, 1) * omega.reshape(1, -1)
    y_values = y.reshape(-1, 1) * omega.reshape(1, -1)
    result = torch.cat(
        [torch.sin(x_values), torch.cos(x_values), torch.sin(y_values), torch.cos(y_values)], dim=1
    )
    return result.unsqueeze(0).to(dtype=dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
        attention: str = "full",
        window_size: int = 8,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention dimension must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.context_dim = context_dim or dim
        self.attention = attention
        self.window_size = window_size
        self.dropout = dropout
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(self.context_dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _window_mask(self, length: int, device: torch.device) -> Tensor | None:
        side = int(math.sqrt(length))
        if side * side != length:
            return None
        coords = torch.stack(
            torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij"),
            dim=-1,
        ).reshape(length, 2)
        distance = (coords[:, None] - coords[None, :]).abs()
        radius = max(1, self.window_size // 2)
        return (distance[..., 0] <= radius) & (distance[..., 1] <= radius)

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        context = x if context is None else context
        batch, query_length, _ = x.shape
        key_length = context.shape[1]
        q = self.to_q(x).view(batch, query_length, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(batch, key_length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, key_length, self.heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.attention == "linear" and context is x:
            q_kernel = F.elu(q.float()) + 1
            k_kernel = F.elu(k.float()) + 1
            if context_mask is not None:
                valid = context_mask[:, None, :, None].to(k_kernel.dtype)
                k_kernel = k_kernel * valid
                v = v * valid.to(v.dtype)
            kv = torch.einsum("bhnd,bhne->bhde", k_kernel, v.float())
            normalizer = torch.einsum("bhnd,bhd->bhn", q_kernel, k_kernel.sum(dim=2)).clamp_min(1e-6)
            output = torch.einsum("bhnd,bhde->bhne", q_kernel, kv)
            output = output / normalizer[..., None]
            output = output.to(x.dtype)
        else:
            mask: Tensor | None = None
            if context_mask is not None:
                mask = context_mask[:, None, None, :].expand(batch, 1, query_length, key_length)
            if self.attention == "window" and context is x:
                local = self._window_mask(query_length, x.device)
                if local is not None:
                    local = local[None, None].expand(batch, 1, query_length, key_length)
                    mask = local if mask is None else (mask & local)
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        output = output.transpose(1, 2).reshape(batch, query_length, self.dim)
        return self.out(output)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio * 2 / 3)
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class GELUFFN(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MixFFN(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.up = nn.Linear(dim, hidden * 2)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: Tensor, grid: tuple[int, int] | None = None) -> Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        value = value * F.silu(gate)
        if grid is not None and grid[0] * grid[1] == x.shape[1]:
            batch, _, channels = value.shape
            spatial = value.transpose(1, 2).reshape(batch, channels, *grid)
            value = self.depthwise(spatial).flatten(2).transpose(1, 2)
        return self.down(value)


class ExpertChoiceMoE(nn.Module):
    """Each expert selects its highest-scoring tokens under a fixed capacity."""

    def __init__(self, dim: int, ratio: float = 4.0, num_experts: int = 4):
        super().__init__()
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, ratio) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.last_balance_loss: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        probabilities = self.router(x).softmax(dim=-1)
        # Route within each image independently. Flattening the batch would
        # make one prompt's denoising trajectory depend on unrelated prompts
        # sampled beside it.
        capacity = max(1, math.ceil(tokens / self.num_experts))
        output = torch.zeros_like(x)
        normalizer = torch.zeros(
            batch, tokens, 1, device=x.device, dtype=x.dtype
        )
        for expert_index, expert in enumerate(self.experts):
            scores, indices = probabilities[:, :, expert_index].topk(
                min(capacity, tokens), dim=1, sorted=False
            )
            selected = x.gather(
                1, indices[..., None].expand(-1, -1, dim)
            )
            weights = scores.to(x.dtype)[..., None]
            output.scatter_add_(
                1,
                indices[..., None].expand(-1, -1, dim),
                expert(selected) * weights,
            )
            normalizer.scatter_add_(
                1, indices[..., None], weights
            )
        routed = output / normalizer.clamp_min(torch.finfo(x.dtype).eps)
        # Unselected tokens retain an identity route; the auxiliary term penalizes
        # a collapsed router while expert choice itself fixes per-expert load.
        routed = torch.where(normalizer > 0, routed, x)
        importance = probabilities.float().mean(dim=(0, 1))
        self.last_balance_loss = (
            importance * self.num_experts - 1.0
        ).square().mean()
        return routed


def build_ffn(kind: str, dim: int, ratio: float, num_experts: int) -> nn.Module:
    if kind == "swiglu":
        return SwiGLU(dim, ratio)
    if kind == "gelu":
        return GELUFFN(dim, ratio)
    if kind == "mix":
        return MixFFN(dim, ratio)
    if kind == "moe":
        return ExpertChoiceMoE(dim, ratio, num_experts)
    raise KeyError(f"Unknown FFN: {kind}")


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x = self.projection(x)
        grid = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), grid


class PatchOutput(nn.Module):
    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, tokens: Tensor, condition: Tensor, grid: tuple[int, int]) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        tokens = self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        patches = self.linear(tokens)
        batch, _, _ = patches.shape
        height, width = grid
        patch = self.patch_size
        patches = patches.view(batch, height, width, patch, patch, self.out_channels)
        return torch.einsum("bhwpqc->bchpwq", patches).reshape(
            batch, self.out_channels, height * patch, width * patch
        )
