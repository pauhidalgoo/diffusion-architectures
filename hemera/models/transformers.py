from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from ..interfaces import HemeraModule, TextCondition
from .components import (
    ExpertChoiceMoE,
    MixFFN,
    MultiHeadAttention,
    PatchEmbed,
    PatchOutput,
    build_ffn,
    sincos_2d_position,
    timestep_embedding,
)


def masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(hidden.dtype)
    return (hidden * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1)


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        text_dim: int,
        cross_attention: bool,
        attention: str,
        ffn: str,
        num_experts: int,
        dropout: float,
        window_size: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attention = MultiHeadAttention(
            dim, heads, dropout=dropout, attention=attention, window_size=window_size
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = build_ffn(ffn, dim, mlp_ratio, num_experts)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attention = (
            MultiHeadAttention(dim, heads, context_dim=text_dim, dropout=dropout)
            if cross_attention
            else None
        )
        self.cross_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: Tensor,
        global_condition: Tensor,
        text: Tensor | None = None,
        text_mask: Tensor | None = None,
        grid: tuple[int, int] | None = None,
        self_mask: Tensor | None = None,
    ) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(
            global_condition
        ).chunk(6, dim=-1)
        normalized = self.norm1(x) * (1 + scale_attn[:, None]) + shift_attn[:, None]
        x = x + gate_attn[:, None] * self.self_attention(
            normalized, context_mask=self_mask
        )
        if self.cross_attention is not None and text is not None:
            x = x + torch.tanh(self.cross_gate) * self.cross_attention(
                self.cross_norm(x), context=text, context_mask=text_mask
            )
        normalized = self.norm2(x) * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        if isinstance(self.ffn, MixFFN):
            update = self.ffn(normalized, grid)
        else:
            update = self.ffn(normalized)
        return x + gate_ffn[:, None] * update


class TransformerDenoiser(HemeraModule):
    """Shared implementation for standard DiT, PixArt, U-ViT and Sana variants."""

    def __init__(self, config: ModelConfig, variant: str):
        super().__init__()
        self.config = config
        self.variant = variant
        self.current_mask_ratio = config.mask_ratio
        dim = config.hidden_size
        self.patch_embed = PatchEmbed(config.in_channels, dim, config.patch_size)
        self.patch_output = PatchOutput(dim, config.in_channels, config.patch_size)
        self.time_mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.pooled_projection = nn.Linear(config.text_dim, dim)
        self.text_projection = nn.Linear(config.text_dim, dim)
        self.patch_mixer = (
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim) if config.efficiency == "microdit" else None
        )

        cross = config.conditioning == "cross"
        self.joint_tokens = config.conditioning == "joint"
        attention = "linear" if variant == "sana" else config.attention
        ffn = "mix" if variant == "sana" else config.ffn
        block_text_dim = config.text_dim if cross else dim
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    dim,
                    config.heads,
                    config.mlp_ratio,
                    block_text_dim,
                    cross,
                    attention,
                    (
                        "moe"
                        if config.efficiency == "moe" and block_index % 2 == 1
                        else ("swiglu" if config.efficiency == "moe" else ffn)
                    ),
                    config.num_experts,
                    config.dropout,
                    config.window_size,
                )
                for block_index in range(config.depth)
            ]
        )
        self.restore_block = (
            AdaLNBlock(
                dim,
                config.heads,
                config.mlp_ratio,
                block_text_dim,
                cross,
                attention,
                ffn,
                config.num_experts,
                config.dropout,
                config.window_size,
            )
            if config.efficiency == "microdit"
            else None
        )
        self.skip_projections = nn.ModuleList(
            [nn.Linear(2 * dim, dim) for _ in range(config.depth // 2)] if variant == "uvit" else []
        )
        self.repa_head = nn.Linear(dim, config.repa_dim) if config.efficiency == "repa" else None

    def _global_condition(self, time: Tensor, condition: TextCondition) -> Tensor:
        # Sampling commonly keeps timesteps in FP32 while the denoiser weights
        # are stored as BF16. Match the embedding to the module rather than the
        # timestep or a standalone BF16 pipeline will fail outside autocast.
        embedding_dtype = self.time_mlp[0].weight.dtype
        time_state = self.time_mlp(
            timestep_embedding(time, self.config.hidden_size).to(embedding_dtype)
        )
        # Keep the conditioning ablation identifiable: pooled text replaces the
        # class vector only in the pooled-AdaLN arm. Cross-attention and joint
        # attention receive text exclusively through their token pathways.
        if self.config.conditioning != "pooled":
            return time_state
        pooled = condition.pooled
        if pooled is None:
            pooled = masked_mean(condition.hidden_states, condition.attention_mask)
        return time_state + self.pooled_projection(pooled.to(time_state.dtype))

    def _apply_patch_mixer(self, tokens: Tensor, grid: tuple[int, int]) -> Tensor:
        if self.patch_mixer is None:
            return tokens
        batch, _, dim = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(batch, dim, *grid)
        return tokens + self.patch_mixer(spatial).flatten(2).transpose(1, 2)

    @staticmethod
    def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
        return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))

    def forward(self, noisy_latents: Tensor, time: Tensor, condition: TextCondition) -> Tensor:
        image_tokens, grid = self.patch_embed(noisy_latents)
        image_tokens = image_tokens + sincos_2d_position(
            *grid, image_tokens.shape[-1], image_tokens.device, image_tokens.dtype
        )
        image_tokens = self._apply_patch_mixer(image_tokens, grid)
        global_condition = self._global_condition(time, condition)
        text_tokens = condition.hidden_states
        text_mask = condition.attention_mask.bool()

        if self.joint_tokens:
            projected_text = self.text_projection(text_tokens)
            tokens = torch.cat([projected_text, image_tokens], dim=1)
            combined_mask = torch.cat(
                [text_mask, torch.ones(image_tokens.shape[:2], device=tokens.device, dtype=torch.bool)], dim=1
            )
            text_length = projected_text.shape[1]
        else:
            tokens = image_tokens
            combined_mask = None
            text_length = 0
        token_mask = combined_mask

        full_tokens: Tensor | None = None
        kept_indices: Tensor | None = None
        if (
            self.training
            and self.config.efficiency == "microdit"
            and self.current_mask_ratio > 0
        ):
            full_tokens = tokens
            image_length = tokens.shape[1] - text_length
            keep = max(1, int(image_length * (1 - self.current_mask_ratio)))
            scores = torch.rand(
                tokens.shape[0], image_length, device=tokens.device
            )
            image_indices = scores.topk(keep, dim=1, sorted=True).indices + text_length
            if text_length:
                text_indices = torch.arange(
                    text_length, device=tokens.device
                )[None].expand(tokens.shape[0], -1)
                kept_indices = torch.cat([text_indices, image_indices], dim=1)
            else:
                kept_indices = image_indices
            tokens = self._gather(tokens, kept_indices)
            if token_mask is not None:
                token_mask = token_mask.gather(1, kept_indices)

        tread_full: Tensor | None = None
        tread_indices: Tensor | None = None
        skips: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            if self.variant == "uvit" and index < len(self.blocks) // 2:
                skips.append(tokens)
            elif self.variant == "uvit" and skips:
                skip = skips.pop()
                tokens = self.skip_projections[index - len(self.blocks) // 2](torch.cat([tokens, skip], dim=-1))

            if (
                self.training
                and self.config.efficiency == "tread"
                and index == len(self.blocks) // 3
            ):
                tread_full = tokens
                image_length = tokens.shape[1] - text_length
                keep = max(1, image_length // 2)
                image_indices = torch.rand(
                    tokens.shape[0], image_length, device=tokens.device
                ).topk(keep, dim=1, sorted=True).indices + text_length
                if text_length:
                    text_indices = torch.arange(
                        text_length, device=tokens.device
                    )[None].expand(tokens.shape[0], -1)
                    tread_indices = torch.cat([text_indices, image_indices], dim=1)
                else:
                    tread_indices = image_indices
                tokens = self._gather(tokens, tread_indices)
                if token_mask is not None:
                    token_mask = token_mask.gather(1, tread_indices)
            if (
                self.training
                and self.config.efficiency == "tread"
                and index == (2 * len(self.blocks)) // 3
                and tread_full is not None
                and tread_indices is not None
            ):
                restored = tread_full.clone()
                restored.scatter_(
                    1, tread_indices[..., None].expand_as(tokens), tokens
                )
                tokens = restored
                token_mask = combined_mask

            block_text = text_tokens if self.config.conditioning == "cross" else None
            block_mask = text_mask if block_text is not None else None
            tokens = block(
                tokens,
                global_condition,
                block_text,
                block_mask,
                grid,
                self_mask=token_mask,
            )
            if self.repa_head is not None and index == max(0, len(self.blocks) // 3 - 1):
                image_for_repa = tokens[:, text_length:] if self.joint_tokens else tokens
                self.last_repa_prediction = self.repa_head(image_for_repa)

        if full_tokens is not None and kept_indices is not None:
            restored = torch.zeros_like(full_tokens)
            restored.scatter_(1, kept_indices[..., None].expand_as(tokens), tokens)
            tokens = self.restore_block(
                restored,
                global_condition,
                text_tokens if self.config.conditioning == "cross" else None,
                text_mask,
                grid,
                self_mask=combined_mask,
            )

        if self.joint_tokens:
            tokens = tokens[:, text_length:]
        return self.patch_output(tokens, global_condition, grid)

    def moe_balance_loss(self) -> Tensor | None:
        losses = []
        for module in self.modules():
            if isinstance(module, ExpertChoiceMoE) and module.last_balance_loss is not None:
                losses.append(module.last_balance_loss)
        return torch.stack(losses).mean() if losses else None

    def set_patch_mask_ratio(self, ratio: float) -> None:
        if not 0.0 <= ratio < 1.0:
            raise ValueError("Patch mask ratio must be in [0, 1)")
        self.current_mask_ratio = ratio


class MMDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float, num_experts: int):
        super().__init__()
        self.image_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.image_qkv = nn.Linear(dim, 3 * dim)
        self.text_qkv = nn.Linear(dim, 3 * dim)
        self.image_out = nn.Linear(dim, dim)
        self.text_out = nn.Linear(dim, dim)
        self.image_ffn = build_ffn("swiglu", dim, ratio, num_experts)
        self.text_ffn = build_ffn("swiglu", dim, ratio, num_experts)
        self.image_ffn_norm = nn.LayerNorm(dim)
        self.text_ffn_norm = nn.LayerNorm(dim)
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, image: Tensor, text: Tensor, text_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, image_length, dim = image.shape
        text_length = text.shape[1]
        iq, ik, iv = self.image_qkv(self.image_norm(image)).chunk(3, dim=-1)
        tq, tk, tv = self.text_qkv(self.text_norm(text)).chunk(3, dim=-1)

        def heads(value: Tensor) -> Tensor:
            return value.view(batch, value.shape[1], self.heads, self.head_dim).transpose(1, 2)

        q = torch.cat([heads(tq), heads(iq)], dim=2)
        k = torch.cat([heads(tk), heads(ik)], dim=2)
        v = torch.cat([heads(tv), heads(iv)], dim=2)
        valid = torch.cat(
            [text_mask, torch.ones(batch, image_length, device=image.device, dtype=torch.bool)], dim=1
        )
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        attention = attention.transpose(1, 2).reshape(batch, text_length + image_length, dim)
        text_update, image_update = attention.split([text_length, image_length], dim=1)
        text = text + self.text_out(text_update)
        image = image + self.image_out(image_update)
        text = text + self.text_ffn(self.text_ffn_norm(text))
        image = image + self.image_ffn(self.image_ffn_norm(image))
        return image, text


class MMDiTDenoiser(HemeraModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.patch_embed = PatchEmbed(config.in_channels, dim, config.patch_size)
        self.patch_output = PatchOutput(dim, config.in_channels, config.patch_size)
        self.text_projection = nn.Linear(config.text_dim, dim)
        self.pooled_projection = nn.Linear(config.text_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.blocks = nn.ModuleList(
            [MMDiTBlock(dim, config.heads, config.mlp_ratio, config.num_experts) for _ in range(config.depth)]
        )

    def forward(self, noisy_latents: Tensor, time: Tensor, condition: TextCondition) -> Tensor:
        image, grid = self.patch_embed(noisy_latents)
        image = image + sincos_2d_position(*grid, image.shape[-1], image.device, image.dtype)
        text = self.text_projection(condition.hidden_states)
        pooled = condition.pooled
        if pooled is None:
            pooled = masked_mean(condition.hidden_states, condition.attention_mask)
        global_condition = self.time_mlp(
            timestep_embedding(time, self.config.hidden_size).to(image.dtype)
        ) + self.pooled_projection(pooled)
        image = image + global_condition[:, None]
        for block in self.blocks:
            image, text = block(image, text, condition.attention_mask.bool())
        return self.patch_output(image, global_condition, grid)
