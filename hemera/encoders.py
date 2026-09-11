from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .config import DataConfig, RepresentationConfig
from .interfaces import TextCondition


class MockTextEncoder(nn.Module):
    """Deterministic lightweight encoder used by local tests."""

    def __init__(self, dim: int, length: int):
        super().__init__()
        self.dim = dim
        self.length = length

    @torch.no_grad()
    def encode(self, prompts: list[str], device: torch.device, dtype: torch.dtype) -> TextCondition:
        hidden = torch.zeros(len(prompts), self.length, self.dim, device=device, dtype=dtype)
        mask = torch.zeros(len(prompts), self.length, device=device, dtype=torch.bool)
        for batch_index, prompt in enumerate(prompts):
            values = prompt.encode("utf-8")[: self.length]
            for token_index, value in enumerate(values):
                positions = torch.arange(self.dim, device=device, dtype=torch.float32)
                hidden[batch_index, token_index] = torch.sin(
                    positions * 0.017 + float(value) * 0.13
                ).to(dtype)
                mask[batch_index, token_index] = True
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return TextCondition(hidden, mask, pooled)


class HFTextEncoder:
    def __init__(self, config: RepresentationConfig, device: torch.device, dtype: torch.dtype):
        try:
            from transformers import (
                AutoTokenizer,
                CLIPTextModel,
                T5EncoderModel,
            )
        except ImportError as error:
            raise RuntimeError("Cloud text encoding requires `transformers`") from error
        self.config = config
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_id)
        if config.text_encoder_type == "clip":
            self.model = CLIPTextModel.from_pretrained(
                config.text_encoder_id, dtype=dtype
            )
        elif config.text_encoder_type == "t5":
            self.model = T5EncoderModel.from_pretrained(
                config.text_encoder_id, dtype=dtype
            )
        else:
            raise ValueError(f"Unsupported cloud text encoder: {config.text_encoder_type}")
        self.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, prompts: list[str], device: torch.device, dtype: torch.dtype) -> TextCondition:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        ).to(device)
        result = self.model(**tokens)
        hidden = result.last_hidden_state.to(dtype)
        mask = tokens["attention_mask"].bool()
        pooled = getattr(result, "pooler_output", None)
        if pooled is None:
            pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return TextCondition(hidden, mask, pooled.to(dtype))


class VAEAdapter:
    def __init__(self, config: RepresentationConfig, device: torch.device, dtype: torch.dtype):
        try:
            from diffusers import AutoencoderDC, AutoencoderKL
        except ImportError as error:
            raise RuntimeError("Cloud VAE preprocessing requires `diffusers`") from error
        self.config = config
        self.device = device
        self.dtype = dtype
        cls = AutoencoderKL if config.vae_type == "kl" else AutoencoderDC
        self.model = cls.from_pretrained(config.vae_id, torch_dtype=dtype)
        self.model.to(device).eval().requires_grad_(False)
        model_config = self.model.config
        self.scaling_factor = float(getattr(model_config, "scaling_factor", 1.0))
        self.shift_factor = float(getattr(model_config, "shift_factor", 0.0) or 0.0)

    @torch.no_grad()
    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.model.encode(images.to(self.device, self.dtype))
        distribution = getattr(encoded, "latent_dist", None)
        if distribution is not None:
            mean = distribution.mean
            logvar = distribution.logvar
        else:
            latent = getattr(encoded, "latent", None)
            if latent is None:
                latent = encoded[0]
            mean = latent
            logvar = torch.full_like(mean, -30.0)
        mean = (mean - self.shift_factor) * self.scaling_factor
        logvar = logvar + 2.0 * torch.log(
            torch.tensor(self.scaling_factor, device=logvar.device, dtype=logvar.dtype)
        )
        return mean, logvar

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        latents = latents / self.scaling_factor + self.shift_factor
        decoded = self.model.decode(latents.to(self.device, self.dtype))
        return getattr(decoded, "sample", decoded[0])


class DINORepresentationEncoder:
    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError("REPA preprocessing requires `transformers`") from error
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, images: list[Any], dtype: torch.dtype) -> Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        output = self.model(**inputs).last_hidden_state
        # DINO-style models prepend a CLS token; REPA aligns spatial patch tokens.
        return output[:, 1:].to(dtype)


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_text_encoder(
    representation: RepresentationConfig,
    data: DataConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> MockTextEncoder | HFTextEncoder:
    if representation.text_encoder_type == "mock":
        return MockTextEncoder(data.text_dim, data.text_length)
    return HFTextEncoder(representation, device, dtype)
