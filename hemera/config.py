from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    dataset_id: str = "pauhidalgoo/photonyx"
    dataset_revision: str | None = None
    image_size: int = 256
    cache_dir: str = "data/cache"
    raw_cache_dir: str = "data/raw-cache"
    manifest_dir: str = "data/manifests"
    shard_size: int = 4096
    max_items: int | None = None
    synthetic_items: int = 32
    latent_channels: int = 4
    latent_size: int = 32
    text_length: int = 77
    text_dim: int = 512
    source_temperature: float = 1.0
    allowed_licenses: list[str] = field(
        default_factory=lambda: [
            "public domain",
            "cc0",
            "cc-by",
            "cc by",
            "creative commons attribution",
        ]
    )


@dataclass
class RepresentationConfig:
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    vae_type: Literal["kl", "dc"] = "kl"
    text_encoder_id: str = "openai/clip-vit-base-patch32"
    text_encoder_type: Literal["clip", "t5", "mock"] = "clip"
    precompute_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    max_text_length: int = 77
    repa_encoder_id: str = "facebook/dinov2-small"
    precompute_repa: bool = False


@dataclass
class ModelConfig:
    backbone: Literal["unet", "dit", "pixart", "uvit", "mmdit", "sana"] = "dit"
    in_channels: int = 4
    hidden_size: int = 192
    depth: int = 6
    heads: int = 6
    patch_size: int = 2
    mlp_ratio: float = 4.0
    text_dim: int = 512
    text_length: int = 77
    dropout: float = 0.0
    conditioning: Literal["pooled", "cross", "joint"] = "pooled"
    ffn: Literal["swiglu", "gelu", "mix", "moe"] = "swiglu"
    attention: Literal["full", "linear", "window"] = "full"
    efficiency: Literal["dense", "microdit", "tread", "repa", "moe"] = "dense"
    mask_ratio: float = 0.5
    mask_finish_ratio: float = 0.1
    window_size: int = 8
    num_experts: int = 4
    repa_dim: int = 384
    max_parameters: int = 50_000_000


@dataclass
class ObjectiveConfig:
    name: Literal["epsilon", "v", "flow"] = "flow"
    train_timesteps: int = 1000
    min_snr_gamma: float = 5.0
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0


@dataclass
class TrainConfig:
    output_dir: str = "runs/smoke"
    seed: int = 42
    steps: int = 100
    batch_size: int = 8
    effective_batch_size: int = 8
    learning_rate: float = 2e-4
    min_learning_rate_ratio: float = 0.1
    warmup_ratio: float = 0.02
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    compile: bool = False
    ema_half_life_images: int = 100_000
    cfg_dropout: float = 0.1
    log_every: int = 10
    sample_every: int = 0
    save_every: int = 50
    resume: str | None = None
    device: str = "auto"
    dataset_mode: Literal["synthetic", "shards"] = "synthetic"


@dataclass
class BudgetConfig:
    max_eur: float = 0.0
    hourly_eur: float = 0.5
    already_spent_eur: float = 0.0
    reserve_minutes: float = 10.0
    save_milestones: bool = False


@dataclass
class EvalConfig:
    seed: int = 1234
    samples: int = 64
    batch_size: int = 8
    num_inference_steps: int = 30
    guidance_scale: float = 3.0
    cmmd_bandwidth: float = 10.0
    prompt_file: str | None = None


@dataclass
class ExperimentConfig:
    name: str = "hemera-smoke"
    data: DataConfig = field(default_factory=DataConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        if self.data.image_size <= 0 or self.data.latent_size <= 0:
            raise ValueError("Image and latent sizes must be positive")
        if self.model.hidden_size % self.model.heads:
            raise ValueError("model.hidden_size must be divisible by model.heads")
        if self.model.in_channels != self.data.latent_channels:
            raise ValueError("model.in_channels must equal data.latent_channels")
        if not 0.0 <= self.train.cfg_dropout < 1.0:
            raise ValueError("train.cfg_dropout must be in [0, 1)")
        if not 0.0 <= self.model.mask_ratio < 1.0:
            raise ValueError("model.mask_ratio must be in [0, 1)")
        if not 0.0 <= self.model.mask_finish_ratio < 1.0:
            raise ValueError("model.mask_finish_ratio must be in [0, 1)")
        if self.train.effective_batch_size < self.train.batch_size:
            raise ValueError("effective_batch_size cannot be smaller than batch_size")
        if self.train.effective_batch_size % self.train.batch_size:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        if self.budget.max_eur < 0 or self.budget.hourly_eur <= 0:
            raise ValueError("Invalid budget values")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _construct(cls: type[T], raw: dict[str, Any] | None) -> T:
    raw = raw or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(unknown)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    top = {"name", "data", "representation", "model", "objective", "train", "budget", "eval"}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(unknown)}")
    cfg = ExperimentConfig(
        name=raw.get("name", "hemera"),
        data=_construct(DataConfig, raw.get("data")),
        representation=_construct(RepresentationConfig, raw.get("representation")),
        model=_construct(ModelConfig, raw.get("model")),
        objective=_construct(ObjectiveConfig, raw.get("objective")),
        train=_construct(TrainConfig, raw.get("train")),
        budget=_construct(BudgetConfig, raw.get("budget")),
        eval=_construct(EvalConfig, raw.get("eval")),
    )
    cfg.validate()
    return cfg


def _load_raw_config(source: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    source = source.resolve()
    if source in seen:
        chain = " -> ".join(str(item) for item in (*seen, source))
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    parent = raw.pop("extends", None)
    if parent is not None:
        parent_path = source.parent / parent
        parent_raw = _load_raw_config(parent_path, (*seen, source))
        raw = _deep_merge(parent_raw, raw)
    return raw


def load_config(path: str | Path) -> ExperimentConfig:
    raw = _load_raw_config(Path(path), ())
    return config_from_dict(raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
