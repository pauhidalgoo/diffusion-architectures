from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from .checkpoint import load_checkpoint
from .config import ExperimentConfig, config_from_dict
from .encoders import MockTextEncoder, VAEAdapter, build_text_encoder, torch_dtype
from .models import build_model
from .objectives import build_objective
from .sampling import sample_latents


class HemeraPipeline:
    def __init__(
        self,
        config: ExperimentConfig,
        model: torch.nn.Module,
        text_encoder,
        vae: VAEAdapter | None,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.model = model.to(device=device, dtype=dtype).eval()
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.objective = build_objective(config.objective)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str = "auto",
        dtype: str = "bfloat16",
        load_vae: bool = True,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> "HemeraPipeline":
        root = cls._resolve_pretrained_root(
            path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        exported = root.is_dir() and (root / "model.safetensors").exists()
        if exported:
            from safetensors.torch import load_file

            config = config_from_dict(
                json.loads((root / "config.json").read_text(encoding="utf-8"))
            )
            payload = {"model": load_file(str(root / "model.safetensors"), device="cpu")}
        else:
            checkpoint_path = root / "checkpoint.pt" if root.is_dir() else root
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = config_from_dict(payload["config"])
        resolved = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        resolved_dtype = torch_dtype(dtype) if resolved.type == "cuda" else torch.float32
        model = build_model(config.model)
        model.load_state_dict(payload["model"])
        if payload.get("ema"):
            named_parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in payload["ema"].items():
                    if name in named_parameters:
                        named_parameters[name].copy_(value.to(named_parameters[name]))
        if config.representation.text_encoder_type == "mock":
            text_encoder = MockTextEncoder(config.data.text_dim, config.data.text_length)
        else:
            text_encoder = build_text_encoder(config.representation, config.data, resolved, resolved_dtype)
        vae = VAEAdapter(config.representation, resolved, resolved_dtype) if load_vae else None
        return cls(config, model, text_encoder, vae, resolved, resolved_dtype)

    @staticmethod
    def _resolve_pretrained_root(
        path: str | Path,
        *,
        revision: str | None,
        cache_dir: str | Path | None,
        local_files_only: bool,
        token: str | bool | None,
    ) -> Path:
        """Resolve a local export directory or download one from the HF Hub."""

        local = Path(path).expanduser()
        if local.exists():
            return local.resolve()
        value = str(path)
        if isinstance(path, Path) or value.startswith((".", "/", "\\")):
            raise FileNotFoundError(f"Hemera checkpoint does not exist: {local}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Loading a Hub model requires `huggingface-hub`; install the "
                "Hemera inference dependencies first."
            ) from error
        downloaded = snapshot_download(
            repo_id=value,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=["model.safetensors", "config.json", "model_index.json"],
        )
        return Path(downloaded)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | Sequence[str],
        negative_prompt: str | Sequence[str] | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
        seed: int | None = None,
        output_type: str = "pil",
    ):
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator or seed, not both")
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        condition = self.text_encoder.encode(prompts, self.device, self.dtype)
        negative = None
        if negative_prompt is not None:
            negatives = (
                [negative_prompt] * len(prompts)
                if isinstance(negative_prompt, str)
                else list(negative_prompt)
            )
            if len(negatives) != len(prompts):
                raise ValueError("negative_prompt batch must match prompt batch")
            negative = self.text_encoder.encode(negatives, self.device, self.dtype)
        latents = sample_latents(
            self.model,
            self.objective,
            (
                len(prompts),
                self.config.data.latent_channels,
                self.config.data.latent_size,
                self.config.data.latent_size,
            ),
            condition,
            num_inference_steps,
            guidance_scale,
            generator,
            unconditional_condition=negative,
        )
        if output_type == "latent":
            return latents
        if self.vae is None:
            raise RuntimeError("Pixel output requested but the VAE was not loaded")
        images = self.vae.decode(latents).clamp(-1, 1).add(1).mul(127.5).byte().cpu()
        pil_images = [
            Image.fromarray(image.permute(1, 2, 0).numpy()) for image in images
        ]
        return pil_images

    def save_pretrained(self, output_dir: str | Path) -> Path:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise RuntimeError("Export requires `safetensors`") from error
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        weights = output / "model.safetensors"
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.model.state_dict().items()
        }
        save_file(state, str(weights), metadata={"format": "pt", "name": "Hemera-Nano"})
        (output / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "HemeraPipeline",
                    "name": "Hemera-Nano",
                    "backbone": self.config.model.backbone,
                    "objective": self.config.objective.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return weights

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: str | None = None,
        commit_message: str = "Upload Hemera model",
    ) -> str:
        """Export and upload this denoiser to a Hugging Face model repository."""

        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("push_to_hub requires `huggingface-hub`") from error
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        with TemporaryDirectory(prefix="hemera-hub-") as temporary:
            self.save_pretrained(temporary)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=temporary,
                commit_message=commit_message,
            )
        return f"https://huggingface.co/{repo_id}"
