from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig, ExperimentConfig
from .encoders import DINORepresentationEncoder, VAEAdapter, build_text_encoder, torch_dtype
from .interfaces import TextCondition
from .runtime import sha256_file

IMAGE_PREPROCESSING = "exif-rgb-center-crop-lanczos-v1"


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("Parquet shard metadata requires `pyarrow`") from error
        return parquet.read_table(path).to_pylist()
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def average_hash(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    bits = values > values.mean()
    packed = np.packbits(bits.reshape(-1))
    return packed.tobytes().hex()


def perceptual_hash(image: Image.Image, size: int = 8, high_frequency_factor: int = 4) -> str:
    try:
        from scipy.fft import dctn
    except ImportError as error:
        raise RuntimeError("Photonyx perceptual deduplication requires `scipy`") from error
    dimension = size * high_frequency_factor
    gray = np.asarray(
        image.convert("L").resize((dimension, dimension), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    frequencies = dctn(gray, type=2, norm="ortho")[:size, :size]
    median = np.median(frequencies.reshape(-1)[1:])
    bits = frequencies > median
    return np.packbits(bits.reshape(-1)).tobytes().hex()


class PerceptualDuplicateIndex:
    """LSH-assisted Hamming lookup; distance ≤4 is guaranteed a shared bucket."""

    def __init__(self, max_distance: int = 4, parts: int = 5):
        self.max_distance = max_distance
        self.parts = parts
        self.values: list[int] = []
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _chunks(self, value: int) -> Iterable[tuple[int, int]]:
        base, remainder = divmod(64, self.parts)
        offset = 0
        for part in range(self.parts):
            width = base + (1 if part < remainder else 0)
            mask = (1 << width) - 1
            yield part, (value >> offset) & mask
            offset += width

    def find_or_add(self, hex_value: str) -> tuple[str, bool]:
        value = int(hex_value, 16)
        candidates: set[int] = set()
        chunks = list(self._chunks(value))
        for chunk in chunks:
            candidates.update(self.buckets.get(chunk, []))
        for index in candidates:
            if (value ^ self.values[index]).bit_count() <= self.max_distance:
                return f"{self.values[index]:016x}", True
        index = len(self.values)
        self.values.append(value)
        for chunk in chunks:
            self.buckets[chunk].append(index)
        return f"{value:016x}", False


def split_for_cluster(cluster: str) -> str:
    value = int(hashlib.sha256(cluster.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    if value < 9_800:
        return "train"
    if value < 9_900:
        return "validation"
    return "test"


def normalized_license(value: Any) -> str:
    return "-".join(str(value or "").strip().lower().replace("_", "-").split())


def license_allowed(value: Any, allowed: Sequence[str]) -> bool:
    license_value = normalized_license(value)
    if any(
        restriction in license_value
        for restriction in (
            "-nc",
            "noncommercial",
            "non-commercial",
            "-nd",
            "no-derivatives",
        )
    ):
        return False
    for candidate in allowed:
        normalized_candidate = normalized_license(candidate)
        if license_value == normalized_candidate:
            return True
        if normalized_candidate == "cc0" and license_value.startswith("cc0-"):
            return True
        if normalized_candidate == "cc-by" and license_value.startswith("cc-by-"):
            return True
        if normalized_candidate == "public-domain" and license_value.startswith("public-domain"):
            return True
    return False


def prepare_square_image(image: Image.Image, size: int) -> Image.Image:
    """Apply EXIF orientation and a deterministic aspect-preserving center crop."""
    if size <= 0:
        raise ValueError("Image size must be positive")
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(
        oriented,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def image_to_tensor(image: Image.Image, size: int) -> Tensor:
    image = prepare_square_image(image, size)
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)


class SyntheticLatentDataset(Dataset[dict[str, Any]]):
    """Small deterministic fixture with text-correlated latent patterns."""

    def __init__(
        self,
        items: int = 32,
        channels: int = 4,
        latent_size: int = 8,
        text_length: int = 8,
        text_dim: int = 32,
        seed: int = 0,
    ):
        generator = torch.Generator().manual_seed(seed)
        self.latents = torch.randn(items, channels, latent_size, latent_size, generator=generator)
        self.text_hidden = torch.randn(items, text_length, text_dim, generator=generator)
        self.text_mask = torch.ones(items, text_length, dtype=torch.bool)
        self.pooled = self.text_hidden.mean(1)
        projection = torch.randn(text_dim, channels, generator=generator) / math.sqrt(text_dim)
        color = self.pooled @ projection
        self.latents = 0.65 * self.latents + 0.35 * color[:, :, None, None]
        self.sources = ["synthetic-a" if index % 2 else "synthetic-b" for index in range(items)]

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "latent_mean": self.latents[index],
            "latent_logvar": torch.full_like(self.latents[index], -30.0),
            "text_hidden": self.text_hidden[index],
            "text_mask": self.text_mask[index],
            "pooled": self.pooled[index],
            "source": self.sources[index],
            "id": f"synthetic-{index:05d}",
        }


class ShardedLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing precompute index: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.shards = [item for item in payload["shards"] if item["split"] == split]
        self.offsets: list[tuple[int, int, dict[str, Any]]] = []
        offset = 0
        for shard in self.shards:
            count = int(shard["count"])
            self.offsets.append((offset, offset + count, shard))
            offset += count
        self.length = offset
        self._cached_path: Path | None = None
        self._cached_tensors: dict[str, Tensor] | None = None
        self._cached_metadata: list[dict[str, Any]] | None = None
        self._verified_paths: set[Path] = set()
        self._source_shards: dict[str, list[tuple[list[int], int]]] | None = None
        self._resident_shards: OrderedDict[
            Path, tuple[dict[str, Tensor], list[dict[str, Any]], int]
        ] = OrderedDict()
        cache_gib = float(os.environ.get("HEMERA_SHARD_CACHE_GIB", "16"))
        self._resident_limit_bytes = max(0, int(cache_gib * 1024**3))
        self._resident_bytes = 0
        self._device_tensors: dict[str, Tensor] | None = None

    def __len__(self) -> int:
        return self.length

    def _locate(self, index: int) -> tuple[int, dict[str, Any]]:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        for start, end, shard in self.offsets:
            if start <= index < end:
                return index - start, shard
        raise IndexError(index)

    def _load(self, shard: dict[str, Any]) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("Reading precomputed shards requires `safetensors`") from error
        path = self.root / shard["file"]
        if path == self._cached_path:
            return
        resident = self._resident_shards.get(path)
        if resident is not None:
            self._resident_shards.move_to_end(path)
            self._cached_tensors, self._cached_metadata, _ = resident
            self._cached_path = path
            return
        expected_hash = shard.get("sha256")
        if path not in self._verified_paths:
            if expected_hash and sha256_file(path) != expected_hash:
                raise RuntimeError(f"Checksum mismatch for {path}")
            metadata_path = self.root / shard["metadata"]
            expected_metadata_hash = shard.get("metadata_sha256")
            if (
                expected_metadata_hash
                and sha256_file(metadata_path) != expected_metadata_hash
            ):
                raise RuntimeError(f"Checksum mismatch for {metadata_path}")
            self._verified_paths.add(path)
        tensors = load_file(str(path), device="cpu")
        metadata_path = self.root / shard["metadata"]
        metadata = _read_metadata(metadata_path)
        shard_bytes = path.stat().st_size + metadata_path.stat().st_size
        if shard_bytes <= self._resident_limit_bytes:
            while (
                self._resident_shards
                and self._resident_bytes + shard_bytes > self._resident_limit_bytes
            ):
                _, (_, _, evicted_bytes) = self._resident_shards.popitem(last=False)
                self._resident_bytes -= evicted_bytes
            self._resident_shards[path] = (tensors, metadata, shard_bytes)
            self._resident_bytes += shard_bytes
        self._cached_tensors = tensors
        self._cached_metadata = metadata
        self._cached_path = path

    def __getitem__(self, index: int) -> dict[str, Any]:
        local, shard = self._locate(index)
        self._load(shard)
        assert self._cached_tensors is not None and self._cached_metadata is not None
        result = {key: value[local] for key, value in self._cached_tensors.items()}
        result.update(self._cached_metadata[local])
        return result

    def collate_indices(
        self,
        indices: Sequence[int],
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Vectorize a shard-local batch without per-example dictionaries."""
        if not indices:
            raise ValueError("Cannot collate an empty batch")
        locations = [self._locate(index) for index in indices]
        shard = locations[0][1]
        if any(location[1]["file"] != shard["file"] for location in locations[1:]):
            return collate_examples(
                [self[index] for index in indices],
                sample_posterior=sample_posterior,
                generator=generator,
            )
        self._load(shard)
        assert self._cached_tensors is not None and self._cached_metadata is not None
        local_indices = torch.tensor(
            [location[0] for location in locations], dtype=torch.long
        )
        tensors = self._cached_tensors
        mean = tensors["latent_mean"].index_select(0, local_indices)
        logvar = tensors["latent_logvar"].index_select(0, local_indices)
        if sample_posterior:
            posterior_noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            latents = mean + torch.exp(0.5 * logvar) * posterior_noise
        else:
            latents = mean
        condition = TextCondition(
            hidden_states=tensors["text_hidden"].index_select(0, local_indices),
            attention_mask=tensors["text_mask"]
            .index_select(0, local_indices)
            .bool(),
            pooled=tensors["pooled"].index_select(0, local_indices),
        )
        metadata = [self._cached_metadata[index] for index in local_indices.tolist()]
        result: dict[str, Any] = {
            "latents": latents,
            "condition": condition,
            "ids": [item.get("id", "") for item in metadata],
            "sources": [item.get("source", "unknown") for item in metadata],
        }
        if "repa_target" in tensors:
            result["repa_target"] = tensors["repa_target"].index_select(
                0, local_indices
            )
        return result

    @property
    def has_device_cache(self) -> bool:
        return self._device_tensors is not None

    def cache_tensors_on_device(
        self,
        device: torch.device,
        dtype: torch.dtype,
        include_repa: bool = False,
    ) -> None:
        """Keep the compact training representation on GPU for paid sweeps."""
        keys = [
            "latent_mean",
            "latent_logvar",
            "text_hidden",
            "text_mask",
            "pooled",
        ]
        if include_repa:
            keys.append("repa_target")
        pieces: dict[str, list[Tensor]] = {key: [] for key in keys}
        for shard in self.shards:
            self._load(shard)
            assert self._cached_tensors is not None
            for key in keys:
                if key not in self._cached_tensors:
                    raise KeyError(f"Precomputed shard is missing {key}")
                pieces[key].append(self._cached_tensors[key])
        cached: dict[str, Tensor] = {}
        for key, values in pieces.items():
            tensor = torch.cat(values, dim=0)
            target_dtype = torch.bool if key == "text_mask" else dtype
            cached[key] = tensor.to(
                device=device, dtype=target_dtype, non_blocking=False
            )
        self._device_tensors = cached

    def collate_device_indices(
        self,
        indices: Sequence[int],
        generator: torch.Generator,
        sample_posterior: bool = True,
    ) -> dict[str, Any]:
        if self._device_tensors is None:
            raise RuntimeError("Device cache has not been initialized")
        tensors = self._device_tensors
        device = tensors["latent_mean"].device
        selected = torch.tensor(indices, dtype=torch.long, device=device)
        mean = tensors["latent_mean"].index_select(0, selected)
        logvar = tensors["latent_logvar"].index_select(0, selected)
        if sample_posterior:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=device,
                generator=generator,
            )
            latents = mean + torch.exp(0.5 * logvar) * noise
        else:
            latents = mean
        result: dict[str, Any] = {
            "latents": latents,
            "condition": TextCondition(
                hidden_states=tensors["text_hidden"].index_select(0, selected),
                attention_mask=tensors["text_mask"]
                .index_select(0, selected)
                .bool(),
                pooled=tensors["pooled"].index_select(0, selected),
            ),
            "ids": [],
            "sources": [],
        }
        if "repa_target" in tensors:
            result["repa_target"] = tensors["repa_target"].index_select(
                0, selected
            )
        return result

    def grouped_indices(self) -> dict[str, list[int]]:
        """Build source groups from metadata without loading tensor shards."""
        groups: dict[str, list[int]] = defaultdict(list)
        offset = 0
        for shard in self.shards:
            metadata_path = self.root / shard["metadata"]
            rows = _read_metadata(metadata_path)
            if len(rows) != int(shard["count"]):
                raise RuntimeError(f"Metadata count mismatch in {metadata_path}")
            for local_index, row in enumerate(rows):
                groups[str(row.get("source", "unknown"))].append(offset + local_index)
            offset += len(rows)
        return dict(groups)

    def _build_sampling_index(self) -> dict[str, list[tuple[list[int], int]]]:
        if self._source_shards is not None:
            return self._source_shards
        source_shards: dict[str, list[tuple[list[int], int]]] = defaultdict(list)
        offset = 0
        for shard in self.shards:
            metadata_path = self.root / shard["metadata"]
            rows = _read_metadata(metadata_path)
            local: dict[str, list[int]] = defaultdict(list)
            for local_index, row in enumerate(rows):
                local[str(row.get("source", "unknown"))].append(offset + local_index)
            for source, indices in local.items():
                source_shards[source].append((indices, len(indices)))
            offset += len(rows)
        self._source_shards = dict(source_shards)
        return self._source_shards

    def deterministic_batch_indices(
        self,
        batch_size: int,
        step: int,
        seed: int,
        source_temperature: float,
    ) -> list[int]:
        """Sample one shard per batch to prevent random-read shard thrashing."""
        generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
        if source_temperature >= 0.999:
            counts = torch.tensor(
                [int(shard["count"]) for shard in self.shards], dtype=torch.float64
            )
            shard_index = int(
                torch.multinomial(counts / counts.sum(), 1, generator=generator)
            )
            start, end, _ = self.offsets[shard_index]
            return (
                torch.randint(end - start, (batch_size,), generator=generator) + start
            ).tolist()

        source_shards = self._build_sampling_index()
        sources = sorted(source_shards)
        source_counts = torch.tensor(
            [sum(count for _, count in source_shards[source]) for source in sources],
            dtype=torch.float64,
        )
        source_weights = source_counts.pow(source_temperature)
        source_index = int(
            torch.multinomial(
                source_weights / source_weights.sum(), 1, generator=generator
            )
        )
        candidates = source_shards[sources[source_index]]
        shard_counts = torch.tensor(
            [count for _, count in candidates], dtype=torch.float64
        )
        selected_shard = int(
            torch.multinomial(shard_counts / shard_counts.sum(), 1, generator=generator)
        )
        indices = candidates[selected_shard][0]
        positions = torch.randint(len(indices), (batch_size,), generator=generator)
        return [indices[position] for position in positions.tolist()]


def deterministic_indices(
    length: int,
    batch_size: int,
    step: int,
    seed: int,
    source_groups: dict[str, list[int]] | None = None,
    source_temperature: float = 1.0,
) -> list[int]:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
    if not source_groups or source_temperature >= 0.999:
        return torch.randint(length, (batch_size,), generator=generator).tolist()
    sources = sorted(source_groups)
    counts = torch.tensor([len(source_groups[source]) for source in sources], dtype=torch.float64)
    probabilities = counts.pow(source_temperature)
    probabilities /= probabilities.sum()
    selected_sources = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
    indices: list[int] = []
    for selected in selected_sources.tolist():
        group = source_groups[sources[selected]]
        position = int(torch.randint(len(group), (1,), generator=generator))
        indices.append(group[position])
    return indices


def deterministic_dataset_indices(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    step: int,
    seed: int,
    source_temperature: float = 1.0,
    groups: dict[str, list[int]] | None = None,
) -> list[int]:
    if isinstance(dataset, ShardedLatentDataset):
        return dataset.deterministic_batch_indices(
            batch_size, step, seed, source_temperature
        )
    return deterministic_indices(
        len(dataset), batch_size, step, seed, groups, source_temperature
    )


def source_groups(dataset: Dataset[dict[str, Any]]) -> dict[str, list[int]]:
    if isinstance(dataset, ShardedLatentDataset):
        return dataset.grouped_indices()
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        groups[str(dataset[index].get("source", "unknown"))].append(index)
    return dict(groups)


def collate_examples(
    examples: list[dict[str, Any]],
    sample_posterior: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    mean = torch.stack([item["latent_mean"] for item in examples])
    logvar = torch.stack([item["latent_logvar"] for item in examples])
    if sample_posterior:
        posterior_noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        latents = mean + torch.exp(0.5 * logvar) * posterior_noise
    else:
        latents = mean
    condition = TextCondition(
        hidden_states=torch.stack([item["text_hidden"] for item in examples]),
        attention_mask=torch.stack([item["text_mask"] for item in examples]).bool(),
        pooled=torch.stack([item["pooled"] for item in examples]),
    )
    result: dict[str, Any] = {
        "latents": latents,
        "condition": condition,
        "ids": [item.get("id", "") for item in examples],
        "sources": [item.get("source", "unknown") for item in examples],
    }
    if all("repa_target" in item for item in examples):
        result["repa_target"] = torch.stack([item["repa_target"] for item in examples])
    return result


def build_dataset(config: ExperimentConfig, split: str = "train") -> Dataset[dict[str, Any]]:
    if config.train.dataset_mode == "synthetic":
        return SyntheticLatentDataset(
            config.data.synthetic_items,
            config.data.latent_channels,
            config.data.latent_size,
            config.data.text_length,
            config.data.text_dim,
            config.train.seed,
        )
    return ShardedLatentDataset(config.data.cache_dir, split)


def _load_photonyx(config: DataConfig):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Photonyx cloud commands require `datasets`") from error
    return load_dataset(
        config.dataset_id,
        split="train",
        revision=config.dataset_revision,
        streaming=True,
    )


def _resume_shard_position(
    shard_lengths: Sequence[int], row_index: int
) -> tuple[int, int]:
    if row_index < 0 or row_index > sum(shard_lengths):
        raise ValueError("Resume row is outside the dataset shard layout")
    offset = 0
    for shard_index, length in enumerate(shard_lengths):
        if row_index < offset + length:
            return shard_index, row_index - offset
        offset += length
    return len(shard_lengths), 0


def _disable_streaming_image_decode(dataset):
    try:
        from datasets import Image as DatasetImage
    except ImportError:
        return dataset
    cast_column = getattr(dataset, "cast_column", None)
    if cast_column is None:
        return dataset
    return cast_column("image", DatasetImage(decode=False))


def _decode_audit_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported streamed image value: {type(value).__name__}")
    if value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as image:
            image.load()
            return image.copy()
    if value.get("path"):
        with Image.open(value["path"]) as image:
            image.load()
            return image.copy()
    raise ValueError("Streamed image has neither bytes nor path")


def _photonyx_rows_from(
    config: DataConfig,
    start_row: int,
    layout_path: Path,
):
    """Stream from an exact global row without replaying preceding Parquet shards."""
    dataset = _load_photonyx(config)
    iterable = getattr(dataset, "_ex_iterable", None)
    files = list(getattr(iterable, "kwargs", {}).get("files", []))
    dataset = _disable_streaming_image_decode(dataset)
    if start_row <= 0:
        yield from enumerate(dataset)
        return
    if not files:
        # Compatibility fallback for a future datasets backend without exposed
        # Parquet sources. It is slow but preserves correctness.
        for row_index, row in enumerate(dataset):
            if row_index >= start_row:
                yield row_index, row
        return

    contract = {
        "dataset": config.dataset_id,
        "revision": config.dataset_revision,
        "files": files,
    }
    shard_lengths: list[int]
    if layout_path.exists():
        saved = json.loads(layout_path.read_text(encoding="utf-8"))
        if {
            "dataset": saved.get("dataset"),
            "revision": saved.get("revision"),
            "files": saved.get("files"),
        } != contract:
            raise ValueError(f"Photonyx shard layout mismatch: {layout_path}")
        shard_lengths = [int(value) for value in saved["shard_lengths"]]
    else:
        try:
            import fsspec
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "Fast audit resume requires fsspec and pyarrow"
            ) from error

        def rows(path: str) -> int:
            with fsspec.open(path, "rb").open() as handle:
                return int(parquet.ParquetFile(handle).metadata.num_rows)

        with ThreadPoolExecutor(max_workers=min(12, len(files))) as pool:
            shard_lengths = list(pool.map(rows, files))
        payload = {
            **contract,
            "shard_lengths": shard_lengths,
            "total_rows": sum(shard_lengths),
        }
        temporary = layout_path.with_suffix(layout_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(layout_path)

    if len(shard_lengths) != len(files):
        raise ValueError(f"Invalid Photonyx shard layout: {layout_path}")
    shard_index, local_skip = _resume_shard_position(
        shard_lengths, start_row
    )
    if shard_index == len(files):
        return
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Photonyx cloud commands require `datasets`") from error
    suffix = load_dataset(
        "parquet",
        data_files={"train": files[shard_index:]},
        split="train",
        streaming=True,
        features=dataset.features,
    )
    # Do not call IterableDataset.skip here. Some datasets releases mishandle
    # exact Arrow row-group boundaries. Enumerating only the partial first
    # shard is inexpensive and was ID-validated against the audit checkpoint.
    for local_index, row in enumerate(suffix):
        if local_index < local_skip:
            continue
        yield start_row + local_index - local_skip, row


def audit_photonyx(config: ExperimentConfig, output: str | Path) -> dict[str, Any]:
    """Stream Photonyx and write a duplicate-aware manifest without caching images."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    duplicate_index = PerceptualDuplicateIndex(max_distance=4)
    seen_ids: set[str] = set()
    accepted = 0
    scanned = 0
    started = time.monotonic()
    progress_path = target.with_suffix(".progress.json")
    existing_records: list[dict[str, Any]] = []
    if target.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        scanned = int(progress.get("scanned", 0))
        counts.update(progress.get("counts", {}))
        parsed = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line
        ]
        existing_records = [
            record for record in parsed if int(record["row_index"]) < scanned
        ]
        if len(existing_records) != int(progress.get("accepted", -1)):
            raise RuntimeError(
                "Audit manifest/progress mismatch; preserve both files and restart "
                "into a new output path"
            )
        accepted = len(existing_records)
        for record in existing_records:
            seen_ids.add(str(record["id"]))
            duplicate_index.find_or_add(str(record["phash"]))
        # Drop any rows written after the last atomic progress snapshot.
        temporary = target.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in existing_records
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
    resumed_from = scanned

    def write_progress() -> None:
        elapsed = time.monotonic() - started
        payload = {
            "scanned": scanned,
            "accepted": accepted,
            "resumed_from": resumed_from,
            "rows_scanned_this_run": scanned - resumed_from,
            "elapsed_seconds": elapsed,
            "rows_per_second": (
                (scanned - resumed_from) / max(elapsed, 1e-8)
            ),
            "counts": dict(counts),
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    if (
        config.data.max_items is not None
        and scanned >= config.data.max_items
    ):
        write_progress()
        summary = {
            "scanned": scanned,
            "accepted": accepted,
            "counts": dict(counts),
            "manifest": str(target),
            "manifest_sha256": sha256_file(target),
        }
        target.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    with target.open("a" if existing_records else "w", encoding="utf-8") as handle:
        rows = _photonyx_rows_from(
            config.data,
            scanned,
            target.with_suffix(".shards.json"),
        )
        for row_index, row in rows:
            if config.data.max_items is not None and row_index >= config.data.max_items:
                break
            scanned = row_index + 1
            try:
                prompt_value = row.get("prompt")
                prompt = (
                    str(prompt_value).strip()
                    if prompt_value is not None
                    else ""
                )
                if not prompt:
                    counts["empty_prompt"] += 1
                    continue
                if not license_allowed(row.get("license"), config.data.allowed_licenses):
                    counts["license_rejected"] += 1
                    continue
                image_id_value = row.get("image_id")
                image_id = (
                    str(image_id_value).strip()
                    if image_id_value is not None
                    else ""
                )
                if not image_id:
                    image_id = f"row-{row_index:09d}"
                    counts["missing_id_replaced"] += 1
                if image_id in seen_ids:
                    counts["duplicate"] += 1
                    continue
                seen_ids.add(image_id)
                image = _decode_audit_image(row["image"])
                cluster, duplicate = duplicate_index.find_or_add(perceptual_hash(image))
                if duplicate:
                    counts["perceptual_duplicate"] += 1
                    continue
                record = {
                    "row_index": row_index,
                    "id": image_id,
                    "prompt": prompt,
                    "source": str(row.get("source", "unknown")),
                    "license": str(row.get("license", "")),
                    "phash": cluster,
                    "split": split_for_cluster(cluster),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[f"split_{record['split']}"] += 1
                counts[f"source_{record['source']}"] += 1
                accepted += 1
            except Exception as error:
                counts["corrupt"] += 1
                counts[f"corrupt_{type(error).__name__}"] += 1
            finally:
                if scanned % 1_000 == 0:
                    handle.flush()
                    write_progress()
    write_progress()
    summary = {
        "scanned": scanned,
        "accepted": accepted,
        "counts": dict(counts),
        "manifest": str(target),
        "manifest_sha256": sha256_file(target),
        "shard_layout_sha256": (
            sha256_file(target.with_suffix(".shards.json"))
            if target.with_suffix(".shards.json").exists()
            else None
        ),
        "dataset": config.data.dataset_id,
        "dataset_revision": config.data.dataset_revision,
        "split_policy": "sha256-phash-cluster-9800-100-100-v1",
        "perceptual_hash": {
            "algorithm": "dct-phash-64",
            "maximum_hamming_distance": 4,
        },
    }
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def verify_audit_manifest(
    config: ExperimentConfig,
    manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Fail closed on an inconsistent audit manifest before paid encoding."""
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Audit manifest is empty")
    required = {
        "row_index",
        "id",
        "prompt",
        "source",
        "license",
        "phash",
        "split",
    }
    ids: set[str] = set()
    phashes: set[str] = set()
    duplicate_verifier = PerceptualDuplicateIndex(max_distance=4)
    row_indices: set[int] = set()
    by_split: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_license: Counter[str] = Counter()
    by_source_split: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_lengths: list[int] = []
    previous_row = -1
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"Audit manifest row {index} is missing {sorted(missing)}"
            )
        row_index = int(row["row_index"])
        if row_index <= previous_row:
            raise ValueError("Audit manifest row indices are not strictly increasing")
        previous_row = row_index
        identifier = str(row["id"])
        phash = str(row["phash"])
        prompt = str(row["prompt"]).strip()
        split = str(row["split"])
        source = str(row["source"])
        license_name = str(row["license"])
        if identifier in ids or row_index in row_indices:
            raise ValueError("Audit manifest contains a duplicate ID or row index")
        if phash in phashes:
            raise ValueError("Audit manifest contains a duplicate pHash cluster")
        _, near_duplicate = duplicate_verifier.find_or_add(phash)
        if near_duplicate:
            raise ValueError(
                "Audit manifest contains perceptual clusters within Hamming distance 4"
            )
        if not prompt:
            raise ValueError("Audit manifest contains an empty prompt")
        if split != split_for_cluster(phash):
            raise ValueError(
                f"Audit manifest row {index} violates deterministic split policy"
            )
        if not license_allowed(license_name, config.data.allowed_licenses):
            raise ValueError(
                f"Audit manifest row {index} has unsupported license {license_name!r}"
            )
        ids.add(identifier)
        phashes.add(phash)
        row_indices.add(row_index)
        by_split[split] += 1
        by_source[source] += 1
        by_license[normalized_license(license_name)] += 1
        by_source_split[source][split] += 1
        prompt_lengths.append(len(prompt))
    ordered_lengths = sorted(prompt_lengths)

    def quantile(fraction: float) -> int:
        return ordered_lengths[
            min(len(ordered_lengths) - 1, int(fraction * len(ordered_lengths)))
        ]

    result = {
        "valid": True,
        "dataset": config.data.dataset_id,
        "dataset_revision": config.data.dataset_revision,
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256_file(manifest),
        "records": len(rows),
        "first_row_index": int(rows[0]["row_index"]),
        "last_row_index": int(rows[-1]["row_index"]),
        "unique_ids": len(ids),
        "unique_phash_clusters": len(phashes),
        "by_split": dict(sorted(by_split.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_license": dict(sorted(by_license.items())),
        "by_source_and_split": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source_split.items())
        },
        "prompt_length_characters": {
            "minimum": ordered_lengths[0],
            "p50": quantile(0.5),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
            "maximum": ordered_lengths[-1],
        },
        "split_policy": "sha256-phash-cluster-9800-100-100-v1",
        "license_policy": config.data.allowed_licenses,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(target)
    return result


def select_manifest_subset(
    records: Sequence[dict[str, Any]],
    max_items: int | None,
    salt: str = "hemera-photonyx-subset-v1",
) -> list[dict[str, Any]]:
    """Select a deterministic source/split-stratified subset of a full manifest."""
    ordered = sorted(records, key=lambda item: int(item["row_index"]))
    if max_items is None or max_items >= len(ordered):
        return ordered
    if max_items <= 0:
        return []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        buckets[(str(record["source"]), str(record["split"]))].append(record)
    keys = sorted(buckets)
    allocations = {key: 0 for key in keys}
    remaining = max_items
    # Retain at least one example from every observed source/split cell whenever
    # the subset is large enough. This prevents rare museum/manual sources from
    # disappearing from validation and test probes.
    if max_items >= len(keys):
        for key in keys:
            allocations[key] = 1
        remaining -= len(keys)
    capacities = {key: len(buckets[key]) - allocations[key] for key in keys}
    total_capacity = sum(capacities.values())
    if remaining and total_capacity:
        exact = {
            key: remaining * capacities[key] / total_capacity
            for key in keys
        }
        for key in keys:
            addition = min(capacities[key], int(math.floor(exact[key])))
            allocations[key] += addition
            remaining -= addition
        remainders = sorted(
            keys,
            key=lambda key: (
                -(exact[key] - math.floor(exact[key])),
                key,
            ),
        )
        while remaining:
            progressed = False
            for key in remainders:
                if allocations[key] < len(buckets[key]):
                    allocations[key] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                break
    selected: list[dict[str, Any]] = []
    for key in keys:
        candidates = sorted(
            buckets[key],
            key=lambda record: hashlib.sha256(
                (
                    salt
                    + ":"
                    + str(record["id"])
                    + ":"
                    + str(record["row_index"])
                ).encode("utf-8")
            ).digest(),
        )
        selected.extend(candidates[: allocations[key]])
    return sorted(selected, key=lambda item: int(item["row_index"]))


def _raw_cache_root(config: ExperimentConfig) -> Path:
    revision = config.data.dataset_revision or "unversioned"
    cache_identity = f"{revision}:{IMAGE_PREPROCESSING}"
    revision_key = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:12]
    return Path(config.data.raw_cache_dir) / revision_key / str(config.data.image_size)


def _raw_image_path(root: Path, row_index: int) -> Path:
    return root / f"{row_index:09d}.png"


def _save_raw_image(image: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resized = prepare_square_image(image, size)
    temporary = path.with_suffix(".png.tmp")
    resized.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)


def _load_raw_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _records_by_parquet_shard(
    selected_records: Sequence[dict[str, Any]],
    shard_lengths: Sequence[int],
) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Map globally indexed manifest records to immutable Parquet shards."""
    result: list[tuple[int, int, list[dict[str, Any]]]] = []
    selected_index = 0
    offset = 0
    for shard_index, length in enumerate(shard_lengths):
        end = offset + int(length)
        records: list[dict[str, Any]] = []
        while (
            selected_index < len(selected_records)
            and int(selected_records[selected_index]["row_index"]) < end
        ):
            record = selected_records[selected_index]
            if int(record["row_index"]) < offset:
                raise ValueError("Selected manifest records are not strictly ordered")
            records.append(record)
            selected_index += 1
        result.append((shard_index, offset, records))
        offset = end
    if selected_index != len(selected_records):
        raise ValueError("Selected manifest row lies outside the Parquet shard layout")
    return result


def _materialize_parquet_shard(
    url: str,
    global_offset: int,
    records: Sequence[dict[str, Any]],
    raw_root: Path,
    image_size: int,
) -> tuple[int, int]:
    """Read only needed row groups from one pinned shard and cache verified PNGs."""
    import bisect

    try:
        import fsspec
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parallel Photonyx materialization requires fsspec and pyarrow"
        ) from error
    pending = [
        record
        for record in records
        if not _raw_image_path(raw_root, int(record["row_index"])).exists()
    ]
    if not pending:
        return len(records), 0
    local_indices = [int(record["row_index"]) - global_offset for record in pending]
    materialized = 0
    with fsspec.open(url, "rb").open() as handle:
        source = parquet.ParquetFile(handle)
        group_offset = 0
        for row_group in range(source.num_row_groups):
            group_rows = int(source.metadata.row_group(row_group).num_rows)
            group_end = group_offset + group_rows
            left = bisect.bisect_left(local_indices, group_offset)
            right = bisect.bisect_left(local_indices, group_end)
            if left != right:
                rows = source.read_row_group(
                    row_group, columns=["image", "prompt", "image_id"]
                ).to_pylist()
                for selection_index in range(left, right):
                    local_index = local_indices[selection_index]
                    record = pending[selection_index]
                    row = rows[local_index - group_offset]
                    validate_streamed_record(global_offset + local_index, row, record)
                    raw_path = _raw_image_path(
                        raw_root, int(record["row_index"])
                    )
                    if not raw_path.exists():
                        _save_raw_image(
                            _decode_audit_image(row["image"]),
                            raw_path,
                            image_size,
                        )
                    materialized += 1
            group_offset = group_end
    return len(records), materialized


def _materialize_selected_parallel(
    selected_records: Sequence[dict[str, Any]],
    manifest_path: Path,
    raw_root: Path,
    image_size: int,
    workers: int,
    progress: Any,
    consume_records: Callable[[Sequence[dict[str, Any]]], None] | None = None,
) -> tuple[int, int]:
    """Materialize pinned shards concurrently and stream completed shard records."""
    if workers <= 1:
        raise ValueError("Parallel materialization requires at least two workers")
    layout_path = manifest_path.with_suffix(".shards.json")
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Parallel materialization requires the audited shard layout: {layout_path}"
        )
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    files = list(layout.get("files", []))
    lengths = [int(value) for value in layout.get("shard_lengths", [])]
    if len(files) != len(lengths) or not files:
        raise ValueError(f"Invalid Photonyx shard layout: {layout_path}")
    assignments = _records_by_parquet_shard(selected_records, lengths)
    jobs = [
        (files[shard_index], offset, records)
        for shard_index, offset, records in assignments
        if records
    ]
    jobs.sort(
        key=lambda job: (
            sum(
                not _raw_image_path(
                    raw_root, int(record["row_index"])
                ).exists()
                for record in job[2]
            ),
            job[1],
        )
    )

    def materialize_with_retry(
        url: str,
        offset: int,
        records: Sequence[dict[str, Any]],
    ) -> tuple[int, int]:
        attempts = 6
        for attempt in range(1, attempts + 1):
            try:
                return _materialize_parquet_shard(
                    url,
                    offset,
                    records,
                    raw_root,
                    image_size,
                )
            except Exception as error:
                if attempt == attempts:
                    raise
                delay = min(2 ** (attempt - 1), 30)
                print(
                    f"Photonyx shard retry {attempt}/{attempts - 1} "
                    f"after {type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    observed = 0
    materialized = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [
            (
                pool.submit(
                    materialize_with_retry,
                    url,
                    offset,
                    records,
                ),
                records,
            )
            for url, offset, records in jobs
        ]
        records_by_future = dict(futures)
        completed = (
            (future, records_by_future[future])
            for future in as_completed(records_by_future)
        )
        for future, records in completed:
            shard_observed, shard_materialized = future.result()
            observed += shard_observed
            materialized += shard_materialized
            progress(observed, materialized, len(jobs))
            if consume_records is not None:
                consume_records(records)
    return observed, materialized


def validate_streamed_record(
    row_index: int,
    row: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Refuse to encode an image if the pinned stream no longer matches audit."""
    image_id_value = row.get("image_id")
    streamed_id = (
        str(image_id_value).strip()
        if image_id_value is not None
        else ""
    )
    if not streamed_id:
        streamed_id = f"row-{row_index:09d}"
    streamed_prompt_value = row.get("prompt")
    streamed_prompt = (
        str(streamed_prompt_value).strip()
        if streamed_prompt_value is not None
        else ""
    )
    if streamed_id != str(record["id"]):
        raise RuntimeError(
            f"Photonyx row {row_index} ID changed after audit: "
            f"{streamed_id!r} != {record['id']!r}"
        )
    if streamed_prompt != str(record["prompt"]):
        raise RuntimeError(
            f"Photonyx row {row_index} prompt changed after audit"
        )


def precompute_photonyx(
    config: ExperimentConfig,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int,
    materialize_workers: int = 1,
) -> dict[str, Any]:
    """Encode audited rows into restartable, fixed-shape Safetensors shards."""
    precompute_started = time.monotonic()
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("Precomputation requires `safetensors`") from error
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "precompute-progress.json"

    def write_initial_progress() -> None:
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "stage": "loading_encoders",
                    "elapsed_seconds": time.monotonic() - precompute_started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(progress_path)

    write_initial_progress()
    dtype = torch_dtype(config.representation.precompute_dtype)
    vae = VAEAdapter(config.representation, device, dtype)
    text_encoder = build_text_encoder(config.representation, config.data, device, dtype)
    repa_encoder = (
        DINORepresentationEncoder(config.representation.repa_encoder_id, device, dtype)
        if config.representation.precompute_repa
        else None
    )
    encoder_load_seconds = time.monotonic() - precompute_started
    all_records = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_records = select_manifest_subset(all_records, config.data.max_items)
    records = {int(item["row_index"]): item for item in selected_records}
    bucket_for_id: dict[str, str] = {}
    bucket_split: dict[str, str] = {}
    bucket_number: dict[str, int] = {}
    bucket_expected: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_records = [
            record for record in selected_records if record["split"] == split
        ]
        for start in range(0, len(split_records), config.data.shard_size):
            number = start // config.data.shard_size
            key = f"{split}:{number:05d}"
            group = split_records[start : start + config.data.shard_size]
            bucket_split[key] = split
            bucket_number[key] = number
            bucket_expected[key] = len(group)
            for record in group:
                bucket_for_id[str(record["id"])] = key
    selection_hash = hashlib.sha256(
        "\n".join(
            f"{record['row_index']}:{record['id']}" for record in selected_records
        ).encode("utf-8")
    ).hexdigest()
    index_path = output / "index.json"
    expected_representation = {
        "vae": config.representation.vae_id,
        "vae_type": config.representation.vae_type,
        "text_encoder": config.representation.text_encoder_id,
        "text_encoder_type": config.representation.text_encoder_type,
        "max_text_tokens": config.representation.max_text_length,
        # Pooled-AdaLN never consumes token-level CLIP states. Retain a
        # one-token fallback equal to the pooled state so TextCondition stays
        # valid while the full Photonyx cache remains GPU-resident.
        "text_storage": (
            "pooled-token-v1"
            if config.model.conditioning == "pooled"
            else "full-token-v1"
        ),
        "repa_encoder": (
            config.representation.repa_encoder_id
            if config.representation.precompute_repa
            else None
        ),
        "image_size": config.data.image_size,
        "latent_channels": config.data.latent_channels,
        "latent_size": config.data.latent_size,
        "selection_count": len(selected_records),
        "selection_sha256": selection_hash,
    }
    if index_path.exists():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        if previous.get("dataset") != config.data.dataset_id:
            raise ValueError("Existing cache belongs to a different dataset")
        if previous.get("revision") != config.data.dataset_revision:
            raise ValueError("Existing cache belongs to a different dataset revision")
        if previous.get("representation") != expected_representation:
            raise ValueError("Existing cache belongs to a different representation")
        shard_index: list[dict[str, Any]] = list(previous.get("shards", []))
        index_statistics: dict[str, Any] = dict(previous.get("statistics", {}))
    else:
        shard_index = []
        index_statistics = {}
    initial_examples = sum(int(shard["count"]) for shard in shard_index)
    completed_ids: set[str] = set()
    for shard in shard_index:
        metadata_path = output / shard["metadata"]
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for existing shard: {metadata_path}")
        completed_ids.update(str(item["id"]) for item in _read_metadata(metadata_path))
    buffers: dict[str, list[Any]] = defaultdict(list)

    def write_index() -> None:
        payload = {
            "dataset": config.data.dataset_id,
            "revision": config.data.dataset_revision,
            "representation": expected_representation,
            "shards": shard_index,
            "statistics": index_statistics,
        }
        temporary = index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(index_path)

    def flush(bucket: str) -> None:
        if not buffers[f"{bucket}:metadata"]:
            return
        split = bucket_split[bucket]
        metadata = buffers[f"{bucket}:metadata"]
        if len(metadata) != bucket_expected[bucket]:
            raise RuntimeError(
                f"Refusing to flush incomplete cache bucket {bucket}: "
                f"{len(metadata)} != {bucket_expected[bucket]}"
            )
        order = sorted(
            range(len(metadata)),
            key=lambda index: int(metadata[index]["row_index"]),
        )
        first_row_index = min(
            int(item["row_index"]) for item in metadata
        )
        stem = f"{split}-{bucket_number[bucket]:05d}"
        tensor_keys = ["latent_mean", "latent_logvar", "text_hidden", "text_mask", "pooled"]
        if buffers[f"{bucket}:repa_target"]:
            tensor_keys.append("repa_target")
        tensors = {
            key: torch.cat(
                [buffers[f"{bucket}:{key}"][index] for index in order],
                dim=0,
            )
            .contiguous()
            .cpu()
            for key in tensor_keys
        }
        tensor_path = output / f"{stem}.safetensors"
        metadata_path = output / f"{stem}.parquet"
        temporary_tensor = tensor_path.with_suffix(".safetensors.tmp")
        temporary_metadata = metadata_path.with_suffix(".parquet.tmp")
        save_file(tensors, str(temporary_tensor))
        temporary_tensor.replace(tensor_path)
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("Precomputation requires `pyarrow` metadata support") from error
        parquet.write_table(
            pa.Table.from_pylist([metadata[index] for index in order]),
            temporary_metadata,
            compression="zstd",
        )
        temporary_metadata.replace(metadata_path)
        shard_index.append(
            {
                "split": split,
                "count": len(metadata),
                "file": tensor_path.name,
                "metadata": metadata_path.name,
                "sha256": sha256_file(tensor_path),
                "metadata_sha256": sha256_file(metadata_path),
                "first_row_index": first_row_index,
            }
        )
        if all("first_row_index" in item for item in shard_index):
            shard_index.sort(key=lambda item: int(item["first_row_index"]))
        write_index()
        for key in tensor_keys + ["metadata"]:
            buffers[f"{bucket}:{key}"].clear()

    raw_root = _raw_cache_root(config)
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_index = raw_root / "index.json"
    raw_contract = {
        "dataset": config.data.dataset_id,
        "revision": config.data.dataset_revision,
        "image_size": config.data.image_size,
        "preprocessing": IMAGE_PREPROCESSING,
    }
    if raw_index.exists():
        if json.loads(raw_index.read_text(encoding="utf-8")) != raw_contract:
            raise ValueError(f"Raw image cache contract mismatch: {raw_index}")
    else:
        temporary = raw_index.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(raw_contract, indent=2), encoding="utf-8")
        temporary.replace(raw_index)

    pending_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    scanned_rows = 0
    materialized_rows = 0
    encoded_rows = 0
    encoding_seconds = 0.0

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def write_progress(stage: str) -> None:
        elapsed = time.monotonic() - precompute_started
        payload = {
            "stage": stage,
            "scanned_rows": scanned_rows,
            "selected_examples": len(selected_records),
            "materialized_rows": materialized_rows,
            "encoded_rows_this_run": encoded_rows,
            "already_completed_rows": initial_examples,
            "elapsed_seconds": elapsed,
            "encoded_examples_per_second": encoded_rows / max(elapsed, 1e-8),
            "pure_encoder_examples_per_second": (
                encoded_rows / max(encoding_seconds, 1e-8)
            ),
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    def queue(row: dict[str, Any], record: dict[str, Any]) -> None:
        nonlocal encoded_rows, encoding_seconds
        if str(record["id"]) in completed_ids:
            return
        bucket = bucket_for_id[str(record["id"])]
        encoded_record = dict(record)
        encoded_record["_cache_bucket"] = bucket
        pending_rows.append((row, encoded_record))
        if len(pending_rows) < batch_size:
            return
        count = len(pending_rows)
        affected_buckets = {
            str(item["_cache_bucket"]) for _, item in pending_rows
        }
        synchronize()
        encoding_started = time.perf_counter()
        _encode_pending(pending_rows, buffers, vae, text_encoder, repa_encoder, config, device, dtype)
        synchronize()
        encoding_seconds += time.perf_counter() - encoding_started
        encoded_rows += count
        pending_rows.clear()
        for affected_bucket in affected_buckets:
            if (
                len(buffers[f"{affected_bucket}:metadata"])
                == bucket_expected[affected_bucket]
            ):
                flush(affected_bucket)

    missing_raw = [
        record
        for record in selected_records
        if not _raw_image_path(raw_root, int(record["row_index"])).exists()
    ]
    if missing_raw and materialize_workers > 1:
        completed_shards = 0

        def materialization_progress(
            observed: int, newly_materialized: int, total_shards: int
        ) -> None:
            nonlocal completed_shards
            completed_shards += 1
            write_progress(
                f"parallel_materialization:{completed_shards}/{total_shards}"
            )

        def consume_materialized(
            shard_records: Sequence[dict[str, Any]],
        ) -> None:
            nonlocal scanned_rows, materialized_rows
            for record in shard_records:
                scanned_rows += 1
                materialized_rows += 1
                raw_path = _raw_image_path(raw_root, int(record["row_index"]))
                if not raw_path.exists():
                    raise RuntimeError(
                        f"Parallel materialization missed Photonyx row "
                        f"{record['row_index']}"
                    )
                queue({"image": _load_raw_image(raw_path)}, record)
                if scanned_rows % 1_000 == 0:
                    write_progress("parallel_materialization_and_encoding")
            write_progress("parallel_materialization_and_encoding")

        _materialize_selected_parallel(
            selected_records,
            Path(manifest_path),
            raw_root,
            config.data.image_size,
            materialize_workers,
            materialization_progress,
            consume_materialized,
        )
    elif missing_raw:
        last_selected_row = max(records) if records else -1
        photonyx = _disable_streaming_image_decode(
            _load_photonyx(config.data)
        )
        for row_index, row in enumerate(photonyx):
            scanned_rows = row_index + 1
            if row_index > last_selected_row:
                break
            record = records.get(row_index)
            if record is None:
                if scanned_rows % 1_000 == 0:
                    write_progress("streaming_and_encoding")
                continue
            validate_streamed_record(row_index, row, record)
            raw_path = _raw_image_path(raw_root, row_index)
            if not raw_path.exists():
                _save_raw_image(
                    _decode_audit_image(row["image"]),
                    raw_path,
                    config.data.image_size,
                )
            materialized_rows += 1
            queue({"image": _load_raw_image(raw_path)}, record)
            if scanned_rows % 1_000 == 0:
                write_progress("streaming_and_encoding")
    else:
        for selected_index, record in enumerate(selected_records, start=1):
            scanned_rows = selected_index
            materialized_rows = selected_index
            raw_path = _raw_image_path(raw_root, int(record["row_index"]))
            queue({"image": _load_raw_image(raw_path)}, record)
            if selected_index % 1_000 == 0:
                write_progress("encoding_raw_cache")
    if pending_rows:
        count = len(pending_rows)
        affected_buckets = {
            str(item["_cache_bucket"]) for _, item in pending_rows
        }
        synchronize()
        encoding_started = time.perf_counter()
        _encode_pending(pending_rows, buffers, vae, text_encoder, repa_encoder, config, device, dtype)
        synchronize()
        encoding_seconds += time.perf_counter() - encoding_started
        encoded_rows += count
        pending_rows.clear()
        for affected_bucket in affected_buckets:
            if (
                len(buffers[f"{affected_bucket}:metadata"])
                == bucket_expected[affected_bucket]
            ):
                flush(affected_bucket)
    for bucket in sorted(bucket_expected):
        if buffers[f"{bucket}:metadata"]:
            flush(bucket)
    total_examples = sum(int(shard["count"]) for shard in shard_index)
    total_storage_bytes = sum(
        (output / shard["file"]).stat().st_size
        + (output / shard["metadata"]).stat().st_size
        for shard in shard_index
    )
    elapsed = time.monotonic() - precompute_started
    new_examples = total_examples - initial_examples
    index_statistics.update(
        {
            "total_examples": total_examples,
            "new_examples_this_run": new_examples,
            "elapsed_seconds_this_run": elapsed,
            "new_examples_per_second": new_examples / max(elapsed, 1e-8),
            "pure_encoder_examples_per_second": (
                new_examples / max(encoding_seconds, 1e-8)
            ),
            "pure_encoding_seconds_this_run": encoding_seconds,
            "encoder_load_seconds_this_run": encoder_load_seconds,
            "total_storage_bytes": total_storage_bytes,
            "bytes_per_example": (
                total_storage_bytes / total_examples if total_examples else 0.0
            ),
        }
    )
    write_index()
    write_progress("complete")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _encode_pending(
    pending: list[tuple[dict[str, Any], dict[str, Any]]],
    buffers: dict[str, list[Any]],
    vae: VAEAdapter,
    text_encoder: Any,
    repa_encoder: DINORepresentationEncoder | None,
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    images = torch.stack([image_to_tensor(row["image"], config.data.image_size) for row, _ in pending])
    prompts = [record["prompt"] for _, record in pending]
    mean, logvar = vae.encode(images)
    text = text_encoder.encode(prompts, device, dtype)
    assert text.pooled is not None
    if config.model.conditioning == "pooled":
        stored_hidden = text.pooled[:, None, :]
        stored_mask = torch.ones(
            (len(pending), 1),
            dtype=torch.bool,
            device=text.pooled.device,
        )
    else:
        stored_hidden = text.hidden_states
        stored_mask = text.attention_mask
    repa = (
        repa_encoder.encode([row["image"].convert("RGB") for row, _ in pending], dtype)
        if repa_encoder is not None
        else None
    )
    for index, (_, record) in enumerate(pending):
        bucket = str(record.get("_cache_bucket", record["split"]))
        buffers[f"{bucket}:latent_mean"].append(mean[index : index + 1].cpu())
        buffers[f"{bucket}:latent_logvar"].append(logvar[index : index + 1].cpu())
        buffers[f"{bucket}:text_hidden"].append(stored_hidden[index : index + 1].cpu())
        buffers[f"{bucket}:text_mask"].append(stored_mask[index : index + 1].cpu())
        buffers[f"{bucket}:pooled"].append(text.pooled[index : index + 1].cpu())
        if repa is not None:
            buffers[f"{bucket}:repa_target"].append(repa[index : index + 1].cpu())
        buffers[f"{bucket}:metadata"].append(
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "source": record["source"],
                "license": record["license"],
                "row_index": record["row_index"],
            }
        )
