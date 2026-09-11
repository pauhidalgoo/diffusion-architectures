from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from PIL import Image

from hemera.config import ModelConfig, config_from_dict, load_config
from hemera.data import (
    ShardedLatentDataset,
    SyntheticLatentDataset,
    average_hash,
    collate_examples,
    deterministic_indices,
    deterministic_dataset_indices,
    image_to_tensor,
    license_allowed,
    prepare_square_image,
    _decode_audit_image,
    _records_by_parquet_shard,
    _resume_shard_position,
    split_for_cluster,
    PerceptualDuplicateIndex,
    audit_photonyx,
    source_groups,
    select_manifest_subset,
    validate_streamed_record,
    verify_audit_manifest,
    _encode_pending,
)
from hemera.runtime import sha256_file, source_tree_sha256
from hemera.interfaces import TextCondition


def test_smoke_config_loads() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.name == "hemera-smoke"
    assert config.model.backbone == "dit"
    assert config.train.dataset_mode == "synthetic"


def test_extended_representation_config_loads() -> None:
    config = load_config("configs/representations/dc-t5.yaml")
    assert config.data.latent_channels == 32
    assert config.model.in_channels == 32
    assert config.data.text_length == 128
    assert config.representation.text_encoder_type == "t5"
    assert not config.budget.save_milestones


def test_only_final_config_retains_budget_milestones() -> None:
    final = load_config("configs/final-hemera-nano.yaml")
    assert final.budget.save_milestones
    assert final.budget.reserve_minutes == 30


def test_pooled_final_cache_compacts_unused_token_states() -> None:
    config = load_config("configs/final-hemera-nano.yaml")

    class VAE:
        def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = images.shape[0]
            shape = (batch, config.data.latent_channels, 2, 2)
            return torch.zeros(shape), torch.full(shape, -30.0)

    class TextEncoder:
        def encode(
            self,
            prompts: list[str],
            device: torch.device,
            dtype: torch.dtype,
        ) -> TextCondition:
            batch = len(prompts)
            hidden = torch.randn(batch, 7, config.data.text_dim)
            pooled = torch.randn(batch, config.data.text_dim)
            return TextCondition(
                hidden,
                torch.ones(batch, 7, dtype=torch.bool),
                pooled,
            )

    pending = [
        (
            {"image": Image.new("RGB", (16, 16), (index, 0, 0))},
            {
                "id": f"id-{index}",
                "prompt": f"prompt {index}",
                "source": "fixture",
                "license": "cc0",
                "row_index": index,
                "split": "train",
            },
        )
        for index in range(2)
    ]
    from collections import defaultdict

    buffers: defaultdict[str, list[object]] = defaultdict(list)
    _encode_pending(
        pending,
        buffers,
        VAE(),
        TextEncoder(),
        None,
        config,
        torch.device("cpu"),
        torch.float32,
    )
    hidden = torch.cat(buffers["train:text_hidden"])
    mask = torch.cat(buffers["train:text_mask"])
    pooled = torch.cat(buffers["train:pooled"])
    assert hidden.shape == (2, 1, config.data.text_dim)
    assert mask.shape == (2, 1)
    torch.testing.assert_close(hidden[:, 0], pooled)


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown ModelConfig"):
        config_from_dict({"model": {"backbone": "dit", "mystery": True}})


def test_nested_config_inheritance(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    middle = tmp_path / "middle.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "train:\n  batch_size: 4\n  effective_batch_size: 8\n",
        encoding="utf-8",
    )
    middle.write_text(
        "extends: base.yaml\ntrain:\n  batch_size: 8\n",
        encoding="utf-8",
    )
    child.write_text("extends: middle.yaml\nname: nested\n", encoding="utf-8")
    config = load_config(child)
    assert config.name == "nested"
    assert config.train.batch_size == 8
    assert config.train.effective_batch_size == 8


def test_manifest_subset_is_deterministic_and_stratified() -> None:
    records = [
        {
            "row_index": index,
            "id": f"id-{index}",
            "source": "large" if index < 90 else "rare",
            "split": ("test" if index % 10 == 0 else "train"),
        }
        for index in range(100)
    ]
    first = select_manifest_subset(records, 20)
    second = select_manifest_subset(list(reversed(records)), 20)
    assert first == second
    assert len(first) == 20
    observed = {(item["source"], item["split"]) for item in first}
    expected = {(item["source"], item["split"]) for item in records}
    assert observed == expected


def test_source_tree_hash_changes_with_runnable_source(tmp_path: Path) -> None:
    package = tmp_path / "hemera"
    package.mkdir()
    source = package / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first


def test_data_audit_resumes_from_atomic_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "image": Image.new("RGB", (16, 16), (20, 40, 80)),
        "prompt": "a blue square",
        "license": "Public Domain",
        "image_id": "one",
        "source": "fixture",
    }
    monkeypatch.setattr("hemera.data._load_photonyx", lambda _: iter([row]))
    config = config_from_dict({"data": {"max_items": 1}})
    manifest = tmp_path / "manifest.jsonl"
    first = audit_photonyx(config, manifest)
    second = audit_photonyx(config, manifest)
    assert first["accepted"] == 1
    assert second["accepted"] == 1
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1
    assert second["manifest_sha256"] == sha256_file(manifest)


def test_resume_shard_position_handles_irregular_shards_and_boundaries() -> None:
    lengths = [5_000, 5_000, 2_699, 5_000, 1_541]
    assert _resume_shard_position(lengths, 0) == (0, 0)
    assert _resume_shard_position(lengths, 5_000) == (1, 0)
    assert _resume_shard_position(lengths, 11_000) == (2, 1_000)
    assert _resume_shard_position(lengths, sum(lengths)) == (len(lengths), 0)


def test_audit_image_decoder_accepts_encoded_bytes() -> None:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (7, 5), (10, 20, 30)).save(buffer, format="PNG")
    decoded = _decode_audit_image({"bytes": buffer.getvalue(), "path": None})
    assert decoded.size == (7, 5)
    assert decoded.getpixel((0, 0)) == (10, 20, 30)


def test_audit_rejects_none_prompt_and_replaces_missing_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "image": Image.new("RGB", (8, 8), (1, 2, 3)),
            "prompt": None,
            "license": "CC0",
            "image_id": "bad-prompt",
            "source": "fixture",
        },
        {
            "image": Image.new("RGB", (8, 8), (10, 20, 30)),
            "prompt": "valid",
            "license": "CC0-1.0",
            "image_id": None,
            "source": "fixture",
        },
    ]
    monkeypatch.setattr("hemera.data._load_photonyx", lambda _: iter(rows))
    config = config_from_dict({"data": {"max_items": 2}})
    manifest = tmp_path / "manifest.jsonl"
    result = audit_photonyx(config, manifest)
    record = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    assert result["accepted"] == 1
    assert result["counts"]["empty_prompt"] == 1
    assert result["counts"]["missing_id_replaced"] == 1
    assert record["id"] == "row-000000001"


def test_precompute_identity_guard_rejects_image_caption_misalignment() -> None:
    record = {"id": "expected", "prompt": "a matching caption"}
    validate_streamed_record(
        7,
        {"image_id": "expected", "prompt": " a matching caption "},
        record,
    )
    with pytest.raises(RuntimeError, match="ID changed"):
        validate_streamed_record(
            7,
            {"image_id": "different", "prompt": "a matching caption"},
            record,
        )
    with pytest.raises(RuntimeError, match="prompt changed"):
        validate_streamed_record(
            7,
            {"image_id": "expected", "prompt": "wrong caption"},
            record,
        )


def test_selected_records_are_partitioned_by_global_parquet_offsets() -> None:
    records = [
        {"row_index": 0},
        {"row_index": 4},
        {"row_index": 5},
        {"row_index": 11},
    ]
    assignments = _records_by_parquet_shard(records, [5, 3, 4])
    assert [(index, offset) for index, offset, _ in assignments] == [
        (0, 0),
        (1, 5),
        (2, 8),
    ]
    assert [
        [record["row_index"] for record in shard_records]
        for _, _, shard_records in assignments
    ] == [[0, 4], [5], [11]]
    with pytest.raises(ValueError, match="outside"):
        _records_by_parquet_shard([{"row_index": 12}], [5, 3, 4])


def test_audit_verifier_reports_source_split_and_rejects_bad_split(
    tmp_path: Path,
) -> None:
    import json

    phash = "0123456789abcdef"
    row = {
        "row_index": 4,
        "id": "fixture",
        "prompt": "a valid prompt",
        "source": "fixture-source",
        "license": "CC0-1.0",
        "phash": phash,
        "split": split_for_cluster(phash),
    }
    manifest = tmp_path / "audit.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = verify_audit_manifest(
        config_from_dict({}), manifest, tmp_path / "verification.json"
    )
    assert result["valid"]
    assert result["by_source_and_split"]["fixture-source"][row["split"]] == 1
    row["split"] = "invalid"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split policy"):
        verify_audit_manifest(
            config_from_dict({}), manifest, tmp_path / "invalid.json"
        )


def test_audit_verifier_rejects_near_duplicate_phashes(tmp_path: Path) -> None:
    import json

    rows = []
    for index, phash in enumerate(
        ("0000000000000000", "000000000000000f")
    ):
        rows.append(
            {
                "row_index": index,
                "id": f"id-{index}",
                "prompt": "valid",
                "source": "fixture",
                "license": "CC0",
                "phash": phash,
                "split": split_for_cluster(phash),
            }
        )
    manifest = tmp_path / "near.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Hamming distance"):
        verify_audit_manifest(
            config_from_dict({}), manifest, tmp_path / "invalid.json"
        )
def test_invalid_head_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        config_from_dict({"model": {"hidden_size": 31, "heads": 4}})


def test_synthetic_collation_and_deterministic_indices() -> None:
    dataset = SyntheticLatentDataset(16, 4, 8, 6, 32, seed=9)
    first = deterministic_indices(len(dataset), 4, step=3, seed=11)
    second = deterministic_indices(len(dataset), 4, step=3, seed=11)
    assert first == second
    batch = collate_examples([dataset[index] for index in first], sample_posterior=False)
    assert batch["latents"].shape == (4, 4, 8, 8)
    assert batch["condition"].hidden_states.shape == (4, 6, 32)

def test_text_condition_index_select_keeps_prompt_tensors_together() -> None:
    condition = TextCondition(
        hidden_states=torch.arange(24).reshape(3, 2, 4),
        attention_mask=torch.tensor(
            [[True, False], [True, True], [False, True]]
        ),
        pooled=torch.arange(12).reshape(3, 4),
    )
    indices = torch.tensor([2, 0, 1])
    selected = condition.index_select(indices)
    torch.testing.assert_close(selected.hidden_states[0], condition.hidden_states[2])
    torch.testing.assert_close(selected.attention_mask[1], condition.attention_mask[0])
    torch.testing.assert_close(selected.pooled[2], condition.pooled[1])


def test_average_hash_and_split_are_stable() -> None:
    image = Image.new("RGB", (32, 32), (120, 30, 200))
    cluster = average_hash(image)
    assert cluster == average_hash(image.copy())
    assert split_for_cluster(cluster) == split_for_cluster(cluster)


@pytest.mark.parametrize("license_name", ["CC0", "cc0-1.0", "CC0 1.0"])
def test_versioned_cc0_licenses_are_allowed(license_name: str) -> None:
    assert license_allowed(license_name, ["cc0"])


@pytest.mark.parametrize(
    "license_name",
    [
        "CC-BY-NC-4.0",
        "CC-BY-ND-4.0",
        "CC BY Non Commercial 4.0",
        "CC BY No Derivatives 4.0",
        None,
    ],
)
def test_restricted_or_missing_licenses_are_rejected(license_name: str | None) -> None:
    assert not license_allowed(license_name, ["cc0", "cc-by"])


def test_image_preprocessing_preserves_aspect_ratio_with_center_crop() -> None:
    image = Image.new("RGB", (80, 40), (255, 0, 0))
    for x in range(20, 60):
        for y in range(40):
            image.putpixel((x, y), (0, 255, 0))
    prepared = prepare_square_image(image, 20)
    assert prepared.size == (20, 20)
    pixels = torch.from_numpy(__import__("numpy").asarray(prepared).copy())
    assert float((pixels[..., 1] > pixels[..., 0]).float().mean()) > 0.9
    tensor = image_to_tensor(image, 20)
    assert tensor.shape == (3, 20, 20)
    assert tensor.min() >= -1 and tensor.max() <= 1


def test_duplicate_index_clusters_close_hashes() -> None:
    index = PerceptualDuplicateIndex(max_distance=4)
    representative, duplicate = index.find_or_add("0000000000000000")
    assert not duplicate
    matched, duplicate = index.find_or_add("000000000000000f")
    assert duplicate
    assert matched == representative
    distributed = PerceptualDuplicateIndex(max_distance=4)
    distributed.find_or_add("0000000000000000")
    four_chunks = (1 << 0) | (1 << 13) | (1 << 26) | (1 << 39)
    _, duplicate = distributed.find_or_add(f"{four_chunks:016x}")
    assert duplicate


def test_sharded_source_groups_read_metadata_without_loading_tensors(tmp_path) -> None:
    metadata = tmp_path / "train-00000.jsonl"
    metadata.write_text(
        '{"id":"a","source":"museum"}\n'
        '{"id":"b","source":"synthetic"}\n'
        '{"id":"c","source":"museum"}\n',
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text(
        '{"shards":[{"split":"train","count":3,'
        '"file":"deliberately-missing.safetensors","metadata":"train-00000.jsonl"}]}',
        encoding="utf-8",
    )
    dataset = ShardedLatentDataset(tmp_path)
    assert source_groups(dataset) == {"museum": [0, 2], "synthetic": [1]}
    first = deterministic_dataset_indices(dataset, 4, 7, 11, 0.5)
    second = deterministic_dataset_indices(dataset, 4, 7, 11, 0.5)
    assert first == second
    assert len({dataset._locate(index)[1]["file"] for index in first}) == 1


def test_parquet_safetensors_shard_is_checksum_verified(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    safetensors = pytest.importorskip("safetensors.torch")
    tensors_path = tmp_path / "train-00000.safetensors"
    metadata_path = tmp_path / "train-00000.parquet"
    safetensors.save_file(
        {
            "latent_mean": torch.zeros(2, 4, 2, 2),
            "latent_logvar": torch.full((2, 4, 2, 2), -30.0),
            "text_hidden": torch.zeros(2, 3, 4),
            "text_mask": torch.ones(2, 3, dtype=torch.bool),
            "pooled": torch.zeros(2, 4),
        },
        tensors_path,
    )
    parquet.write_table(
        pa.Table.from_pylist(
            [{"id": "a", "source": "x"}, {"id": "b", "source": "y"}]
        ),
        metadata_path,
    )
    (tmp_path / "index.json").write_text(
        __import__("json").dumps(
            {
                "shards": [
                    {
                        "split": "train",
                        "count": 2,
                        "file": tensors_path.name,
                        "metadata": metadata_path.name,
                        "sha256": sha256_file(tensors_path),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = ShardedLatentDataset(tmp_path)
    assert dataset[1]["id"] == "b"
    assert dataset[1]["latent_mean"].shape == (4, 2, 2)
