from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch
from PIL import Image

from hemera.benchmarks import (
    chunked_nearest_neighbors,
    clip_cmmd,
    hpsv2_scores,
    nearest_neighbors,
)
from hemera.evaluation import (
    analyze_human_study,
    cmmd,
    compare_confirmation_benchmarks,
    compare_paired_benchmarks,
    create_blinded_human_study,
    create_blinded_human_study_from_manifests,
    percentile_selection_scores,
)
from hemera.runtime import sha256_file
from hemera.protocols import human_study_prompts, safety_bias_prompts
from hemera.representation_probe import (
    _retrieval_metrics,
    _stratified_probe_indices,
)


def test_cmmd_prefers_identical_features() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(32, 16, generator=generator)
    shifted = features + 2 * torch.randn(32, 16, generator=generator)
    assert cmmd(features, features) < cmmd(features, shifted)


def test_cmmd_matches_official_biased_scaled_formula() -> None:
    real = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    generated = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    bandwidth = 2.0

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distances = (left[:, None] - right[None]).square().sum(-1)
        return torch.exp(-distances / (2 * bandwidth**2))

    expected = 1000 * (
        kernel(real, real).mean()
        + kernel(generated, generated).mean()
        - 2 * kernel(real, generated).mean()
    )
    assert cmmd(real, generated, bandwidth) == pytest.approx(float(expected))


def test_clip_cmmd_uses_l2_normalized_embeddings() -> None:
    generator = torch.Generator().manual_seed(29)
    real = torch.randn(16, 8, generator=generator)
    generated = torch.randn(16, 8, generator=generator)
    scaled_real = real * torch.linspace(0.5, 3.0, len(real))[:, None]
    scaled_generated = generated * torch.linspace(4.0, 1.0, len(generated))[:, None]
    assert clip_cmmd(scaled_real, scaled_generated) == pytest.approx(
        clip_cmmd(real, generated), abs=1e-5
    )


def test_chunked_nearest_neighbors_matches_dense() -> None:
    generator = torch.Generator().manual_seed(37)
    generated = torch.randn(7, 11, generator=generator)
    training = torch.randn(23, 11, generator=generator)
    generated_ids = [f"g-{index}" for index in range(len(generated))]
    training_ids = [f"t-{index}" for index in range(len(training))]
    dense = nearest_neighbors(
        generated, training, generated_ids, training_ids, top_k=4
    )
    chunked = chunked_nearest_neighbors(
        generated,
        training,
        generated_ids,
        training_ids,
        top_k=4,
        chunk_size=5,
    )
    for dense_row, chunked_row in zip(dense, chunked, strict=True):
        assert chunked_row["generated_id"] == dense_row["generated_id"]
        assert [
            item["training_id"] for item in chunked_row["neighbors"]
        ] == [item["training_id"] for item in dense_row["neighbors"]]
        assert [
            item["cosine_similarity"] for item in chunked_row["neighbors"]
        ] == pytest.approx(
            [item["cosine_similarity"] for item in dense_row["neighbors"]]
        )


def test_selection_uses_vram_after_quality_and_throughput_tie() -> None:
    rows = [
        {
            "name": "larger",
            "cmmd": 1.0,
            "hpsv2": 0.2,
            "clip_score": 0.3,
            "images_per_second": 10.0,
            "peak_vram_bytes": 200,
        },
        {
            "name": "smaller",
            "cmmd": 1.0,
            "hpsv2": 0.2,
            "clip_score": 0.3,
            "images_per_second": 10.0,
            "peak_vram_bytes": 100,
        },
    ]
    assert percentile_selection_scores(rows)[0]["name"] == "smaller"


def test_paired_benchmark_bootstrap_detects_consistent_improvement(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(41)
    real = torch.nn.functional.normalize(
        torch.randn(32, 16, generator=generator), dim=-1
    )
    baseline_generated = torch.nn.functional.normalize(
        real + 0.8 * torch.randn(32, 16, generator=generator), dim=-1
    )
    candidate_generated = torch.nn.functional.normalize(
        real + 0.05 * torch.randn(32, 16, generator=generator), dim=-1
    )
    row_ids = [f"row-{index}" for index in range(len(real))]

    def write(name: str, generated: torch.Tensor, score: float) -> Path:
        feature_path = tmp_path / f"{name}.features.safetensors"
        save_file(
            {
                "real_clip": real,
                "generated_clip": generated,
                "text_clip": real.clone(),
                "clip_alignment": torch.full((len(real),), score),
                "hpsv2": torch.full((len(real),), score),
            },
            str(feature_path),
        )
        metrics_path = tmp_path / f"{name}.json"
        metrics_path.write_text(
            __import__("json").dumps(
                {
                    "cmmd_estimator": {"bandwidth": 10.0},
                    "paired_features": {
                        "path": str(feature_path),
                        "sha256": sha256_file(feature_path),
                        "row_ids": row_ids,
                    },
                }
            ),
            encoding="utf-8",
        )
        return metrics_path

    baseline = write("baseline", baseline_generated, 0.1)
    candidate = write("candidate", candidate_generated, 0.3)
    result = compare_paired_benchmarks(
        candidate,
        baseline,
        tmp_path / "comparison.json",
        bootstrap_samples=200,
        seed=9,
    )
    assert result["candidate_beats_baseline"]
    assert result["positive_at_95_confidence"]


def test_confirmation_bootstrap_clusters_seeds_by_prompt(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file
    import json

    generator = torch.Generator().manual_seed(43)
    real = torch.nn.functional.normalize(
        torch.randn(20, 8, generator=generator), dim=-1
    )
    ids = [f"prompt-{index}" for index in range(len(real))]

    def write(name: str, generated: torch.Tensor, score: float) -> Path:
        feature_path = tmp_path / f"{name}.safetensors"
        save_file(
            {
                "real_clip": real,
                "generated_clip": generated,
                "text_clip": real.clone(),
                "clip_alignment": torch.full((len(real),), score),
                "hpsv2": torch.full((len(real),), score),
            },
            str(feature_path),
        )
        metrics = tmp_path / f"{name}.json"
        metrics.write_text(
            json.dumps(
                {
                    "cmmd_estimator": {"bandwidth": 10.0},
                    "paired_features": {
                        "path": str(feature_path),
                        "sha256": sha256_file(feature_path),
                        "row_ids": ids,
                    },
                }
            ),
            encoding="utf-8",
        )
        return metrics

    candidates, baselines = [], []
    for seed in range(3):
        candidates.append(
            write(
                f"candidate-{seed}",
                torch.nn.functional.normalize(
                    real
                    + 0.02
                    * torch.randn(real.shape, generator=generator),
                    dim=-1,
                ),
                0.3,
            )
        )
        baselines.append(
            write(
                f"baseline-{seed}",
                torch.nn.functional.normalize(
                    real
                    + 0.8
                    * torch.randn(real.shape, generator=generator),
                    dim=-1,
                ),
                0.1,
            )
        )
    result = compare_confirmation_benchmarks(
        candidates,
        baselines,
        tmp_path / "confirmation.json",
        bootstrap_samples=200,
        seed=5,
    )
    assert result["candidate_beats_baseline_in_every_seed"]
    assert result["aggregate_positive_at_95_confidence"]
    assert result["confirmation_gate_passed"]


def test_ridge_retrieval_probe_recovers_shared_semantics() -> None:
    generator = torch.Generator().manual_seed(19)
    text = torch.randn(100, 12, generator=generator)
    projection = torch.randn(12, 16, generator=generator)
    image = text @ projection
    metrics = _retrieval_metrics(text, image, [f"id-{index}" for index in range(100)])
    assert metrics["recall_at_1"] > 0.9
    assert metrics["mean_reciprocal_rank"] > 0.9


def test_representation_probe_sampling_is_source_stratified_and_stable() -> None:
    groups = {
        "large": list(range(90)),
        "medium": list(range(90, 99)),
        "rare": [99],
    }
    first = _stratified_probe_indices(groups, 20)
    second = _stratified_probe_indices(groups, 20)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert 99 in first
    assert any(index >= 90 for index in first)


def test_frozen_prompt_protocols_are_balanced_and_unique() -> None:
    human = human_study_prompts()
    safety = safety_bias_prompts()
    assert len(human) == 200
    assert len(safety) == 100
    assert len({row["id"] for row in human}) == 200
    assert len({row["prompt"] for row in human}) == 200
    human_counts = __import__("collections").Counter(
        row["category"] for row in human
    )
    safety_counts = __import__("collections").Counter(
        row["category"] for row in safety
    )
    assert set(human_counts.values()) == {20}
    assert set(safety_counts.values()) == {10}


def test_hpsv2_scores_pairs_each_image_with_its_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for index, value in enumerate((1, 2)):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), (value, 0, 0)).save(path)
        paths.append(str(path))

    class FakeModel:
        def load_state_dict(self, state):
            assert state == {}

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, images, tokens):
            return {
                "image_features": images.flatten(1),
                "text_features": tokens.float(),
            }

    model_dict = {}
    img_score = types.ModuleType("hpsv2.img_score")
    img_score.model_dict = model_dict

    def initialize_model():
        model_dict.update(
            {
                "model": FakeModel(),
                "preprocess_val": lambda image: torch.tensor(
                    [float(image.getpixel((0, 0))[0])]
                ),
            }
        )

    img_score.initialize_model = initialize_model
    hps = types.ModuleType("hpsv2")
    hps.img_score = img_score
    open_clip = types.ModuleType("hpsv2.src.open_clip")
    open_clip.get_tokenizer = lambda _: (
        lambda prompts: torch.tensor([[len(prompt)] for prompt in prompts])
    )
    utils = types.ModuleType("hpsv2.utils")
    utils.hps_version_map = {"v2.1": "checkpoint.pt"}
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *_args, **_kwargs: "checkpoint.pt"
    monkeypatch.setitem(sys.modules, "hpsv2", hps)
    monkeypatch.setitem(sys.modules, "hpsv2.img_score", img_score)
    monkeypatch.setitem(sys.modules, "hpsv2.src.open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "hpsv2.utils", utils)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {"state_dict": {}})

    scores = hpsv2_scores(
        paths, ["aa", "bbb"], batch_size=2, device=torch.device("cpu")
    )
    torch.testing.assert_close(scores, torch.tensor([2.0, 6.0]))
    assert model_dict == {}


def test_selection_score_and_human_study(tmp_path: Path) -> None:
    ranked = percentile_selection_scores(
        [
            {"name": "a", "cmmd": 0.1, "hpsv2": 0.4, "clip_score": 0.3, "images_per_second": 10},
            {"name": "b", "cmmd": 0.2, "hpsv2": 0.2, "clip_score": 0.1, "images_per_second": 20},
        ]
    )
    assert ranked[0]["name"] == "a"

    path = tmp_path / "study.csv"
    create_blinded_human_study(["p1", "p2"], ["a1", "a2"], ["b1", "b2"], path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(',,,', ',left,rater,', 1)
    # Parsing an unrated study remains well-defined.
    result = analyze_human_study(path)
    assert result["decisive_votes"] == 0


def test_human_study_manifest_alignment_is_enforced(tmp_path: Path) -> None:
    import json

    def write(name: str, prompt: str) -> Path:
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "shared",
                    "prompt": prompt,
                    "category": "test",
                    "generation_index": 0,
                    "image": f"{name}.png",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    model_a = write("a", "same prompt")
    model_b = write("b", "same prompt")
    study = tmp_path / "study.csv"
    create_blinded_human_study_from_manifests(model_a, model_b, study)
    assert "category" in study.read_text(encoding="utf-8").splitlines()[0]
    write("b", "different prompt")
    with pytest.raises(ValueError, match="Prompt mismatch"):
        create_blinded_human_study_from_manifests(model_a, model_b, study)


def test_human_study_manifests_can_rebase_remote_image_paths(
    tmp_path: Path,
) -> None:
    import csv
    import json

    def write_manifest(name: str) -> tuple[Path, Path]:
        image_root = tmp_path / f"{name}-images"
        image_root.mkdir()
        local_image = image_root / f"{name}.png"
        local_image.write_bytes(b"image")
        manifest = tmp_path / f"{name}.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "id": "shared",
                    "prompt": "same prompt",
                    "category": "test",
                    "generation_index": 0,
                    "image": f"/remote/machine/{name}.png",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest, image_root

    model_a, root_a = write_manifest("a")
    model_b, root_b = write_manifest("b")
    study = tmp_path / "rebased.csv"
    create_blinded_human_study_from_manifests(
        model_a,
        model_b,
        study,
        model_a_image_root=root_a,
        model_b_image_root=root_b,
    )
    rows = list(csv.DictReader(study.open(encoding="utf-8")))
    paths = {
        Path(row[field])
        for row in rows
        for field in ("left_image", "right_image")
    }
    assert paths == {(root_a / "a.png").resolve(), (root_b / "b.png").resolve()}
