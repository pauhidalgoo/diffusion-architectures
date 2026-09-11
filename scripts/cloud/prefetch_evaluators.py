"""Download final-evaluation weights without taking memory from the trainer."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    for model_id in (
        "openai/clip-vit-large-patch14-336",
        "google/siglip-base-patch16-256",
    ):
        path = snapshot_download(model_id)
        print(f"cached {model_id}: {path}", flush=True)

    from hpsv2 import img_score
    from hpsv2.utils import hps_version_map

    checkpoint = hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    print(f"cached HPSv2 checkpoint: {checkpoint}", flush=True)
    # HPSv2's initializer also obtains its OpenCLIP ViT-H/14 base weights.
    # Keep CUDA hidden so this cannot contend for the training GPU.
    img_score.initialize_model()
    img_score.model_dict.clear()
    print("cached HPSv2 OpenCLIP base weights", flush=True)

    from torchmetrics.image.fid import FrechetInceptionDistance

    FrechetInceptionDistance(feature=2048, normalize=False)
    print("cached canonical FID weights", flush=True)


if __name__ == "__main__":
    main()
