"""Publish the verified Hemera-Nano export and model card to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = (
    "3a033dd0ade8178003adaa43270711b382200ec6dffa6f4e565a6c9d4763c6ee"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_upload_tree(destination: Path, model_dir: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    required_model_files = ("model.safetensors", "config.json", "model_index.json")
    for name in required_model_files:
        source = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)

    actual_hash = sha256(destination / "model.safetensors")
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(
            f"Refusing to publish unexpected weights: {actual_hash}"
        )

    shutil.copy2(ROOT / "MODEL_CARD.md", destination / "README.md")
    for name in ("LICENSE", "LICENSE_AUDIT.md", "FINAL_REPORT.md"):
        shutil.copy2(ROOT / name, destination / name)
    shutil.copy2(
        ROOT / "requirements-inference.txt", destination / "requirements.txt"
    )
    shutil.copytree(
        ROOT / "assets" / "showcase", destination / "assets" / "showcase"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="pauhidalgoo/hemera-nano")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model" / "hemera-nano")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Publish Hemera-Nano v1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    with TemporaryDirectory(prefix="hemera-hf-") as temporary:
        upload = Path(temporary) / "upload"
        build_upload_tree(upload, args.model_dir.resolve())
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=upload,
            commit_message=args.commit_message,
        )
    print(result)


if __name__ == "__main__":
    main()
