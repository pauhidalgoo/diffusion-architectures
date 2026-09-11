"""Repair the HPSv2 1.2.0 wheel's omitted OpenAI CLIP BPE asset."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import os
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/openai/CLIP/"
    "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/"
    "clip/bpe_simple_vocab_16e6.txt.gz"
)
SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def main() -> None:
    spec = importlib.util.find_spec("hpsv2")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install hpsv2 before preparing its tokenizer asset")
    destination = (
        Path(spec.origin).parent
        / "src"
        / "open_clip"
        / "bpe_simple_vocab_16e6.txt.gz"
    )
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        print(f"HPSv2 tokenizer asset already verified: {destination}")
        return
    with urlopen(URL, timeout=120) as response:
        payload = response.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != SHA256:
        raise RuntimeError(
            f"HPSv2 tokenizer SHA-256 mismatch: expected {SHA256}, got {observed}"
        )
    # Validate the compressed stream before replacing anything in site-packages.
    if not gzip.decompress(payload):
        raise RuntimeError("HPSv2 tokenizer asset is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    print(f"Installed verified HPSv2 tokenizer asset: {destination}")


if __name__ == "__main__":
    main()
