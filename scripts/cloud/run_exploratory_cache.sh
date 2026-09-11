#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
mkdir -p logs reports
exec > >(tee -a logs/exploratory-cache.log) 2>&1
source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.348576158}" exploratory
CACHE_VERIFIED=0
finish_cache() {
  status=$?
  if [[ "$status" == "0" && "$CACHE_VERIFIED" != "1" ]]; then
    status=1
  fi
  printf "%s\n" "$status" > reports/exploratory-cache-exit-code
  hemera_budget_report
}
trap finish_cache EXIT
rm -f reports/exploratory-cache-exit-code

if [[ ! -f data/cache/index.json ]] || \
   [[ "$(python -c "import json; print(json.load(open('data/cache/precompute-progress.json')).get('stage', ''))" 2>/dev/null || true)" != "complete" ]]; then
  hemera_run python run.py precompute \
    --config configs/experiment-base.yaml \
    --manifest data/manifests/photonyx.jsonl \
    --output data/cache \
    --batch-size 128 \
    --materialize-workers 32
fi

python - <<'PY'
import json
from pathlib import Path

from hemera.config import load_config
from hemera.data import ShardedLatentDataset
from hemera.runtime import sha256_file

config = load_config("configs/experiment-base.yaml")
root = Path("data/cache")
index = json.loads((root / "index.json").read_text(encoding="utf-8"))
representation = index["representation"]
expected = {
    "vae": config.representation.vae_id,
    "vae_type": config.representation.vae_type,
    "text_encoder": config.representation.text_encoder_id,
    "text_encoder_type": config.representation.text_encoder_type,
    "selection_count": config.data.max_items,
    "latent_channels": config.data.latent_channels,
    "latent_size": config.data.latent_size,
}
for key, value in expected.items():
    if representation.get(key) != value:
        raise SystemExit(
            f"Cache representation mismatch for {key}: "
            f"{representation.get(key)!r} != {value!r}"
        )
if not representation.get("repa_encoder"):
    raise SystemExit("Exploratory cache is missing required REPA targets")
for shard in index["shards"]:
    for name_key, hash_key in (("file", "sha256"), ("metadata", "metadata_sha256")):
        path = root / shard[name_key]
        if sha256_file(path) != shard[hash_key]:
            raise SystemExit(f"Checksum mismatch: {path}")
dataset = ShardedLatentDataset(root, split="train")
sample = dataset[0]
if tuple(sample["latent_mean"].shape) != (
    config.data.latent_channels,
    config.data.latent_size,
    config.data.latent_size,
):
    raise SystemExit(
        f"Unexpected latent shape: {tuple(sample['latent_mean'].shape)}"
    )
if sample["text_hidden"].shape[0] != config.data.text_length:
    raise SystemExit("Unexpected text-token length")
if "repa_target" not in sample:
    raise SystemExit("Decoded cache row is missing REPA target")
print(
    json.dumps(
        {
            "verified_examples": index["statistics"]["total_examples"],
            "selection_sha256": representation["selection_sha256"],
            "index_sha256": sha256_file(root / "index.json"),
            "latent_shape": list(sample["latent_mean"].shape),
            "text_shape": list(sample["text_hidden"].shape),
            "repa_shape": list(sample["repa_target"].shape),
        },
        indent=2,
    )
)
PY
CACHE_VERIFIED=1
