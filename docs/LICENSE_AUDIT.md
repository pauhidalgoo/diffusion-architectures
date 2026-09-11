# Hemera-Nano release license record

**Release status:** public release approved. Hemera's original code and
denoiser weights are released under Apache-2.0.

**Decision recorded:** 2026-09-11.

This file records the concrete release decision and the separation between
files distributed by Hemera and frozen third-party components downloaded at
runtime. It is a provenance record, not legal advice.

## Release decision

The owner of Photonyx and Hemera has explicitly declared Photonyx to be CC-BY
and authorized public release of the trained Hemera-Nano denoiser. Based on
that declaration, this repository no longer treats the missing machine-readable
license field in Photonyx's pinned Hugging Face metadata as a publication
blocker.

The public release applies Apache-2.0 to:

- Hemera's original source code;
- `model/hemera-nano/model.safetensors`, containing only the trained
  29,756,564-parameter Hemera denoiser;
- Hemera-authored configuration, evaluation, and documentation files.

Photonyx is attributed by name and linked in the repository README, model card,
configuration, and Hugging Face metadata. Users of the dataset itself must
follow its CC-BY terms.

## Distributed and runtime components

| Component | Pinned identifier | Included in Hemera weights? | Release treatment |
|---|---|---:|---|
| Photonyx | `35978bff3d6f38f4e432c25506168aaf8bd7e26f` | No | Training dataset; CC-BY as declared by its owner and attributed in this release. |
| Hemera-Nano denoiser | SHA-256 `3a033dd0ade8178003adaa43270711b382200ec6dffa6f4e565a6c9d4763c6ee` | Yes | Apache-2.0. |
| DC-AE-Lite f32c32 | `dc-ai/dc-ae-lite-f32c32-diffusers` | No | Downloaded separately by Diffusers on first use; upstream terms apply. |
| OpenAI CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | No | Downloaded separately by Transformers on first use; upstream terms apply. |
| DINOv2-small | `facebook/dinov2-small` | No | Evaluation-only dependency; not required for inference or redistributed. |

The released Safetensors file was hash-verified against the final export. It
does not bundle the VAE, CLIP, dataset images, dataset captions, optimizer
state, or evaluator weights.

## Dataset record

The immutable training audit accepted 427,260 of 449,240 rows and kept only
rows labeled CC0, CC0-1.0, or public domain in the pinned materialized dataset.
The final accepted split was 418,792 train, 4,278 validation, and 4,190 test
rows. The audited manifest hash is
`7d4fa4f570640747d957876a52b307bfd3aab460477bd74099c65a55d32692cf`.

The live Hugging Face repository metadata did not expose a top-level license
field when checked on the decision date. This metadata gap is documented for
reproducibility, but the dataset owner's direct CC-BY declaration controls this
project's release decision.

## References

- [Photonyx dataset](https://huggingface.co/datasets/pauhidalgoo/photonyx)
- [Photonyx source repository](https://github.com/pauhidalgoo/photonyx-dataset)
- [DC-AE-Lite checkpoint](https://huggingface.co/dc-ai/dc-ae-lite-f32c32-diffusers)
- [DC-Gen source](https://github.com/dc-ai-projects/DC-Gen)
- [OpenAI CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32)
- [OpenAI CLIP source license](https://github.com/openai/CLIP/blob/main/LICENSE)
- [Apache License 2.0](LICENSE)

## Continuing obligations

Public release does not remove privacy, publicity, personality, trademark,
consumer-protection, or platform-policy concerns. Dataset corrections and
takedown requests should continue to be honored, and model outputs must not be
presented as factual evidence or used for identity-sensitive or safety-critical
decisions.
