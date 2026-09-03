# Model assets and reproducibility

This repository archives every model artifact trained by this project. The
frozen third-party foundation checkpoints are intentionally referenced by
their Hugging Face identifiers instead of being redistributed.

## Project-trained model artifacts

The sealed Task-D probe weights are stored in:

- `rescue_audit/sealed_taskD_primary/bilinear.npz`
- `rescue_audit/sealed_taskD_primary/cheap_only.npz`
- `rescue_audit/sealed_taskD_primary/no_state_interaction.npz`

Additional fitted/audit caches are under
`rescue_audit/results/auditB_cache/`. All `.npz` and `.npy` files are tracked
with Git LFS. File hashes, array shapes, dtypes, and numerical summaries are
recorded in `rescue_audit/results/full_inventory/asset_manifest.csv` and the
corresponding JSON cards below `rescue_audit/results/full_inventory/cards/`.

## Frozen foundation checkpoints

| Role | Hugging Face model ID | Revision |
|---|---|---|
| Primary OpenWebText backbone | `kuleshov-group/mdlm-owt` | Repository default used by the original runs |
| Independent OpenWebText backbone | `louaaron/sedd-small` | Repository default used by the original runs |
| Verifiable-task backbone | `nvidia/Nemotron-Labs-Diffusion-3B` | `0d51902da1f8869f83413ce642fab402fa5641e0` |

The exact Nemotron revision is enforced in `src/nemotron_local.py`. MDLM and
SEDD were loaded from the named repositories by the original collection
scripts; their full run settings and model metadata are preserved in the
per-run JSON outputs and logs.

Foundation checkpoint caches are not committed because they are third-party
distributions governed by their upstream licenses. Reproduction fetches them
from their canonical repositories; all downstream data, seeds, fitted probes,
scores, detailed per-batch outputs, plots, logs, and analysis produced by this
project are included here.
