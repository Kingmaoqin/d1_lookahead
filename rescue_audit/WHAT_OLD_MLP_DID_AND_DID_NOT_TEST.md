# What the old MLP did and did not test

## What it actually did

- Code: `src/probes.py:155-230`, called by `scripts/exp1_decodability.py`.
- Objective: pointwise mean-squared error on a standardized scalar target.
- Inputs: one validation-selected layer of `cheap + [h_i; h_global]` in the main exp1 path. The later diagnostic discussed in the report also evaluated `cheap + h_i` at a selected layer.
- Architecture: two hidden transformations (`d -> min(512,d) -> width -> 1`) with GELU.
- Search: learning rate × weight decay × width 256/512, selected by validation R² with early stopping.
- Seeds: the exp1 caller does not pass an initialization seed, so every split uses the default `seed=0`; this is not a 3–5 initialization study.
- Layer/timestep: a single layer selected by validation for the pooled dataset. It does not exhaustively fit nonlinear models at every layer×timestep cell.
- Normalization: train-only feature mean/std and train-only target mean/std; this part is appropriate.
- Test use: hyperparameters are selected on validation, not test, in the MLP function itself.

## What it did not test

- No bilinear `h_i^T W h_global` or other explicit candidate×state interaction.
- No FiLM/gating in which `h_global` changes how `h_i` is interpreted.
- No relational feature set `[h_i,h_g,h_i-h_g,|h_i-h_g|,h_i*h_g]`.
- No action-token embedding, unembedding vector, position×token interaction, or candidate token alternatives.
- No temporal `h_t-h_{t-1}` features.
- No learned or sparse mixture over layers.
- No direct pairwise, RankNet/Siamese, or listwise ranking loss.
- No kernel, boosting, PLS, or CCA diagnostic.
- No parameter-count-matched nonlinear cheap-only search beyond the narrow control comparison.
- No 3–5 random initialization robustness.
- No systematic capacity/sample-complexity or cross-timestep/dataset transfer test.

## Correct interpretation

The old MLP rules out only a narrow claim: on the tested pooled split and selected single layer, that particular pointwise MSE MLP did not improve candidate ranking over cheap controls. It does **not** rule out nonlinear, relational, temporal, action-conditioned, ranking-objective, or distributed-across-layer representations. The rescue study must not cite it as a general nonlinear impossibility result.
