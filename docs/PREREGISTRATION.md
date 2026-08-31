# Pre-registration — Direction 1 interim kill gate

Written **2026-08-19, before Experiment 1 was run on the full label set**.
Only the 6-prompt smoke run (216 examples, used solely to check that the code
executes) and the label-variance pilot had been seen at the time of writing.
The pilot informed the ESTIMAND (horizon, estimator, K) and nothing else; no
probe result influenced any threshold below.

## Frozen estimand (set in `src/policy.py` / `src/collect.py`)

| item | value |
|---|---|
| substrate | `kuleshov-group/mdlm-owt` (Cornell, Apache-2.0), frozen |
| `pi_ref` | ancestral uniform-random-order unmasking, 1 commit/step, top-k 50, temperature 1.0 |
| sequence | L=256, prefix 64 observed, 192 masked |
| `G` | Rao-Blackwellised Path-LL, full-softmax scoring |
| horizon | H = 16 `pi_ref` commits after the decision point |
| `K` | 8 CRN-coupled paired rollouts |
| labels | `A_full` (incl. the action's own log-prob) and `A_future` (downstream only) |
| splits | document-level train 0.6 / val 0.15 / test 0.25 |
| probe | linear ridge on `[h_{i,t}; h_global,t]`; alpha and layer chosen on VAL only |

All labels are policy-relative `A^{pi_ref}`, never `A*`.

## Primary statistic

    Delta_R2 = R2(cheap + output-dist + hidden) - R2(cheap + output-dist)

where "cheap" = C1 (confidence scalars) + C2 (trajectory) + C3 (top-16
log-probs and a fixed 64-d JL projection of the full log-probability vector).

## Minimum effect sizes to PASS the gate

A pass requires **all four**:

- **G1 incremental prediction.** `Delta_R2 >= 0.010` absolute **and** the
  cluster-bootstrap (over documents) 95% CI on `Delta_R2` excludes zero.
- **G2 candidate ranking.** Within-state pairwise concordance of the
  hidden-augmented probe exceeds the best cheap-control baseline by
  `>= 0.020` absolute, with the same sign in **all 3** split seeds.
- **G3 matched candidates.** After matching on confidence, entropy, margin,
  temporal KL, flips, persistence, timestep, mask ratio and the
  output-distribution projection, the hidden probe orders matched pairs at
  `>= 0.55` accuracy **and** beats the cheap-control score by `>= 0.03`,
  with the bootstrap CI on the gap excluding zero, over `>= 3` resamples.
- **G4 robustness.** `Delta_R2 > 0` in at least half of the recorded diffusion
  timestep bins, and in the naturally-sampled (non-oversampled) held-out
  stratum considered alone. The effect must not come from one prompt family
  or one narrow timestep slice.

## Reported but explicitly NOT decisive

- Conditional mutual information (estimator-sensitive; secondary diagnostic
  only, per the brief).
- The `<=2-layer` MLP. If only the MLP clears the thresholds, the correct
  conclusion is "not linearly represented", and the linear claim is dead.

## Attenuation

Label noise bounds the attainable `R2`. The noise ceiling
`(Var(mean_k A) - Var_k(A)/K) / Var(mean_k A)` is reported alongside every
`R2`, and `Delta_R2` is additionally reported as a fraction of that ceiling.
Thresholds above apply to the RAW `Delta_R2`, which is the conservative choice.

## If the gate fails

Stop. Do not add RL, SAEs, a Transformer-sized controller, joint fine-tuning,
or model capacity of any kind to make the signal appear.

---

## Amendment 1 — 2026-08-19, before any real-label probe was fitted

**Change.** The augmented model is now a **two-block ridge** with a separate
regularisation strength for the control block and the hidden block (`gamma`
grid includes 0, selected on validation), instead of a single shared `alpha`
over the concatenated features.

**Why.** A synthetic self-test (`scripts/pipeline_selftest.py`, 400 docs x 6
states x 6 candidates, 768-d hidden, label SNR 3) showed that with a
**pure-noise** hidden block the shared-alpha probe reports

    Delta_R2 = -0.0605

purely from the added dimensionality — one alpha cannot regularise a 100-d and
a 1536-d block simultaneously. That is not conservatism, it is a miscalibrated
statistic: the null is not centred at zero, so neither the threshold nor the
bootstrap CI means what the pre-registration says it means. With the two-block
form the same null gives `Delta_R2 = -0.0004`, while a genuine linear hidden
signal is still recovered at `Delta_R2 = +0.3210` with the correct layer
identified, and a purely NONLINEAR hidden signal is correctly **not** claimed
by the linear probe (`+0.0004`).

**Basis.** Synthetic data only. No real label had been used to fit any probe at
the time of this amendment; label collection was still running.

**Thresholds unchanged.** G1–G4 in the table above stand exactly as written.
Because the `gamma = 0` point reproduces the controls-only fit, the augmented
model now nests the baseline, so `Delta_R2` can only be negative through
validation noise. `scripts/exp1_decodability.py --null global` is retained as
the empirical negative control on the real features.

---

## Correction 1 — 2026-08-19, control-feature defect found and repaired

**Defect.** In the first collection run the trajectory-stability controls
`flip_count` and `persistence` were accumulated inside `snapshot()`, which runs
only at the 6 recorded checkpoints, instead of at every denoising step. They
were therefore coarsened: `flip_count` counted argmax changes observed at the
checkpoints rather than along the whole trajectory.

**Why this matters.** These are the TraceLock-style realised-trajectory
stability signals — part of the C2 control block the hidden representation must
beat. A coarsened control is a WEAKER control, which biases `Delta_R2`,
`Delta_concordance` and the matched-candidate test **in favour of the study's
own hypothesis**. It is not a conservative error.

**Repair.** `pi_ref` trajectories are deterministic given (prompt window,
rollout seed), and recomputing these statistics needs no rollouts — only the
forward passes along the reference trajectory. `scripts/repair_trajectory_
features.py` re-walks every trajectory, recomputes both statistics per step,
and writes them back over the two affected columns. All other features, all
hidden states and all labels are untouched. The script fails loudly if any
stored record cannot be matched, which would indicate the trajectory was not
reproduced exactly.

The source defect is fixed in `src/collect.py` (`update_hist` now takes the
mask and accumulates per step), so re-running collection from scratch produces
correct features directly.

**All experiments below are reported on the REPAIRED features.**

---

## Correction 2 — 2026-08-20, FATAL label defect found by audit and repaired

An independent audit found that the Q and V branches consumed **different
numbers of masked positions** (17 vs 16). Two consequences, both fatal to the
labels as originally collected:

1. The per-token denominators differed, and the `(1/17 - 1/16)` mismatch made
   the primary label satisfy, exactly,
   `A_full == ((logp_action - V) + 16*A_future)/17`  (corr 1.000000).
   `corr(A_full, -V) = +0.589`: the advantage label was over half **state
   value** by construction — the very quantity it had to be independent of for
   the study's conclusion to mean anything.
2. The branches depleted the mask pool unequally, so committing an easy token
   appeared harmful merely because it removed an easy commit from the future
   pool. `corr(A_future, logp_action) = -0.440`, worsening to -0.710 late in
   the trajectory.

**Repair.** The V-branch now runs `H+1` commits so both branches consume `H+1`
positions; `rollout` returns the score of its own first commit so the
downstream-only label compares `H` post-decision commits per side. All 14,400
labels were re-collected and every experiment re-run. See `docs/AUDIT.md`.

**Thresholds unchanged.** G1-G4 stand exactly as pre-registered.

**Effect on the labels.** The corrected labels are noisier, because part of the
old "signal" was the contamination: `A_full` SNR 3.11 -> 2.05 (ceiling 0.757 ->
0.672), `A_future` SNR 1.52 -> 0.74 (ceiling 0.603 -> 0.426). `A_future` now
has SNR below 1 and its null is therefore **uninformative**, not evidence of
absence.

## Correction 3 — Phase-S confidence rule, and the addition of arm 2

The Phase-S confidence decoder ranked by the **sampled** token's probability
rather than the standard **top-1 maximum**. Under the correct rule the collapse
is real but roughly half as severe (distinct-2 0.447, not 0.202). Re-qualified
at n=48, a genuinely confidence-driven order still FAILS the pre-registered S2
coherence gate (order_temp 0.5/1.0/1.5 give distinct-2 0.502/0.577/0.600 against
a 0.626 threshold); only `order_temp >= 2.0` passes.

The substrate conclusion therefore stands, but the justification was weaker
than reported. A second study arm was added with a confidence-ordered `pi_ref`
at `order_temp = 2.0` — the most confidence-driven setting that qualifies — to
test whether the verdict depends on the reference policy. This is an ADDITION;
the pre-registered ancestral arm remains primary.

---

## Correction 4 — 2026-08-20, three residual defects fixed before the final run

Disclosed as limitations in the previous round; now repaired rather than
excused.

1. **`A_future` was underpowered.** At `K = 8` its SNR was 0.74 (ancestral) and
   0.59 (confidence) — below 1, so the null was uninformative rather than
   evidence of absence. Rollout noise falls as `1/K`; `K` is raised to **24**,
   which projects `A_future` to SNR ~2.2 / ~1.8 and lifts its `R^2` ceiling
   from 0.43 to ~0.69. Cost: 3x the rollout compute.

2. **The fixed horizon was not comparable across timestep bins.** `H = 16`
   covered 9% of the remaining mask at 10% diffusion progress but 55% at 85%,
   so labels from different bins were not on the same footing — which
   undermined the layer x timestep heatmap and G4. `record_fracs` is capped at
   **0.60**, keeping `H/remaining` within `[0.09, 0.21]` in every bin. Cost:
   late-trajectory states are no longer covered, and this is stated as a scope
   limit rather than papered over.

3. **G4 took the per-bin maximum over 13 layers**, a selection-biased statistic
   that a noisy grid passes almost for free. It now evaluates at the
   **validation-selected layer**.

4. **The bootstrap held both fitted models fixed** and resampled only the test
   set, so it captured test-sampling variability but not fitting variability
   and its CI was too narrow. This matters specifically for **G1, the only gate
   that passes**: an understated CI makes "excludes 0" too easy. A `--refit_boot`
   bootstrap that re-splits the documents and re-fits both models on every
   replicate is added and reported alongside the fixed-model CI.

**Thresholds unchanged.** G1-G4 stand exactly as pre-registered.
