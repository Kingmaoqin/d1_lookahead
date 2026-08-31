# Direction 1 — "Lookahead Without Looking Ahead"

Phenomenon-first, minimal-compute test of one narrow claim:

> A frozen diffusion LM's representation linearly exposes a **rollout-defined**
> value of alternative future decoding actions, beyond what its exposed output
> distribution and cheap trajectory signals reveal.

This is a **representation finding first**. No decoder is shipped, no model is
trained, and the direction is designed to be killed cheaply.

## Substrate

`kuleshov-group/mdlm-owt` — MDLM (Sahoo et al., NeurIPS 2024), Cornell /
Cornell Tech, Apache-2.0, trained on OpenWebText. Non-Chinese-origin, so the
brief's hard provenance rule is satisfied. 169.6M params, 12 layers, d=768,
GPT-2 tokenizer + a `[MASK]` id. **Frozen throughout.**

`src/mdlm_local.py` is a flash-attn-free reimplementation (the released
`modeling_mdlm.py` hard-depends on `flash_attn`, absent here). Validated in
Phase S — see below.

## Layout

```
src/
  mdlm_local.py   MDLM without flash_attn; loads the released weights
  crn.py          counter-based RNG -> common random numbers across branches
  policy.py       pi_ref, Path-LL, Rao-Blackwellised scoring, rollouts
  collect.py      snapshots, candidate strata, paired Q/V/A rollouts
  features.py     C1 / C2 / C3 control hierarchy + hidden extraction
  dataset.py      shard loading, feature blocks, document-level splits
  probes.py       ridge probes, MLP control, metrics, cluster bootstrap
scripts/
  phase_s_qualify.py    Phase S substrate qualification gates
  collect_labels.py     Phase 0A oracle-label collection
  label_diagnostics.py  oracle-label + rollout-variance deliverable
  pipeline_selftest.py  synthetic positive/negative control for the analysis
  exp1_decodability.py  Experiment 1: Delta_R2 + layer x timestep heatmap
  exp2_matched.py       Experiment 2: matched candidates
  kill_gate.py          mechanical evaluation of the pre-registered gate
  make_figures.py       deliverable figures
docs/
  PREREGISTRATION.md    effect sizes, fixed before Experiment 1 was run
  PRIOR_ART_UPDATE.md   arXiv re-search 2026-08-19; positioning
  FINDINGS.md           results and verdict
```

## Audit and corrections

An independent audit (`docs/AUDIT.md`) found a **fatal defect in the oracle
labels**: the Q and V branches consumed different numbers of masked positions,
which injected the state value into the primary label
(`A_full == ((logp_action - V) + 16*A_future)/17`, corr 1.000000;
`corr(A_full, -V) = +0.589`). All labels were re-collected and every experiment
re-run. A second arm with a confidence-ordered `pi_ref` was added after the
audit showed the Phase-S confidence-collapse claim had been measured with a
non-standard ordering rule. Superseded results are kept under
`results/exp1/`, `results/exp2/`; the corrected ones are `results/exp1_fixed/`,
`results/exp2_fixed/` (ancestral) and `results/exp1_conf/`, `results/exp2_conf/`
(confidence).

## Reproduce

```bash
conda activate llm     # torch 2.7.0+cu126, transformers 4.53.2, sklearn, scipy

python scripts/phase_s_qualify.py                       # ~3 min
# arm 1 -- ancestral pi_ref (pre-registered, primary)
python scripts/collect_labels.py --n_prompts 200 --offset   0 --tag a2
python scripts/collect_labels.py --n_prompts 200 --offset 200 --tag b2
# arm 2 -- confidence-ordered pi_ref (added after audit)
python scripts/collect_labels.py --n_prompts 200 --offset   0 --tag c1 \
       --order confidence --order_temp 2.0
python scripts/collect_labels.py --n_prompts 200 --offset 200 --tag c2 \
       --order confidence --order_temp 2.0

python scripts/pipeline_selftest.py                      # must print VALIDATED
python scripts/label_diagnostics.py --tags a2 b2
python scripts/exp1_decodability.py --tags a2 b2 --out results/exp1_fixed
python scripts/exp1_decodability.py --tags a2 b2 --null global   # must give ~0
python scripts/exp1b_within_state.py A_pertok a2,b2      # the decisive split
python scripts/exp2_matched.py --tags a2 b2 --ladder
python scripts/exp2_matched.py --tags a2 b2 --tag fixed
python scripts/kill_gate.py A_pertok _fixed
EXP1_DIR=results/exp1_fixed EXP2_DIR=results/exp2_fixed \
  FIG_DIR=results/figures_fixed \
  EXP1B_NPZ=results/exp1b_heatmaps_a2b2.npz python scripts/make_figures.py
```

## How the analysis is kept honest

- **Document-level splits** everywhere; candidates within a state and states
  within a trajectory never straddle a split.
- **Two-block ridge.** The control block and the hidden block get separate
  regularisation strengths, with `gamma = 0` in the grid, so the augmented
  model *nests* the baseline. With a single shared `alpha`, adding a
  pure-noise 1536-d hidden block costs 0.06 `R^2` outright — `Delta_R2` would
  be miscalibrated rather than merely conservative.
- **Synthetic self-test.** `pipeline_selftest.py` builds data of the same shape
  with a known answer: pure-noise hidden must give `Delta_R2 ~ 0` (measured
  -0.0004), a real linear hidden signal must be found at the right layer
  (+0.3210 at the true layer 8), and a purely nonlinear hidden signal must NOT
  be claimed by the linear probe (+0.0004).
- **Label-permutation control on the real features** — measured
  `Delta_R2 = -0.0019` with every metric at chance.
- **`within_state_r2`.** `R^2` after removing each state's mean from label and
  prediction. `h_global` is constant within a state, so it scores exactly zero
  here; only per-candidate structure survives. This separates "the probe knows
  this *state* is good" from "the probe knows which *candidate* is better",
  which is the decision a scheduler actually faces.
- **Noise ceiling** reported next to every `R^2`, since label noise caps what
  any predictor can reach.

## Frozen estimand

See `docs/PREREGISTRATION.md`. In brief: `pi_ref` is MDLM's native ancestral
sampler (uniform random unmasking order, one commit per step, top-k 50,
temperature 1.0); `G` is a Rao-Blackwellised POKE-style Path-LL over a
horizon of `H=16` commits; advantages are estimated from `K=8` CRN-coupled
paired rollouts. Every label is policy-relative `A^{pi_ref}`, never `A*`.

## Two substrate facts that shaped the design

1. **Confidence-ordered decoding is degraded on MDLM-owt.** Under the standard
   top-1 rule, per-sample distinct-2 is 0.447 against 0.867 for real text
   (0.706 for random order). A genuinely confidence-driven order still fails
   the pre-registered coherence gate; only heavily noised confidence
   (`order_temp >= 2.0`) passes. `pi_ref` is therefore the ancestral sampler — which also keeps the advantage label from being mechanically
   entangled with confidence. This is a Phase-S substrate finding and it
   weakens the "confidence decoding" baseline any later Phase-0B would use.
2. **Both Q/V branches must consume the same number of masked positions.**
   Running V for `H` commits against Q's `H+1` injects `-V/(H+1)` into the
   advantage label and makes committing an easy token look harmful because it
   depletes the future pool. This was the audit's fatal finding; see
   `docs/AUDIT.md`.
3. **The naive Monte-Carlo Path-LL label is unusable at this scale.** Its
   rollout noise EXCEEDS the between-candidate signal (measured SNR 0.1–0.6).
   Rao-Blackwellising the per-commit score — replacing the realized token's
   log-probability with its conditional expectation, which has the same
   expectation by the tower property — raises SNR to 2–4. Without this the
   study would have produced a false KILL.
