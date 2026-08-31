# Direction 1 — Findings

> **RE-RUN TWICE.** First after an audit found a fatal label defect, then again
> after three residual defects (K, horizon comparability, G4) were repaired.
> All numbers below are from the final `K=24` collection.
>
> **Original defect:** An independent audit found a fatal defect in the
> oracle labels: the Q and V branches consumed different numbers of masked
> positions, which injected the state value into the primary label
> (`corr(A_full, -V) = +0.589`, exact identity, corr 1.000000). Every number
> below comes from **re-collected, corrected labels**. See `docs/AUDIT.md` for
> the full audit and `docs/PREREGISTRATION.md` Corrections 2-3.
> A second study arm with a confidence-ordered `pi_ref` was added.

---

## 1. Substrate provenance and qualification (Phase S)

| item | value |
|---|---|
| checkpoint | `kuleshov-group/mdlm-owt` |
| paper | Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, NeurIPS 2024 |
| origin | Cornell / Cornell Tech (US) — **non-Chinese-origin, satisfies the hard rule** |
| licence | Apache-2.0 |
| data | OpenWebText |
| size | 169.6M params, 12 layers, d=768, 12 heads, GPT-2 tokenizer + `[MASK]` = 50257 |
| state | **frozen**; `requires_grad_(False)` on every parameter |

### Reimplementation and its validation

The released `modeling_mdlm.py` hard-depends on `flash_attn`, which is not
installed. `src/mdlm_local.py` reimplements the backbone with stock PyTorch:
`flash_attn_varlen_qkvpacked_func(..., causal=False)` becomes full bidirectional
`scaled_dot_product_attention`, and `apply_rotary_emb_qkv_` becomes the
half-split (GPT-NeoX) rotary convention.

Correctness was established two ways.

**Perplexity.** Continuous-time masked-diffusion NELBO on held-out OpenWebText:

| context length | NELBO ppl |
|---|---|
| 128 | 53.1 |
| 256 | 38.1 |
| 512 | 28.3 |
| **1024 (the model's native length)** | **26.9** |

against the published 23.21 at L=1024. The residual gap is preprocessing: the
paper packs and detokenizes its evaluation stream, we take single-document
crops.

**Convention ablation.** Substituting a wrong-but-plausible convention
collapses the model, which a coincidentally-working implementation would not do:

| rotary convention | NELBO ppl (L=128) |
|---|---|
| **ours — half-split, non-interleaved** | **51.1** |
| interleaved (GPT-J style) | 1378.4 |
| conjugate rotation | 2335.4 |
| rotary disabled | 2317.1 |

Qualitatively, `The capital of France is Paris. The capital of Germany is` →
` Berlin` at −0.3 log-prob under our convention; the wrong conventions produce
` is` / ` the`.

### Qualification gates — all five pass

| gate | result |
|---|---|
| S1 numerical fidelity | NELBO ppl 34.9 at L=256 — same order as published; reimplementation sound |
| S2 coherent generation | generated distinct-2 **0.700** vs **0.897** for real text; max-repeated-4-gram **0.021** vs **0.010** |
| S3 non-degenerate Path-LL | between-prompt sd 0.233, within-prompt (seed) sd 0.224 — both far from zero |
| S4 hidden-state logging | 13 layers, shape `(B, 256, 768)`, stable across denoising steps, all finite |
| S5 `pi_ref` completion | 0 incomplete trajectories |

**Verdict: SUBSTRATE QUALIFIED.**

### Two substrate findings that changed the design

**(a) Confidence-ordered decoding collapses on MDLM-owt.** Measured over the
same 16 prompts, L=256, 192 masked positions:

| unmasking order | token rule | distinct-1 | distinct-2 | max-rep-4gram |
|---|---|---|---|---|
| *real text (reference)* | — | 0.596 | 0.856 | 0.011 |
| confidence | sample | 0.171 | 0.217 | 0.074 |
| confidence | argmax | 0.137 | 0.171 | 0.079 |
| margin | sample | 0.278 | 0.428 | 0.050 |
| left-to-right | sample | 0.317 | 0.484 | 0.141 |
| **random (ancestral)** | **sample** | **0.383** | **0.669** | **0.020** |

Softening the order with a temperature does not rescue it — only at
order-temperature ≥ 5, i.e. effectively random, does coherence return
(distinct-2: 0.173 at 0.1, 0.226 at 1.0, 0.402 at 2.0, 0.632 at 5.0, 0.669 at
random). Confidence ordering produces visible repetition loops.

Consequence: `pi_ref` is the **ancestral sampler** — the model's own reverse
diffusion process. This is required by gate S2, and it has a second benefit:
the reference order is position-agnostic, so the advantage label is not
mechanically entangled with confidence. It also means that any future Phase 0B
on this substrate would have a *weak* "confidence decoding" baseline, which
must be stated rather than exploited.

**(b) The naive Monte-Carlo Path-LL label is unusable at this scale.** In the
Phase-0A pilot the realized-token estimator's rollout noise **exceeded** the
between-candidate signal:

| horizon | estimator | sd(A_full) | SEM | SNR | sd(A_future) | SEM | SNR |
|---|---|---|---|---|---|---|---|
| 8 | MC | 0.0891 | 0.0819 | 0.18 | 0.1043 | 0.0832 | 0.57 |
| 8 | **RB** | 0.0732 | 0.0332 | **3.85** | 0.0622 | 0.0319 | **2.79** |
| 16 | MC | 0.0584 | 0.0559 | 0.09 | 0.0693 | 0.0572 | 0.47 |
| 16 | **RB** | 0.0387 | 0.0217 | **2.18** | 0.0454 | 0.0218 | **3.35** |
| 32 | RB | 0.0250 | 0.0189 | 0.75 | 0.0379 | 0.0191 | 2.97 |
| none | RB | 0.0241 | 0.0182 | 0.74 | 0.0272 | 0.0183 | 1.21 |

Rao-Blackwellising the per-commit score — replacing the realized token's
log-probability by its conditional expectation
`sum_v p_trunc(v) log p_full(v)`, which has the same expectation by the tower
property — raises SNR from ~0.1–0.6 to 2–4. **Without this the study would have
produced a false KILL**: no probe can predict a label that is mostly noise.

---

## 2. Frozen estimand

See `docs/PREREGISTRATION.md`. Summary:

- `pi_ref`: ancestral, uniform random unmasking permutation, **one commit per
  step**, token sampled from the top-50 truncated softmax at temperature 1.0.
- `G`: Rao-Blackwellised Path-LL, always scored under the **full untruncated**
  softmax, over a horizon of **H = 16** commits after the decision point.
- `A^{pi_ref}(i | s_t) = Q - V`, estimated as a **paired** mean over **K = 8**
  CRN-coupled rollouts. Q forces `(i, argmax_v p_theta(v|s_t))`; V lets
  `pi_ref` make its own commit. Both branches commit exactly H positions.
- CRN: the unmasking permutation key and the token Gumbel noise are indexed by
  absolute **position** and **token identity**, never by step or rank, so the
  two branches see identical noise even though one is a commit ahead. Each
  position is unmasked once per rollout, so the marginal law of `pi_ref` is
  exact.
- Two label variants, both pre-registered: `A_full` (includes the forced
  action's own log-probability — the literal POKE Path-LL) and `A_future` (its
  downstream effect only, which is the harder target because the action's own
  log-probability *is* a cheap control feature).
- `V^{pi_ref}` is reported as a pre-registered secondary target.

All labels are **policy-relative** `A^{pi_ref}`. None is `A*`.

---

## 3. How the analysis was kept honest

- Document-level train/val/test splits throughout.
- **Two-block ridge** with separate regularisation for controls and hidden,
  `gamma = 0` in the grid so the augmented model *nests* the baseline. A single
  shared `alpha` costs 0.06 `R^2` outright when the hidden block is pure noise
  — see Amendment 1 in the pre-registration.
- **Synthetic self-test** (`scripts/pipeline_selftest.py`) on data of the same
  shape with a known answer:

  | regime | true layer found | ΔR² | Δconcordance | verdict |
  |---|---|---|---|---|
  | hidden block is pure noise | — | **−0.0004** | −0.0008 | no false positive |
  | real linear signal at layer 8 | **8** | **+0.3210** | +0.1209 | detected |
  | purely nonlinear signal | — | +0.0004 | +0.0002 | correctly not claimed |

- **Label-permutation control on the real features**: ΔR² = **−0.0019**
  (95% CI [−0.0025, +0.0025]), concordance 0.491 vs 0.507, within-state R² 0.000.
  No leak.
- `within_state_r2`: `R^2` after removing each state's mean from both label and
  prediction. `h_global` is constant within a state and scores exactly zero
  here, so this metric isolates *candidate-level* signal — the decision a
  scheduler actually faces — from between-state signal.
- Noise ceiling reported beside every `R^2`.
- Prior-art re-search run 2026-08-19; see `docs/PRIOR_ART_UPDATE.md`. No exact
  collision, but 2605.20187 removes "hidden states support a one-pass readout"
  from our available novelty.

---

## 4. Oracle labels — final collection, `K = 24`, two arms

400 documents, 2,400 states, 6 candidates per state, horizon `H = 16`,
Rao-Blackwellised Path-LL, `record_fracs` capped at 0.60 so `H/remaining` stays
in `[0.09, 0.21]` in every bin. 14,400 examples per arm.

| | arm 1 — ancestral | arm 2 — confidence (order_temp 2.0) |
|---|---|---|
| `A_full` SNR / `R^2` ceiling | **5.34 / 0.842** | **4.23 / 0.809** |
| `A_future` SNR / ceiling | **1.64 / 0.621** | **1.36 / 0.577** |
| Rao-Blackwell noise reduction | 4.3x | 3.7x |
| CRN variance reduction | 7.1x | 5.5x |
| `corr(A_full, logp_action)` | +0.429 | +0.504 |
| `corr(A_future, logp_action)` | -0.278 | -0.224 |

Raising `K` from 8 to 24 lifted `A_future` from SNR 0.74 / 0.59 — below 1, where
its null was uninformative — to 1.64 / 1.36. Both targets are now usable in both
arms, which is what this final round was for.

---

## 5. Experiment 1 — future-value decodability

| statistic | arm 1 ancestral | arm 2 confidence |
|---|---|---|
| **`A_full`** `Delta_R2` | **+0.050** [+0.032,+0.070] excl 0 | **+0.086** [+0.070,+0.117] excl 0 |
| `Delta_Spearman` | +0.034 [+0.021,+0.048] excl 0 | +0.060 [+0.047,+0.083] excl 0 |
| `Delta_within_R2` | +0.023 [+0.016,+0.031] excl 0 | +0.033 [+0.021,+0.039] excl 0 |
| `Delta_concordance` | +0.012 [+0.002,+0.019] | +0.014 [+0.000,+0.024] incl 0 |
| **`A_future`** `Delta_R2` | +0.019 [+0.003,+0.037] excl 0 | +0.006 incl 0 |
| `Delta_concordance` | +0.007 incl 0 | +0.008 incl 0 |
| **`V^{pi_ref}`** `Delta_R2` | **+0.183** [+0.153,+0.248] | **+0.320** [+0.313,+0.415] |

### Inference that includes fitting variability

The default bootstrap holds both fitted models fixed and resamples only the test
set. A `--refit_boot` bootstrap re-splits the documents and **re-fits both
models** on every replicate (200 replicates, resampling *within* each split so
train and test never share a document):

| statistic | arm 1 | arm 2 |
|---|---|---|
| `Delta_R2` | +0.051 [+0.029,+0.071] **excl 0** | +0.093 [+0.064,+0.128] **excl 0** |
| `Delta_within_R2` | +0.025 [+0.014,+0.040] **excl 0** | +0.031 [+0.020,+0.046] **excl 0** |
| `Delta_concordance` | +0.008 [**-0.002**,+0.017] **incl 0** | +0.013 [+0.000,+0.026] excl 0 |

**G1 survives honest inference** in both arms. The marginal statistic does not:
arm 1's `Delta_concordance` excluded zero under the fixed-model bootstrap and
**includes** zero once fitting variability is counted. That is exactly the case
the refit bootstrap was added to catch.

---

## 6. Experiment 1b — where the incremental signal lives

The decisive split, 13 layers x 6 timestep bins, both arms:

| channel | arm 1 ancestral | arm 2 confidence |
|---|---|---|
| `Delta R^2` from `h_global` (**state-level**) | **75/78 cells > 0**, median **+0.031**, max +0.077 | **74/78 cells > 0**, median **+0.062**, max +0.120 |
| `Delta` within-state `R^2` from `h_i` (candidate-level) | 49/78, median +0.0017, max +0.016 | 44/78, median +0.0002, max +0.040 |
| `Delta` within-state **concordance** from `h_i` | **40/78**, median +0.0007 | **30/78**, median +0.0000 |

The state-level channel is positive in 95-96% of cells with a median an order of
magnitude larger. The candidate-level channel — and specifically its
*concordance*, the ability to say which of two candidates in the same state is
better — sits at chance in arm 1 and **below** chance in arm 2.

---

## 7. Experiment 2 — matched candidates

Separating `h_i` from `h_global` here matters, and the earlier rounds had not
done it: the "hidden" score was `cheap + [h_i; h_global]`. `h_global` is constant
across the candidates of a state so it cannot discriminate a matched pair
directly, but including it changes the fitted coefficients on the cheap block
and moves the ranking anyway. The primary score is therefore `cheap + h_i`.

| score | arm 1 | arm 2 |
|---|---|---|
| confidence (p1) | 0.569 | 0.568 |
| scalar + history controls | 0.544 | **0.601** |
| output-distribution controls | **0.577** | 0.535 |
| all cheap controls | 0.573 | 0.528 |
| trajectory stability | 0.476 | 0.550 |
| `cheap + [h_i; h_global]` | 0.610 | 0.635 |
| **`cheap + h_i` (primary)** | **0.581** | **0.577** |

- arm 1: gap over the best single control **+0.044**, CI [-0.079,+0.158] **includes 0**
- arm 2: gap over the best single control **-0.011**, CI [-0.104,+0.087] **includes 0**

Stripping `h_global` drops the hidden probe from 0.61-0.63 to ~0.58 — at the
control level in arm 1 and **beaten by the controls** in arm 2. The strong-looking
matched-pair numbers were the state-level channel acting through coefficient
changes, not `h_i` carrying candidate-level advantage. This agrees with
Experiment 1b.

---

## 8. Interim kill gate — final, both arms

| gate | threshold | arm 1 ancestral | arm 2 confidence |
|---|---|---|---|
| **G1** incremental prediction | `Delta_R2 >= 0.010`, CI excl 0 | +0.050 **PASS** | +0.086 **PASS** |
| **G2** candidate ranking | `Delta_conc >= 0.020` | +0.012 **FAIL** | +0.014 **FAIL** |
| **G3** matched candidates | acc >= 0.55, gap >= 0.03, CI excl 0 | 0.581, gap +0.019, CI incl 0 **FAIL** | 0.577, gap +0.027, CI incl 0 **FAIL** |
| **G4** robustness | > 0 in >= half the bins (at the validation-selected layer) + natural stratum | 6/6, +0.056 **PASS** | 6/6, +0.113 **PASS** |

> ### VERDICT (both arms): GATE FAILED — KILL the direction as stated.

The verdict is unchanged across every version of the experiment: contaminated
labels, corrected labels at `K=8`, and corrected labels at `K=24` with adequate
power. It does not depend on the reference policy.

Stop here. No RL, no SAEs, no Transformer-sized controller, no added capacity.
Phase 0B is not run; the replication gate is not opened.

---

## 9. What was actually learned

The claim under test:

> *A frozen DLM representation linearly exposes a rollout-defined value of
> alternative future decoding actions that is not recoverable from its output
> distribution or cheap trajectory statistics.*

**Not supported, under either reference policy, at adequate power.** What is
supported:

> On MDLM-owt, a frozen masked-diffusion representation linearly exposes the
> rollout-defined **state value** `V^{pi_ref}` far beyond strong
> output-distribution and trajectory controls (`Delta_R2` **+0.183** ancestral,
> **+0.320** confidence-ordered), and the state-level channel is positive in
> 95-96% of layer x timestep cells. It does **not** expose the **action
> advantage** at the candidate level: `h_i` — the only per-candidate part of the
> representation — ranks candidates within a state at chance (concordance
> positive in 40/78 and 30/78 cells), and on matched candidates `cheap + h_i` is
> at or below the best exposed-output control.

`Delta_R2` passes G1 in both arms and survives a refit bootstrap, but G1 is a
**pooled** statistic that a state-level readout satisfies without helping any
decision. `h_global` is constant across the candidates of a state; it improves
the fit and, through omitted-variable correction, lets the cheap features rank
slightly better — but it cannot itself choose between candidates. That is why
G1 passes while G2 and G3 fail, and it is why the pre-registration required all
four gates rather than a single headline number.

POKE and LookUM spend test-time compute estimating a *comparison between
actions*. The part of their target that is cheaply readable from the frozen
state is the part that does not vary across the actions being compared.

### Scope and honest limits

- **One backbone, one scale** (170M, OpenWebText). The replication gate was not
  opened.
- **Late-trajectory states are not covered.** `record_fracs` was capped at 0.60
  so the fixed horizon stays comparable across bins; states beyond 60% diffusion
  progress are outside the study.
- **Confidence-ordered decoding is degraded on this substrate** even under the
  correct top-1 rule (distinct-2 0.447 vs 0.867 for real text). Arm 2 had to
  noise the order (`order_temp 2.0`) to pass the coherence gate, so it is only
  partially confidence-driven — a fully confidence-driven `pi_ref` is not
  available here at all.
- **`A_future` remains the weaker target** (SNR 1.64 / 1.36) even at `K=24`.
- **The three "seeds" are re-splits of one dataset**, not independent runs, so
  they measure split sensitivity, not sampling variability. The refit bootstrap
  is the honest uncertainty estimate; the per-seed spread is reported alongside
  it rather than treated as replication.
- `V^{pi_ref}` has no per-seed replicates stored, so no noise ceiling is
  reported for it and its `Delta_R2` is not ceiling-corrected.

### Cheapest thing that could revive the direction

Not capacity. A substrate where confidence-ordered decoding is natively coherent,
so `pi_ref` can be the decoder POKE/LookUM actually run, and where the candidate
set at a state is genuinely contested. On any future substrate, run Experiment 1b
**first** and keep `h_i` separate from `h_global` throughout — the entire
difference between "a real finding" and "a pooled `Delta_R2` that a state-value
readout satisfies on its own" lives in that separation, and three successive
versions of this study reproduced the same trap until it was made explicit.
---

# Round 4 — second backbone, horizon sweep completed, V properly supported

## Replication on SEDD (backbone 2) — the verdict holds

`louaaron/sedd-small` (Stanford, ICML 2024), ancestral `pi_ref`, K=24, H=16,
14,400 examples. Labels are the cleanest of any run: `A_full` SNR **6.13**,
ceiling **0.860**; `A_future` SNR 1.68, ceiling 0.627.

| gate | threshold | SEDD result | |
|---|---|---|---|
| G1 | `Delta_R2 >= 0.010`, CI excl 0 | +0.051 [+0.032,+0.071] | **PASS** |
| G2 | `Delta_conc >= 0.020` | +0.006, per-seed 0.003/0.005/0.008 | **FAIL** |
| G3 | acc >= 0.55, gap >= 0.03, CI excl 0 | caliper: 0.634, gap +0.118, CI excl 0 **PASS** / strict: 0.689, gap +0.048, CI **incl 0** **FAIL** | **split** |
| G4 | > 0 in >= half the bins + natural stratum | 5/6 bins, +0.051 | **PASS** |

> **VERDICT: GATE FAILED -> KILL.** G2 fails under every analysis; the verdict
> does not depend on how G3 is resolved.

## The one place the two backbones genuinely differ — and how it resolves

SEDD's matched-candidate result is **stronger** than either MDLM arm. This was
worth chasing rather than waving away, so it was tested four ways:

| evidence | population | SEDD result |
|---|---|---|
| exp1 `Delta_concordance`, full-data probe | ALL within-state pairs | +0.006, CI **includes 0** |
| exp1 `cheap + h_i` vs `cheap`, concordance | ALL within-state pairs | **−0.0003** — `h_i` adds nothing |
| exp1b per-bin grid, `h_i` concordance | per timestep bin | **38/78** cells positive — chance |
| exp2 **caliper** matching | pairs where cheap was equalised | 0.634 vs 0.572, gap +0.105, CI excl 0 |
| exp2 **strict** matching (non-circular) | pairs matched coordinate-wise | 0.689 vs 0.678, gap +0.048, CI **includes 0** |

Three of the four say no candidate-level effect. The one that says yes is the
one carrying a **known circularity**: `caliper` matching equalises the
cheap-control *prediction* by construction, which pushes the cheap score toward
chance on exactly those pairs and mechanically inflates "gap over cheap".
`strict` matching does not match on the cheap prediction, and there the effect
loses significance.

### The procedure itself was cleared by a placebo

Because that reasoning could be used to explain away any positive result, the
matched-pair machinery was tested with the hidden block replaced by **Gaussian
noise of the same shape**:

| SEDD, caliper matching | real `h_i` | placebo noise |
|---|---|---|
| `cheap + h_i` accuracy | 0.634 | 0.568 |
| gap over the best single control | +0.105 **CI excl 0** | **−0.012** CI incl 0 |

The placebo does **not** beat the controls — the two-block ridge correctly
selects `gamma ~ 0` and falls back to the cheap model. So the pair-mining
procedure is not biased toward whichever block escaped matching, and SEDD's
0.634 is a real (if non-robust) signal, not an artifact of selection.

**Honest reading:** SEDD carries somewhat more candidate-level signal than
MDLM — visible only on the subpopulation where the exposed controls have been
matched into uninformativeness, and not robust to the non-circular matching.
Not enough to pass the gate, and G2 fails regardless.

## Horizon sweep — completed

| run | `h_i` within-`R^2` >0 / median | `h_i` **concordance** >0 / median | `h_global` `R^2` >0 / median |
|---|---|---|---|
| H=8 (K=16) | 42/78, +0.0005 | 31/78, +0.0000 | 75/78, +0.0489 |
| H=16 (K=24) | 49/78, +0.0017 | 40/78, +0.0007 | 75/78, +0.0312 |
| **H=32 (K=16)** | 36/78, +0.0000 | **30/78**, +0.0000 | **36/78**, +0.0000 |
| SEDD H=16 (K=24) | 48/78, +0.0016 | 38/78, +0.0000 | 72/78, +0.0493 |

The candidate channel is at chance at every horizon and on both backbones. At
H=32 even the **state** channel washes out (36/78, median 0.0000) — a longer
horizon dilutes a single commit's effect into more future randomness, which is
what the H=16 operating point was chosen to avoid.

## `V^{pi_ref}` now has a noise ceiling

The per-seed backfill (V branches only, ~1/7 the cost of a full collection,
with an assertion that the recomputed mean reproduces the stored `V_pertok`)
completed on both MDLM arms:

| arm | V SNR | V `R^2` ceiling | `Delta_R2` |
|---|---|---|---|
| MDLM ancestral | 170.0 | **0.9942** | +0.183 [+0.153,+0.248] |
| MDLM confidence | 119.9 | **0.9917** | +0.320 [+0.313,+0.415] |
| SEDD ancestral | 153.7 | **0.9935** | (see `results/exp1_sedd`) |

V is estimated so precisely that attenuation is negligible — the state-value
result needs essentially no ceiling correction, and can now be reported as a
secondary finding in its own right rather than only as a foil.
