# Independent audit — findings, corrections, and what changed

Two reviewer agents were run independently (one on implementation correctness,
one on experimental design). Both were cut off by a session limit before
finishing; their completed checks are reported here, the remainder were run
directly. Every finding below was re-verified from scratch, not taken on trust.

---

## FATAL 1 — the primary label was algebraically contaminated with the state value

**Verified identity**, exact to floating point:

```
A_full  ==  ( (logp_action - V) + 16 * A_future ) / 17
            max|err| 6.0e-08     corr 1.000000
```

**Cause.** The Q-branch averaged over 17 commits (1 forced + 16 rollout); the
V-branch over 16. The resulting `(1/17 - 1/16)` denominator mismatch injects a
`-V/17` term straight into the label.

**Measured consequence.** `corr(A_full, -V) = +0.589` — the "advantage" label was
over half state value **by construction**.

**What it invalidated.** The previous headline was that `Delta_R2 = +0.061` was
real but came from the state-level channel, so the representation "exposes V,
not A". That reasoning was circular: the hidden state predicts `V` at
`R^2 = 0.93`, so it necessarily predicted the `V` component that had been
injected into the target. The old Experiment-1b claim that "the *entire*
incremental signal is state-level" was an artifact — with clean labels the
`h_global` channel collapses from **+0.112 to +0.036**.

## FATAL 2 — the two branches consumed different numbers of positions

Q filled 17 masked positions, V filled 16. Committing an "easy" (high
log-probability) token therefore looked harmful purely because it removed an
easy commit from the future pool.

**Measured.** `corr(A_future, logp_action) = -0.440` overall, degrading
monotonically with diffusion progress to **-0.710** at 85%. `A_future` was
substantially measuring "how easy is this token", negated — not planning value.

## SERIOUS 3 — the Phase-S confidence-collapse claim was overstated ~2x

The Phase-S decoder ranked positions by the log-probability of the **sampled**
token. The standard convention (MaskGIT / LLaDA, and the decoder POKE and LookUM
actually operate on) ranks by the **top-1 maximum** probability.

Controlled A/B, identical prompts, identical seeds, only the ordering score
varying:

| ordering rule | distinct-1 | distinct-2 | distinct-3 | max-rep-4gram |
|---|---|---|---|---|
| real text | 0.608 | 0.867 | 0.924 | 0.010 |
| random / ancestral | 0.415 | 0.706 | 0.817 | 0.020 |
| confidence = **top-1 max prob** (standard) | 0.298 | **0.447** | 0.506 | 0.092 |
| confidence = sampled-token prob (what was used) | 0.158 | **0.202** | 0.221 | 0.111 |

**The conclusion survives, the magnitude did not.** Re-qualifying the standard
rule at the full n=48 against the pre-registered S2 gate
(`d2 > 0.70 x real`, `maxrep4 < 3.0 x real`  ->  `d2 > 0.626`, `maxrep4 < 0.027`):

| `pi_ref` | distinct-2 | max-rep-4gram | S2 |
|---|---|---|---|
| ancestral | 0.697 | 0.022 | **PASS** |
| confidence, order_temp 0.5 | 0.502 | 0.055 | FAIL |
| confidence, order_temp 1.0 | 0.577 | 0.030 | FAIL |
| confidence, order_temp 1.5 | 0.600 | 0.032 | FAIL |
| confidence, order_temp 2.0 | 0.627 | 0.025 | **PASS** |

A genuinely confidence-driven order still fails the coherence gate under the
correct rule. Only heavily noised confidence qualifies. Arm 2 of the study uses
`order_temp = 2.0`, the most confidence-driven setting that passes.

## MODERATE 4 — the horizon is not comparable across timestep bins

`H = 16` covers 9% of the remaining mask at 10% diffusion progress but 55% at
85%. Labels from different bins are not on the same footing, which weakens the
layer x timestep heatmap and G4.

## MINOR 5 — G4 as coded takes the max over 13 layers per bin

A selection-biased statistic; the max over 13 layers is almost always positive.
Recomputed at the single validation-selected layer the verdict is unchanged
(all six bins still positive), so G4's outcome stands, but the criterion was
sloppy as written.

---

## Verified CORRECT (so it is on record that these were checked)

| component | check | result |
|---|---|---|
| `crn.py` splitmix64 | vs reference implementation | exact match; shift masks correct; int64 wrap correct |
| CRN independence | across token ids / positions / seeds | no correlation beyond `1/sqrt(n)`; Gumbel-argmax uniform, chi2 p=0.87 |
| `mdlm_local.py` rotary | vs flash_attn non-interleaved reference | **error exactly 0.0**; v untouched |
| `modulate_fused` | which of the two `modulate` defs was captured | the first (no-unsqueeze) — matches the reimplementation |
| `expected_logp` | vs Monte-Carlo `E[log p_full]` at T = 0.5 / 1 / 2 | within 1.1 sigma |
| `sample_tokens` | vs `softmax(logits/T)` | chi2 p = 0.41 / 0.24 / 0.20 |
| `ridge_path` Gram path | vs sklearn Ridge | rel. err 1e-7 (fp64), 1e-6 (GPU fp32) |
| 2-block nesting | `gamma = 0` vs controls-only fit | predictions identical, max diff 0.0 |
| document splits | states spanning >1 document | 0; doc/prompt 1:1 |
| **Q/V CRN coupling** | **no-op test: force Q to take V's own action** | **30/30 identical trajectories, error exactly 0.000e+00, under all three order rules** |
| branch coupling | genuine actions | 96-97% token agreement; per-seed Path-LL corr 0.925 |

## Power — the candidate-level null is real, not underpowered

| label | scope | SNR | attainable `R^2` |
|---|---|---|---|
| `A_full` | total | 3.11 | 0.757 |
| `A_full` | **within-state** | **2.51** | **0.715** |
| `A_future` | within-state | 1.74 | 0.635 |

Within-state variance is 51% (`A_full`) and 71% (`A_future`) of total, and the
observed CI half-width on `Delta_within_R2` is ±0.008, so the study could detect
an effect of ~0.017. The candidate-level result is a genuine negative.

---

## The fix

Both branches must consume the same number of positions:

```
Q-branch: force (i, x_hat_i), then H  pi_ref commits   -> H+1 positions
V-branch:                          H+1 pi_ref commits  -> H+1 positions
A_full   = [ logp_action + S_Q - S_V ] / (H+1)
A_future = [ S_Q - (S_V - v_first) ] / H
```

`rollout` now returns `first_ll`, the score of the branch's own first commit, so
the downstream-only label compares H commits made after the decision on each
side. Verified after the fix: `V_n == n_commit == 17`; the contaminating
identity is broken (corr 1.000000 -> 0.931); `corr(A_future, logp_action)`
improves from -0.440 to -0.171.

All labels were re-collected (14,400 examples) and every experiment re-run.

## Honest cost of the fix

The corrected labels are **noisier**, because the old "signal" partly *was* the
contamination:

| label | SNR before | SNR after | ceiling before | ceiling after |
|---|---|---|---|---|
| `A_full` | 3.11 | **2.05** | 0.757 | 0.672 |
| `A_future` | 1.52 | **0.74** | 0.603 | 0.426 |

`A_future` now has SNR < 1, so that arm is genuinely underpowered and its null
must be reported as uninformative rather than as evidence of absence.

---

# Round 3 — residual defects repaired, final `K = 24` run

## Repaired

| # | defect | fix | measured effect |
|---|---|---|---|
| 1 | `A_future` underpowered (SNR 0.74 / 0.59, below 1 — null uninformative) | `K` 8 -> **24** | SNR **1.64 / 1.36**, ceiling 0.43 -> **0.62** / 0.37 -> **0.58**; `A_full` SNR -> 5.34 / 4.23 |
| 2 | fixed horizon not comparable across bins (`H/remaining` 0.09 at 10% progress vs 0.55 at 85%) | `record_fracs` capped at **0.60** | `H/remaining` now in [0.09, 0.21] everywhere |
| 3 | G4 took the per-bin **max over 13 layers** — selection-biased | evaluate at the **validation-selected** layer | verdict unchanged (6/6 bins both arms) |
| 4 | bootstrap held both fitted models fixed — CI too narrow, and G1 was the only passing gate | `--refit_boot`: re-split + **re-fit** per replicate | G1 survives; arm 1 `Delta_concordance` flips from "excludes 0" to **includes 0** |
| 5 | Experiment 2's "hidden" score was `cheap + [h_i; h_global]`, inconsistent with 1b's candidate-level test | primary score is now `cheap + h_i` | 0.610 -> **0.581** (arm 1), 0.635 -> **0.577** (arm 2) — at or below the controls |

## A defect introduced by the fix itself, then caught

The first `--refit_boot` implementation drew documents with replacement from the
whole pool and split *afterwards*, so a duplicated document could land in both
train and test. Symptom: the refit point estimate came out at **+0.0748** against
+0.0499 from the fixed-model bootstrap — a bootstrap that adds variance should
not move the point estimate that far. Rewritten to resample *within* each
disjoint split, with an assertion on train/test document overlap; the estimate
returned to **+0.0506**, matching the fixed-model value.

## Final checks

- **Negative control** on the `K=24` data (labels globally permuted):
  `Delta_R2` -0.0005, `Delta_within_R2` +0.0004, `Delta_concordance` +0.0013 —
  all null, as required.
- **Verdict unchanged** across all three versions of the experiment
  (contaminated labels, corrected at `K=8`, corrected at `K=24`) and across both
  reference policies: **G1 PASS, G2 FAIL, G3 FAIL, G4 PASS -> KILL**.

The direction did not survive, and the reason sharpened at every round: the
frozen representation carries the rollout-defined **state value** (`Delta_R2`
+0.183 / +0.320, positive in 95-96% of layer x timestep cells) and not the
**action advantage** (`h_i` concordance positive in 40/78 and 30/78 cells).

---

# Round 6 — task-utility collection (2026-08-30)

## Defect 12 — record points tested with exact equality, silently skipping half

```python
if filled in record_at:      # record_at = {24, 56, 88, 120}
```

`filled` counts positions committed so far, and block diffusion commits SEVERAL
positions per step, so the counter jumps (22 -> 27) straight over a record
point. Measured on the first 150-prompt run: the four points were hit 332 / 264
/ 236 / 224 times, i.e. only **1.96 of 4 per prompt**, yielding 1,056 examples
against a designed 2,400.

**What it does and does not cost.** The states that WERE recorded are entirely
valid -- labels, features and hidden states are correct. What is lost is
*coverage*: 56% fewer samples, and timestep coverage thinned by an arbitrary
rather than a systematic rule. It is a power loss, not a bias.

**Fixed** by triggering on crossing a threshold rather than landing on it.

**Deliberately NOT applied to the run in flight.** taskB was mid-collection when
this was found; patching the code would have made the two halves
non-homogeneous, which is a worse problem than the sample loss. Both halves use
the same (buggy) sampling rule and are therefore comparable; the fix applies to
any future collection.

## Label quality on the real data — the best of the project

| metric | task-utility (`A_task`) | best Path-LL run |
|---|---|---|
| SNR | **9.7** | 5.34 |
| `R^2` ceiling | **0.907** | 0.842 |
| fraction of non-zero labels | 27.1% | — |
| within-state share of variance | 55.4% | 62% (but on a 6-8% differential) |

27.1% of candidate commits change whether the final answer is right, and 18.2%
change it by at least 0.25. This is the effect-size structure the Path-LL study
never had: one token flipping a verifiable outcome is a 100% swing, against the
~3% differential that the earlier advantage label was measuring.
