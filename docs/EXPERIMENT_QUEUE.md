# Experiment queue — completing Direction 1

The interim kill gate FAILED on both reference policies. Per the brief, that
closes **Phase 0B** (quality-vs-wall-clock Pareto) and any decoder work: *"If
killed: stop. Do not add RL, SAEs, a Transformer controller, or joint
fine-tuning."* Nothing below re-opens them.

What remains is the work that makes a NEGATIVE result defensible. A negative
result's weakness is the opposite of a positive one's: it is attacked with
"you didn't look hard enough" and "one model proves nothing", so the queue
targets exactly those two objections.

## Done

| item | result |
|---|---|
| Phase S — MDLM (backbone 1) | 5/5 gates PASS |
| Phase S — SEDD (backbone 2) | 5/5 gates PASS; conditional NLL *better* than MDLM at every mask ratio (1.54 vs 1.86 at r=0.2), distinct-2 0.756 vs 0.697 |
| Labels — ancestral `pi_ref`, K=24 | 14,400 ex; `A_full` SNR 5.34, `A_future` SNR 1.64 |
| Labels — confidence `pi_ref`, K=24 | 14,400 ex; SNR 4.23 / 1.36 |
| Experiments 1, 1b, 2, kill gate | both arms: G1 PASS, G2 FAIL, G3 FAIL, G4 PASS -> **KILL** |
| Negative control (permuted labels) | `Delta_R2` -0.0005 — null, no leak |
| **Nonlinear headroom (rebuilt MLP)** | closes the "linear probe too weak" objection — see below |
| **Horizon sensitivity H=8** | same verdict as H=16 |

### Nonlinear headroom — the objection this closes

The original MLP control scored *below* the linear controls (0.450 vs 0.518),
which is diagnostic of an undertrained control, not of absent information: the
targets have sd ~0.06, so MSE ~4e-3 and `weight_decay=1e-2` dominated the
gradient. Rebuilt with a standardised target and a hyper-parameter sweep, the
MLP now matches the linear baseline on the controls (0.5186 vs 0.5162), so it
is a credible upper bound. With that fixed:

| model | within-state `R^2` | concordance |
|---|---|---|
| cheap (linear) | 0.5588 | 0.7788 |
| cheap + `h_i` (linear) | 0.5641 | 0.7758 |
| cheap + `h_i` (**MLP**) | 0.4779 | 0.7522 |

Adding `h_i` fails to improve candidate ranking **linearly or nonlinearly** —
the MLP is worse, overfitting the extra 768 dimensions. Same in arm 2
(0.7663 -> 0.7691 linear, 0.7479 MLP).

### Horizon sensitivity

| run | `h_i` within-`R^2` cells>0 / median | `h_i` **concordance** cells>0 / median | `h_global` `R^2` cells>0 / median |
|---|---|---|---|
| H=8 (K=16) | 42/78, +0.0005 | **31/78**, +0.0000 | **75/78**, +0.0489 |
| H=16 (K=24) | 49/78, +0.0017 | **40/78**, +0.0007 | **75/78**, +0.0312 |

The candidate channel is at chance at both horizons; the state channel is
systematic at both. The verdict is not horizon-specific.

## Running

| item | GPUs | ETA |
|---|---|---|
| **SEDD backbone-2 collection** (ancestral, K=24, H=16, 400 prompts) | 1, 2 | ~3h |
| **H=32 horizon sensitivity** (ancestral, K=16, 400 prompts) | 0, 3 | ~4h |

Both are followed by `label_diagnostics` -> `exp1` -> `exp1b` -> `exp2` ->
`kill_gate`, i.e. the minimum phenomenon set the brief specifies for a
replication: layer/timestep decodability, incremental `Delta_R2`, matched
candidates.

## Still outstanding

1. **`V^{pi_ref}` per-seed replicates for the MDLM arms.** `V` is stored only as
   a mean over seeds, so it has no noise ceiling and its `Delta_R2` cannot be
   ceiling-corrected. `collect.py` now stores `V_pertok_seeds`, so the SEDD run
   will have them; the two MDLM arms need a cheap V-only backfill (V branches
   are `B*K` chains against `N*K` for Q, so ~1/7 the cost of a full collection).
   Required before `V` can be reported as a finding in its own right.
2. **Task-utility target on verifiable tasks.** The brief asks for a
   task-utility label (exact answer / unit-test reward) as an external-validity
   check. **Not attainable on either backbone**: MDLM-owt and SEDD-small are
   both unconditional OpenWebText models with no instruction or task ability.
   This is a substrate limit, reported as a scope restriction rather than
   approximated with a proxy.

## Deliberately NOT queued

- Phase 0B decoder / Pareto experiments — gate failed.
- Any added capacity (RL, SAEs, controller, fine-tuning) — explicitly forbidden
  after a kill.

---

# Round 5 — third backbone qualified, task-utility unblocked (2026-08-27/28)

## What changed

The one deliverable the brief asked for and neither of the first two backbones
could provide -- a **task-utility label on verifiable tasks** -- is now
reachable. `nvidia/Nemotron-Labs-Diffusion-3B` (NVIDIA, US, non-Chinese-origin)
is instruction-tuned AND has a native masked-diffusion mode, so no AR->DLM LoRA
conversion is needed.

## Phase S — SUBSTRATE QUALIFIED (4/4)

| gate | check | result |
|---|---|---|
| S1 | task capability non-degenerate | GSM8K **59.9%** over 48 prompts x 8 rollouts (native `generate` reference 66.7%) |
| **S3** | **within-prompt reward variance** | variance **0.0534**; **15/48 = 31.2% of prompts have MIXED outcomes** (accuracy 0.25-0.875); between-prompt variance 0.187 |
| S4 | hidden states consistent | 26 layers via forward hooks, stable |
| S5 | rollouts complete | 0 incomplete, mean NFE 386 |

S3 was the gate that could have killed this outright: if `pi_ref` were near
deterministic then every prompt would be always-right or always-wrong, every
`A_task` would be 0, and there would be nothing to probe.

## Substrate facts established

- `mask_token_id = 100`; 26 layers; hidden 3072; **vocab 131072** (2.6x MDLM,
  so every top-k and projection is chunked)
- a plain `forward()` runs **bidirectional** attention -- `modeling_ministral.py`
  builds a causal mask only when `use_causal_mask=True` is passed. Verified:
  changing text AFTER a masked position moves that position's prediction by
  0.95 in max probability
- `output_hidden_states=True` returns **None**; per-layer states are captured
  with forward hooks on the decoder layers
- `pi_ref` is Nemotron's native **block diffusion** (block 32, threshold 0.9),
  temperature **0.2** -- a sweep showed temperature is the dominant factor for
  task accuracy (T=1.0 -> 25.0%, T=0.5 -> 54.2%, **T=0.2 -> 62.5%**). Greedy
  also reaches 62.5% but destroys the stochasticity CRN pairing needs.

## Hardware finding — sharding is the wrong answer here

The checkpoint is 7.35 GB of weights against ~7-8 GB of free memory on
co-tenanted GPUs; a first Phase-S run was killed mid-flight. Sharding across 3
GPUs is stable (~3 GB weights each, 3.4 GB peak) but **3.5-4.6x slower in total
throughput**, because pipeline stages run one at a time and two thirds of the
GPUs idle:

| approach | total throughput | headroom |
|---|---|---|
| sharded over 3 GPUs, B=24 | 1.0 chains/s | 2-3 GB |
| **3 independent single-GPU jobs, B=2** | **4.6 chains/s** | 0.2 GB |

Chosen: single-GPU with an OOM-resilient rollout (`safe_block_rollout`) that
halves the micro-batch on OOM and, if even one sequence will not fit, waits up
to 180 s for the co-tenant rather than dying and losing the run. It asserts the
returned row count, which is exactly the failure that silently dropped data on
the MDLM backbone. Both paths are kept: `load_nemotron(shard=True|False)`.

## Ready infrastructure

- `src/nemotron_local.py` — backbone wrapper, hook-based hidden states, optional sharding
- `src/nemotron_policy.py` — block-diffusion `pi_ref`, CRN sampling, `safe_block_rollout`
- `src/collect_task.py` — snapshots, candidate strata, GSM8K reward, JL projection
- `scripts/phase_s_nemotron.py` — the qualification above

## Still to build

The collection driver and the analysis. **One design point must not be missed:**
of 48 prompts, 19 are always correct and 14 always wrong — `A_task` is
identically 0 on both. Collecting naively would make ~69% of labels a constant
zero and floor the detectable variance. Prompts should be **pre-screened for
mixed outcomes** as an oversampled stratum while a naturally-sampled held-out
set is preserved — the same pattern the existing design already uses for the
"informative" candidate stratum.
