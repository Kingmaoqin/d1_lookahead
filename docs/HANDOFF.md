# Handoff — continue Direction 1: the task-utility experiment

Paste the block below into a fresh agent window. Everything it needs to know is
in it; the repo docs it points at are the source of truth for anything else.

---

You are continuing a research project. **Read these four files first, in this
order, before touching anything:**

```
/home/xqin5/diffusion_LLM/d1_lookahead/docs/FINDINGS.md            # what was found, incl. the KILL verdict
/home/xqin5/diffusion_LLM/d1_lookahead/docs/PREREGISTRATION.md     # frozen thresholds + 4 corrections
/home/xqin5/diffusion_LLM/d1_lookahead/docs/AUDIT.md               # 3 rounds of audit, 10 bugs found
/home/xqin5/diffusion_LLM/d1_lookahead/docs/EXPERIMENT_QUEUE.md    # what is done / gated / next  <-- Round 5 is your task
```

## The one-paragraph situation

The study tested whether a frozen diffusion LM's hidden states linearly encode a
rollout-defined **action advantage** `A(i|s_t)` beyond strong output-distribution
controls. Across three label versions, two reference policies, two independent
backbones (MDLM-owt, SEDD-small) and three horizons, the pre-registered gate
**FAILED (G1 pass, G2 fail, G3 fail, G4 pass) -> KILL**. What the representation
*does* linearly encode is the **state value** `V` (`Delta_R2` +0.183 / +0.320,
positive in 95-96% of layer x timestep cells), not the candidate-level advantage
(`h_i` ranks candidates at chance: 40/78, 30/78, 38/78 cells positive).

**Phase 0B and the decoder/Pareto experiments are CLOSED by protocol.** The brief
says: *"If killed: stop. Do not add RL, SAEs, a Transformer controller, or joint
fine-tuning."* Do not re-open them. Do not add model capacity to make a signal
appear.

## Your task, and why it is not a protocol violation

Exactly one pre-registered deliverable was never completed: the brief's
**secondary external-validity check** — *"On verifiable tasks only, separately
construct a task-utility version of the label using exact answer / unit-test
reward."* It was blocked because MDLM-owt and SEDD-small are unconditional
OpenWebText models with no task to be right or wrong about. The PI has confirmed
that completing it is **not** a violation of the kill clause, and has approved
changing `pi_ref` for the new backbone.

A third backbone was found and qualified for exactly this: **nvidia/Nemotron-Labs-Diffusion-3B**
(NVIDIA, US — satisfies the hard "no Chinese-origin models" rule, which excludes
LLaDA and Dream). It is instruction-tuned AND natively masked-diffusion, so the
brief's fallback of an AR->DLM LoRA conversion is unnecessary.

**Phase S is already done and PASSED 4/4** (`results/phase_s_nemotron/report.json`).
Do not redo it. Key numbers you will need:
- GSM8K accuracy 59.9% over 48 prompts x 8 rollouts (native `generate` = 66.7%)
- **31.2% of prompts (15/48) have MIXED outcomes** across rollout seeds;
  19/48 always correct, 14/48 always wrong
- within-prompt reward variance 0.0534, between-prompt 0.187

## What to build

`scripts/collect_task_labels.py` — the collection driver — and then the analysis.
The infrastructure below is written, tested, and ready to import:

| file | what it gives you |
|---|---|
| `src/nemotron_local.py` | `load_nemotron(device, shard=False)` -> wrapper whose `forward(ids, output_hidden_states=True)` returns `(logits, [26 hidden states])`. `MASK_TOKEN_ID = 100`. |
| `src/nemotron_policy.py` | `BlockPiRefConfig`, `block_rollout`, **`safe_block_rollout`** (use this one), `make_state`, `topk_logprobs`, `sample_tokens`, `expected_logp` |
| `src/collect_task.py` | `TaskCollectConfig`, `snapshot`, `new_hist`, `update_hist`, `pick_candidates`, `task_reward`, `gold_answer`, `project_logprobs` |
| `src/features.py` | `assemble_cheap` (C1/C2/C3 blocks), `assemble_hidden`, `ProjCache` |
| `src/probes.py` | `fit_linear`, `fit_linear_2block`, `predict_2block`, `fit_mlp`, `r2_score`, `within_state_r2`, `within_state_concordance`, `noise_ceiling`, `cluster_bootstrap` |
| `src/dataset.py` | `load_labels`, `block`, `doc_splits` |

### The estimand (mirror `collect.py`'s docstring style — write it down and freeze it)

```
A_task(i | s_t) = P(correct | commit (i, x_hat_i) at s_t, then pi_ref)
                - P(correct |                              pi_ref)
```
estimated by CRN-coupled paired rollouts run **to completion**, reward = exact
match on the GSM8K final numeric answer. Also record `A_full` (Path-LL
advantage) from the same rollouts — one set of rollouts, two labels.

**Rolling to completion is what makes this clean:** both branches fill every
remaining masked position, so they are automatically matched in positions
consumed. The denominator mismatch and depletion artifact that had to be
repaired on MDLM (see AUDIT.md defects 2 and 3) **cannot arise here**. Do not
reintroduce a truncated horizon.

### Settings already validated — do not re-tune without a reason

```
pi_ref     : block diffusion, block_length=32, threshold=0.9,
             temperature=0.2, top_k=50        <- temperature is the dominant
             factor: T=1.0 -> 25.0%, T=0.5 -> 54.2%, T=0.2 -> 62.5% GSM8K
gen_len    : 160
K          : 8 CRN-coupled paired rollouts (raise if label SNR < 1)
n_cand     : 4 per state (2 uniform-random masked + 2 high-confidence x unstable)
record_fracs: (0.15, 0.35, 0.55, 0.75) of the generation region
env        : conda run -n p08_skilloverload    <- NOT `llm`; needs transformers>=5
```

### The design point that will sink you if you miss it

Of 48 prompts, 19 are always correct and 14 always wrong — `A_task` is
**identically 0** on both. Collecting naively makes ~69% of labels a constant
zero and floors the detectable variance. **Pre-screen prompts by running K
rollouts and keeping the mixed ones as an oversampled stratum, while preserving
a naturally-sampled held-out set.** Record the stratum per example
(`stratum` field), exactly as the existing collector does, so held-out metrics
can be reported on the natural stratum separately. This mirrors the
"informative candidate stratum" pattern already in `collect.py`.

### Hardware — read this before you launch anything

The GPUs are shared with another tenant that occupies ~73 GB of each 80 GB card
and **fluctuates**. The checkpoint is 7.35 GB of weights against ~7-8 GB free.

- **Use single-GPU jobs with `safe_block_rollout`, not sharding.** Sharding
  works (`load_nemotron(shard=True)`, ~3 GB/GPU, 3.4 GB peak) but measured
  **4.6x lower total throughput** because pipeline stages run one at a time.
- Always launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and pick
  the GPU with most free memory at launch time.
- Micro-batch 2 is the ceiling at L~280 on one GPU. `safe_block_rollout` halves
  it on OOM and waits up to 180 s for the co-tenant rather than dying.
- Budget: ~40 rollout chains per state at ~0.65 s/chain -> ~26 s/state,
  ~104 s/prompt at 4 states. 100 prompts across 3 GPUs ~= 1 hour.
- **`pgrep -f <pattern> | xargs kill` will kill your own shell** if the pattern
  appears in your command line. Use `pgrep -f "[p]attern"`. This was hit twice.

### Analysis, once labels exist

Reuse the existing scripts — they are backbone-agnostic once the shards match
`dataset.load_labels`'s expected keys:
```
scripts/label_diagnostics.py     # FIRST. If A_task SNR < 1 the null is uninformative
                                 # and you must raise K before interpreting anything
scripts/exp1_decodability.py     # --refit_boot 200 for honest CIs (includes fitting variance)
scripts/exp1b_within_state.py    # THE decisive split: h_i (candidate-level) vs h_global (state-level)
scripts/exp2_matched.py          # run BOTH --mode caliper AND --mode strict; also --placebo
scripts/kill_gate.py A_task _<suffix>
```

**Run `exp1b` before drawing any conclusion from a pooled `Delta_R2`.** Three
successive versions of this study reproduced the same trap: pooled `Delta_R2` is
positive and significant every time, and every time it decomposes into the
state-level channel. `h_global` is constant within a state and cannot rank
candidates; only `h_i` can. Keep them separate throughout.

For `exp2`, note that `caliper` matching equalises the cheap-control prediction
by construction and therefore inflates "gap over cheap" — report the
non-circular `strict` mode alongside, and run `--placebo` (hidden block replaced
by Gaussian noise) to confirm the procedure is not biased before believing any
positive result.

### Standards this project holds itself to

- The backbone stays **frozen** (`requires_grad_(False)`). Only probes are fitted.
- Document-level splits everywhere; never let a prompt straddle train/test.
- Report a **noise ceiling** next to every `R^2`. A label with SNR < 1 gives an
  uninformative null, not evidence of absence — say so explicitly.
- Run the **negative control** (`--null global`) and expect `Delta_R2 ~ 0`.
- When you find a bug, record it in `docs/AUDIT.md` with what it would have
  changed had it not been caught — including bugs you introduce yourself. Three
  of the ten found so far were self-inflicted during fixes.
- Do not report a result whose sign depends on an analysis choice without saying
  so and flagging it for the PI.

### Deliverables

Update `docs/FINDINGS.md`, `docs/EXPERIMENT_QUEUE.md` and `docs/AUDIT.md`, then
update the illustrated report artifact at
**https://claude.ai/code/artifact/95b0fc3b-64e6-46f7-8ae5-6dad5de34acc**
(read it first with the Artifact tool using that `url`, then republish to the
same URL so the link is preserved). Reply to the user in **Chinese**.
