# Prior-art re-search — run 2026-08-19 (arXiv API, sorted by submission date)

Queries run, per the brief: *future value probe diffusion language model*,
*amortized lookahead dLLM*, *hidden-state path value diffusion LM*,
*value prediction unmasking diffusion language model*, plus schedule/unmasking-policy sweeps.

## Verdict: NO exact collision

Nothing found claims that a rollout-defined future path value is *linearly
decodable from a frozen DLM state*. The direction survives. Three items must
nevertheless be added to the positioning table — one of them materially
sharpens what we are allowed to claim.

## New entries required

| Work | arXiv | Date | What it establishes | Required distinction |
|---|---|---|---|---|
| **Neural Estimation of Pairwise MI in Masked Discrete Sequence Models** | 2605.20187 | 2026-01 | Trains a head **on the hidden states of a pretrained MDM** to predict the full pairwise conditional-MI matrix in **one forward pass**, and uses it for MI-guided parallel decoding | **The most important new neighbour.** It already establishes that an MDM's hidden states support a useful single-pass readout, so "hidden states support a cheap readout" is *not* available to us as novelty. Their target is **dependency structure**, supervised by MI computed from the model's own conditional distributions — an output-distribution-derived quantity. Ours is a **rollout-defined scalar future path value / action advantage**, which is not a functional of the exposed conditionals at `s_t` at all. We must state this contrast explicitly. |
| **Ripple-Pivot Search (RPS)** | 2608.11742 | 2026-08-12 | Identifies a "ripple effect": committing a mid-entropy pivot sharply reduces uncertainty elsewhere. Chooses *where* and *what* to decode by **test-time lookahead evaluation of downstream benefit** | Conceptually the closest decision to our `A^{pi_ref}(i | s_t)` — the downstream benefit of committing position `i` with a token. But it **searches** for that benefit at test time. We test whether it is already readable. Strengthens motivation; does not pre-empt the representation claim. Note it also reports pivots at *mid* entropy, i.e. the quantity is explicitly not monotone in confidence — supportive of our matched-candidate design. |
| **Adaptive Multi-Step Lookahead Decoding for DLMs** | 2607.15655 | 2026-07-17 | Adaptive rollout depth driven by candidate-score variance; re-triggers lookahead from informative states | Same family as POKE / LookUM: test-time future search. Same distinction. |

Also noted, not competing: *Certified-optimal unmasking schedules via unmasking
growth complexity* (2608.13520, schedule theory, no learned readout);
*Stopping Computation for Converged Tokens* (2602.06412); *Stop the Flip-Flop*
(2602.06161) — both realized-trajectory stability, the TraceLock distinction
already in the brief.

## Consequence for the framing rule

The brief already forbids "hidden states contain more than confidence".
2605.20187 forces a second forbidden sentence:

> ~~"Hidden states of a masked diffusion model support a useful one-pass readout."~~

That is now established prior art. The surviving claim is only:

> **A frozen DLM representation linearly exposes a ROLLOUT-DEFINED value of
> alternative future decoding actions — a quantity that is not a functional of
> its exposed output distribution at `s_t` — beyond what that output
> distribution and cheap trajectory signals reveal.**
