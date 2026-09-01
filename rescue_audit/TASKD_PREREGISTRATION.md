# Task-D task-utility expansion preregistration

Locked before any `labels_taskD` shard or screening file is created/read.

## Data

* Backbone/task: Nemotron Diffusion 3B on GSM8K exact-answer utility.
* Prompt range: offsets 700–1299 (`offset=700`, `n_screen=600`).
* Collect 570 prompts, four recorded states per prompt, six candidates per state.
* K=8 CRN-paired rollouts per candidate.
* Expected new size: 2,280 states; combined with taskC: 3,000 states.
* No prompt overlaps taskA/B (0–439) or taskC (440–699 screened range).

## Frozen primary design

Before Task-D labels are read, fit on taskC only:

1. document split seed 0: 60% train / 15% validation / 25% internal test;
2. choose layer by validation performance of `cheap+H_i`, exactly as in the broad
   screen;
3. fit rank-4 bilinear `cheap + H_i + H_g + H_i^TUV^TH_g` with the quick-screen
   training budget (25 epochs, patience 8, seed 0);
4. fit the cheap-only and no-state-interaction controls under the same data and
   training/selection protocol;
5. freeze preprocessing and weights, then apply once to every Task-D state.

Primary endpoint: within-state pairwise concordance difference, bilinear minus
cheap-only. A positive confirmation requires point estimate at least +0.020 and
document-cluster bootstrap 95% CI excluding 0.

Relational endpoint: bilinear minus no-state-interaction. It is positive if the
point estimate is at least +0.010 and the 95% CI excludes 0.

## Secondary designs

* Refit the same locked rank-4 family within Task-D using three fixed document
  splits (seeds 0,1,2), with layer chosen on each validation split only.
* Run five-fold out-of-fold predictions on combined taskC+taskD, so every prompt
  is evaluated exactly once.
* Report candidate natural/informative strata and prompt natural/mixed strata.
* Report shuffled-h_g, Gaussian-h_g, and no-state-interaction controls.

The frozen primary decides cross-offset transport. Secondary refitting decides
whether the signal exists but requires domain adaptation. These outcomes must
not be conflated.

## Tie handling

A_task is discrete. Concordance uses only candidate pairs with unequal observed
advantages. Top-1 is secondary and must be tie-aware and restricted separately
to states with nonzero candidate-value span.
