# Task-E SVAMP candidate-utility preregistration

Locked after substrate qualification but before any candidate-level SVAMP label
is collected or read.

## Qualified substrate

The fixed qualification sample (100 prompts, K=8) produced 69.75% accuracy,
23% mixed prompts, and mean within-prompt reward variance 0.0496. SVAMP therefore
passes the non-degeneracy gate.

## Collection

* Dataset: all 300 SVAMP test prompts are screened; collect 200 prompts with
  mixed prompts oversampled and prompt stratum retained.
* Four early/mid states per prompt, six candidates per state, K=8 CRN rollouts.
* Expected 800 states / 4,800 candidate examples.
* Same frozen Nemotron backbone and π_ref as GSM8K.

## Locked analysis

For document splits 0,1,2:

1. select the hidden layer on validation only using `cheap+H_i`;
2. fit rank-4 bilinear, no-state-interaction, cheap-only, shuffled-h_g, and
   Gaussian-h_g under the same 25-epoch budget;
3. report concordance on held-out documents and a five-fold OOF aggregate;
4. primary positive threshold: bilinear−cheap at least +0.020 with document
   bootstrap CI excluding 0;
5. relational threshold: bilinear−no-state-interaction at least +0.010 with CI
   excluding 0.

Candidate and prompt strata are secondary. Task dependence is assessed by the
direction and magnitude relative to GSM8K Task C/D, not by requiring identical
layers or absolute baseline concordance.
