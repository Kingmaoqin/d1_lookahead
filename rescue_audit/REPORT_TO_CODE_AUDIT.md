# Report-to-code audit

Status: Phase R0, in progress. Updated 2026-08-31.

## Prior report access and reconstruction

- Requested URL: `https://claude.ai/code/artifact/95b0fc3b-64e6-46f7-8ae5-6dad5de34acc`.
- Browser attempt: blocked by the browsing safety layer; search fallback was blocked by `robots.txt` and exposed only the Claude login page. The remote page was therefore **not** counted as read at that point.
- A direct HTML snapshot was later recovered at `rescue_audit/PRIOR_REPORT_SNAPSHOT.html` and converted to `rescue_audit/PRIOR_REPORT_SNAPSHOT.md`.
- The user then supplied the authoritative downloaded text at `文档内容`. After whitespace normalization, that full text is an exact substring of the Markdown snapshot (`SequenceMatcher` ratio 0.99817); the extra snapshot content is only a source/title header. The report has now been read in full.
- Local sources cross-checked: `README.md`, `docs/FINDINGS.md`, `docs/AUDIT.md`, `docs/PREREGISTRATION.md`, `docs/EXPERIMENT_QUEUE.md`, `docs/HANDOFF.md`, `docs/PRIOR_ART_UPDATE.md`.

Important version fact: the Artifact/user-supplied report is newer than `docs/FINDINGS.md`. It contains the 08-29 AR-to-DLM experiment and the 08-30 Nemotron value-readout study; those are not fully represented in `FINDINGS.md`.

## Headline-result provenance

| Headline result | Collection / implementation | Analysis | Primary artifacts | Reproduction/audit status |
|---|---|---|---|---|
| MDLM substrate qualified | `scripts/phase_s_qualify.py`, `src/mdlm_local.py`, `src/policy.py` | qualification script | `results/phase_s/phase_s_report.json` | Historical artifact present; independent backbone equivalence not yet rerun in R0. |
| Original labels were contaminated by state value and depletion | historical `src/policy.py`, `src/collect.py`; corrected paths at `src/policy.py:194-222`, `src/collect.py:245-283` | algebra and diagnostics in `docs/AUDIT.md` | superseded `data/labels_a,b`, `results/exp1`; corrected `a2,b2` and final `a3,b3` | Algebra and corrected branch counts are under independent Auditor A review. Old results retained, correctly marked superseded. |
| Final MDLM ancestral: G1 pass, G2/G3 fail, G4 pass | `scripts/collect_labels.py`, `src/collect.py`, `src/policy.py` | `exp1_decodability.py`, `exp1b_within_state.py`, `exp2_matched.py`, `kill_gate.py` | `results/exp1_k24`, `exp1b_within_state_a3b3.json`, `exp2_k24`, `kill_gate_A_pertok_ANC.json` | Numerical artifact exists. **Not yet accepted as fully reproduced:** new audits found C2 timing mismatch, incomplete refit selection, and Exp2 test-set selection/estimand mixing. |
| Final MDLM confidence arm gives the same KILL pattern | same, tags `c3,d3`, confidence order | same | `results/exp1_k24conf`, `exp1b_within_state_c3d3.json`, `exp2_k24conf`, `kill_gate_A_pertok_CONF.json` | Same caveats as ancestral arm. |
| Candidate channel is at chance; state channel is systematic | hidden extraction in `src/features.py`; final label tags above | `exp1b_within_state.py` | `results/exp1b_within_state_{a3b3,c3d3,s1s2}.json` | Headline counts 40/78, 30/78, 38/78 are traceable. New rescue specifically tests relational/nonlinear alternatives that old additive readout did not cover. |
| State value V is strongly linearly readable | V labels from collection plus `scripts/backfill_v_seeds.py` | `exp1_decodability.py` | final MDLM/SEDD exp1 JSON; backfill logs | ΔR² +0.183/+0.320 and ceilings 0.9942/0.9917 are traceable. Old `FINDINGS.md` contains a stale “no ceiling” limitation that the later report corrects. |
| Horizon H=8/16/32 does not rescue candidate readout | `collect_labels.py` tags `h8a,h8b,h32a,h32b` | `exp1b_within_state.py` | `results/exp1b_within_state_h8ah8b.json`, `_h32ah32b.json` | Historical artifacts present; not a test of H=4/12/24/48 or relational probes. |
| SEDD replication gives same overall KILL; caliper G3 positive but strict not significant | `src/sedd_local.py`, standard collector tags `s1,s2` | exp1/1b/2/kill gate | `results/exp1_sedd`, `exp2_sedd`, `exp2_seddstrict`, `exp2_seddplacebo`, SEDD kill-gate JSON | Numerical discrepancy across matching definitions is real. New Auditor B found additional test-set baseline selection and seed/CI estimand problems in Exp2; old G3 evidence must be recomputed. |
| Nemotron-3B substrate qualified on GSM8K | `src/nemotron_local.py`, `src/nemotron_policy.py`, `scripts/phase_s_nemotron.py` | same script | `results/phase_s_nemotron/report.json`, `rewards.npy` | Traceable: 59.9% accuracy and 15/48 mixed prompts. No R0 rerun yet. |
| Nemotron candidate-level `A_task` shows no hidden increment | `scripts/collect_task_labels.py`, `src/collect_task.py` | exp1/exp1b | `data/labels_taskA,taskB`, `results/exp1_task`, `exp1b_within_state_taskAtaskB.json` | Core γ=0 result is traceable. **Provenance gap:** report's AUC 0.669→0.669 and SNR 9.727/ceiling 0.907 have no saved matching analysis artifact; `exp1_task` stores ceiling NaN. Must be independently recomputed. C2 history is also one state late. |
| Nemotron V_reward predicts final correctness but mostly prompt difficulty | `scripts/collect_v_readout.py`, `scripts/analyze_v_readout.py` | `analyze_v_readout.py` | `data/v_readout_results.json`, `labels_vreadA,B` | Fully traceable: hidden AUC .8896 vs cheap .8229, ΔR² .1317, negative within-prompt R², selection curve. This is V/difficulty evidence, not candidate advantage. |
| AR-to-DLM conversion improves NELBO but fails generation/task gates | sibling project `/home/xqin5/diffusion_LLM/ar2diff/` | sibling evaluation scripts | `ar2diff/results/eval_convert.json`, train/eval logs | Outside this directory but provenance located. Not part of rescue P0-P13 screen. |

## The 17 known report defects

The complete table was read in the user-supplied report. The historical set is: MC label SNR; Q/V denominator mismatch; unequal branch depletion; shared ridge alpha; under-updated trajectory controls; wrong confidence-order statistic; leaky first refit bootstrap; OOM retry row loss; undertrained MLP; top-k vocabulary-ID indexing; unsafe `pgrep`; `None` attention mask reverting to causal; exact-equality block record points; missing `A_task_seeds` mapping; AUC name shadowing; ceiling tuple formatting; scalar-only bootstrap. The report's item “8” is a placebo validation, not an additional defect.

## New R0 discrepancies (not in the prior report)

1. **Current online history features are still one state late.** `collect_labels.py` and `collect_task_labels.py` snapshot before `update_hist`, whereas `repair_trajectory_features.py` establishes the intended current-transition-before-snapshot definition. Labels and hidden states are unaffected; C2 controls are shifted/weakened. Existing datasets require deterministic backfill or a conservative C2 exclusion sensitivity analysis.
2. **Exp2 selects the best control on the test matched pairs.** `scripts/exp2_matched.py:262-270` uses test accuracy to pick `best_ctrl`, then reports/bootstrap-compares on the same pairs. That is test-set selection leakage.
3. **Exp2 mixes estimands across seeds.** The headline accuracy averages three split seeds, while inference at `:223-276` uses only the final seed's pairs/scores. The G3 point estimate and interval therefore do not describe the same statistic.
4. **The kill gate ignores refit-bootstrap output.** `scripts/kill_gate.py` reads the fixed-fit test bootstrap for G1. The so-called refit bootstrap fixes the original split pools and a median layer rather than rerunning the complete validation selection, so it does not capture full pipeline-selection uncertainty.
5. **The advertised synthetic self-test is currently broken.** `pipeline_selftest.py` passes an unsupported `epochs=` argument to `fit_mlp`.
6. **Old MLP evidence is narrower than the report's prose suggests.** It is single-layer MSE on concatenated/additive features, usually one initialization; it does not test candidate×global interactions, ranking losses, temporal deltas, multi-layer mixtures, or 3–5 random initializations.
7. **Result directories preserve a known-invalid refit result next to the repaired result.** `results/exp1_k24*` can be mistaken for the repaired refit outputs stored separately under `results/refit_fixed_*`.
8. **A_task headline diagnostics lack a saved direct artifact.** AUC and label-quality numbers must be recomputed from shards rather than cited as independently reproducible output.
9. **Old task recording is selection-biased, not merely smaller.** Exact threshold hits depend on how many positions the block policy commits, which itself depends on confidence/trajectory dynamics. The 264/296 observed task states are therefore a selected state population; old taskA/B cannot be a natural or confirmatory holdout.
10. **Task C1 duplicated logp1 as logit_max.** `collect_task.py` populated `lg_top` with normalized log probabilities. This weakens the task cheap baseline; fixed for future collections on 2026-08-31.
11. **Confidence-arm random order is a persistent random-utility policy.** Its per-position Gumbel is reused across steps. The resulting Q/V label is valid for that explicitly defined policy, but it is not the same as redrawing a categorical confidence order at every step. Historical prose calling it standard stochastic MaskGIT ordering is too broad.
12. **Matched-pair counts were doubled.** `mine_pairs()` emitted both `(i,j)` and `(j,i)`; a3+b3 seed 0 reports 152 rows but only 76 unique unordered pairs. Accuracy is unchanged, but displayed sample size/power is wrong.
13. **Exp2 matching has already been tuned on the old test data.** It uses test-label scale and labels to define the conditional matched population, and the ladder explored several tolerances/modes on the same tests. Those results are exploratory even after code repair.
14. **The three “seeds” are overlapping re-splits, not replications.** Across 400 documents, test-test overlaps are 21/29/28 and 207 documents are test in one split but train in another. No individual split leaks, but seed spread is not an independent-repeat uncertainty measure.
15. **Several visualization/error bars are not inferential intervals.** `make_figures.py` uses standard deviation across the three overlapping re-splits. These must not be presented as SE/CI.
16. **Phase-S SEDD hidden consistency is weaker than described.** The check repeats the same unmodified state three times rather than following three trajectory states.

## Independent Auditor A/B discrepancy summary

- Agreement: both auditors independently conclude that existing results are suitable for exploratory reconstruction, not fresh confirmation.
- Auditor A found the core Q/V/A algebra, seed pairing, RB expectation, ancestral order, hidden extraction, and task reward algebra correct. Its new concerns are data/control extraction and the precise confidence-policy estimand.
- Auditor B found no within-split document leakage and accepted the basic metric definitions, but found invalid/partial uncertainty accounting, test-set selection in matching, pair duplication, repeated holdout reuse, and an overly broad interpretation of the old MLP.
- Neither auditor's findings currently show that a robust candidate-level relational signal exists. They do show that the old negative evidence is less final than the prior report claimed, exactly motivating P0-P13 plus a genuinely untouched holdout.

## Current reproducibility verdict

The old high-level qualitative conclusion may still survive, but the current repository does **not** yet support treating every headline as cleanly reproducible. Labels/hidden states appear usable; controls and some inference paths require repair. R0 will not overwrite historical results. Repaired reruns will use new `rescue_audit/results/` paths and the experiment registry.
