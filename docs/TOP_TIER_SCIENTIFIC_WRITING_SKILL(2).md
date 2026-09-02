# Top-Tier Scientific Paper Writing Skill
## Universal Framing, Evidence, Story, and Reviewer-Impact Protocol

**Use for:** AI/ML/NLP, biomedical informatics, causal inference, digital health, computer vision, point-cloud/LiDAR, engineering, statistics, and other empirical/methodological research.

**Goal:** write papers that are scientifically faithful, memorable, contribution-forward, evidence-dense, visually coherent, appropriately confident, and precise about scope without becoming defensive.

---

# 1. Core philosophy

A paper is not a chronological record of research activity. It is a designed argument:

> **Important gap → precise research object → credible solution/experiment → decisive evidence → scientific implication.**

The reader should not reconstruct the contribution from logs, failed experiments, caveats, or scattered analyses.

**Central rule:**
> **Claim first. Evidence second. Interpretation third. Boundary at the appropriate scope.**

Scientific caution means claiming exactly what the evidence supports. It does **not** mean weakening every supported finding with immediate disclaimers.

---

# 2. Determine what the paper actually is

Before drafting, classify the highest-level contribution genuinely supported by the work.

## A. Method / Algorithm
Existing methods fail at X → introduce Y → Y provides missing technical capability → experiments establish gains.

## B. Benchmark / Evaluation Framework / Resource
Current evaluation cannot measure X → introduce Y → Y operationalizes X → validation shows why X matters.

## C. Empirical Discovery
The field assumes/does not know X → controlled evidence reveals Y → Y changes understanding/evaluation/practice.

## D. System / Platform
Current workflow lacks X → integrated system Y provides X → end-to-end evidence establishes utility.

## E. Applied / Clinical / Scientific Method
Important domain problem X → technical approach Y → evidence shows scientifically or clinically meaningful value.

## F. Theory / Formal Analysis
Formal characterization, theorem, bound, impossibility result, or conceptual framework.

## G. Dataset / Corpus
New data asset + collection methodology + documentation + quality analysis + demonstrated utility.

### Contribution elevation test

Ask:

> Is this merely one experiment, or has the experiment instantiated a reusable method, framework, benchmark, protocol, or resource?

Use the stronger framing **only when the reusable artifact actually exists**.

---

# 3. Freeze the one-sentence thesis

Before writing prose, answer:

> **What should a competent reviewer remember one week after reading this paper?**

The sentence should contain:
1. scientific object,
2. core capability/contrast,
3. main result,
4. implication.

Template:

> **We introduce/show X, which does Y, demonstrating Z.**

This sentence is the manuscript's narrative checksum. Every main section must support it.

---

# 4. Build a contribution pyramid

Do not present a flat list.

1. **Field-level gap:** what capability/knowledge is missing?
2. **Reusable contribution:** method, metric, benchmark, dataset, system, framework.
3. **Technical mechanism:** what makes it work?
4. **Empirical validation:** what evidence establishes it?
5. **Broader implication:** what changes for the field?

Bad:
> metric A + analysis B + experiment C + dataset D.

Better:
> framework X, enabled by A/B, validated by C, establishing D.

---

# 5. Write for the reviewer's decision

A selective-venue reviewer asks:

1. What is the paper about?
2. Why does the problem matter?
3. What is genuinely new?
4. Is the technical logic correct?
5. Is the evidence convincing?
6. Why publish this instead of another strong paper?
7. What can readers use or learn?

The manuscript should make a strong “Reasons to Accept” section easy to write.

---

# 6. Use Context → Content → Conclusion

Apply C-C-C to the paper, sections, and most paragraphs.

- **Context:** why this exists.
- **Content:** method/evidence.
- **Conclusion:** what the reader should remember.

Avoid paragraphs that end on trivia or caveats when their purpose is to establish a result.

---

# 7. Abstract: complete story, not compressed audit log

Order:

1. problem/gap,
2. method/framework/test,
3. experimental scope,
4. headline result,
5. strongest robustness,
6. scientific implication.

**Do not end the abstract on a limitation.**

Weak:
> However, X is not perfectly isolated...

Better:
> X is not a pure manipulation, so the claim applies to the full controlled condition. **Even within that scope, the result shows that existing evaluation misses Y.**

The final sentence should normally be the take-home contribution.

---

# 8. Introduction: four jobs

## Block 1 — Important problem
Start near the actual scientific tension, not with generic field hype.

## Block 2 — Exact unresolved gap
Use related work to form the gap, not to inventory citations.

## Block 3 — Why the problem is technically nontrivial
Explain why a naive method or standard evaluation is inadequate.

## Block 4 — Solution + headline evidence + contribution hierarchy
State the core result early. List 2–4 contributions with labels such as:
- Benchmark
- Method
- Metric
- Empirical Finding
- Resource

Avoid ending this block with extended defensive hedging.

---

# 9. Related Work is a novelty argument

Organize by research axis, not paper-by-paper chronology.

For each paragraph:
1. what the literature solves,
2. what remains missing,
3. exactly how this paper differs.

End each paragraph by sharpening **your novelty**.

---

# 10. Method: idea before machinery

Recommended order:
1. setting/problem,
2. main technical object,
3. core algorithm/estimator/framework,
4. why naive alternatives fail,
5. inference/training/evaluation,
6. implementation details.

Main-text implementation detail is justified only when necessary for:
- interpretation,
- reproducibility,
- confound control,
- or a meaningful design decision.

Move low-level engineering detail to appendix.

---

# 11. Robustness is a design strength, not an apology

Do not frame robustness as:
> We worried the result might be false, so we checked...

Where appropriate, frame it as part of the method/framework:

> The evaluation includes a robustness verification pipeline comprising...

Possible components:
- negative/placebo controls,
- permutations,
- alternative specifications,
- alternative matching,
- missingness analysis,
- threshold sensitivity,
- influence analysis,
- leave-one-out,
- held-out confirmation.

This turns “extra checks” into methodological value.

---

# 12. Result headings must state findings

Weak:
- Reward Analysis
- Ablation
- Subgroup Analysis
- Robustness

Strong:
- Reward Remains Equivalent
- All Process Representations Exceed Baseline Drift
- The Effect Persists under Outcome Concordance
- Registration Error Drives the Failure Mode
- Calibration Improves Most in the Low-Data Regime

A reader scanning headings should understand the result sequence.

---

# 13. Result paragraph formula

For every major result:

1. **Claim**
2. **Evidence**
3. **Strongest robustness**
4. **Interpretation**

Do not end each paragraph with a list of things the result does not prove.

For complex empirical papers, organize Results as **Observations/Findings**, not as a sequence of statistical procedures.

---

# 14. Defensive hedging control — mandatory

## Central rule

> **Do not hedge the existence of a result that the evidence clearly supports. Hedge only the boundary of interpretation.**

Good:
> The intervention significantly changed X.

Later:
> X measures structural difference rather than clinical benefit.

Bad:
> The intervention changed X, although this does not establish A, does not prove B, is not evidence of C, and may not generalize to D.

## Placement hierarchy

- Main result sentence: minimal hedging.
- End of subsection: one scope sentence if necessary.
- Discussion: interpretive boundaries.
- Limitations: complete caveats.

Do not repeat the same limitation in Abstract + Introduction + Method + Results + Discussion + Conclusion + Limitations.

---

# 15. Defensive phrase audit

Search for:

- does not identify
- does not establish
- does not imply
- not a claim about
- not evidence of
- cannot conclude
- cannot infer
- should not be interpreted
- not necessarily
- not automatically
- only descriptive
- not causal
- not independent
- not a replication
- we do not claim
- we cannot rule out

For each occurrence ask:
1. Is it necessary **here**?
2. Is it already said elsewhere?
3. Can it become a positive scope statement?

Examples:

Defensive:
> This metric does not measure route quality.

Better:
> This metric measures route separation; route quality is evaluated with task-specific outcome metrics.

Defensive:
> We cannot claim that component X alone caused the effect.

Better:
> The causal estimand applies to the full intervention package.

Defensive:
> These subsets are not independent replications.

Better:
> These overlapping subsets test whether the effect persists under outcome concordance.

---

# 16. Discussion: meaning, not repetition

Discussion should answer:

1. What changes scientifically?
2. Why was the method/evaluation necessary?
3. How should researchers/practitioners use the result?
4. What remains unknown?
5. What is the natural next step?

Recommended structure:
- Scientific implication
- Methodological implication
- Practical use cases
- Scope

Do not simply restate Results.

---

# 17. Include practical use cases

A paper feels impactful when readers know what to do with it.

Examples:

**Method:** when to use it, what failure it detects, what decision it supports.  
**Benchmark:** model development, comparison, auditing, condition discovery.  
**Biomedical:** patient stratification, treatment decision support, validation.  
**Engineering:** inspection, optimization, planning, robustness testing.

Do not invent capabilities the work does not support.

---

# 18. Limitations: concentrated, strong, non-repetitive

A limitation paragraph should contain:

> limitation → consequence for claim → natural remedy.

Example:

> The intervention jointly changes urgency, length, and imperative form, so causal attribution applies to the full condition rather than urgency alone. A follow-up should length-match the prefix and hold the rendering policy fixed.

Say it once well.

Do not leak the same limitation throughout the manuscript.

---

# 19. Strong-claim verification pass

Flag:

- first
- novel
- state-of-the-art
- benchmark
- preregistered
- causal
- generalizable
- robust
- comprehensive
- clinically meaningful
- scalable
- publicly available
- reproducible

Require evidence for each.

### “First”
Prefer:
> To our knowledge, this is the first X to jointly do A and B.

Verify literature.

### “Benchmark”
Require a reusable protocol/resource:
- tasks/protocol,
- metrics,
- code,
- documentation,
- reproducible interface.

### “Preregistered”
Use only with a genuine time-stamped registration.
Otherwise use:
- prespecified,
- prospectively specified,
- fixed before data collection.

### “Robust”
Name the dimensions:
> robust across thresholds, cohorts, and alternative estimators.

---

# 20. Never write research autobiography

Avoid main-text structures such as:
- first we tried...
- then it failed...
- after many rounds...
- we discovered...
- an earlier pipeline had a bug...

Readers need the final scientific logic, not the research diary.

Earlier failures belong only where they justify a design decision, explain provenance, or matter scientifically.

---

# 21. Figures are part of the argument

Before drawing each figure, write:

> **This figure should make the reader understand ______ in five seconds.**

Typical roles:
- **Figure 1:** problem/framework/motivating example.
- **Figure 2:** flagship result.
- **Figure 3:** strongest alternative explanation / robustness.
- **Figure 4:** mechanism or practical interpretation.

Rules:
1. consistent visual grammar,
2. same method/condition = same encoding,
3. do not rely only on color,
4. use vector PDF where possible,
5. readable final-size labels,
6. avoid default plotting aesthetics,
7. avoid PowerPoint-style box clutter,
8. direct labels where possible,
9. panel titles state conclusions,
10. do not mix incompatible units on one axis,
11. captions explain reading + takeaway,
12. remove decorative complexity.

---

# 22. Tables: exact comparisons only

Use main-text tables for:
- primary model comparison,
- headline results,
- ablation,
- sample accounting,
- key subgroup consistency.

Do not dump configuration logs or every metric into main text.

---

# 23. Freeze terminology

Before drafting create:

| Concept | Preferred term | Abbreviation | Forbidden synonyms | Definition |

One concept should not drift among robustness/stability/consistency/divergence/sensitivity unless these are formally distinct.

Terminology drift signals conceptual immaturity.

---

# 24. Acronym discipline

Expand at first use unless universally exempt by venue convention.

Audit:
- Abstract
- Main text
- Tables
- Captions
- Appendix

Avoid unnecessary acronyms.

---

# 25. Quantitative reporting

For major claims report:
- effect size,
- uncertainty,
- independent unit/cluster,
- test,
- multiplicity adjustment when relevant.

Do not rely only on p-values.

Do not call nonsignificance “equivalence.”

Do not call structural difference “better” unless outcome quality supports it.

---

# 26. Distinguish evidence types

**Primary evidence:** tests main hypothesis.  
**Robustness evidence:** tests sensitivity to analysis/sampling.  
**Mechanism evidence:** explains how/where effect arises.  
**Scope evidence:** defines where effect holds/fails.

Do not present shared-data robustness slices as independent replications.

---

# 27. Main text vs appendix

Main text must contain what a reviewer needs for:
- novelty,
- correctness,
- main claim,
- main implication.

Appendix can contain:
- exhaustive configurations,
- task inventories,
- long metric definitions,
- secondary subgroup tables,
- full sensitivity grids,
- implementation lineage,
- extra examples.

Rule:
> If the reviewer needs it to judge the paper, it belongs in the paper.

---

# 28. Page budget is an argument budget

Do not fill pages mechanically.

Spend space on:
- the gap,
- the reusable contribution,
- the decisive result,
- the strongest robustness,
- practical/scientific meaning.

Do not let:
- configuration detail,
- repetitive caveats,
- procedural history,
- citation inventories

displace the main story.

---

# 29. Results-first writing workflow

1. Freeze evidence and claim ledger.
2. Design final main figures/tables.
3. Write Results around them.
4. Write Method required to understand Results.
5. Write Introduction to motivate exactly that Method and Results.
6. Write Discussion for implications.
7. Write Abstract last.
8. Write Limitations after claims are frozen.

This prevents the paper from promising a story that the evidence cannot support.

---

# 30. Mandatory claim–evidence ledger

Before final drafting:

| Claim | Evidence | Exact result | Main/Appendix | Allowed wording | Prohibited stronger wording |
|---|---|---|---|---|---|

Every strong sentence in:
- title,
- abstract,
- contribution list,
- result headings,
- conclusion

must map to this ledger.

---

# 31. Reviewer-retelling test

Without looking at the paper, write the likely reviewer summary.

Weak:
> This paper studies X and runs several experiments.

Target:
> The paper introduces X to address Y. It demonstrates Z across A/B/C. The key implication is D.

The paper must be retellable.

---

# 32. Reviewer-excitement test

Ask:

1. What is surprising?
2. What is reusable?
3. What changes practice?
4. What would future papers cite this for?
5. What belongs in “major reasons to publish”?

If the answer is only:
> the experiments are careful,

the framing is too weak.

---

# 33. Universal anti-patterns

## Audit-report prose
Too much provenance, eligibility language, caveats, internal statuses.

## Defensive hedging everywhere
Move boundaries to Discussion/Limitations.

## Chronological storytelling
Use logical scientific structure.

## Flat contributions
Use a contribution pyramid.

## Analysis-driven Results
Use finding-driven headings.

## No practical role
Add legitimate use cases.

## More experiments = stronger paper
Only add experiments that close a claim gap or rule out a credible alternative explanation.

## Strong numbers without conceptual conclusion
Every main result needs a knowledge-level interpretation.

## Decorative figures
Every main figure must answer a research question.

## Limitation leakage
Do not repeat the same caveat throughout the manuscript.

---

# 34. Field-adaptive patterns

## AI/ML Method
Gap → method → technical mechanism → benchmark evidence → ablation/robustness → implication.

## Biomedical/Clinical AI
Clinical problem → technical gap → model → discrimination/calibration/utility → subgroup robustness → clinical scope.

## Causal Inference
Identification/estimation problem → assumptions → estimator → simulation + real data → sensitivity → scope of causal claim.

## Computer Vision / 3D / LiDAR
Operational failure → geometry/visibility/registration/planning method → public-data evidence → ablation/efficiency → operational implication.

## Benchmark / Resource
Evaluation gap → benchmark/protocol → metric/task design → validation → robustness → practical use cases.

## System
Workflow gap → integrated architecture → capability demonstration → system evaluation → deployment boundary.

---

# 35. Mandatory final editing passes

### Pass A — Thesis consistency
Every section supports the one-sentence thesis.

### Pass B — Contribution elevation
Check whether the work is under-framed as “an experiment” when it genuinely provides a reusable method/framework/resource.

### Pass C — Defensive hedging
Audit and remove repetitive self-limitation.

### Pass D — Claim verification
Verify first/benchmark/preregistered/causal/generalization/public-release claims.

### Pass E — Finding headings
Major Results headings state conclusions.

### Pass F — Figure narrative
Figures tell the same story as the text.

### Pass G — Terminology
One concept = one preferred term.

### Pass H — Acronyms
Expand at first use.

### Pass I — Self-containment
No critical evidence hidden in appendix.

### Pass J — Reviewer simulation
Write:
- 3-sentence summary,
- 3 strongest reasons to accept,
- 3 strongest weaknesses.

If the reasons to accept are not obvious from the manuscript, rewrite.

---

# 36. Strict AI instructions

An AI using this skill MUST:

1. Read all supplied evidence before drafting.
2. Identify paper type before framing.
3. Produce the one-sentence thesis and contribution pyramid first.
4. Never invent methods, experiments, novelty, or resources.
5. Never hide negative or contradictory evidence.
6. Never write the paper as a chronological project log.
7. Never repeat the same limitation throughout the paper.
8. Never end every Results paragraph with a caveat.
9. Never hedge a statistically/scientifically supported result merely to sound cautious.
10. Never call a difference “better” unless quality evidence supports it.
11. Never call nonsignificance “equivalence.”
12. Never call prespecification “preregistration” without a real registration.
13. Never use “first” without literature verification.
14. Never call something a benchmark unless a reusable evaluation artifact exists.
15. State the main quantitative result early.
16. Make the strongest legitimate claim supported by evidence.
17. Put claim boundaries primarily in Discussion/Limitations.
18. Treat figures and subsection titles as part of the argument.
19. Distinguish primary findings, robustness, mechanism, and scope.
20. End Abstract and Conclusion on contribution, not self-defense.

---

# 37. AI pre-draft output

Before writing the manuscript, the AI must internally produce:

### Paper identity
- Primary paper type:
- Reusable artifact:
- Central gap:
- One-sentence thesis:
- Strongest result:
- Strongest robustness evidence:
- Main implication:
- Primary limitation:

### Contribution pyramid
1. Field gap:
2. Reusable contribution:
3. Technical mechanism:
4. Empirical validation:
5. Broader implication:

### Main figure plan
- Fig. 1:
- Fig. 2:
- Fig. 3:
- Fig. 4:

### Claim-risk list
- first:
- benchmark:
- preregistered:
- causal:
- generalizes:
- publicly available:

Do not begin full drafting until this plan is coherent.

---

# 38. Gold-standard tone

The target tone is:

> **Confident about evidence. Precise about scope. Economical about limitations.**

Not marketing.
Not legalistic defense.
Not a research diary.
Not rebuttal prose.

A strong scientific paper should make the reviewer think:

> **The authors know exactly what they found, why it matters, how they established it, and where the claim stops.**
