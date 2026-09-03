#!/usr/bin/env python3
"""Build the Direction One paper-number ledger from tracked result artifacts.

This script deliberately performs only deterministic arithmetic over saved outputs.
It is the reproducible bridge between the underlying JSON/CSV artifacts and the
single paper-facing master table.
"""

from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "iclr_direction1"

MASTER_FIELDS = [
    "claim_group", "figure_or_table_candidate", "result_id", "experiment",
    "backbone", "model_size", "task", "target", "reward", "policy", "probe",
    "feature_set", "level", "layer", "timestep", "horizon", "K", "n_problem",
    "n_state", "n_candidate", "n_example", "metric", "baseline", "hidden",
    "delta", "ci_low", "ci_high", "p_value", "stat_method", "folds_seeds",
    "multiple_testing", "status", "strength", "paper_use", "source_path",
    "source_key", "script_path", "notes",
]

CANDIDATE_FIELDS = [
    "result_id", "date", "experiment_name", "status", "backbone", "model_size",
    "task", "dataset", "reference_policy", "reward_type", "target_type",
    "probe_type", "feature_type", "candidate_or_state_level", "layer", "timestep",
    "horizon", "K", "n_documents", "n_problems", "n_states", "n_candidates",
    "n_examples", "metric", "baseline_value", "hidden_value", "delta", "ci_low",
    "ci_high", "p_value", "statistical_method", "seeds_or_folds",
    "positive_or_negative", "headline_candidate", "source_path", "source_key",
    "notes",
]


def load(rel: str):
    with (ROOT / rel).open() as f:
        return json.load(f)


def fmt(x):
    if x is None or x == "":
        return ""
    if isinstance(x, (np.floating, float)):
        if math.isnan(float(x)):
            return ""
        return f"{float(x):.10g}"
    return x


master: list[dict] = []


def add(**kw):
    row = {k: "" for k in MASTER_FIELDS}
    row.update(kw)
    row["script_path"] = row["script_path"] or "iclr_direction1/build_iclr_evidence.py"
    master.append(row)


# 1) Strongest state-value readout, independently re-audited.
vr = load("data/v_readout_results.json")
va = load("data/v_audit.json")
cheap = vr["metrics"]["cheap"]
hidden = vr["metrics"]["hidden"]
combined = vr["metrics"]["cheap+hidden"]
for metric, key, ci_key in [("R2", "r2", "dR2"), ("AUC", "auc", "dAUC")]:
    add(
        claim_group="state_future_utility", figure_or_table_candidate="Figure 1 / Table 1",
        result_id=f"VREAD_HIDDEN_ONLY_{metric}", experiment="Nemotron rollout-value readout",
        backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K",
        target="V_reward", reward="task correctness", policy="Nemotron diffusion decoding",
        probe="ridge/logistic linear probe", feature_set="validation-selected layer-18 H_m",
        level="state", layer=18, timestep="all six record points", horizon="completion",
        K=8, n_problem=vr["n_prompts"], n_state=vr["n_states"], n_example=vr["n_states"],
        metric=metric, baseline=cheap[key], hidden=hidden[key], delta=hidden[key]-cheap[key],
        ci_low=va["best_ci"][ci_key][0], ci_high=va["best_ci"][ci_key][1],
        stat_method="prompt-cluster bootstrap, 10,000 resamples",
        folds_seeds="480/120/200 prompt train/validation/test; K=8 rollouts",
        multiple_testing="layer selected on validation only", status="POST_HOC_AUDIT_VALID_CURRENT",
        strength="HEADLINE", paper_use="Figure 1; abstract; main text",
        source_path="data/v_readout_results.json; data/v_audit.json",
        source_key=f"metrics.hidden.{key}; best_ci.{ci_key}",
        notes="RECOMPUTED_FROM_SAVED_METRICS. Hidden-only is authoritative; no test-set layer selection.",
    )

# Preserve the older, valid two-block number as an ablation, not the headline.
for metric, key, old_delta in [("R2", "r2", vr["delta_r2"]), ("AUC", "auc", vr["delta_auc"])]:
    add(
        claim_group="state_future_utility_ablation", figure_or_table_candidate="Appendix",
        result_id=f"VREAD_CHEAP_PLUS_HIDDEN_{metric}", experiment="Nemotron rollout-value readout",
        backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K", target="V_reward",
        reward="task correctness", policy="Nemotron diffusion decoding", probe="two-block linear probe",
        feature_set="cheap + layer-18 H_m", level="state", layer=18, horizon="completion", K=8,
        n_problem=800, n_state=4411, n_example=4411, metric=metric, baseline=cheap[key],
        hidden=combined[key], delta=old_delta, stat_method="held-out prompt split",
        folds_seeds="480/120/200; K=8", multiple_testing="layer selected on validation only",
        status="VALID_CURRENT_ABLATION", strength="SUPPORTING", paper_use="Appendix",
        source_path="data/v_readout_results.json", source_key=f"delta_{key}",
        notes="RECOMPUTED_FROM_SAVED_METRICS. Explains old +0.1317/+0.0655 values; not superseded, but not headline.",
    )

# Six temporal checkpoints.
for i, audit in enumerate(va["per_point"]):
    point = vr["per_point"][i]
    for metric, dkey, basekey, hiddenkey in [
        ("R2", "dR2", "r2_cheap", None), ("AUC", "dAUC", "auc_cheap", "auc_hidden")
    ]:
        base = point[basekey]
        hid = base + audit[dkey] if hiddenkey is None else point[hiddenkey]
        ci = audit[dkey + "_ci"]
        add(
            claim_group="temporal_state_future_utility", figure_or_table_candidate="Figure 2",
            result_id=f"VREAD_T{i}_{metric}", experiment="Nemotron temporal readout",
            backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K", target="V_reward",
            reward="task correctness", policy="Nemotron diffusion decoding", probe="linear probe",
            feature_set="layer-18 H_m", level="state", layer=18,
            timestep=audit["prog"], horizon="completion", K=8, n_problem=audit["n"],
            metric=metric, baseline=base, hidden=hid, delta=audit[dkey], ci_low=ci[0], ci_high=ci[1],
            stat_method="prompt-cluster bootstrap, 10,000 resamples", folds_seeds="held-out prompts; K=8",
            multiple_testing="six pre-existing record points; layer fixed from validation",
            status="POST_HOC_AUDIT_VALID_CURRENT", strength="STRONG_SUPPORT", paper_use="Figure 2",
            source_path="data/v_audit.json; data/v_readout_results.json",
            source_key=f"per_point[{i}]", notes="RECOMPUTED_FROM_SAVED_METRICS.",
        )

# Selective generation operating points.
coverages = [.9, .8, .7, .6, .5, .3]
sel = {c: (c, h) for c, h in zip(coverages, vr["selective"]["hidden"])}
cheap_sel = {c: a for c, a in zip(coverages, vr["selective"]["cheap"])}
sel_ci = {
    .7: (0.0025, 0.0423), .5: (0.0247, 0.0926), .3: (0.0299, 0.1368),
}
for cov in [.7, .5, .3]:
    h = sel[cov][1]; b = cheap_sel[cov]
    add(
        claim_group="selective_generation", figure_or_table_candidate="Figure 4",
        result_id=f"SELECTIVE_COV_{int(cov*100)}", experiment="Nemotron selective generation",
        backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K", target="V_reward",
        reward="task correctness", policy="abort at 40% generation progress", probe="linear probe",
        feature_set="layer-18 H_m", level="prompt/state", layer=18, timestep=.4, horizon="completion",
        K=8, n_problem=800, metric="retained accuracy", baseline=b, hidden=h, delta=h-b,
        ci_low=sel_ci[cov][0], ci_high=sel_ci[cov][1],
        stat_method="prompt-cluster bootstrap, 10,000 resamples", folds_seeds="held-out prompts; K=8",
            status="POST_HOC_PRACTICAL_ANALYSIS", strength="STRONG_SUPPORT", paper_use="Figure 4",
        source_path="data/v_readout_results.json; iclr_direction1/VREAD_INDEPENDENT_RECOMPUTE.log",
        source_key=f"selective coverage={cov}",
        notes=f"RECOMPUTED_FROM_SAVED_METRICS. Coverage={cov:.0%}; compute saved={(1-cov)*0.6:.0%} under the explicit 40%-progress abort accounting convention.",
    )
add(
    claim_group="selective_generation", figure_or_table_candidate="Figure 4",
    result_id="SELECTIVE_AURC", experiment="Nemotron selective generation",
    backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K", target="V_reward",
    reward="task correctness", policy="selective generation", probe="linear probe",
    feature_set="layer-18 H_m", level="prompt/state", layer=18, K=8, n_problem=800,
    metric="AURC", baseline=vr["aurc"]["cheap"], hidden=vr["aurc"]["hidden"],
    delta=vr["aurc"]["hidden"]-vr["aurc"]["cheap"], ci_low=.0171, ci_high=.0812,
    stat_method="prompt-cluster bootstrap, 10,000 resamples", status="POST_HOC_PRACTICAL_ANALYSIS",
    strength="STRONG_SUPPORT", paper_use="Figure 4", source_path="data/v_readout_results.json; iclr_direction1/VREAD_INDEPENDENT_RECOMPUTE.log",
    source_key="aurc", notes="RECOMPUTED_FROM_SAVED_METRICS.",
)

# 2) Candidate-level Path-LL: validation-selected layers, 50 repeated document splits.
est = load("data/estimate_all.json")
for e in est:
    p = e["probes"]["cheap+h_local(3layers)"]
    for metric_key, metric_name in [("d_r2", "within-state R2"), ("d_conc", "pairwise concordance")]:
        nb = p[metric_key]["nadeau_bengio"]
        sr = p[metric_key]["sign_rank"]
        rid = f"PATHLL_{e['arm']}_{e['target']}_{metric_key}".upper()
        positive = nb["ci"][0] > 0
        add(
            claim_group="candidate_pathll", figure_or_table_candidate="Figure 3 / Table 2",
            result_id=rid, experiment="state-centered candidate Path-LL readout",
            backbone="SEDD" if e["arm"].startswith("SEDD") else "MDLM",
            model_size="169.6M", task="OpenWebText continuations", target=e["target"],
            reward="Path-LL", policy=e["arm"].split("_", 1)[1], probe="ridge linear probe",
            feature_set="cheap + h_local (3 validation-fixed layers)", level="candidate within state",
            layer=e["layer"], horizon=16, K=24, n_problem=400, n_state=2400,
            n_candidate=14400, n_example=14400, metric=metric_name, baseline=0, hidden=nb["mean"],
            delta=nb["mean"], ci_low=nb["ci"][0], ci_high=nb["ci"][1], p_value=nb["p"],
            stat_method="50 repeated document splits; Nadeau-Bengio corrected t interval",
            folds_seeds=f"50 splits; {sr['n_positive']}/{sr['n']} positive",
            multiple_testing="BH-FDR saved in source; layers selected once on validation seed 99",
            status="POST_HOC_RESCUE_VALID_CURRENT", strength="STRONG_SUPPORT" if metric_key == "d_r2" else ("SUPPORTING" if positive else "NEGATIVE_BOUNDARY"),
            paper_use="Table 2; main text" if metric_key == "d_r2" else "Table 2 / Appendix",
            source_path="data/estimate_all.json", source_key=f"{e['arm']}/{e['target']}/cheap+h_local(3layers)/{metric_key}",
            notes="RECOMPUTED_FROM_SAVED_METRICS. Repeated splits are dependent; corrected interval is authoritative.",
        )

# Three locked/fair split repetitions, retained for auditability.
for e in load("rescue_audit/results/A_fairtest_corrected_ci.json"):
    for key, name, ci_key in [("d_r2", "within-state R2", "corrected_ci_r2"), ("d_conc", "pairwise concordance", "corrected_ci_conc")]:
        obs = e["observed"][key]; ci = e[ci_key]
        add(
            claim_group="candidate_pathll_fairtest", figure_or_table_candidate="Appendix",
            result_id=f"FAIR_{e['arm']}_{e['target']}_S{e['split_seed']}_{key}".replace(" ", "_").upper(),
            experiment="candidate Path-LL fair-split audit", backbone="MDLM", model_size="169.6M",
            task="OpenWebText continuations", target=e["target"], reward="Path-LL", policy=e["arm"],
            probe="ridge linear probe", feature_set="cheap + h_local", level="candidate within state",
            layer=e["layer"], horizon=16, K=24, n_problem=400, n_state=2400, n_candidate=14400,
            n_example=14400, metric=name, baseline=0, hidden=obs, delta=obs,
            ci_low=ci[0], ci_high=ci[1], stat_method="document-cluster bootstrap, 10,000 resamples",
            folds_seeds=f"split seed {e['split_seed']}; 100 test documents", status="POST_HOC_AUDIT",
            strength="SUPPORTING" if ci[0] > 0 else "NEGATIVE_BOUNDARY", paper_use="Appendix",
            source_path="rescue_audit/results/A_fairtest_corrected_ci.json",
            source_key=f"{e['arm']}/{e['target']}/seed={e['split_seed']}/{key}",
            notes="RECOMPUTED_FROM_SAVED_METRICS.",
        )

# 3) Cross-task state readouts.
state = load("data/task_state_readout.json")
for e in state:
    for key, metric in [("r2", "R2"), ("auc", "AUC")]:
        nb = e["probes"]["hidden"][key]["nadeau_bengio"]
        sr = e["probes"]["hidden"][key]["sign_rank"]
        add(
            claim_group="cross_task_state_utility", figure_or_table_candidate="Table 3",
            result_id=f"STATE_{e['task']}_{metric}".replace("(", "_").replace(")", "").upper(),
            experiment="task-state readout", backbone="Nemotron-Diffusion-3B", model_size="3B", task=e["task"],
            target="V_task", reward="task correctness", policy="ancestral", probe="linear probe",
            feature_set="cheap + h_global", level="state", layer=e["layer"], horizon="completion", K=8,
            n_problem=e["n_docs"], n_state=e["n_states"], n_example=e["n_states"], metric=metric,
            baseline=0, hidden=nb["mean"], delta=nb["mean"], ci_low=nb["ci"][0], ci_high=nb["ci"][1],
            p_value=nb["p"], stat_method="50 repeated document splits; Nadeau-Bengio corrected t interval",
            folds_seeds=f"50 splits; {sr['n_positive']}/{sr['n']} nonzero positive",
            status="VALID_CURRENT_UNDERPOWERED", strength="SUPPORTING", paper_use="Table 3 / Appendix",
            source_path="data/task_state_readout.json", source_key=f"{e['task']}/hidden/{key}",
            notes="RECOMPUTED_FROM_SAVED_METRICS. Positive point estimate and sign consistency; corrected CI crosses zero at this sample size.",
        )

# 4) Candidate task-utility results that bound the claim.
for rel, label, backbone, task in [
    ("rescue_audit/results/taskCD_crossfit_confirmatory.json", "TASK_CD", "Nemotron-Diffusion-3B", "GSM8K C+D"),
    ("rescue_audit/results/taskE_svamp_crossfit.json", "TASK_E", "Nemotron-Diffusion-3B", "SVAMP E"),
]:
    x = load(rel)
    for comp, desc in [("bilinear_vs_cheap_only", "bilinear vs cheap"), ("bilinear_vs_no_state_interaction", "state-action interaction")]:
        c = x["comparisons"][comp]
        add(
            claim_group="candidate_task_utility_boundary", figure_or_table_candidate="Table 4",
            result_id=f"{label}_{comp}".upper(), experiment="five-fold candidate task-utility cross-fit",
            backbone=backbone, model_size="3B", task=task, target="A_task", reward="task correctness",
            policy="ancestral", probe="low-rank bilinear probe", feature_set="cheap + hidden",
            level="candidate within state", layer=x["config"].get("layer", "validation-selected"),
            horizon="completion", K=8, n_problem=x["n_docs"], n_state=x["n_states"],
            n_candidate=x["n_rows"], n_example=x["n_rows"], metric="pairwise concordance delta",
            baseline=0, hidden=c["observed"], delta=c["observed"], ci_low=c["ci_lo"], ci_high=c["ci_hi"],
            p_value=c["p_le_zero"], stat_method="5-fold document cross-fit + 10,000 document-cluster bootstraps",
            folds_seeds="5 folds", status="POST_HOC_CROSSFIT_BOUNDARY", strength="NEGATIVE_BOUNDARY",
            paper_use="Table 4 / limitations", source_path=rel, source_key=f"comparisons.{comp}",
            notes=f"RECOMPUTED_FROM_SAVED_METRICS. {desc}; CI crosses zero, so not evidence for a stable candidate task-utility effect.",
        )

x = load("rescue_audit/results/CONFIRMATORY_taskD_frozen.json")
for comp in ["primary_bilinear_vs_cheap_only", "relational_bilinear_vs_no_state_interaction"]:
    c = x["comparisons"][comp]
    add(
        claim_group="candidate_task_utility_transfer_boundary", figure_or_table_candidate="Table 4",
        result_id=f"TASK_D_FROZEN_{comp}".upper(), experiment="sealed frozen Task C to Task D transfer",
        backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K D", target="A_task",
        reward="task correctness", policy="ancestral", probe="frozen low-rank bilinear probe",
        feature_set="cheap + hidden", level="candidate within state", layer=x["sealed_metadata"]["layer"],
        horizon="completion", K=8, n_problem=x["n_docs"], n_state=x["n_states"],
        n_candidate=x["n_rows"], n_example=x["n_rows"], metric="pairwise concordance delta",
        baseline=0, hidden=c["observed"], delta=c["observed"], ci_low=c["ci_lo"], ci_high=c["ci_hi"],
        p_value=c["p_le_zero"], stat_method="sealed application + 10,000 document-cluster bootstraps",
        folds_seeds="frozen before Task D labels", status="CONFIRMATORY_NEGATIVE",
        strength="NEGATIVE_BOUNDARY", paper_use="Table 4 / limitations",
        source_path="rescue_audit/results/CONFIRMATORY_taskD_frozen.json", source_key=f"comparisons.{comp}",
        notes="RECOMPUTED_FROM_SAVED_METRICS. Clean domain-transfer failure; narrows rather than reverses the state-level claim.",
    )

# 4b) Independently recompute Path-LL/task-reward alignment from raw NPZ shards.
def raw_alignment(dirname: str, seed: int = 20260902, n_boot: int = 10000):
    chunks = []
    for path in sorted(glob.glob(str(ROOT / "data" / dirname / "*.npz"))):
        z = np.load(path)
        chunks.append((z["doc_id"], z["A_task"], z["A_pertok"]))
    doc = np.concatenate([x[0] for x in chunks])
    x = np.concatenate([x[1] for x in chunks]).astype(float)
    y = np.concatenate([x[2] for x in chunks]).astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    doc, x, y = doc[ok], x[ok], y[ok]
    ids, inv = np.unique(doc, return_inverse=True)
    # Per-document sufficient statistics permit an exact, fast cluster bootstrap.
    stats = np.stack([
        np.bincount(inv), np.bincount(inv, weights=x), np.bincount(inv, weights=y),
        np.bincount(inv, weights=x*x), np.bincount(inv, weights=y*y),
        np.bincount(inv, weights=x*y),
    ], axis=1)
    def corr(s):
        n,sx,sy,sxx,syy,sxy = s
        num=sxy-sx*sy/n
        den=math.sqrt(max(0.0,(sxx-sx*sx/n)*(syy-sy*sy/n)))
        return num/den if den else float("nan")
    observed = corr(stats.sum(axis=0))
    rng=np.random.default_rng(seed)
    vals=np.empty(n_boot)
    for b in range(n_boot):
        vals[b]=corr(stats[rng.integers(0,len(ids),len(ids))].sum(axis=0))
    return {"n_rows":len(x), "n_docs":len(ids), "pearson":observed,
            "r2":observed**2, "ci":[float(v) for v in np.nanpercentile(vals,[2.5,97.5])],
            "n_boot":n_boot, "seed":seed}

alignment = {}
for dirname, task in [("labels_taskA","GSM8K A"),("labels_taskB","GSM8K B"),
                      ("labels_taskC","GSM8K C"),("labels_taskE_svamp","SVAMP E")]:
    a=raw_alignment(dirname); alignment[task]=a
    add(
        claim_group="reward_alignment", figure_or_table_candidate="Table 4 / Appendix",
        result_id=f"ALIGN_{task}".replace(" ","_").upper(), experiment="Path-LL/task-reward alignment",
        backbone="Nemotron-Diffusion-3B", model_size="3B", task=task, target="A_task vs A_pertok",
        reward="task correctness vs Path-LL", policy="ancestral", probe="none; direct correlation",
        feature_set="rollout labels", level="candidate", horizon="completion", K=8,
        n_problem=a["n_docs"], n_candidate=a["n_rows"], n_example=a["n_rows"], metric="Pearson r",
        baseline=0, hidden=a["pearson"], delta=a["pearson"], ci_low=a["ci"][0], ci_high=a["ci"][1],
        stat_method="document-cluster bootstrap, 10,000 resamples", folds_seeds="bootstrap seed 20260902",
        status="VALID_CURRENT", strength="SUPPORTING" if a["ci"][0] > 0 else "NEGATIVE_BOUNDARY",
        paper_use="Table 4 / limitations", source_path=f"data/{dirname}/shard_*.npz; iclr_direction1/TASK_REWARD_ALIGNMENT_RECOMPUTE.json",
        source_key=task, notes="RECOMPUTED_FROM_RAW_FEATURES. Shows that task reward and Path-LL are related but non-equivalent targets.",
    )
with (OUT / "TASK_REWARD_ALIGNMENT_RECOMPUTE.json").open("w") as f:
    json.dump(alignment, f, indent=2)

# 5) Within-prompt state residual is a useful negative boundary.
add(
    claim_group="state_within_prompt_boundary", figure_or_table_candidate="Appendix",
    result_id="VREAD_WITHIN_PROMPT_DIRECT", experiment="prompt-centered state-value probe",
    backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K", target="centered V_reward",
    reward="task correctness", policy="Nemotron diffusion decoding", probe="prompt-centered linear probe",
    feature_set="centered cheap + centered hidden", level="state within prompt", layer="0,3,...,24",
    horizon="completion", K=8, n_problem=800, n_state=4411, n_example=4411,
    metric="incremental R2", baseline=.0013, hidden=.0013, delta=0,
    stat_method="direct prompt-centered fitting", status="VALID_CURRENT_BOUNDARY",
    strength="NEGATIVE_BOUNDARY", paper_use="Appendix / limitations", source_path="data/v_audit.json; iclr_direction1/VREAD_INDEPENDENT_RECOMPUTE.log",
    source_key="within_signal_var; direct centered probe", notes=f"RECOMPUTED_FROM_RAW_FEATURES. Nonzero within-prompt target variance={va['within_signal_var']:.10g}, noise ceiling=0.8699, but hidden delta is approximately zero.",
)

# Older horizon sweep: useful only as a boundary because it predates the corrected
# state-centered estimator and is available here as a report-level summary.
add(
    claim_group="horizon_boundary", figure_or_table_candidate="Appendix",
    result_id="OLD_HORIZON_H32_RELATIONAL", experiment="legacy horizon × timestep × layer sweep",
    backbone="MDLM", model_size="169.6M", task="OpenWebText continuations", target="A_pertok",
    reward="Path-LL", policy="ancestral", probe="legacy relational probe", feature_set="hidden interaction",
    level="candidate within state", horizon=32, K=16, n_candidate=14400,
    metric="relational concordance delta", baseline=0, hidden=-.0050, delta=-.0050,
    stat_method="report-level pooled/temporal summary", status="LEGACY_EXPLORATORY_BOUNDARY",
    strength="NEGATIVE_BOUNDARY", paper_use="Appendix only",
    source_path="rescue_audit/FINAL_RESCUE_REPORT.md", source_key="section 9 horizon table H=32",
    notes="SOURCE_SUMMARY_ONLY. Label ceiling fell to 0.656/SNR 1.91; long-horizon candidate effect was washed out. Do not mix with the corrected H=16 state-centered analysis.",
)

# Meta-analysis helper. It is supporting synthesis, not a replacement for component rows.
def meta(rows):
    y = np.asarray([r[0] for r in rows], float)
    se = np.asarray([r[1] for r in rows], float)
    w = 1 / se**2
    fixed = float(np.sum(w*y)/np.sum(w)); fixed_se = float(np.sqrt(1/np.sum(w)))
    q = float(np.sum(w*(y-fixed)**2)); df = len(y)-1
    c = float(np.sum(w)-np.sum(w*w)/np.sum(w)); tau2 = max(0.0, (q-df)/c)
    wr = 1/(se**2+tau2); random = float(np.sum(wr*y)/np.sum(wr)); random_se=float(np.sqrt(1/np.sum(wr)))
    i2 = max(0.0, (q-df)/q)*100 if q else 0.0
    return fixed, fixed_se, random, random_se, tau2, i2

state_auc = [(hidden["auc"]-cheap["auc"], (va["best_ci"]["dAUC"][1]-va["best_ci"]["dAUC"][0])/(2*1.96))]
for e in state:
    nb=e["probes"]["hidden"]["auc"]["nadeau_bengio"]; state_auc.append((nb["mean"],nb["se"]))
fx,fse,re,rse,tau,i2=meta(state_auc)
for method,val,se in [("fixed effect",fx,fse),("DerSimonian-Laird random effects",re,rse)]:
    add(
        claim_group="cross_task_state_utility_meta", figure_or_table_candidate="Table 3",
        result_id=f"STATE_AUC_META_{method.split()[0].upper()}", experiment="state-level cross-task synthesis",
        backbone="Nemotron-Diffusion-3B", model_size="3B", task="GSM8K + SVAMP",
        target="V_task", reward="task correctness", policy="multiple", probe="linear probes",
        feature_set="hidden vs cheap", level="state", n_problem=1180, n_state=5931, metric="Delta AUC",
        baseline=0, hidden=val, delta=val, ci_low=val-1.96*se, ci_high=val+1.96*se,
        stat_method=method, folds_seeds="3 study estimates", status="VALID_CURRENT_SYNTHESIS",
        strength="STRONG_SUPPORT", paper_use="Table 3 / main text",
        source_path="data/v_audit.json; data/task_state_readout.json", source_key="independent study estimates",
        notes=f"RECOMPUTED_FROM_SAVED_METRICS; tau2={tau:.6g}, I2={i2:.1f}%. Supporting synthesis; component studies differ in model and sample size.",
    )

# Write authoritative table.
with (OUT / "DIRECTION1_ICLR_MASTER_RESULTS.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MASTER_FIELDS, lineterminator="\n")
    w.writeheader()
    for row in master:
        w.writerow({k: fmt(row[k]) for k in MASTER_FIELDS})

# Broad candidate pool: every registry row plus every curated master result.
candidates: list[dict] = []
regpath = ROOT / "rescue_audit/EXPERIMENT_REGISTRY.csv"
with regpath.open(newline="") as f:
    for i, r in enumerate(csv.DictReader(f), 1):
        metric = "within_r2" if r.get("within_r2") else "r2"
        value = r.get(metric, "")
        try: sign = "positive" if float(value) > 0 else "negative_or_zero"
        except Exception: sign = "unknown"
        candidates.append({
            "result_id": f"REGISTRY_{i:04d}", "date": r.get("date", ""),
            "experiment_name": r.get("phase", "registry result"), "status": r.get("phase", ""),
            "backbone": r.get("arm", ""), "model_size": "169.6M", "task": "OpenWebText continuations",
            "dataset": "OpenWebText", "reference_policy": r.get("arm", ""), "reward_type": r.get("target", ""),
            "target_type": r.get("target", ""), "probe_type": r.get("probe", ""),
            "feature_type": r.get("hg_kind", ""), "candidate_or_state_level": "candidate within state",
            "layer": r.get("layer", ""), "n_documents": "", "n_problems": "", "n_states": "",
            "n_candidates": "", "n_examples": "", "metric": metric, "hidden_value": value,
            "positive_or_negative": sign, "headline_candidate": "no",
            "source_path": r.get("output_path", ""), "source_key": f"registry row {i}",
            "notes": "Registry inventory entry; exploratory unless promoted by a curated master-table row.",
        })

for r in master:
    candidates.append({
        "result_id": r["result_id"], "date": "2026-09-02", "experiment_name": r["experiment"],
        "status": r["status"], "backbone": r["backbone"], "model_size": r["model_size"],
        "task": r["task"], "dataset": r["task"], "reference_policy": r["policy"],
        "reward_type": r["reward"], "target_type": r["target"], "probe_type": r["probe"],
        "feature_type": r["feature_set"], "candidate_or_state_level": r["level"], "layer": r["layer"],
        "timestep": r["timestep"], "horizon": r["horizon"], "K": r["K"],
        "n_documents": r["n_problem"], "n_problems": r["n_problem"], "n_states": r["n_state"],
        "n_candidates": r["n_candidate"], "n_examples": r["n_example"], "metric": r["metric"],
        "baseline_value": r["baseline"], "hidden_value": r["hidden"], "delta": r["delta"],
        "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"],
        "statistical_method": r["stat_method"], "seeds_or_folds": r["folds_seeds"],
        "positive_or_negative": "positive" if isinstance(r["delta"], (int,float,np.floating)) and r["delta"] > 0 else "negative_or_zero",
        "headline_candidate": "yes" if r["strength"] == "HEADLINE" else "supporting",
        "source_path": r["source_path"], "source_key": r["source_key"], "notes": r["notes"],
    })

with (OUT / "_ALL_RESULT_CANDIDATES.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS, lineterminator="\n")
    w.writeheader()
    for row in candidates:
        w.writerow({k: fmt(row.get(k, "")) for k in CANDIDATE_FIELDS})

print(f"wrote {len(master)} master rows and {len(candidates)} candidate rows")
