"""
R0 静态审计的数值复核 + 网格正格数复算。

不依赖任何旧分析代码的结论，直接从盘上标签重算：
  1. 两分支位置数是否真的相等 (V_n == n_commit == H+1)
  2. 污染恒等式是否已被打破 corr(A_full, -V)
  3. 耗尽效应 corr(A_future, logp_action)
  4. 噪声天花板 / SNR（独立实现，不调用 probes.noise_ceiling）
  5. exp1b 网格的正格数（复算报告里的 40/78、75/78 等）
  6. 标签方差的 within/between-state 分解
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "rescue_audit", "results")
os.makedirs(OUT, exist_ok=True)


def load(tags, keys=None):
    files = []
    for t in tags:
        files += sorted(glob.glob(os.path.join(ROOT, "data", f"labels_{t}",
                                               "shard_*.npz")))
    parts = [np.load(f) for f in files]
    ks = set(parts[0].files)
    for p in parts:
        ks &= set(p.files)
    if keys is not None:
        ks &= set(keys)
    d = {k: np.concatenate([p[k] for p in parts], 0) for k in sorted(ks)}
    d["state_id"] = (d["prompt_row"].astype(np.int64) * 10_000
                     + d["step"].astype(np.int64))
    return d


def ceiling_snr(seeds):
    """独立实现：obs = var(mean_k), noise = mean_n(var_k/K)."""
    ybar = seeds.mean(1)
    K = seeds.shape[1]
    noise = float((seeds.var(1, ddof=1) / K).mean())
    obs = float(ybar.var())
    sig = max(obs - noise, 0.0)
    return dict(ceiling=sig / max(obs, 1e-12), snr=sig / max(noise, 1e-12),
                noise_var=noise, obs_var=obs)


def var_decomp(y, sid):
    """within / between state 方差分解."""
    y = y.astype(np.float64)
    order = np.argsort(sid, kind="stable")
    ys, ss = y[order], sid[order]
    bounds = np.flatnonzero(np.diff(ss)) + 1
    groups = np.split(ys, bounds)
    means = np.array([g.mean() for g in groups])
    ns = np.array([len(g) for g in groups])
    within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    grand = y.mean()
    between = float((ns * (means - grand) ** 2).sum())
    tot = within + between
    return dict(within_frac=within / tot, between_frac=between / tot)


ARMS = {
    "MDLM_anc_K24_H16": ["a3", "b3"],
    "MDLM_conf_K24_H16": ["c3", "d3"],
    "SEDD_anc_K24_H16": ["s1", "s2"],
    "MDLM_anc_K16_H8": ["h8a", "h8b"],
    "MDLM_anc_K16_H32": ["h32a", "h32b"],
    "MDLM_anc_K8_v2": ["a2", "b2"],
    "MDLM_anc_K8_v1_CONTAMINATED": ["a", "b"],
}

KEYS = ["A_pertok", "A_future", "V_pertok", "Q_pertok", "logp_action",
        "n_commit", "V_n", "A_full_seeds", "A_future_seeds",
        "V_pertok_seeds", "prompt_row", "step", "doc_id", "stratum",
        "position"]

res = {}
for name, tags in ARMS.items():
    try:
        d = load(tags, KEYS)
    except Exception as e:                                    # noqa: BLE001
        res[name] = {"error": str(e)}
        print(f"[{name}] SKIP {e}")
        continue
    n = len(d["A_pertok"])
    r = {"tags": tags, "n_examples": int(n),
         "n_states": int(len(np.unique(d["state_id"]))),
         "n_docs": int(len(np.unique(d["doc_id"])))}

    # --- 1. 两分支位置数 ---
    if "V_n" in d and "n_commit" in d:
        r["branch_lengths"] = {
            "V_n_unique": np.unique(d["V_n"]).tolist()[:6],
            "n_commit_unique": np.unique(d["n_commit"]).tolist()[:6],
            "V_n_equals_n_commit_frac":
                float((d["V_n"] == d["n_commit"]).mean())}

    # --- 2/3. 污染与耗尽 ---
    r["corr_Afull_negV"] = float(np.corrcoef(d["A_pertok"],
                                             -d["V_pertok"])[0, 1])
    r["corr_Afuture_logpaction"] = float(
        np.corrcoef(d["A_future"], d["logp_action"])[0, 1])
    r["corr_Afull_logpaction"] = float(
        np.corrcoef(d["A_pertok"], d["logp_action"])[0, 1])

    # 污染恒等式：A_full ?= [(l_i - V) + H*A_future]/(H+1)
    H = int(round(float(np.median(d["n_commit"])))) - 1
    lhs = d["A_pertok"]
    rhs = ((d["logp_action"] - d["V_pertok"]) + H * d["A_future"]) / (H + 1)
    r["contamination_identity"] = {
        "H": H,
        "max_abs_err": float(np.abs(lhs - rhs).max()),
        "corr": float(np.corrcoef(lhs, rhs)[0, 1])}

    # --- 4. 天花板 / SNR ---
    for tgt, sk in (("A_full", "A_full_seeds"), ("A_future", "A_future_seeds"),
                    ("V", "V_pertok_seeds")):
        if sk in d:
            r[f"ceiling_{tgt}"] = ceiling_snr(d[sk])

    # --- 6. 方差分解 ---
    for tgt in ("A_pertok", "A_future"):
        r[f"vardecomp_{tgt}"] = var_decomp(d[tgt], d["state_id"])

    # 标签基本量
    r["label_sd"] = {t: float(d[t].std()) for t in
                     ("A_pertok", "A_future", "V_pertok")}
    r["stratum_counts"] = {int(k): int(v) for k, v in
                           zip(*np.unique(d["stratum"], return_counts=True))}
    res[name] = r
    print(f"[{name}] n={n} corr(A,-V)={r['corr_Afull_negV']:+.3f} "
          f"corr(Afut,logp)={r['corr_Afuture_logpaction']:+.3f} "
          f"Vn==ncommit {r.get('branch_lengths',{}).get('V_n_equals_n_commit_frac')}",
          flush=True)

# --- 5. exp1b 网格正格数复算 ---
grid = {}
for suffix, label in (("_a3b3", "MDLM_anc"), ("_c3d3", "MDLM_conf"),
                      ("_s1s2", "SEDD_anc"), ("_h8ah8b", "MDLM_H8"),
                      ("_h32ah32b", "MDLM_H32"), ("_a2b2", "MDLM_K8_v2"),
                      ("_taskAtaskB", "Nemotron_task")):
    p = os.path.join(ROOT, "results", f"exp1b_heatmaps{suffix}.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p)
    g = {}
    for k in ("within_r2_h_local", "concordance_h_local", "r2_h_global"):
        M = z[k]
        fin = np.isfinite(M)
        g[k] = {"n_cells": int(fin.sum()),
                "n_positive": int((M[fin] > 0).sum()),
                "median": float(np.median(M[fin])),
                "max": float(M[fin].max()),
                "shape": list(M.shape)}
    grid[label] = g
    print(f"[grid {label}] conc h_i {g['concordance_h_local']['n_positive']}"
          f"/{g['concordance_h_local']['n_cells']}  "
          f"h_g R2 {g['r2_h_global']['n_positive']}"
          f"/{g['r2_h_global']['n_cells']}")

json.dump(res, open(os.path.join(OUT, "R0_static_checks.json"), "w"),
          indent=2, default=float)
json.dump(grid, open(os.path.join(OUT, "R0_gridcounts.json"), "w"),
          indent=2, default=float)
print("\nwrote", OUT)
