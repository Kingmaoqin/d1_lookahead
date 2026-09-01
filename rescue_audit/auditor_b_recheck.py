"""Auditor B —— 独立统计复核（全部 POST-HOC，不改动任何预注册结论）。

阶段 1 (--stage fit)   ：按 5 个设计各拟合一族探针，把 test 预测缓存到 npz。
阶段 2 (--stage analyze)：全部推断在缓存上做，纯 numpy，可秒级重跑。

设计：
  D1_L6      预注册 primary：old(a3,b3) 上拟合冻结 → freshA 全部 1200 states
  D1_L8      同上但 layer 8（post-hoc 层）
  MDLM_D2_L8 a3+b3 自身文档划分，layer 8（复现 stratified_concordance.py）
  SEDD_D2_L8 s1+s2 同上
  FRESH_D2_L8 freshA 自身划分 同上
  D1_L8_STRAT  = D1_L8 上的分层分析（1200 states，分层发现的高功效重复）
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

ALPHAS = np.logspace(-2, 9, 34)
GAMMAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
TARGET = "A_pertok"
CACHE = os.path.join(HERE, "results", "auditB_cache")


# ------------------------------------------------------------------ fitting --
def build_blocks(d_tr, i_tr, i_va, d_te, i_te, layer, pca_d=32):
    def get(d, idx):
        return {"cheap": RD.cheap_block(d)[idx], "hi": RD.h_i(d, layer)[idx],
                "hg": RD.h_g(d, layer)[idx]}
    tr, va, te = get(d_tr, i_tr), get(d_tr, i_va), get(d_te, i_te)
    out = {"raw": {}, "pca": {}}
    for nm in ("cheap", "hi", "hg"):
        out["raw"][nm] = {"train": tr[nm], "val": va[nm], "test": te[nm]}
    for nm in ("hi", "hg"):
        pca = RD.TrainPCA(pca_d, whiten=True).fit(tr[nm])
        out["pca"][nm] = {k: pca.transform(out["raw"][nm][k])
                          for k in ("train", "val", "test")}
    mu = tr["cheap"].mean(0, keepdims=True)
    sd = tr["cheap"].std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    out["pca"]["cheap"] = {k: ((out["raw"]["cheap"][k] - mu) / sd
                               ).astype(np.float32) for k in out["raw"]["cheap"]}
    return out


def fit_family(B, y, sid, epochs=400, seeds=3):
    """cheap / additive_pca / cheap+H / rank_cheap，全部按预注册口径。

    同时保存 val 预测，以便审计"用什么准则选 best_cheap"。
    """
    sel = P.make_selector("within_r2", sid["val"])
    raw, pca = B["raw"], B["pca"]
    Zi = {k: pca["hi"][k] for k in pca["hi"]}
    Zg = {k: pca["hg"][k] for k in pca["hg"]}
    Xa = {k: np.concatenate([Zi[k], Zg[k]], 1) for k in Zi}
    Hf = {k: np.concatenate([raw["hi"][k], raw["hg"][k]], 1) for k in Zi}
    out = {}
    m = P.fit_ridge(raw["cheap"]["train"], y["train"], raw["cheap"]["val"],
                    y["val"], raw["cheap"]["test"], sel, ALPHAS)
    out["cheap"] = m
    out["additive_pca"] = P.fit_ridge_2block(
        pca["cheap"]["train"], Xa["train"], y["train"], pca["cheap"]["val"],
        Xa["val"], y["val"], pca["cheap"]["test"], Xa["test"], sel,
        ALPHAS, GAMMAS)
    out["cheap+H"] = P.fit_ridge_2block(
        raw["cheap"]["train"], Hf["train"], y["train"], raw["cheap"]["val"],
        Hf["val"], y["val"], raw["cheap"]["test"], Hf["test"], sel,
        ALPHAS, GAMMAS)
    F = {"c": pca["cheap"], "hi": pca["hi"], "hg": pca["hg"]}
    rn = P._Runner({k: F[k]["train"] for k in F}, y["train"], sid["train"],
                   {k: F[k]["val"] for k in F}, y["val"], sid["val"],
                   {k: F[k]["test"] for k in F}, loss_kind="pairwise")
    d_c = pca["cheap"]["train"].shape[1]
    mm = rn.run(lambda: P.MLPProbe([d_c], (256,), keys=("c",)), sel,
                seeds=tuple(range(seeds)), epochs=epochs)
    if mm is not None:
        out["rank_cheap"] = mm
    return out


DESIGNS = {
    "D1_L6": dict(train=["a3", "b3"], test="freshA", layer=6, mode="transfer"),
    "D1_L8": dict(train=["a3", "b3"], test="freshA", layer=8, mode="transfer"),
    "MDLM_D2_L8": dict(train=["a3", "b3"], test=None, layer=8, mode="refit"),
    "SEDD_D2_L8": dict(train=["s1", "s2"], test=None, layer=8, mode="refit"),
    "FRESH_D2_L8": dict(train=["freshA"], test=None, layer=8, mode="refit"),
    "MDLM_D2_L6": dict(train=["a3", "b3"], test=None, layer=6, mode="refit"),
}


def stage_fit(names, epochs):
    os.makedirs(CACHE, exist_ok=True)
    for nm in names:
        cfg = DESIGNS[nm]
        out_p = os.path.join(CACHE, nm + ".npz")
        if os.path.exists(out_p):
            print("[skip]", nm); continue
        d_tr = RD.load_labels(cfg["train"])
        sp = RD.doc_splits(d_tr, seed=0)
        RD.check_split_disjoint(d_tr, sp)
        if cfg["mode"] == "transfer":
            d_te = RD.load_labels([cfg["test"]])
            ov = set(np.unique(d_tr["prompt_row"]).tolist()) & \
                set(np.unique(d_te["prompt_row"]).tolist())
            assert not ov, "prompt overlap"
            i_te = np.arange(len(d_te[TARGET]))
        else:
            d_te, i_te = d_tr, sp["test"]
        B = build_blocks(d_tr, sp["train"], sp["val"], d_te, i_te,
                         cfg["layer"])
        y = {"train": d_tr[TARGET][sp["train"]], "val": d_tr[TARGET][sp["val"]],
             "test": d_te[TARGET][i_te]}
        sid = {"train": d_tr["state_id"][sp["train"]],
               "val": d_tr["state_id"][sp["val"]],
               "test": d_te["state_id"][i_te]}
        fam = fit_family(B, y, sid, epochs=epochs)
        z = {"y": y["test"], "sid": sid["test"], "doc": d_te["doc_id"][i_te],
             "stratum": d_te["stratum"][i_te],
             "seeds": d_te["A_full_seeds"][i_te].astype(np.float64),
             "y_val": y["val"], "sid_val": sid["val"],
             "conf": d_te["C1"][i_te][:, 0]}
        meta = {}
        for k, v in fam.items():
            z["pred_" + k] = v["pred_test"]
            z["predval_" + k] = v["pred_val"]
            meta[k] = {"val_score_within_r2": float(v["val_score"]),
                       "hp": v.get("hp")}
        np.savez_compressed(out_p, **z)
        json.dump(meta, open(out_p.replace(".npz", "_meta.json"), "w"),
                  indent=1, default=float)
        print("[fit]", nm, {k: round(v["val_score"], 5)
                            for k, v in fam.items()})


# ----------------------------------------------------------------- analysis --
def pair_table(y, pred, groups, strat):
    """返回每个 state 的 (conc_sum, n_pairs) 逐层，形状 (S,4) 4=nat/inf/cross/all."""
    S = len(groups)
    C = np.zeros((S, 4)); N = np.zeros((S, 4))
    KEY = {(0, 0): 0, (1, 1): 1}
    for si, g in enumerate(groups):
        yy, pp, ss = y[g], pred[g], strat[g]
        n = len(g)
        for a in range(n):
            for b in range(a + 1, n):
                dy = yy[a] - yy[b]
                if abs(dy) <= 1e-9:
                    continue
                dp = pp[a] - pp[b]
                v = 0.5 if abs(dp) <= 1e-12 else float(np.sign(dy) == np.sign(dp))
                k = KEY.get((int(ss[a]), int(ss[b])), 2)
                C[si, k] += v; N[si, k] += 1
                C[si, 3] += v; N[si, 3] += 1
    return C, N


def micro(C, N, rows=None):
    if rows is not None:
        C, N = C[rows], N[rows]
    n = N.sum(0)
    return np.where(n > 0, C.sum(0) / np.maximum(n, 1), np.nan)


def load(nm):
    z = np.load(os.path.join(CACHE, nm + ".npz"))
    return {k: z[k] for k in z.files}


STRATA = ["within_nat", "within_inf", "cross", "all"]
EOF_MARK = None


def _groups(sid):
    _, g = M.group_slices(sid)
    return g


def _state_doc(groups, doc):
    return np.array([doc[g[0]] for g in groups])


def _perm_p(diff_c, diff_n, cluster=None, n_perm=20000, seed=0, alt="greater"):
    """符号翻转置换。diff_c=每 state 的 (conc_h - conc_c) 计数差，diff_n=配对数。

    cluster=None -> state 级翻转；否则同一 cluster 内的 state 共用一个符号。
    返回 (obs, p, null_sd, null_matrix_flags)
    """
    rng = np.random.default_rng(seed)
    tot_n = diff_n.sum()
    obs = diff_c.sum() / tot_n
    if cluster is None:
        S = np.sign(rng.integers(0, 2, size=(n_perm, len(diff_c))) * 2 - 1)
        null = (S * diff_c[None, :]).sum(1) / tot_n
    else:
        uc, inv = np.unique(cluster, return_inverse=True)
        S = (rng.integers(0, 2, size=(n_perm, len(uc))) * 2 - 1)[:, inv]
        null = (S * diff_c[None, :]).sum(1) / tot_n
    p = (null >= obs).mean() if alt == "greater" else (np.abs(null) >= abs(obs)).mean()
    return float(obs), float(max(p, 1.0 / n_perm)), float(null.std()), null


def _doc_boot(Ch, Nh, Cc, Nc, state_doc, n_boot=4000, seed=0):
    """文档级聚类自助（固定模型）。返回 (K,) 每个分层的 mean/lo/hi/se。"""
    rng = np.random.default_rng(seed)
    docs = np.unique(state_doc)
    idx_by = {d: np.where(state_doc == d)[0] for d in docs}
    samp = np.zeros((n_boot, Ch.shape[1]))
    for b in range(n_boot):
        pick = rng.choice(docs, len(docs), replace=True)
        rows = np.concatenate([idx_by[d] for d in pick])
        n = Nh[rows].sum(0)
        samp[b] = (Ch[rows].sum(0) - Cc[rows].sum(0)) / np.maximum(n, 1)
    lo, hi = np.percentile(samp, [2.5, 97.5], axis=0)
    return samp, lo, hi, samp.std(0)


def analyze_stratified(names, hidden="cheap+H", base="cheap", n_boot=4000,
                       n_perm=20000):
    """9 格（3 数据 × 3 子集）的分层 Δconcordance + 多重性校正。"""
    from rlib import stats2 as S
    res = {}
    null_cols, obs_vec, keys, p_state, p_doc = [], [], [], [], []
    for nm in names:
        z = load(nm)
        g = _groups(z["sid"])
        sdoc = _state_doc(g, z["doc"])
        Ch, Nh = pair_table(z["y"], z["pred_" + hidden], g, z["stratum"])
        Cc, Nc = pair_table(z["y"], z["pred_" + base], g, z["stratum"])
        assert np.allclose(Nh, Nc)
        mh, mc = micro(Ch, Nh), micro(Cc, Nc)
        samp, lo, hi, se = _doc_boot(Ch, Nh, Cc, Nc, sdoc, n_boot=n_boot)
        ent = {"n_pairs": Nh.sum(0).tolist(), "n_states": int(len(g)),
               "base": mc.tolist(), "hidden": mh.tolist(),
               "delta": (mh - mc).tolist(), "boot_lo": lo.tolist(),
               "boot_hi": hi.tolist(), "boot_se": se.tolist(), "cells": {}}
        dC = Ch - Cc
        for k, kn in enumerate(STRATA):
            o1, p1, sd1, _ = _perm_p(dC[:, k], Nh[:, k], None, n_perm, 0)
            o2, p2, sd2, nl2 = _perm_p(dC[:, k], Nh[:, k], sdoc, n_perm, 0)
            ent["cells"][kn] = {
                "delta": o1, "p_state_flip": p1, "null_sd_state": sd1,
                "p_doc_flip": p2, "null_sd_doc": sd2,
                "design_effect": (sd2 / sd1) ** 2,
                "boot_ci": [lo[k], hi[k]], "boot_se": float(se[k])}
            if kn != "all":
                keys.append(f"{nm}/{kn}"); obs_vec.append(o1)
                null_cols.append(nl2); p_state.append(p1); p_doc.append(p2)
        res[nm] = ent
    null = np.stack(null_cols, 1)                      # (B, 9) 同一 arm 共用翻转
    obs = np.array(obs_vec)
    wy = S.westfall_young(obs, null, "greater")
    _, bh_state = S.bh_fdr(np.array(p_state))
    _, by_state = S.by_fdr(np.array(p_state))
    _, bh_doc = S.bh_fdr(np.array(p_doc))
    _, by_doc = S.by_fdr(np.array(p_doc))
    res["multiplicity"] = {
        "keys": keys, "delta": obs.tolist(),
        "p_raw_state_flip": p_state, "p_raw_doc_flip": p_doc,
        "BH_on_state_flip": bh_state.tolist(),
        "BH_on_doc_flip": bh_doc.tolist(), "BY_on_doc_flip": by_doc.tolist(),
        "westfall_young_doc_flip": wy.tolist(),
        "note": "WY 用文档级翻转的同一副本；同一 arm 的 3 个子集共用符号向量"}
    return res


ENDPOINTS = ("concordance", "pairwise_auc", "kendall_tau", "within_spearman",
             "top1", "regret_norm", "within_r2")


def per_state_suffstats(y, pred, groups):
    """每个 state 的充分统计量；所有 7 个端点都是它们的可加函数。

    与 rlib/metrics.py 的定义逐字一致（已数值核对）。
    """
    from scipy import stats as sps
    S = len(groups)
    out = {k: np.full(S, np.nan) for k in
           ("cc", "cn", "tau", "sp", "top1", "regn", "sse", "sst")}
    for i, g in enumerate(groups):
        yy = y[g].astype(np.float64); pp = pred[g].astype(np.float64)
        c, n = M._pair_stats(yy, pp)
        out["cc"][i] = c; out["cn"][i] = n
        if len(g) >= 3:
            t = sps.kendalltau(yy, pp).statistic
            out["tau"][i] = t
            if np.std(pp) >= 1e-12:
                out["sp"][i] = sps.spearmanr(yy, pp).statistic
        b = int(np.argmax(yy))
        out["top1"][i] = 1.0 if int(np.argmax(pp)) == b else 0.0
        r = float(yy.max() - yy[int(np.argmax(pp))])
        rg = float(yy.max() - yy.min())
        out["regn"][i] = r / rg if rg > 1e-12 else 0.0
        yc = yy - yy.mean(); pc = pp - pp.mean()
        out["sse"][i] = float(((yc - pc) ** 2).sum())
        out["sst"][i] = float((yc ** 2).sum())
    return out


def _agg(st, rows):
    cc, cn = st["cc"][rows], st["cn"][rows]
    ok = cn > 0
    return {
        "concordance": cc[ok].sum() / cn[ok].sum(),
        "pairwise_auc": np.nanmean(np.where(ok, cc / np.maximum(cn, 1), np.nan)),
        "kendall_tau": np.nanmean(st["tau"][rows]),
        "within_spearman": np.nanmean(st["sp"][rows]),
        "top1": st["top1"][rows].mean(),
        "regret_norm": -st["regn"][rows].mean(),
        "within_r2": 1.0 - st["sse"][rows].sum() / st["sst"][rows].sum()}


def analyze_endpoints(nm, a="additive_pca", b="cheap", n_boot=4000, seed=0):
    """同一份预测上比较 7 个端点的功效（文档级自助 SE + z + MDE80）。"""
    z = load(nm)
    y, sid, doc = z["y"], z["sid"], z["doc"]
    groups = _groups(sid)
    sdoc = _state_doc(groups, doc)
    sa = per_state_suffstats(y, z["pred_" + a], groups)
    sb = per_state_suffstats(y, z["pred_" + b], groups)
    allrows = np.arange(len(groups))
    va, vb = _agg(sa, allrows), _agg(sb, allrows)
    obs = {k: va[k] - vb[k] for k in ENDPOINTS}
    rng = np.random.default_rng(seed)
    docs = np.unique(sdoc)
    idx_by = {d: np.where(sdoc == d)[0] for d in docs}
    samples = {k: np.empty(n_boot) for k in ENDPOINTS}
    for bi in range(n_boot):
        pick = rng.choice(docs, len(docs), replace=True)
        rows = np.concatenate([idx_by[d] for d in pick])
        xa, xb = _agg(sa, rows), _agg(sb, rows)
        for k in ENDPOINTS:
            samples[k][bi] = xa[k] - xb[k]
    out = {}
    for k in ENDPOINTS:
        v = samples[k][np.isfinite(samples[k])]
        lo, hi = np.percentile(v, [2.5, 97.5])
        se = float(v.std())
        out[k] = {"baseline": float(vb[k]), "treatment": float(va[k]),
                  "delta": float(obs[k]), "se": se,
                  "ci": [float(lo), float(hi)],
                  "z": float(obs[k] / max(se, 1e-12)),
                  "mde80_onesided": 2.486 * se,
                  "boot_p_le0": float((v <= 0).mean())}
    return out


SEL_CRIT = ("within_r2", "concordance", "top1", "regret_norm")


def analyze_baseline(nm, n_boot=4000, seed=0):
    """best_cheap 的选择准则敏感性 + best-of-two 选择的方向与大小。"""
    z = load(nm)
    yv, sv = z["y_val"], z["sid_val"]
    gv = _groups(sv)
    y, s = z["y"], z["sid"]
    g = _groups(s)
    cands = ["cheap", "rank_cheap"]
    allp = cands + ["additive_pca", "cheap+H"]
    stv = {c: per_state_suffstats(yv, z["predval_" + c], gv) for c in cands}
    stt = {c: per_state_suffstats(y, z["pred_" + c], g) for c in allp}
    rv = np.arange(len(gv)); rt = np.arange(len(g))
    val = {c: _agg(stv[c], rv) for c in cands}
    test = {c: _agg(stt[c], rt) for c in allp}
    out = {"val_scores": {c: {k: float(val[c][k]) for k in SEL_CRIT}
                          for c in cands},
           "test_scores": {c: {k: float(test[c][k]) for k in ENDPOINTS}
                           for c in allp},
           "by_criterion": {}}
    for k in SEL_CRIT:
        pick = max(cands, key=lambda c: val[c][k])
        out["by_criterion"][k] = {
            "best_cheap": pick,
            "primary_delta_conc": float(test["additive_pca"]["concordance"]
                                        - test[pick]["concordance"]),
            "delta_conc_cheapH": float(test["cheap+H"]["concordance"]
                                       - test[pick]["concordance"]),
            "primary_delta_top1": float(test["additive_pca"]["top1"]
                                        - test[pick]["top1"]),
            "primary_delta_regret_norm":
                float(test["additive_pca"]["regret_norm"]
                      - test[pick]["regret_norm"])}
    docv = sv // 10_000
    sdv = np.array([docv[x[0]] for x in gv])
    docs = np.unique(sdv)
    idx_by = {d: np.where(sdv == d)[0] for d in docs}
    rng = np.random.default_rng(seed)
    rec = {k: [] for k in SEL_CRIT}
    wins = {k: 0 for k in SEL_CRIT}
    for _ in range(n_boot):
        pick = rng.choice(docs, len(docs), replace=True)
        rows = np.concatenate([idx_by[d] for d in pick])
        sc = {c: _agg(stv[c], rows) for c in cands}
        for k in SEL_CRIT:
            w = max(cands, key=lambda c: sc[c][k])
            wins[k] += (w == "rank_cheap")
            rec[k].append(test[w]["concordance"])
    out["selection_stability"] = {
        k: {"P_pick_rank_cheap": wins[k] / n_boot,
            "E_test_conc_of_selected": float(np.mean(rec[k])),
            "test_conc_fixed_cheap": float(test["cheap"]["concordance"]),
            "selection_gain_over_fixed_cheap":
                float(np.mean(rec[k])) - float(test["cheap"]["concordance"])}
        for k in SEL_CRIT}
    return out


def _conc_to_rho(c):
    return np.sin(np.pi * (np.asarray(c, float) - 0.5))


def _rho_to_conc(r):
    return 0.5 + np.arcsin(np.clip(r, -1, 1)) / np.pi


def analyze_disattenuation(nm, probes=("cheap", "additive_pca", "cheap+H",
                                       "rank_cheap"), n_rep=40, seed=0):
    """用分半可靠性把 Δconcordance 折算到"无噪声标签"尺度。"""
    z = load(nm)
    y, s = z["y"], z["sid"]
    g = _groups(s)
    seeds = z["seeds"]
    K = seeds.shape[1]; half = K // 2
    rng = np.random.default_rng(seed)
    cs = []
    for _ in range(n_rep):
        pm = rng.permutation(K)
        a = seeds[:, pm[:half]].mean(1); b = seeds[:, pm[half:2 * half]].mean(1)
        cs.append(M.concordance(b, a, s, g))
    c_half = float(np.mean(cs))
    r_half = float(_conc_to_rho(c_half))                     # = r(m=K/2)
    nsr = half * (1.0 / max(r_half, 1e-9) - 1.0)             # sigma_n^2/sigma_s^2
    r_K = 1.0 / (1.0 + nsr / K)
    att = np.sqrt(r_K)
    out = {"conc_split_half_m%d" % half: c_half, "r_half": r_half,
           "noise_to_signal": nsr, "reliability_at_K": r_K,
           "attenuation_factor_sqrt_r": att,
           "conc_ceiling_perfect_predictor": float(_rho_to_conc(att)),
           "probes": {}}
    for p in probes:
        if "pred_" + p not in z:
            continue
        c = M.concordance(y, z["pred_" + p], s, g)
        rho_obs = float(_conc_to_rho(c))
        rho_true = min(rho_obs / att, 1.0)
        out["probes"][p] = {"conc": float(c), "rho_obs": rho_obs,
                            "rho_true": rho_true,
                            "conc_disattenuated": float(_rho_to_conc(rho_true))}
    return out


def analyze_decision_power(se, thresh=0.020, obs=None):
    """预注册判定规则（点估计 ≥ 阈值）本身的功效，与 Δ<阈值 的等价性检验。"""
    from scipy import stats as sps
    out = {"se": se, "threshold": thresh}
    out["power_vs_H0_zero_at_true_thresh"] = float(
        1 - sps.norm.cdf(1.645 - thresh / se))
    for d in (0.010, 0.015, 0.020, 0.025, 0.030, 0.040):
        out[f"P_pass_pointestimate_rule_if_true_{d:.3f}"] = float(
            1 - sps.norm.cdf((thresh - d) / se))
    out["true_effect_needed_for_80pct_pass"] = thresh + 0.8416 * se
    if obs is not None:
        zz = (obs - thresh) / se
        out["observed"] = obs
        out["equivalence_p_H0_delta_ge_thresh"] = float(sps.norm.cdf(zz))
        out["upper_95_onesided"] = obs + 1.645 * se
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS))
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "AUDITOR_B_recheck.json"))
    a = ap.parse_args()
    if a.stage in ("fit", "all"):
        stage_fit(a.designs, a.epochs)
    if a.stage in ("analyze", "all"):
        rep = {}
        rep["stratified"] = analyze_stratified(
            ["MDLM_D2_L8", "SEDD_D2_L8", "FRESH_D2_L8"])
        rep["stratified_highpower_D1"] = analyze_stratified(
            ["D1_L8", "D1_L6"])
        rep["stratified_vs_bestcheap_D1L8"] = analyze_stratified(
            ["D1_L8"], hidden="cheap+H", base="rank_cheap")
        rep["endpoints_D1_L6"] = analyze_endpoints("D1_L6")
        rep["endpoints_D1_L8"] = analyze_endpoints("D1_L8")
        rep["baseline_D1_L6"] = analyze_baseline("D1_L6")
        rep["baseline_D1_L8"] = analyze_baseline("D1_L8")
        rep["disattenuation_D1_L6"] = analyze_disattenuation("D1_L6")
        rep["disattenuation_MDLM_D2_L8"] = analyze_disattenuation("MDLM_D2_L8")
        se = rep["endpoints_D1_L6"]["concordance"]["se"]
        rep["decision_power_D1_L6"] = analyze_decision_power(
            se, 0.020, rep["endpoints_D1_L6"]["concordance"]["delta"])
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
