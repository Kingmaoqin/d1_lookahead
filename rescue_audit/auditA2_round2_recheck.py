"""Auditor-A 第二轮：对 round-2 修正代码本身的复核 + 更正实现。

不覆盖任何既有文件。四个 stage：

  nest      `probes3.fit_ridge_blocks` 的 record_slices 嵌套逻辑 vs 一份独立的
            numpy float64 穷举参考实现。验证 kron ⊇ additive ⊇ cheap_only，
            且每个切片取到的确实是其可行域内的 val 最优。

  boot      **A2-1（MAJOR）修正**：`strat_ci.py` 的聚类自助里，
            `M.group_slices(sid_te[idx])` 按 state_id 归组，于是同一个 document
            被抽到 m 次时，它的每个 state 的 6 行会被**合并成一个 6m 行的组**，
            产生 m² 份配对而不是 m 份。正确做法是把每一份拷贝当作独立的 state。
            这里同时给出 orig（m²）与 corrected（m）两版 CI。
            两版都用解析权重实现（已验证与逐行重算逐位相同）。

  capacity  **A2-2（MAJOR）修正**：`nonlinear_retest.py` 声称给每个关系型探针
            配了"同容量的 cheap-only 对照"，但 relmlp_w128 有 94,977 参数而
            cheaponly_mlp_w128 只有 13,057（7.3×）。于是"关系型 MLP 输给
            cheap-only MLP"与"容量更大因而过拟合更重"完全混淆。
            这里换成**形状与参数量逐位相同**的证伪对照：
              real      : rel = [h_i, h_g, h_i−h_g, |h_i−h_g|, h_i⊙h_g]
              shuffle_hg: 同上，但 h_g 在 state 之间整体置换（h_i 不动）
                          —— 只摧毁"候选×状态"的关系结构
              gauss     : 整块 rel 换成同均值同方差的高斯噪声（纯容量地板）
            三者共用同一套 lr/wd/seed/epochs 网格。

  arcsin    label_reliability_ceiling 里 conc = 0.5 + arcsin(rho)/pi 这一步的
            经验校准：用真实标签的经验边际 + 受控噪声，看公式偏多少。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P2           # noqa: E402
from rlib import probes3 as P3           # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

ARMS = {"MDLM_anc": ["a3", "b3"], "SEDD_anc": ["s1", "s2"],
        "FRESH_MDLM_anc": ["freshA"]}
KEYS = ("within_nat", "within_inf", "cross", "all")
RES = os.path.join(HERE, "results")


# ============================================================== stage: nest ==
def stage_nest():
    import itertools
    rng = np.random.default_rng(0)
    p = (5, 7, 11)
    mk = lambda n: [rng.standard_normal((n, d)).astype(np.float32) for d in p]
    Btr, Bva, Bte = mk(300), mk(120), mk(150)
    w = [rng.standard_normal(d) for d in p]
    f = lambda B: B[0] @ w[0] + 0.7 * B[1] @ w[1] + 0.4 * B[2] @ w[2]
    ytr = f(Btr) + 0.5 * rng.standard_normal(300)
    yva = f(Bva) + 0.5 * rng.standard_normal(120)
    sel = lambda y, q: -float(np.mean((np.asarray(y) - np.asarray(q)) ** 2))
    gm, gi = [0.0, 0.3, 1.0, 3.0], [0.0, 0.1, 1.0]
    SL = {"cheap_only": [True, False, False], "additive": [True, True, False],
          "kron": [True, True, True]}
    alphas = np.logspace(-2, 4, 13)
    res = P3.fit_ridge_blocks(Btr, ytr, Bva, yva, Bte, sel, [[1.0], gm, gi],
                              alphas=alphas, record_slices=SL)

    def brute(allow):
        best = None
        for combo in itertools.product([1.0], gm, gi):
            act = [g > 0 for g in combo]
            if not any(act) or any(act[b] and not allow[b] for b in range(3)):
                continue
            Xt, Xv = [], []
            for b in [b for b in range(3) if act[b]]:
                mu = Btr[b].mean(0, keepdims=True)
                sd = Btr[b].std(0, keepdims=True); sd[sd < 1e-8] = 1.0
                Xt.append(combo[b] * (Btr[b] - mu) / sd)
                Xv.append(combo[b] * (Bva[b] - mu) / sd)
            Xt = np.concatenate(Xt, 1).astype(np.float64)
            Xv = np.concatenate(Xv, 1).astype(np.float64)
            ym = ytr.mean()
            lam, V = np.linalg.eigh(Xt.T @ Xt)
            Vtb = V.T @ (Xt.T @ (ytr - ym))
            for a in alphas:
                s = sel(yva, Xv @ (V @ (Vtb / (np.clip(lam, 0, None) + a))) + ym)
                if best is None or s > best[0]:
                    best = (s, combo, a)
        return best

    out = {}
    for k, allow in SL.items():
        b = brute(allow)
        g = res[k]["hp"]["gammas"]
        out[k] = {"impl_val": res[k]["val_score"], "brute_val": b[0],
                  "abs_diff": abs(res[k]["val_score"] - b[0]),
                  "impl_gammas": g, "brute_gammas": list(b[1]),
                  "slice_legal": not any((g[i] > 0) and not allow[i]
                                         for i in range(3))}
        print(f"  {k:11s} impl {res[k]['val_score']:.8f}  brute {b[0]:.8f}"
              f"  |diff| {out[k]['abs_diff']:.2e}  legal={out[k]['slice_legal']}")
    out["nesting_kron_ge_additive"] = bool(
        res["kron"]["val_score"] >= res["additive"]["val_score"] - 1e-12)
    out["nesting_additive_ge_cheap"] = bool(
        res["additive"]["val_score"] >= res["cheap_only"]["val_score"] - 1e-12)
    print("  nesting:", out["nesting_kron_ge_additive"],
          out["nesting_additive_ge_cheap"])
    return out


# ============================================================== stage: boot ==
def _per_state_pair_counts(y, pred, sid, strat):
    _, groups = M.group_slices(sid)
    C = {k: [] for k in KEYS}; N = {k: [] for k in KEYS}
    for g in groups:
        yy, pp, ss = y[g], pred[g], strat[g]
        n = len(g); acc = {k: [0.0, 0] for k in KEYS}
        for a in range(n):
            for b in range(a + 1, n):
                dy = yy[a] - yy[b]
                if abs(dy) <= 1e-9:
                    continue
                dp = pp[a] - pp[b]
                v = 0.5 if abs(dp) <= 1e-12 else float(np.sign(dy) == np.sign(dp))
                key = ("cross" if ss[a] != ss[b]
                       else ("within_nat" if ss[a] == 0 else "within_inf"))
                acc[key][0] += v; acc[key][1] += 1
                acc["all"][0] += v; acc["all"][1] += 1
        for k in KEYS:
            C[k].append(acc[k][0]); N[k].append(acc[k][1])
    return ({k: np.array(C[k]) for k in KEYS},
            {k: np.array(N[k], float) for k in KEYS}, groups)


def _boot(Cc, Ch, N, sdoc, mode, n_boot, seed):
    """mode='orig'  复刻 strat_ci.py（重复文档被合并成一个大 state -> 权重 m²）
       mode='fixed' 正确的聚类自助（重复文档的每份拷贝是独立的 state -> 权重 m）"""
    cl = np.unique(sdoc); ci = np.searchsorted(cl, sdoc)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in KEYS}
    for _ in range(n_boot):
        mult = np.bincount(rng.choice(len(cl), len(cl), replace=True),
                           minlength=len(cl)).astype(float)
        w = mult[ci]
        if mode == "orig":
            w = w * w
        for k in KEYS:
            den = float((w * N[k]).sum())
            acc[k].append(float(((w * Ch[k]).sum() - (w * Cc[k]).sum()) / den)
                          if den > 0 else np.nan)
    out = {}
    for k in KEYS:
        v = np.array([x for x in acc[k] if np.isfinite(x)])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"mean": float(v.mean()), "sd": float(v.std()),
                  "ci_lo": float(lo), "ci_hi": float(hi),
                  "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def stage_boot(n_boot=20000, layer=8, target="A_pertok"):
    rep = {}
    for arm, tags in ARMS.items():
        d = RD.load_labels(tags); sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        _, ga, _ = RD.state_groups(d["state_id"]); sid = SC.split_sid(d, sp)
        y = SC.split_targets(d, sp, target)
        prep = SC.prepare(d, sp, layer, pca_dim=128, groups=ga)
        sel = P2.make_selector("within_r2", sid["val"])
        mc = P2.fit_ridge(prep["raw"]["cheap"]["train"], y["train"],
                          prep["raw"]["cheap"]["val"], y["val"],
                          prep["raw"]["cheap"]["test"], sel)
        H = {k: np.concatenate([prep["raw"]["hi"][k], prep["raw"]["hg"][k]], 1)
             for k in ("train", "val", "test")}
        mh = P2.fit_ridge_2block(prep["raw"]["cheap"]["train"], H["train"],
                                 y["train"], prep["raw"]["cheap"]["val"],
                                 H["val"], y["val"],
                                 prep["raw"]["cheap"]["test"], H["test"], sel)
        te = sp["test"]; strat = d["stratum"][te]; doc = d["doc_id"][te]
        Cc, N, groups = _per_state_pair_counts(y["test"], mc["pred_test"],
                                               sid["test"], strat)
        Ch, N2, _ = _per_state_pair_counts(y["test"], mh["pred_test"],
                                           sid["test"], strat)
        for k in KEYS:
            assert np.array_equal(N[k], N2[k])
        sdoc = np.array([doc[g[0]] for g in groups])
        for g in groups:
            assert len(np.unique(doc[g])) == 1, "state spans two documents"
        pt = {k: float((Ch[k].sum() - Cc[k].sum()) / N[k].sum()) for k in KEYS}
        o = _boot(Cc, Ch, N, sdoc, "orig", 1000, 0)      # 复刻原设置
        f = _boot(Cc, Ch, N, sdoc, "fixed", n_boot, 1)
        rep[arm] = {"n_test_docs": int(len(np.unique(doc))),
                    "n_test_states": int(len(groups)),
                    "n_pairs": {k: int(N[k].sum()) for k in KEYS},
                    "gamma_hidden_block": float(mh["gamma"]),
                    "point": pt, "orig_m2_weighting": o, "corrected": f}
        print(f"===== {arm}  docs={rep[arm]['n_test_docs']} "
              f"states={rep[arm]['n_test_states']}")
        for k in KEYS:
            print(f"   {k:11s} point {pt[k]:+.4f} | ORIG mean {o[k]['mean']:+.4f} "
                  f"CI[{o[k]['ci_lo']:+.4f},{o[k]['ci_hi']:+.4f}] "
                  f"excl0={str(o[k]['ci_excludes_zero']):5s} sd={o[k]['sd']:.4f}"
                  f" | FIXED mean {f[k]['mean']:+.4f} "
                  f"CI[{f[k]['ci_lo']:+.4f},{f[k]['ci_hi']:+.4f}] "
                  f"excl0={str(f[k]['ci_excludes_zero']):5s} sd={f[k]['sd']:.4f}",
                  flush=True)
    return rep


# ========================================================== stage: capacity ==
def _shuffle_hg_states(hg, sid, rng):
    """h_g 在 state 之间整体置换（h_g 在 state 内恒定，已核验）。"""
    _, groups = M.group_slices(sid)
    X = np.array(hg, copy=True)
    perm = rng.permutation(len(groups))
    src = [hg[groups[p][0]] for p in perm]
    for i, g in enumerate(groups):
        X[g] = src[i]
    return X


def stage_capacity(arm="MDLM_anc", layer=8, target="A_pertok", widths=(128, 512),
                   epochs=3000, patience=500, seeds=2, scen=None, syn_seed=12345):
    lrs, wds = (3e-2, 1e-2, 3e-3), (0.0, 1e-2)
    sd_t = tuple(range(seeds))
    d = RD.load_labels(ARMS[arm])
    sp = RD.doc_splits(d, seed=0); RD.check_split_disjoint(d, sp)
    _, groups, _ = RD.state_groups(d["state_id"])
    sid = SC.split_sid(d, sp)
    L = 9 if scen else layer
    prep = SC.prepare(d, sp, L, pca_dim=128, groups=groups)
    if scen:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "syn", os.path.join(HERE, "synthetic_tests.py"))
        syn = importlib.util.module_from_spec(spec); spec.loader.exec_module(syn)
        rng = np.random.default_rng(syn_seed)          # 显式种子（原代码用 hash()）
        y_all = syn.build(scen, RD.h_i(d, L), RD.h_g(d, L), d["state_id"],
                          rng, snr=2.0)
        y = {k: y_all[v] for k, v in sp.items()}
    else:
        y = SC.split_targets(d, sp, target)

    rng = np.random.default_rng(2026)
    hi = prep["pca"]["hi"]; hg = prep["pca"]["hg"]; ch = prep["pca"]["cheap"]
    variants = {}
    variants["real"] = {k: RD.relational_block(hi[k], hg[k])
                        for k in ("train", "val", "test")}
    hg_s = {k: _shuffle_hg_states(hg[k], sid[k], rng)
            for k in ("train", "val", "test")}
    variants["shuffle_hg"] = {k: RD.relational_block(hi[k], hg_s[k])
                              for k in ("train", "val", "test")}
    ref = variants["real"]["train"]
    mu, sg = ref.mean(0, keepdims=True), ref.std(0, keepdims=True)
    variants["gauss"] = {k: (mu + sg * rng.standard_normal(
        variants["real"][k].shape)).astype(np.float32)
        for k in ("train", "val", "test")}

    sel = P2.make_selector("within_r2", sid["val"])
    _, g_te = M.group_slices(sid["test"])
    d_c = ch["train"].shape[1]; d_r = variants["real"]["train"].shape[1]
    out = {}
    mc = P2.fit_ridge(prep["raw"]["cheap"]["train"], y["train"],
                      prep["raw"]["cheap"]["val"], y["val"],
                      prep["raw"]["cheap"]["test"], sel)
    out["cheap_ridge"] = M.full_report(y["test"], mc["pred_test"], sid["test"],
                                       g_te)
    print(f"[{arm} {'syn:'+scen if scen else 'real'} L{L}] cheap ridge conc="
          f"{out['cheap_ridge']['concordance']:.4f} "
          f"within={out['cheap_ridge']['within_r2']:+.4f}", flush=True)
    for w in widths:
        for vname, V in variants.items():
            feats = {"x": V, "c": ch}
            rn = P3.Runner3({k: feats[k]["train"] for k in feats}, y["train"],
                            sid["train"],
                            {k: feats[k]["val"] for k in feats}, y["val"],
                            sid["val"],
                            {k: feats[k]["test"] for k in feats},
                            loss_kind="mse")
            m = rn.run(lambda: P2.MLPProbe([d_r, d_c], (w,), keys=("x", "c")),
                       sel, seeds=sd_t, epochs=epochs, lrs=lrs, wds=wds,
                       patience=patience)
            name = f"relmlp_w{w}__{vname}"
            r = M.full_report(y["test"], m["pred_test"], sid["test"], g_te)
            r["hp"] = m["hp"]; r["n_params"] = m["n_params"]
            r["best_epoch"] = m["best_epoch"]; r["val_score"] = m["val_score"]
            out[name] = r
            print(f"    {name:28s} n_par={m['n_params']:7d} val={m['val_score']:+.4f}"
                  f" conc={r['concordance']:.4f} within={r['within_r2']:+.4f}"
                  f" top1={r['top1']:.4f} ep={m['best_epoch']}", flush=True)
        a = out[f"relmlp_w{w}__real"]; b = out[f"relmlp_w{w}__shuffle_hg"]
        c = out[f"relmlp_w{w}__gauss"]
        out[f"delta_w{w}"] = {
            "real_minus_shuffle_hg": {k: a[k] - b[k] for k in
                                      ("concordance", "within_r2", "top1")},
            "real_minus_gauss": {k: a[k] - c[k] for k in
                                 ("concordance", "within_r2", "top1")},
            "params_equal": bool(a["n_params"] == b["n_params"] == c["n_params"])}
        print(f"    >> w{w} 容量完全匹配={out[f'delta_w{w}']['params_equal']}  "
              f"Δconc(real−shufHg)={out[f'delta_w{w}']['real_minus_shuffle_hg']['concordance']:+.4f}"
              f"  Δconc(real−gauss)={out[f'delta_w{w}']['real_minus_gauss']['concordance']:+.4f}",
              flush=True)
    return out


# ============================================================ stage: arcsin ==
def stage_arcsin():
    f = lambda r: 0.5 + np.arcsin(np.clip(r, -1, 1)) / np.pi
    rep = {}
    for arm, tags in ARMS.items():
        d = RD.load_labels(tags, keys=["prompt_row", "step", "doc_id",
                                       "A_pertok", "A_full_seeds"])
        sid = d["state_id"]; _, groups, _ = RD.state_groups(sid)
        S = d["A_full_seeds"].astype(np.float64); K = S.shape[1]
        yv = d["A_pertok"].astype(np.float64)
        yc = RD.center_within_state(yv, sid, groups)
        rng = np.random.default_rng(0); sdv = yc.std()
        gauss = {}
        for a in (0.5, 0.7, 0.8, 0.9, 0.9395, 0.97):
            p = a * yc + np.sqrt(1 - a * a) * sdv * rng.standard_normal(len(yc))
            e = M.concordance(yc, p, sid, groups)
            gauss[f"{a:.4f}"] = {"empirical": float(e), "arcsin": float(f(a)),
                                 "err": float(e - f(a))}
        E = S - S.mean(1, keepdims=True)
        rr, ee = [], []
        for _ in range(20):
            nz = E[:, rng.integers(0, K, K)].mean(1)
            yt = RD.center_within_state(yv + nz, sid, groups)
            rr.append(float(np.corrcoef(yc, yt)[0, 1]))
            ee.append(M.concordance(yt, yc, sid, groups))
        rep[arm] = {"label_eq_seed_mean_maxabs":
                    float(np.abs(yv - S.mean(1)).max()),
                    "gaussian_noise_probe": gauss,
                    "real_noise_probe": {"rho": float(np.mean(rr)),
                                         "empirical": float(np.mean(ee)),
                                         "arcsin": float(f(np.mean(rr))),
                                         "err": float(np.mean(ee) - f(np.mean(rr)))}}
        print(f"== {arm}")
        for k, v in gauss.items():
            print(f"   rho={k}  emp {v['empirical']:.4f}  arcsin {v['arcsin']:.4f}"
                  f"  err {v['err']:+.4f}")
        r = rep[arm]["real_noise_probe"]
        print(f"   real-noise: rho={r['rho']:.4f} emp {r['empirical']:.4f} "
              f"arcsin {r['arcsin']:.4f} err {r['err']:+.4f}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["nest", "boot", "capacity", "arcsin"])
    ap.add_argument("--arm", default="MDLM_anc")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--scen", default=None)
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "nest":
        r = stage_nest()
    elif a.stage == "boot":
        r = stage_boot(n_boot=a.n_boot, layer=a.layer)
    elif a.stage == "capacity":
        r = stage_capacity(arm=a.arm, layer=a.layer, scen=a.scen,
                           widths=tuple(a.widths))
    else:
        r = stage_arcsin()
    tag = a.stage + (f"_{a.arm}" + (f"_{a.scen}" if a.scen else "_real")
                     if a.stage == "capacity" else "")
    out = a.out or os.path.join(RES, f"auditA2_{tag}.json")
    json.dump(r, open(out, "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
