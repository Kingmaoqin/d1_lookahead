"""
分层 concordance：候选集的分层结构会不会让 cheap 基线"不劳而获"？

每个 state 的 6 个候选是 3 个 natural（掩码位均匀抽样）+ 3 个 informative
（高置信 × 高不稳定）。全部 15 个配对里，**natural × informative 的 9 个跨层
配对**可能仅凭置信度就能分开——而置信度正是 C1 的第一个特征。
若如此，cheap 的 0.7786 里有相当一部分是这种"分层可分性"，
真正的候选级决策（同层内部比较）被稀释，隐藏态的空间被人为压缩。

本脚本把 concordance 拆成三部分并分别评：
    within-natural   ：3 个 natural 之间的 3 个配对
    within-informative：3 个 informative 之间的 3 个配对
    cross            ：natural × informative 的 9 个配对
并给出各自的 cheap / cheap+H / 完美预测器上限。

注：`stratum` 字段 0=natural, 1=informative（见 src/collect.py:pick_candidates）。
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

ARMS = {"MDLM_anc": ["a3", "b3"], "SEDD_anc": ["s1", "s2"],
        "FRESH_MDLM_anc": ["freshA"]}


def conc_by_stratum(y, pred, groups, strat):
    """返回 {within_nat, within_inf, cross, all} 的 concordance。"""
    acc = {k: [0.0, 0] for k in ("within_nat", "within_inf", "cross", "all")}
    for g in groups:
        yy, pp, ss = y[g], pred[g], strat[g]
        n = len(g)
        for a in range(n):
            for b in range(a + 1, n):
                dy = yy[a] - yy[b]
                if abs(dy) <= 1e-9:
                    continue
                dp = pp[a] - pp[b]
                v = 0.5 if abs(dp) <= 1e-12 else float(np.sign(dy) == np.sign(dp))
                if ss[a] == ss[b]:
                    key = "within_nat" if ss[a] == 0 else "within_inf"
                else:
                    key = "cross"
                acc[key][0] += v; acc[key][1] += 1
                acc["all"][0] += v; acc["all"][1] += 1
    return {k: (c / n if n else float("nan")) for k, (c, n) in acc.items()}, \
           {k: n for k, (c, n) in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["MDLM_anc", "SEDD_anc",
                                                  "FRESH_MDLM_anc"])
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--n_rep", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "stratified_concordance.json"))
    args = ap.parse_args()
    rep = {"config": vars(args), "arms": {}}

    for arm in args.arms:
        d = RD.load_labels(ARMS[arm])
        sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        _, groups_all, _ = RD.state_groups(d["state_id"])
        sid = SC.split_sid(d, sp)
        y = SC.split_targets(d, sp, args.target)
        prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups_all)
        sel = P.make_selector("within_r2", sid["val"])
        m_ch = P.fit_ridge(prep["raw"]["cheap"]["train"], y["train"],
                           prep["raw"]["cheap"]["val"], y["val"],
                           prep["raw"]["cheap"]["test"], sel)
        H = {k: np.concatenate([prep["raw"]["hi"][k], prep["raw"]["hg"][k]], 1)
             for k in ("train", "val", "test")}
        m_h = P.fit_ridge_2block(prep["raw"]["cheap"]["train"], H["train"],
                                 y["train"], prep["raw"]["cheap"]["val"],
                                 H["val"], y["val"],
                                 prep["raw"]["cheap"]["test"], H["test"], sel)
        te = sp["test"]
        _, g_te = M.group_slices(sid["test"])
        strat = d["stratum"][te]
        yt = y["test"]

        c_cheap, npairs = conc_by_stratum(yt, m_ch["pred_test"], g_te, strat)
        c_hid, _ = conc_by_stratum(yt, m_h["pred_test"], g_te, strat)
        # 只用置信度（C1 第一列 p1）作为排序器，量化"分层可分性"
        p1 = d["C1"][te][:, 0]
        c_p1, _ = conc_by_stratum(yt, p1, g_te, strat)
        # 分半可达上限（同一分层口径）
        sk = "A_full_seeds"
        seeds = d[sk][te].astype(np.float64)
        K = seeds.shape[1]; half = K // 2
        rng = np.random.default_rng(0)
        acc = {k: [] for k in ("within_nat", "within_inf", "cross", "all")}
        for _ in range(args.n_rep):
            perm = rng.permutation(K)
            ya = seeds[:, perm[:half]].mean(1)
            yb = seeds[:, perm[half:2 * half]].mean(1)
            c, _ = conc_by_stratum(yb, ya, g_te, strat)
            for k in acc:
                acc[k].append(c[k])
        c_split = {k: float(np.mean(v)) for k, v in acc.items()}

        ent = {"n_pairs": npairs, "cheap": c_cheap, "cheap+H": c_hid,
               "confidence_only": c_p1, "split_half_reliability": c_split,
               "delta_hidden": {k: c_hid[k] - c_cheap[k] for k in c_cheap}}
        rep["arms"][arm] = ent
        print(f"\n===== {arm} (layer {args.layer})")
        print(f"  {'子集':<18}{'配对数':>8}{'仅置信度':>10}{'cheap':>9}"
              f"{'cheap+H':>10}{'Δhidden':>10}{'分半可靠性':>12}")
        for k in ("within_nat", "within_inf", "cross", "all"):
            print(f"  {k:<18}{npairs[k]:>8}{c_p1[k]:>10.4f}{c_cheap[k]:>9.4f}"
                  f"{c_hid[k]:>10.4f}{c_hid[k]-c_cheap[k]:>+10.4f}"
                  f"{c_split[k]:>12.4f}")
        json.dump(rep, open(args.out, "w"), indent=2, default=float)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
