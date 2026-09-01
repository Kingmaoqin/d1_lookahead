"""
标签的**排序可靠性天花板**——本轮否定结论最关键的一个前提检查。

动机：全项目所有的 Δconcordance 都是"探针 vs cheap 控制组"的差。但没有人问过
一个更基本的问题：

    给定标签本身是 K=24 条推演的蒙特卡洛均值，**任何**预测器在 within-state
    排序上最多能到多少 concordance？

如果这个上限只有 0.79，而 cheap 已经拿到 0.7786，那么"没有探针能超过 cheap"
就是**测量精度**的陈述，不是表征的陈述——整个 KILL 判定就要重写。
如果上限是 0.89，那么 cheap 之上还有 0.11 的空间没被任何探针拿到，负结果
才真正是关于表征的。

做法（不依赖任何高斯近似）：
  1. 把 K 条种子随机分成两个不相交的半 A / 半 B，各自求均值得到 Ā_A、Ā_B。
     两者信号相同、seed 子集不重叠；这是可靠性诊断，不是 oracle 标签。
  2. `concordance(y = Ā_B, pred = Ā_A)` 是 K/2 噪声水平下的**分半可靠性**，
     不是预测器的上限。
  3. 对 K/2 = 2, 3, 4, 6, 12 都做一遍，画出 concordance 随每半种子数的曲线，
     用显式可靠性模型解释曲线；不对 1/m 作线性“无噪声上限”外推。
  4. 同时把 cheap / cheap+H 探针也拿 Ā_B 当标签评一次，使噪声口径可比。

另外报告一个高斯参照：若 (pred, label) 联合高斯，
    concordance = 0.5 + arcsin(rho) / pi
把 within-state 噪声天花板（可解释方差比例）折算成 concordance 上限，
与经验值对照。
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

ARMS = {"MDLM_anc": ["a3", "b3"], "MDLM_conf": ["c3", "d3"],
        "SEDD_anc": ["s1", "s2"], "FRESH_MDLM_anc": ["freshA"]}
SEEDKEY = {"A_pertok": "A_full_seeds", "A_future": "A_future_seeds"}


def split_half_concordance(seeds, sid, groups, m, n_rep, rng):
    """把 K 条种子随机分成两个不相交的 m 条子集，返回 concordance 的分布。"""
    K = seeds.shape[1]
    assert 2 * m <= K
    out = []
    for _ in range(n_rep):
        perm = rng.permutation(K)
        a, b = perm[:m], perm[m:2 * m]
        ya = seeds[:, a].mean(1)
        yb = seeds[:, b].mean(1)
        out.append(M.concordance(yb, ya, sid, groups))
    return np.array(out)


def gaussian_conc_from_rho(rho):
    return 0.5 + np.arcsin(np.clip(rho, -1, 1)) / np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["MDLM_anc", "SEDD_anc",
                                                  "FRESH_MDLM_anc"])
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--n_rep", type=int, default=40)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "label_reliability.json"))
    args = ap.parse_args()
    rep = {"config": vars(args), "arms": {}}

    for arm in args.arms:
        d = RD.load_labels(ARMS[arm])
        sid = d["state_id"]
        uniq, groups, _ = RD.state_groups(sid)
        sk = SEEDKEY[args.target]
        seeds = d[sk].astype(np.float64)
        K = seeds.shape[1]
        rng = np.random.default_rng(0)
        ent = {"tags": ARMS[arm], "K": int(K), "n_states": int(len(uniq)),
               "n_rows": int(len(sid)), "curve": {}}

        # ---- 1. 分半曲线 ----
        ms = [m for m in (2, 3, 4, 6, 8, 12) if 2 * m <= K]
        for m in ms:
            v = split_half_concordance(seeds, sid, groups, m, args.n_rep, rng)
            ent["curve"][str(m)] = {"mean": float(v.mean()),
                                    "sd": float(v.std()),
                                    "n_rep": int(len(v))}
            print(f"[{arm}] 每半 m={m:2d} 条种子: concordance "
                  f"{v.mean():.4f} ± {v.std():.4f}", flush=True)

        # ---- 2. 从分半曲线反推噪声模型，再算“完美预测器”的上限 ----
        #
        # 【一处必须写清楚的自我修正】第一版这里对 conc(1/m) 做线性外推，
        # 得到 0.79 并把它当作“concordance 上限”。那是**错的**，两重错误：
        #   (a) conc 关于 1/m 是凸的，线性外推在 1/m→0 处系统性偏低；
        #   (b) 更根本的是它回答错了问题——分半 concordance 衡量的是
        #       “一个**同样带噪**的 A 测量去排序”，不是“完美预测器”。
        #       m→∞ 时两半都无噪，该曲线的真极限是 1.0，不是任何有限值。
        #
        # 正确做法：分半相关就是可靠性 r(m) = σ²_s / (σ²_s + σ²_n/m)。
        # 若 (预测, 标签) 近似联合高斯，concordance = 0.5 + arcsin(ρ)/π。
        # 于是：
        #   * 带噪预测器（m 条种子）vs 带噪标签（m 条种子）: ρ = r(m)
        #   * **完美**预测器 vs 实际使用的 K 条种子标签:      ρ = sqrt(r(K))
        # 后者才是任何探针在本数据上的可达上限。
        x = np.array([1.0 / m for m in ms])
        y = np.array([ent["curve"][str(m)]["mean"] for m in ms])
        ent["curve_fit_note"] = ("conc(1/m) 是凸的，线性外推无意义；"
                                 "改用可靠性模型 + arcsin 变换")
        # 用 within-state 噪声天花板给出 σ²_n/σ²_s
        wc = RD.within_state_noise_ceiling(seeds, sid, groups)
        rK = float(np.clip(wc["ceiling"], 1e-9, 1 - 1e-9))   # r(K)
        ratio = (1.0 / rK - 1.0) * K                          # σ²_n / σ²_s
        ent["noise_to_signal_ratio"] = float(ratio)
        pred_curve = {str(m): float(gaussian_conc_from_rho(1.0 / (1.0 + ratio / m)))
                      for m in ms}
        ent["curve_predicted_by_model"] = pred_curve
        err = max(abs(pred_curve[str(m)] - ent["curve"][str(m)]["mean"])
                  for m in ms)
        ent["curve_model_max_abs_error"] = float(err)
        print(f"[{arm}] 噪声/信号比 = {ratio:.2f}; 分半曲线的模型预测值与实测"
              f"最大偏差 {err:.4f}", flush=True)
        for m in ms:
            print(f"          m={m:2d} 实测 {ent['curve'][str(m)]['mean']:.4f}"
                  f"  模型 {pred_curve[str(m)]:.4f}", flush=True)

        # ---- 3. 完美预测器的 concordance 上限 ----
        ent["within_ceiling_r2"] = wc
        rho_max = float(np.sqrt(rK))
        ent["conc_ceiling_perfect_predictor"] = float(
            gaussian_conc_from_rho(rho_max))
        print(f"[{arm}] within-state R² 天花板 r(K)={rK:.4f} "
              f"-> 完美预测器与 K={K} 标签的相关 ρ={rho_max:.4f} "
              f"-> **concordance 上限 {ent['conc_ceiling_perfect_predictor']:.4f}**",
              flush=True)

        # ---- 4. 探针在同一口径下的表现 ----
        sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        s_ = SC.split_sid(d, sp)
        y_full = {k: d[args.target][v] for k, v in sp.items()}
        prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups)
        sel = P.make_selector("within_r2", s_["val"])
        m_ch = P.fit_ridge(prep["raw"]["cheap"]["train"], y_full["train"],
                           prep["raw"]["cheap"]["val"], y_full["val"],
                           prep["raw"]["cheap"]["test"], sel)
        H = {k: np.concatenate([prep["raw"]["hi"][k], prep["raw"]["hg"][k]], 1)
             for k in ("train", "val", "test")}
        m_h = P.fit_ridge_2block(prep["raw"]["cheap"]["train"], H["train"],
                                 y_full["train"], prep["raw"]["cheap"]["val"],
                                 H["val"], y_full["val"],
                                 prep["raw"]["cheap"]["test"], H["test"], sel)
        te = sp["test"]
        _, g_te = M.group_slices(s_["test"])
        # 用全 K 标签评（常规口径）
        c_cheap = M.concordance(y_full["test"], m_ch["pred_test"],
                                s_["test"], g_te)
        c_hid = M.concordance(y_full["test"], m_h["pred_test"],
                              s_["test"], g_te)
        ceil = ent["conc_ceiling_perfect_predictor"]
        ent["probe_vs_full_label"] = {
            "cheap": c_cheap, "cheap+H": c_hid, "ceiling": ceil,
            "headroom_above_cheap": float(ceil - c_cheap),
            "hidden_gain": float(c_hid - c_cheap),
            "frac_of_headroom_captured_by_hidden":
                float((c_hid - c_cheap) / max(ceil - c_cheap, 1e-9))}
        print(f"[{arm}] 全 K 标签口径: cheap {c_cheap:.4f}  cheap+H {c_hid:.4f}"
              f"  上限 {ceil:.4f}  ->  cheap 之上还剩 {ceil-c_cheap:+.4f}，"
              f"隐藏态吃掉其中 "
              f"{100*(c_hid-c_cheap)/max(ceil-c_cheap,1e-9):.1f}%", flush=True)
        # 用半 B 标签评（与分半上限同噪声口径）
        half = K // 2
        rng2 = np.random.default_rng(7)
        cb, chh, ceil_te = [], [], []
        for _ in range(args.n_rep):
            perm = rng2.permutation(K)
            a, b = perm[:half], perm[half:2 * half]
            yb = seeds[te][:, b].mean(1)
            ya = seeds[te][:, a].mean(1)
            cb.append(M.concordance(yb, m_ch["pred_test"], s_["test"], g_te))
            chh.append(M.concordance(yb, m_h["pred_test"], s_["test"], g_te))
            ceil_te.append(M.concordance(yb, ya, s_["test"], g_te))
        ent["on_test_half_label"] = {
            "cheap": float(np.mean(cb)), "cheap+H": float(np.mean(chh)),
            "split_half_reliability": float(np.mean(ceil_te)),
            "gap_over_cheap_NOT_HEADROOM": float(np.mean(ceil_te) - np.mean(cb))}
        print(f"[{arm}] test 集上（半 B 标签为真值）: cheap "
              f"{np.mean(cb):.4f}  cheap+H {np.mean(chh):.4f}  "
              f"分半可靠性 {np.mean(ceil_te):.4f}  "
              f"与 cheap 的差 {np.mean(ceil_te)-np.mean(cb):+.4f}", flush=True)

        rep["arms"][arm] = ent
        json.dump(rep, open(args.out, "w"), indent=2, default=float)

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
