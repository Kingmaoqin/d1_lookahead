"""
合成自测 A–F（任务书 §10）。

全部建在**真实的隐藏态几何**上（取 a3/b3 的第 L 层 h_i / h_g、真实的
state/doc 结构），只把标签换成已知构造。这样测的是"探针族在这份数据的
真实协方差结构下能不能读出该种结构"，而不是在理想高斯数据上自欺。

期望模式（不符即说明探针族有问题，任何真实数据上的零结果都不可信）：

  A 线性     : 线性成功、非线性也成功
  B 双线性   : h_i 线性失败、[h_i;h_g] 加性线性失败、**双线性成功**、
               shuffled h_g 失败                      <-- 本轮最关键自测
  D 仅排序   : 池化回归被状态级方差淹没、**排序探针恢复候选顺序**
               并且 within_r2 选择准则明显优于 pooled_r2 选择准则
  E 非线性   : 线性弱、MLP/核成功
  F 零假设   : 全部 Δ≈0、置换校准、无系统性假阳性

  C 时序     : 需要 h_{i,t−1}，盘上不存在 -> 标记为 PENDING（R2 阶段补采）
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

OUT = os.path.join(HERE, "synthetic")
os.makedirs(OUT, exist_ok=True)


def global_pca(X, dim, seed=0):
    """标签构造用的方向基（不是探针的一部分，可以用全量数据）。"""
    Xc = X - X.mean(0, keepdims=True)
    G = Xc.T @ Xc
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:dim]
    return V[:, idx], np.sqrt(np.clip(w[idx], 1e-12, None) / len(X))


def scale_to_snr(signal, snr, rng):
    """给定 signal，加噪使 var(signal)/var(noise) = snr。"""
    s = signal - signal.mean()
    sd_n = np.sqrt(max(s.var(), 1e-12) / max(snr, 1e-9))
    return s + rng.normal(0, sd_n, size=len(s))


def build(scenario, hi, hg, sid, rng, snr=2.0, dim=64):
    """返回 y。hi/hg 为原始 768 维。"""
    Ui, si = global_pca(hi, dim)
    Ug, sg = global_pca(hg, dim)
    Zi = (hi - hi.mean(0, keepdims=True)) @ Ui / si[None, :]
    Zg = (hg - hg.mean(0, keepdims=True)) @ Ug / sg[None, :]

    if scenario == "A_linear":
        w = rng.normal(size=dim)
        sig = Zi @ w
    elif scenario == "B_bilinear":
        r = 4
        U = rng.normal(size=(dim, r)); V = rng.normal(size=(dim, r))
        sig = ((Zi @ U) * (Zg @ V)).sum(1)
    elif scenario == "D_rank_only":
        # 状态级大信号（只由 h_g 决定）+ 状态内小信号（只由 h_i 决定）。
        # 噪声必须按 **候选级** 成分定标：否则 5 倍大的状态级成分会把噪声抬到
        # 淹没候选信号的水平，这个场景就退化成"信号根本不存在"，测不出任何
        # 关于 pooled-vs-ranking 选择准则的东西。
        v = rng.normal(size=dim); w = rng.normal(size=dim)
        state_part = Zg @ v
        cand_part = Zi @ w
        state_part = state_part / (state_part.std() + 1e-12)
        cand_part = cand_part - _state_mean(cand_part, sid)
        cand_part = cand_part / (cand_part.std() + 1e-12)
        noise = rng.normal(0, 1.0 / np.sqrt(max(snr, 1e-9)), size=len(Zi))
        return 5.0 * state_part + 1.0 * cand_part + noise
    elif scenario == "E_nonlinear":
        # 频率必须温和：第一版用 sin(Zi@u/sqrt(dim)*3)，自变量方差约 9，
        # 得到的是高频、在本样本量下**不可学**的函数——那测的是样本量，
        # 不是探针。改为把投影标准化到单位方差后再过温和非线性。
        u = rng.normal(size=dim); v = rng.normal(size=dim); w = rng.normal(size=dim)
        zi = Zi @ u; zi /= (zi.std() + 1e-12)
        zg = Zg @ v; zg /= (zg.std() + 1e-12)
        zw = Zi @ w; zw /= (zw.std() + 1e-12)
        sig = np.tanh(1.5 * zi) + 0.7 * (zw ** 2 - 1.0) + 0.5 * np.cos(zg)
    elif scenario == "F_null":
        return rng.normal(size=len(hi))
    else:
        raise KeyError(scenario)
    return scale_to_snr(sig, snr, rng)


def _state_mean(x, sid):
    _, groups = M.group_slices(sid)
    out = np.zeros_like(x)
    for g in groups:
        out[g] = x[g].mean()
    return out


PROBE_SETS = {
    "A_linear":    ["P0", "P2", "P4"],
    "B_bilinear":  ["P0", "P2", "P3", "P4", "P7"],
    "D_rank_only": ["P0", "P2", "P7", "P8", "P9"],
    "E_nonlinear": ["P0", "P4", "P5", "P6"],
    "F_null":      ["P0", "P2", "P4", "P7"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a3", "b3"])
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--scenarios", nargs="+",
                    default=["A_linear", "B_bilinear", "D_rank_only",
                             "E_nonlinear", "F_null"])
    ap.add_argument("--out", default=os.path.join(OUT, "synthetic_results.json"))
    args = ap.parse_args()

    print(f"[syn] loading {args.tags} layer {args.layer}", flush=True)
    d = RD.load_labels(args.tags, keys=["H_i", "H_g", "C1", "C2", "C3",
                                        "prompt_row", "step", "doc_id",
                                        "stratum", "A_pertok"])
    sp = RD.doc_splits(d, seed=0)
    RD.check_split_disjoint(d, sp)
    _, groups, _ = RD.state_groups(d["state_id"])
    hi = RD.h_i(d, args.layer)
    hg = RD.h_g(d, args.layer)
    sid_all = d["state_id"]

    prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups)
    sid = SC.split_sid(d, sp)
    doc = {k: d["doc_id"][v] for k, v in sp.items()}
    rng_ctl = np.random.default_rng(999)
    ctl = SC.make_controls(prep, sid, rng_ctl)

    results = {"config": vars(args), "scenarios": {}}
    for sc in args.scenarios:
        t0 = time.time()
        rng = np.random.default_rng(hash(sc) % (2 ** 31))
        y_all = build(sc, hi, hg, sid_all, rng, snr=args.snr)
        y = {k: y_all[v] for k, v in sp.items()}
        entry = {}
        for selkind in ("pooled_r2", "within_r2"):
            R = SC.run_probes(prep, y, sid, doc, selkind,
                              which=PROBE_SETS[sc],
                              seeds=tuple(range(args.seeds)),
                              epochs=args.epochs)
            entry[selkind] = SC.score_all(R, y["test"], sid["test"])
            print(f"  [{sc}/{selkind}] {time.time()-t0:.0f}s "
                  f"{len(entry[selkind])} probes", flush=True)
        # B 场景的关键对照：shuffled h_global 必须归零
        if sc == "B_bilinear":
            R = SC.run_probes(prep, y, sid, doc, "within_r2",
                              which=["P0", "P2"],
                              seeds=tuple(range(args.seeds)),
                              epochs=args.epochs,
                              hg_override=ctl["shuffle_hg"])
            entry["ctl_shuffle_hg"] = SC.score_all(R, y["test"], sid["test"])
            R = SC.run_probes(prep, y, sid, doc, "within_r2",
                              which=["P0", "P2"],
                              seeds=tuple(range(args.seeds)),
                              epochs=args.epochs,
                              hg_override=ctl["gauss_hg"])
            entry["ctl_gauss_hg"] = SC.score_all(R, y["test"], sid["test"])
        results["scenarios"][sc] = entry
        json.dump(results, open(args.out, "w"), indent=2, default=float)
        print(f"[syn] {sc} done in {time.time()-t0:.0f}s", flush=True)

    results["scenarios"]["C_temporal"] = {
        "status": "PENDING",
        "reason": "需要 h_{i,t-1}；Phase-0A 分片只存了当前时间步的隐藏态。"
                  "轨迹在给定种子下完全确定，可在 R2 阶段重走轨迹补采。"}
    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
