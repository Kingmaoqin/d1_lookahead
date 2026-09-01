"""
关系型假设的**修正版**检验（回应 Auditor-A 的 F1/F2）。

原实现的两个缺陷：
  F1  `[Zi, Zg, vec(Zi⊗Zg)]` 三部分塞进**同一个**隐藏块，共享一个 gamma。
      64 个主效应列与 1024 个交互列被同一强度收缩 —— 这是原项目 defect-4
      的复刻。实测 `kron_pca` 被迫选 gamma≈0.003–0.03（重度收缩），
      而同数据上 `additive_pca` 选 gamma=3.0，两者根本不在同一个惩罚点上，
      "kron < additive" 因此**不能**解释为"交互项无用"。
  F2  gamma 网格上界 3.0，而 `additive_pca` 在 6 处中有 5 处选中 3.0，
      **撞上界**。隐藏块想要更大权重却拿不到。

修正：`rlib/probes3.fit_ridge_blocks`，每块独立 gamma（上界 100），
并且**加性与 kron 从同一次搜索里取嵌套切片** —— 加性 = 交互块 gamma=0。
于是 kron 在验证集上按构造 ≥ 加性，任何 test 上的劣势都只能来自过拟合，
而不是惩罚错配。

先在**合成 B（植入双线性信号）**上验证修复确实能把信号读出来，
再上真实数据。合成 F（零假设）必须仍然归零。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P2           # noqa: E402
from rlib import probes3 as P3           # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

ARMS = {"MDLM_anc": ["a3", "b3"], "MDLM_conf": ["c3", "d3"],
        "SEDD_anc": ["s1", "s2"], "FRESH_MDLM_anc": ["freshA"]}

SLICES = {"cheap_only": [True, False, False],
          "additive": [True, True, False],
          "kron": [True, True, True]}


def build(prep, kron_d):
    Zi = {k: prep["pca"]["hi"][k][:, :kron_d] for k in ("train", "val", "test")}
    Zg = {k: prep["pca"]["hg"][k][:, :kron_d] for k in ("train", "val", "test")}
    main = {k: np.concatenate([Zi[k], Zg[k]], 1) for k in Zi}
    inter = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
                 ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
    return prep["pca"]["cheap"], main, inter


def run_one(prep, y, sid, kron_d, criterion, gm, gi):
    cheap, main, inter = build(prep, kron_d)
    sel = P2.make_selector(criterion, sid["val"])
    res = P3.fit_ridge_blocks(
        [cheap["train"], main["train"], inter["train"]], y["train"],
        [cheap["val"], main["val"], inter["val"]], y["val"],
        [cheap["test"], main["test"], inter["test"]], sel,
        gamma_grids=[[1.0], gm, gi], record_slices=SLICES)
    _, g_te = M.group_slices(sid["test"])
    out = {}
    for name, m in res.items():
        if m is None:
            continue
        out[name] = M.full_report(y["test"], m["pred_test"], sid["test"], g_te)
        out[name]["hp"] = m["hp"]
        out[name]["val_score"] = m["val_score"]
        out[name]["n_params"] = m["n_params"]
    return out


def deltas(o):
    d = {}
    if "kron" in o and "additive" in o:
        d["relational"] = {k: o["kron"][k] - o["additive"][k]
                           for k in ("concordance", "within_r2", "top1",
                                     "regret_norm_mean")}
    if "additive" in o and "cheap_only" in o:
        d["hidden"] = {k: o["additive"][k] - o["cheap_only"][k]
                       for k in ("concordance", "within_r2", "top1",
                                 "regret_norm_mean")}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="real")
    ap.add_argument("--arm", default="MDLM_anc")
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 8, 9])
    ap.add_argument("--kron_d", type=int, default=32)
    ap.add_argument("--criterion", default="within_r2")
    ap.add_argument("--hg_kinds", nargs="+", default=["hg", "hgm"])
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--scenarios", nargs="+",
                    default=["B_bilinear", "F_null", "A_linear"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        HERE, "results", f"relational_recheck_{args.mode}_{args.arm}.json")
    t0 = time.time()
    # 主效应块的网格必须含 0，否则 "cheap_only" 这个嵌套切片不可达
    gm = [0.0] + list(P3.GAMMAS_MAIN)
    gi = list(P3.GAMMAS_INT)
    rep = {"config": vars(args), "gamma_main": gm, "gamma_int": gi,
           "results": {}}

    if args.mode == "synthetic":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "syn", os.path.join(HERE, "synthetic_tests.py"))
        syn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(syn)
        d = RD.load_labels(["a3", "b3"],
                           keys=["H_i", "H_g", "C1", "C2", "C3", "prompt_row",
                                 "step", "doc_id", "A_pertok"])
        sp = RD.doc_splits(d, seed=0)
        _, groups, _ = RD.state_groups(d["state_id"])
        hi, hg = RD.h_i(d, 9), RD.h_g(d, 9)
        prep = SC.prepare(d, sp, 9, pca_dim=args.pca_dim, groups=groups)
        sid = SC.split_sid(d, sp)
        for sc in args.scenarios:
            rng = np.random.default_rng(hash(sc) % (2 ** 31))
            y_all = syn.build(sc, hi, hg, d["state_id"], rng, snr=2.0)
            y = {k: y_all[v] for k, v in sp.items()}
            o = run_one(prep, y, sid, args.kron_d, args.criterion, gm, gi)
            rep["results"][sc] = {"scores": o, "deltas": deltas(o)}
            dd = deltas(o)
            print(f"[syn {sc}] cheap {o['cheap_only']['concordance']:.4f} | "
                  f"additive {o['additive']['concordance']:.4f} "
                  f"(within {o['additive']['within_r2']:+.4f}, "
                  f"γ={o['additive']['hp']['gammas']}) | "
                  f"kron {o['kron']['concordance']:.4f} "
                  f"(within {o['kron']['within_r2']:+.4f}, "
                  f"γ={o['kron']['hp']['gammas']})  ->  "
                  f"RELATIONAL Δconc {dd['relational']['concordance']:+.4f} "
                  f"Δwithin {dd['relational']['within_r2']:+.4f}", flush=True)
            json.dump(rep, open(out, "w"), indent=2, default=float)
    else:
        d = RD.load_labels(ARMS[args.arm])
        sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        _, groups, _ = RD.state_groups(d["state_id"])
        sid = SC.split_sid(d, sp)
        y = SC.split_targets(d, sp, args.target)
        for hk in args.hg_kinds:
            for L in args.layers:
                prep = SC.prepare(d, sp, L, pca_dim=args.pca_dim,
                                  hg_kind=hk, groups=groups)
                o = run_one(prep, y, sid, args.kron_d, args.criterion, gm, gi)
                key = f"{hk}_L{L}"
                rep["results"][key] = {"scores": o, "deltas": deltas(o)}
                dd = deltas(o)
                print(f"[{args.arm} {key}] cheap "
                      f"{o['cheap_only']['concordance']:.4f} | additive "
                      f"{o['additive']['concordance']:.4f} "
                      f"(γ={o['additive']['hp']['gammas']}) | kron "
                      f"{o['kron']['concordance']:.4f} "
                      f"(γ={o['kron']['hp']['gammas']})", flush=True)
                print(f"    hidden Δconc {dd['hidden']['concordance']:+.4f} "
                      f"Δtop1 {dd['hidden']['top1']:+.4f}   |   "
                      f"RELATIONAL Δconc "
                      f"{dd['relational']['concordance']:+.4f} "
                      f"Δtop1 {dd['relational']['top1']:+.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(rep, open(out, "w"), indent=2, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
