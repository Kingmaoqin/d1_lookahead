"""
非凸探针的加预算重测（回应 Auditor-A 的 F3）。

F3 实测证据：在 103 处保存的最优超参里，`lr` 取网格最大值 3e-3 的比例是
**74%** —— 模型要更大的学习率，而 `probes2._Runner` 只跑 400 步全批 AdamW。
因此 P2 双线性 / P3 FiLM / P4 关系型 MLP / P8 RankNet 的负结果**不能排除欠训**。

（P2 双线性另有一个凸的闭式等价物——Kronecker 岭回归，见
`relational_recheck.py`，它必然收敛且同样给 ≈0。但 FiLM 与关系型 MLP
没有凸等价物，只能靠加预算重测。）

用 `rlib/probes3.Runner3`：3000 步、lr 上探到 3e-2、wd 含 0.0、patience 500、
`calib="within"`（在 within-state 中心化空间里定标，这才是 within_r2 的
正确尺度）、listwise 目标改为 within-state 标准化。

每个探针都配一个**同容量的 cheap-only 对照**，并且先在合成 B / F 上验证
"加预算之后确实能读出植入信号、且在零假设上不假阳性"。
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

ARMS = {"MDLM_anc": ["a3", "b3"], "SEDD_anc": ["s1", "s2"],
        "FRESH_MDLM_anc": ["freshA"]}


def fit_family(prep, y, sid, criterion, seeds, epochs, lrs, wds, patience):
    sel = P2.make_selector(criterion, sid["val"])
    pca = prep["pca"]
    d_c = pca["cheap"]["train"].shape[1]
    d_i = pca["hi"]["train"].shape[1]
    d_g = pca["hg"]["train"].shape[1]
    F = {"c": pca["cheap"], "hi": pca["hi"], "hg": pca["hg"]}
    rel = {k: RD.relational_block(pca["hi"][k], pca["hg"][k])
           for k in ("train", "val", "test")}
    d_r = rel["train"].shape[1]
    out = {}

    def run(name, feats, make, loss, calib="pooled"):
        rn = P3.Runner3({k: feats[k]["train"] for k in feats}, y["train"],
                        sid["train"], {k: feats[k]["val"] for k in feats},
                        y["val"], sid["val"],
                        {k: feats[k]["test"] for k in feats},
                        loss_kind=loss, calib=calib)
        m = rn.run(make, sel, seeds=seeds, epochs=epochs, lrs=lrs, wds=wds,
                   patience=patience)
        if m is not None:
            out[name] = m
            print(f"    {name:26s} val={m['val_score']:+.4f} "
                  f"hp={m['hp']} best_ep={m['best_epoch']}", flush=True)

    # --- 低秩双线性（有凸等价物，作交叉验证用） ---
    for rk in (2, 4, 8, 16):
        run(f"bilinear_r{rk}", F,
            lambda rk=rk: P3_bilinear(d_i, d_g, rk, d_c), "mse")
    # --- FiLM（无凸等价物） ---
    run("film", F, lambda: P3_film(d_i, d_g, d_c), "mse")
    # --- 关系型 MLP（无凸等价物） + 同容量 cheap-only 对照 ---
    for w in (128, 512):
        run(f"relmlp_w{w}", {"x": rel, "c": pca["cheap"]},
            lambda w=w: P2.MLPProbe([d_r, d_c], (w,), keys=("x", "c")), "mse")
        run(f"cheaponly_mlp_w{w}", {"c": pca["cheap"]},
            lambda w=w: P2.MLPProbe([d_c], (w,), keys=("c",)), "mse")
    # --- 成对排序：候选 vs 候选+状态条件 ---
    run("rank_cand", F, lambda: P2.MLPProbe([d_i], (256,), keys=("hi",)),
        "pairwise", calib="within")
    run("rank_cand_state", F,
        lambda: P2.MLPProbe([d_i, d_g], (256,), keys=("hi", "hg")),
        "pairwise", calib="within")
    run("rank_cheap", F, lambda: P2.MLPProbe([d_c], (256,), keys=("c",)),
        "pairwise", calib="within")
    run("rank_cheap_hidden", F,
        lambda: P2.MLPProbe([d_c, d_i, d_g], (256,), keys=("c", "hi", "hg")),
        "pairwise", calib="within")
    return out


def P3_bilinear(d_i, d_g, rk, d_c):
    return P2.BilinearC(d_i, d_g, rk, d_c)


def P3_film(d_i, d_g, d_c):
    return P2.FiLM(d_i, d_g, hid=64, d_c=d_c)


def score(out, y_te, sid_te):
    _, g = M.group_slices(sid_te)
    r = {}
    for k, m in out.items():
        r[k] = M.full_report(y_te, m["pred_test"], sid_te, g)
        r[k]["hp"] = m["hp"]; r[k]["best_epoch"] = m["best_epoch"]
        r[k]["n_params"] = m["n_params"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="real")
    ap.add_argument("--arm", default="MDLM_anc")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--criterion", default="within_r2")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=500)
    ap.add_argument("--scenarios", nargs="+", default=["B_bilinear", "F_null"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        HERE, "results", f"nonlinear_retest_{args.mode}_{args.arm}.json")
    lrs = (3e-2, 1e-2, 3e-3)
    wds = (0.0, 1e-2)
    seeds = tuple(range(args.seeds))
    rep = {"config": vars(args), "lrs": list(lrs), "wds": list(wds),
           "results": {}}
    t0 = time.time()

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
        prep = SC.prepare(d, sp, 9, pca_dim=128, groups=groups)
        sid = SC.split_sid(d, sp)
        for sc in args.scenarios:
            print(f"\n=== synthetic {sc} ===", flush=True)
            rng = np.random.default_rng(hash(sc) % (2 ** 31))
            y_all = syn.build(sc, hi, hg, d["state_id"], rng, snr=2.0)
            y = {k: y_all[v] for k, v in sp.items()}
            o = fit_family(prep, y, sid, args.criterion, seeds, args.epochs,
                           lrs, wds, args.patience)
            rep["results"][sc] = score(o, y["test"], sid["test"])
            json.dump(rep, open(out, "w"), indent=2, default=float)
    else:
        d = RD.load_labels(ARMS[args.arm])
        sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        _, groups, _ = RD.state_groups(d["state_id"])
        sid = SC.split_sid(d, sp)
        y = SC.split_targets(d, sp, args.target)
        prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups)
        # cheap 岭回归基线（闭式）
        sel = P2.make_selector(args.criterion, sid["val"])
        mc = P2.fit_ridge(prep["raw"]["cheap"]["train"], y["train"],
                          prep["raw"]["cheap"]["val"], y["val"],
                          prep["raw"]["cheap"]["test"], sel)
        _, g_te = M.group_slices(sid["test"])
        base = M.full_report(y["test"], mc["pred_test"], sid["test"], g_te)
        rep["cheap_ridge_baseline"] = base
        print(f"[{args.arm} L{args.layer}] cheap ridge conc="
              f"{base['concordance']:.4f} within={base['within_r2']:+.4f}",
              flush=True)
        o = fit_family(prep, y, sid, args.criterion, seeds, args.epochs,
                       lrs, wds, args.patience)
        rep["results"][f"L{args.layer}"] = score(o, y["test"], sid["test"])
        json.dump(rep, open(out, "w"), indent=2, default=float)

    print(f"\ndone in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
