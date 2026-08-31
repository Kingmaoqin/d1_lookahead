"""
Phase R1 —— 在**现有数据**上做 P0–P13 的 broad exploratory screen。

三段式，控制算力：
  A. 层扫描  : 13 层 × {便宜探针族} ，只看 **验证集** 分数，按每种选择准则
               各挑一个层。旧实验只按 pooled R² 挑层（S1）。
  B. 全族    : 在选中的层上跑 P0/P1/P2k/P2/P3/P4/P5/P6/P7/P7k/P8/P9/P13。
  C. 证伪    : shuffled h_g / Gaussian h_g / Gaussian h_i / within-state 标签置换。

一切选择只用 val；test 只在最后评一次。多划分种子（默认 3）。
所有结果（含失败的、难看的）写入 EXPERIMENT_REGISTRY.csv。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

ARMS = {
    "MDLM_anc": ["a3", "b3"],
    "MDLM_conf": ["c3", "d3"],
    "SEDD_anc": ["s1", "s2"],
    "MDLM_H8": ["h8a", "h8b"],
    "MDLM_H32": ["h32a", "h32b"],
}
SEEDKEY = {"A_pertok": "A_full_seeds", "A_future": "A_future_seeds"}


def layer_sweep(d, sp, sid, doc, target, criteria, layers, groups,
                pca_dim, kron_d, log=print):
    """A 段：只用验证集分数，为每种准则挑层。"""
    y = SC.split_targets(d, sp, target)
    val = {c: {} for c in criteria}
    for L in layers:
        prep = SC.prepare(d, sp, L, pca_dim=pca_dim, groups=groups)
        Zi = {k: prep["pca"]["hi"][k][:, :kron_d] for k in
              ("train", "val", "test")}
        Zg = {k: prep["pca"]["hg"][k][:, :kron_d] for k in
              ("train", "val", "test")}
        KR = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
                  ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
        Xk = {k: np.concatenate([Zi[k], Zg[k], KR[k]], 1) for k in Zi}
        Xa = {k: np.concatenate([prep["raw"]["hi"][k], prep["raw"]["hg"][k]], 1)
              for k in ("train", "val", "test")}
        for c in criteria:
            sel = P.make_selector(c, sid["val"])
            m_add = P.fit_ridge_2block(
                prep["raw"]["cheap"]["train"], Xa["train"], y["train"],
                prep["raw"]["cheap"]["val"], Xa["val"], y["val"],
                prep["raw"]["cheap"]["test"], Xa["test"], sel)
            m_kr = P.fit_ridge_2block(
                prep["pca"]["cheap"]["train"], Xk["train"], y["train"],
                prep["pca"]["cheap"]["val"], Xk["val"], y["val"],
                prep["pca"]["cheap"]["test"], Xk["test"], sel)
            val[c][L] = {"additive": m_add["val_score"],
                         "kron": m_kr["val_score"]}
        log(f"    layer {L:2d} " + "  ".join(
            f"{c}: add {val[c][L]['additive']:+.4f} kron "
            f"{val[c][L]['kron']:+.4f}" for c in criteria), flush=True)
    best = {}
    for c in criteria:
        best[c] = {
            "additive": int(max(layers, key=lambda L: val[c][L]["additive"])),
            "kron": int(max(layers, key=lambda L: val[c][L]["kron"]))}
    return best, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="MDLM_anc", choices=list(ARMS))
    ap.add_argument("--targets", nargs="+", default=["A_pertok", "A_future"])
    ap.add_argument("--criteria", nargs="+",
                    default=["pooled_r2", "within_r2", "concordance"])
    ap.add_argument("--split_seeds", type=int, default=3)
    ap.add_argument("--probe_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--kron_dims", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--hg_kind", default="hg", choices=["hg", "hgm", "both"])
    ap.add_argument("--which", nargs="+", default=None)
    ap.add_argument("--skip_controls", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tags = ARMS[args.arm]
    out = args.out or os.path.join(HERE, "results",
                                   f"R1_{args.arm}_{args.hg_kind}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t00 = time.time()

    print(f"[R1] arm={args.arm} tags={tags} hg={args.hg_kind}", flush=True)
    d = RD.load_labels(tags)
    _, groups, _ = RD.state_groups(d["state_id"])
    layers = args.layers or list(range(d["n_layers"]))
    report = {"config": vars(args), "arm": args.arm, "tags": tags,
              "n_examples": int(len(d["A_pertok"])),
              "n_states": int(len(np.unique(d["state_id"]))),
              "n_docs": int(len(np.unique(d["doc_id"]))),
              "results": {}}

    # 噪声天花板（池化与 within-state 两种）
    for tgt in args.targets:
        sk = SEEDKEY.get(tgt)
        if sk and sk in d:
            report[f"ceiling_{tgt}"] = RD.noise_ceiling(d[sk])
            report[f"within_ceiling_{tgt}"] = RD.within_state_noise_ceiling(
                d[sk], d["state_id"], groups)

    rows = []
    for tgt in args.targets:
        report["results"][tgt] = {}
        for s in range(args.split_seeds):
            sp = RD.doc_splits(d, seed=s)
            RD.check_split_disjoint(d, sp)
            sid = SC.split_sid(d, sp)
            doc = {k: d["doc_id"][v] for k, v in sp.items()}
            y = SC.split_targets(d, sp, tgt)
            print(f"\n[R1] === target={tgt} split_seed={s} "
                  f"({time.time()-t00:.0f}s) ===", flush=True)

            print("  [A] layer sweep", flush=True)
            best, valgrid = layer_sweep(
                d, sp, sid, doc, tgt, args.criteria, layers, groups,
                args.pca_dim, args.kron_dims[-1])
            print(f"  [A] chosen layers: {best}", flush=True)

            ent = {"best_layers": best, "layer_val_grid": valgrid,
                   "by_criterion": {}}
            for c in args.criteria:
                L = best[c]["kron"]
                prep = SC.prepare(d, sp, L, pca_dim=args.pca_dim,
                                  hg_kind=args.hg_kind, groups=groups)
                print(f"  [B] criterion={c} layer={L}", flush=True)
                R = SC.run_probes(prep, y, sid, doc, c, which=args.which,
                                  seeds=tuple(range(args.probe_seeds)),
                                  epochs=args.epochs,
                                  kron_dims=tuple(args.kron_dims))
                sc = SC.score_all(
                    R, y["test"], sid["test"],
                    ceiling=report.get(f"ceiling_{tgt}", {}).get("ceiling"),
                    within_ceiling=report.get(f"within_ceiling_{tgt}",
                                              {}).get("ceiling"))
                ent["by_criterion"][c] = {"layer": L, "probes": sc}
                for k, v in sc.items():
                    rows.append(dict(arm=args.arm, target=tgt, split_seed=s,
                                     criterion=c, layer=L, probe=k,
                                     hg_kind=args.hg_kind,
                                     **{kk: v.get(kk) for kk in
                                        ("r2", "within_r2", "concordance",
                                         "top1", "regret_norm_mean",
                                         "kendall_tau", "val_score",
                                         "n_params")}))
                print(f"  [B] {len(sc)} probes scored "
                      f"({time.time()-t00:.0f}s)", flush=True)

            # ---- C 段：证伪对照（只在主准则、主层上做一次） ----
            if not args.skip_controls and s == 0:
                cmain = "within_r2" if "within_r2" in args.criteria \
                    else args.criteria[0]
                L = best[cmain]["kron"]
                prep = SC.prepare(d, sp, L, pca_dim=args.pca_dim,
                                  hg_kind=args.hg_kind, groups=groups)
                rng = np.random.default_rng(20260831)
                ctl = SC.make_controls(prep, sid, rng)
                ent["controls"] = {}
                sub = ["P0", "P2k"]
                for cname in ("shuffle_hg", "gauss_hg", "gauss_hi"):
                    ov = ctl[cname]
                    if cname == "gauss_hi":
                        # gauss_hi 要替换的是 hi，不是 hg：直接改 prep 的副本
                        prep2 = {"raw": dict(prep["raw"]),
                                 "pca": dict(prep["pca"]),
                                 "idx": prep["idx"]}
                        prep2["raw"]["hi"] = ov["raw"]
                        prep2["pca"]["hi"] = ov["pca"]
                        R = SC.run_probes(prep2, y, sid, doc, cmain,
                                          which=sub,
                                          seeds=tuple(range(args.probe_seeds)),
                                          epochs=args.epochs,
                                          kron_dims=tuple(args.kron_dims))
                    else:
                        R = SC.run_probes(prep, y, sid, doc, cmain,
                                          which=sub,
                                          seeds=tuple(range(args.probe_seeds)),
                                          epochs=args.epochs,
                                          kron_dims=tuple(args.kron_dims),
                                          hg_override=ov)
                    ent["controls"][cname] = SC.score_all(
                        R, y["test"], sid["test"])
                    print(f"  [C] control {cname} done "
                          f"({time.time()-t00:.0f}s)", flush=True)
                # 标签置换（within-state）
                yp = SC.permute_labels_within_state(y, sid, rng)
                R = SC.run_probes(prep, yp, sid, doc, cmain, which=sub,
                                  seeds=tuple(range(args.probe_seeds)),
                                  epochs=args.epochs,
                                  kron_dims=tuple(args.kron_dims))
                ent["controls"]["label_perm_within"] = SC.score_all(
                    R, yp["test"], sid["test"])
                print(f"  [C] label permutation done "
                      f"({time.time()-t00:.0f}s)", flush=True)

            report["results"][tgt][f"seed{s}"] = ent
            json.dump(report, open(out, "w"), indent=2, default=float)

    # ---- 登记簿 ----
    reg = os.path.join(HERE, "EXPERIMENT_REGISTRY.csv")
    new = not os.path.exists(reg)
    with open(reg, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["date",
                                                                "phase",
                                                                "output_path"])
        if new:
            w.writeheader()
        for r in rows:
            r = dict(r); r["date"] = time.strftime("%Y-%m-%d")
            r["phase"] = "R1_exploratory"; r["output_path"] = out
            w.writerow(r)
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\n[R1] done in {time.time()-t00:.0f}s -> {out}")
    print(f"[R1] registry rows appended: {len(rows)} -> {reg}")


if __name__ == "__main__":
    main()
