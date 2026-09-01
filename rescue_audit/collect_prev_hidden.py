"""
补采上一时间步的隐藏态 h_{i,t-1} / h_{g,t-1}（任务书 §6 P11、§10 合成 C）。

为什么以前做不了：Phase-0A 的分片只存了**当前**时间步的隐藏态，
所以"表征是否编码时序差分 h_{i,t} − h_{i,t-1}"这个假设**从未被检验过**。

为什么现在能做：π_ref 的轨迹在给定种子下**完全确定**（order key 与 token
Gumbel 都由 (seed, 位置, token id) 散列而来，不含状态）。因此可以原样重走
一遍轨迹，只做前向、不做任何 rollout，把上一步的隐藏态截下来。
成本约为原采集的 1/50。

**自校验（必须通过，否则本文件产出的一切作废）**：重走时同样截下**当前**步的
h_i / h_g，与盘上已存的 `H_i` / `H_g` 逐元素比对。若轨迹没有精确重现，
两者不会相等。这一条同时验证了"轨迹确定性"这个前提本身。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from mdlm_local import load_mdlm                                   # noqa: E402
from sedd_local import load_sedd                                   # noqa: E402
import data as datamod                                             # noqa: E402
import collect as C                                                # noqa: E402
from policy import (forward_raw, topk_logprobs, unmask_order,      # noqa: E402
                    sample_tokens, make_initial_state, order_score)

from rlib import rdata as RD                                       # noqa: E402

ARMS = {"a3": dict(offset=0, backbone="mdlm", order="ancestral", order_temp=1.0),
        "b3": dict(offset=200, backbone="mdlm", order="ancestral", order_temp=1.0),
        "s1": dict(offset=0, backbone="sedd", order="ancestral", order_temp=1.0),
        "s2": dict(offset=200, backbone="sedd", order="ancestral", order_temp=1.0),
        "freshA": dict(offset=400, backbone="mdlm", order="ancestral",
                       order_temp=1.0)}


@torch.no_grad()
def rewalk(model, windows, doc_ids, idx, cfg, pcfg, dev, seed_offset,
           want_steps, want_pos):
    """重走一批 prompt 的轨迹，在记录点截取 t 与 t-1 的隐藏态。

    want_steps: set[int] 需要记录的 step
    want_pos:   dict[(prompt_row, step)] -> np.array 候选位置
    返回 list[dict]
    """
    w = torch.as_tensor(windows[idx], device=dev)
    ids, mask = make_initial_state(w, pcfg, dev)
    B, L = ids.shape
    n_steps = L - cfg.prefix_len
    positions = torch.arange(L, device=dev, dtype=torch.int64)
    traj_seeds = torch.arange(B, device=dev, dtype=torch.int64) + seed_offset
    order = unmask_order(traj_seeds, positions)

    prev_h = None                      # 上一步的 hidden（list of (B,L,D)）
    rows = []
    for step in range(n_steps):
        need = step in want_steps
        logits, hs = forward_raw(model, ids, output_hidden_states=True)
        cur_h = [hs[l] for l in cfg.store_layers]
        lp_top, tk_idx, lse, _ = topk_logprobs(logits, cfg.top_k, 32)

        if need and prev_h is not None:
            for bi in range(B):
                key = (int(idx[bi]), step)
                pos = want_pos.get(key)
                if pos is None or len(pos) == 0:
                    continue
                pt = torch.as_tensor(pos, device=dev, dtype=torch.long)
                hi_t = np.stack([cur_h[l][bi][pt].float().cpu().numpy()
                                 for l in range(len(cur_h))], 1)
                hi_p = np.stack([prev_h[l][bi][pt].float().cpu().numpy()
                                 for l in range(len(prev_h))], 1)
                hg_t = np.stack([cur_h[l][bi].mean(0).float().cpu().numpy()
                                 for l in range(len(cur_h))], 0)
                hg_p = np.stack([prev_h[l][bi].mean(0).float().cpu().numpy()
                                 for l in range(len(prev_h))], 0)
                rows.append(dict(
                    prompt_row=np.full(len(pos), int(idx[bi]), np.int64),
                    step=np.full(len(pos), step, np.int32),
                    position=np.asarray(pos, np.int64),
                    H_i_now=hi_t.astype(np.float16),
                    H_i_prev=hi_p.astype(np.float16),
                    H_g_now=np.repeat(hg_t[None], len(pos), 0).astype(np.float16),
                    H_g_prev=np.repeat(hg_p[None], len(pos), 0).astype(np.float16)))

        # 推进一步 π_ref —— 与 scripts/collect_labels.py 逐行一致
        proposed, _ = sample_tokens(lp_top, tk_idx, traj_seeds, positions, pcfg)
        raw = order_score(lp_top, mask, order, traj_seeds, positions, pcfg)
        score = torch.where(mask, raw, torch.full_like(raw, -1e30))
        sel = torch.zeros_like(mask)
        sel.scatter_(1, score.argmax(1, keepdim=True), True)
        sel &= mask
        ids = torch.where(sel, proposed, ids)
        mask = mask & ~sel
        prev_h = cur_h
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a3", choices=list(ARMS))
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--traj_batch", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_tag", default=None)
    args = ap.parse_args()
    spec = ARMS[args.tag]
    out_tag = args.out_tag or f"prev_{args.tag}"
    outdir = os.path.join(ROOT, "data", f"labels_{out_tag}")
    os.makedirs(outdir, exist_ok=True)

    # 已存分片里记录了哪些 (prompt_row, step, position)
    d = RD.load_labels([args.tag], keys=["prompt_row", "step", "position",
                                         "H_i", "H_g"])
    want_pos = {}
    for pr, st, po in zip(d["prompt_row"], d["step"], d["position"]):
        want_pos.setdefault((int(pr), int(st)), []).append(int(po))
    want_pos = {k: np.array(v) for k, v in want_pos.items()}
    want_steps = {k[1] for k in want_pos}
    print(f"[prev] {args.tag}: {len(want_pos)} states, steps={sorted(want_steps)}",
          flush=True)

    cfg = C.CollectConfig(order=spec["order"], order_temp=spec["order_temp"],
                          traj_batch=args.traj_batch)
    pcfg = cfg.pi()
    dev = args.device
    model, _ = (load_sedd(device=dev) if spec["backbone"] == "sedd"
                else load_mdlm(device=dev))
    windows, doc_ids = datamod.get_windows(seq_len=cfg.seq_len, n_docs=3000)
    splits = datamod.doc_level_split(doc_ids)
    order = np.concatenate([splits["train"], splits["val"], splits["test"]])
    order = order[spec["offset"]:spec["offset"] + args.n_prompts]

    t0 = time.time()
    allrows = []
    for b0 in range(0, len(order), cfg.traj_batch):
        idx = order[b0:b0 + cfg.traj_batch]
        seed_offset = cfg.seed_base + spec["offset"] * 131 + b0
        allrows += rewalk(model, windows, doc_ids, idx, cfg, pcfg, dev,
                          seed_offset, want_steps, want_pos)
        done = b0 + len(idx)
        el = time.time() - t0
        print(f"  prompts {done}/{len(order)}  {el:.0f}s  "
              f"eta {el / done * (len(order) - done):.0f}s", flush=True)

    out = {k: np.concatenate([r[k] for r in allrows], 0) for k in allrows[0]}
    # ---- 自校验：重走得到的 H_i_now 必须与盘上已存的 H_i 一致 ----
    key_new = list(zip(out["prompt_row"].tolist(), out["step"].tolist(),
                       out["position"].tolist()))
    key_old = {(int(a), int(b), int(c)): i for i, (a, b, c) in
               enumerate(zip(d["prompt_row"], d["step"], d["position"]))}
    matched = [(i, key_old[k]) for i, k in enumerate(key_new) if k in key_old]
    ii = np.array([a for a, _ in matched]); jj = np.array([b for _, b in matched])
    dif_i = np.abs(out["H_i_now"][ii].astype(np.float32)
                   - d["H_i"][jj].astype(np.float32))
    dif_g = np.abs(out["H_g_now"][ii].astype(np.float32)
                   - d["H_g"][jj].astype(np.float32))
    check = {"n_rows_new": int(len(key_new)), "n_matched": int(len(matched)),
             "max_abs_diff_H_i": float(dif_i.max()),
             "max_abs_diff_H_g": float(dif_g.max()),
             "frac_exact_H_i": float((dif_i == 0).mean()),
             "verdict": "PASS" if float(dif_i.max()) < 1e-2 else "FAIL"}
    print(f"\n[prev] 自校验：匹配 {check['n_matched']}/{check['n_rows_new']} 行；"
          f"max|ΔH_i| = {check['max_abs_diff_H_i']:.3g}，"
          f"max|ΔH_g| = {check['max_abs_diff_H_g']:.3g}  -> "
          f"{check['verdict']}", flush=True)
    if check["verdict"] != "PASS":
        print("[prev] 轨迹未能精确重现，产出不可用。", flush=True)

    np.savez_compressed(os.path.join(outdir, "shard_000.npz"), **out)
    json.dump({"tag": args.tag, "out_tag": out_tag, "spec": spec,
               "n_prompts": args.n_prompts, "selfcheck": check,
               "seconds": time.time() - t0},
              open(os.path.join(outdir, "meta.json"), "w"), indent=2)
    print(f"[prev] wrote {outdir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
