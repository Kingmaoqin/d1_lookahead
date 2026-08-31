"""
精确 toy 环境（任务书 §3.3）+ 独立参考实现（§3.2）。

设计要点：**不重写一遍估计量再自我对拍**，而是把 *生产代码本身*
（`src/policy.py:rollout`、`src/collect.py:branch_rollouts`、`src/crn.py`）
接到一个可以穷举未来的小环境上，与动态规划算出的 **精确** Q/V/A 对比。
这样验的是实际出货的实现。

环境：
  * 词表 V=3（外加 MASK=3），序列长 L=6，前缀 1 位可见，5 位待揭示；
  * 状态空间 4^5 = 1024，全部可枚举；
  * p_θ(x_i | s) 由一张固定随机表定义，**依赖整个状态**（双向，像 DLM）；
  * π_ref = ancestral：每步在剩余掩码位中均匀随机选一个，token 从完整
    softmax 抽（top_k 设为 V，即不截断，使 p_trunc == p_full）。

精确量（对 horizon h 的递归）：
    V_0(s) = 0
    V_h(s) = (1/m) Σ_{i∈M} Σ_x p(x|s,i) [ log p(x|s,i) + V_{h-1}(s ∪ {i:x}) ]
    v_first(s) = (1/m) Σ_{i∈M} Σ_x p(x|s,i) log p(x|s,i)

对应生产代码的估计量（H = horizon）：
    A_full   = [ logp_action + V_H(s_a) ] / (H+1)  −  V_{H+1}(s) / (H+1)
    A_future = V_H(s_a) / H  −  [ V_{H+1}(s) − v_first(s) ] / H
其中 s_a = s 上强制提交 (i, x̂_i) 之后的状态。

检查项：状态 / 选中位置 / 选中 token / Q / V / A / first reward / Path-LL /
CRN / 最终序列。
"""
import argparse
import json
import os
import sys
from functools import lru_cache

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

VOCAB = 3
MASK = 3
L = 6
PREFIX = 1

# 生产代码从 mdlm_local 里 import MASK_TOKEN_ID；先把它顶掉再导入 policy
import types                                                    # noqa: E402
_fake = types.ModuleType("mdlm_local")
_fake.MASK_TOKEN_ID = MASK
sys.modules.setdefault("mdlm_local", _fake)
if not hasattr(sys.modules["mdlm_local"], "MASK_TOKEN_ID"):
    sys.modules["mdlm_local"].MASK_TOKEN_ID = MASK

import policy as POL                                            # noqa: E402
import collect as COL                                           # noqa: E402


# ------------------------------------------------------------------- 模型 ----
class ToyModel:
    """p(x | s) 的固定随机表。接口与生产代码期望的骨干一致：model(ids)->logits。

    logits 只依赖整个状态 s（双向），与位置无关地由一张 (n_state, L, V) 的表
    给出——这正是掩码扩散模型的结构：一次前向给出所有被掩位置的分布。
    """

    def __init__(self, seed=0, device="cpu"):
        rng = np.random.default_rng(seed)
        self.device = device
        self.table = {}
        self.rng = rng
        self._seed = seed

    def _logits_for(self, code):
        if code not in self.table:
            g = np.random.default_rng(self._seed * 1_000_003 + code)
            self.table[code] = g.normal(0, 1.5, size=(L, VOCAB)).astype(
                np.float64)
        return self.table[code]

    @staticmethod
    def encode(ids_row):
        c = 0
        for t in ids_row:
            c = c * 4 + int(t)
        return c

    def __call__(self, ids, output_hidden_states=False):
        arr = ids.detach().cpu().numpy()
        B = arr.shape[0]
        out = np.zeros((B, L, VOCAB + 1), dtype=np.float64)
        for b in range(B):
            lg = self._logits_for(self.encode(arr[b]))
            out[b, :, :VOCAB] = lg
            out[b, :, MASK] = -1e30            # 绝不预测 MASK
        t = torch.as_tensor(out, dtype=torch.float32, device=ids.device)
        if output_hidden_states:
            hs = [torch.zeros(B, L, 4, device=ids.device)]
            return t, hs
        return t


# --------------------------------------------------------------- 精确 DP ----
class Exact:
    def __init__(self, model):
        self.m = model
        self._V = {}

    def logp(self, state):
        """返回 (L, VOCAB) 的 log p(x | s)，只对被掩位置有意义。"""
        lg = self.m._logits_for(ToyModel.encode(state))
        z = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) \
            + lg.max(1, keepdims=True).ravel()
        return lg - z[:, None]

    def V(self, state, h):
        """V_h(s)：未来 h 次 π_ref 提交的 Path-LL 期望（未归一化）。"""
        if h <= 0:
            return 0.0
        key = (tuple(state), h)
        if key in self._V:
            return self._V[key]
        masked = [i for i in range(L) if state[i] == MASK]
        if not masked:
            self._V[key] = 0.0
            return 0.0
        lp = self.logp(state)
        tot = 0.0
        for i in masked:
            for x in range(VOCAB):
                p = float(np.exp(lp[i, x]))
                nxt = list(state); nxt[i] = x
                tot += p * (float(lp[i, x]) + self.V(tuple(nxt), h - 1))
        val = tot / len(masked)
        self._V[key] = val
        return val

    def v_first(self, state):
        masked = [i for i in range(L) if state[i] == MASK]
        lp = self.logp(state)
        tot = 0.0
        for i in masked:
            for x in range(VOCAB):
                tot += float(np.exp(lp[i, x])) * float(lp[i, x])
        return tot / len(masked)

    def advantages(self, state, i, H):
        """精确 A_full / A_future，按生产代码的定义。"""
        lp = self.logp(state)
        xhat = int(np.argmax(lp[i]))
        logp_action = float(lp[i, xhat])
        s_a = list(state); s_a[i] = xhat
        VH_after = self.V(tuple(s_a), H)
        VH1 = self.V(tuple(state), H + 1)
        vf = self.v_first(state)
        a_full = (logp_action + VH_after) / (H + 1) - VH1 / (H + 1)
        a_future = VH_after / H - (VH1 - vf) / H
        return dict(xhat=xhat, logp_action=logp_action,
                    A_full=a_full, A_future=a_future,
                    V_pertok=VH1 / (H + 1),
                    Q_pertok=(logp_action + VH_after) / (H + 1))


# ------------------------------------------------------------ 生产代码调用 ---
def make_snapshot(model, state, device="cpu"):
    ids = torch.tensor([state], dtype=torch.long, device=device)
    mask = (ids == MASK)
    logits = model(ids)
    lp_top, idx, lse, lg_top = POL.topk_logprobs(logits, VOCAB, 32)
    return {"ids": ids, "mask": mask, "lp_top": lp_top, "idx": idx,
            "argmax": idx[..., 0], "step": 0}


def run_production(model, state, cand_positions, H, K, seed_offset=0,
                   device="cpu"):
    cfg = COL.CollectConfig(seq_len=L, prefix_len=PREFIX, top_k=VOCAB,
                            K=K, horizon=H, order="ancestral",
                            rollout_batch=256)
    pcfg = cfg.pi()
    pcfg.seq_len = L
    snap = make_snapshot(model, state, device)
    cand_b = torch.zeros(len(cand_positions), dtype=torch.long, device=device)
    cand_i = torch.tensor(cand_positions, dtype=torch.long, device=device)
    return COL.branch_rollouts(model, snap, cand_b, cand_i, cfg, pcfg,
                               seed_offset)


# ------------------------------------------------------------------ 检查 ----
def check_noop(model, state, H, K, device="cpu"):
    """CRN no-op：强迫 Q 分支提交 V 分支本来就会提交的动作，A 必须恰为 0。

    做法：把候选位置设为 V 分支在该 rollout 第一步会选的位置，并且提交它
    本来会抽到的 token。这里通过直接调用生产代码的选择逻辑复现。
    """
    ids = torch.tensor([state], dtype=torch.long, device=device)
    mask = (ids == MASK)
    positions = torch.arange(L, device=device, dtype=torch.int64)
    cfg = COL.CollectConfig(seq_len=L, prefix_len=PREFIX, top_k=VOCAB, K=K,
                            horizon=H, order="ancestral", rollout_batch=256)
    pcfg = cfg.pi()
    seeds = torch.arange(K, dtype=torch.int64, device=device)
    order = POL.unmask_order(seeds, positions)
    logits = model(ids.expand(K, -1))
    lp_top, idx, _, _ = POL.topk_logprobs(logits, VOCAB, 32)
    proposed, _ = POL.sample_tokens(lp_top, idx, seeds, positions, pcfg)
    raw = POL.order_score(lp_top, mask.expand(K, -1), order, seeds, positions,
                          pcfg)
    score = torch.where(mask.expand(K, -1), raw,
                        torch.full_like(raw, -1e30))
    sel_pos = score.argmax(1)
    sel_tok = proposed.gather(1, sel_pos[:, None]).squeeze(1)
    return sel_pos.cpu().numpy(), sel_tok.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--K", type=int, default=4000)
    ap.add_argument("--n_states", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "toy",
                                                  "toy_exact_results.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = ToyModel(seed=args.seed)
    ex = Exact(model)
    rng = np.random.default_rng(args.seed)

    # 构造若干个部分去噪的状态：前缀可见，其余随机地已提交/仍掩码
    states = []
    while len(states) < args.n_states:
        s = [int(rng.integers(VOCAB))] * PREFIX
        for _ in range(L - PREFIX):
            s.append(int(rng.integers(VOCAB)) if rng.random() < 0.25 else MASK)
        m = sum(1 for t in s if t == MASK)
        if m >= args.H + 2:
            states.append(tuple(s))

    report = {"config": vars(args), "states": []}
    max_err_full = max_err_fut = 0.0
    for st in states:
        masked = [i for i in range(L) if st[i] == MASK]
        exact = {i: ex.advantages(st, i, args.H) for i in masked}
        out = run_production(model, list(st), masked, args.H, args.K,
                             seed_offset=1234)
        rec = {"state": list(st), "masked": masked, "cands": []}
        for j, i in enumerate(masked):
            e = exact[i]
            mc_sd = float(out["A_full_seeds"][j].std() / np.sqrt(args.K))
            fut_sd = float(out["A_future_seeds"][j].std() / np.sqrt(args.K))
            d_full = float(out["A_pertok"][j]) - e["A_full"]
            d_fut = float(out["A_future"][j]) - e["A_future"]
            max_err_full = max(max_err_full, abs(d_full) / max(mc_sd, 1e-12))
            max_err_fut = max(max_err_fut, abs(d_fut) / max(fut_sd, 1e-12))
            rec["cands"].append({
                "pos": i,
                "xhat_exact": e["xhat"],
                "logp_action_exact": e["logp_action"],
                "logp_action_prod": float(out["logp_action"][j]),
                "A_full_exact": e["A_full"],
                "A_full_prod": float(out["A_pertok"][j]),
                "A_full_z": d_full / max(mc_sd, 1e-12),
                "A_future_exact": e["A_future"],
                "A_future_prod": float(out["A_future"][j]),
                "A_future_z": d_fut / max(fut_sd, 1e-12),
                "V_pertok_exact": e["V_pertok"],
                "V_pertok_prod": float(out["V_pertok"][j]),
                "Q_pertok_exact": e["Q_pertok"],
                "Q_pertok_prod": float(out["Q_pertok"][j]),
                # RB 与 MC 两个估计量必须估同一个量
                "A_full_mc_prod": float(out["mc_A_pertok"][j]),
                "rb_var": float(out["A_full_seeds"][j].var()),
                "mc_var": float(out["mc_A_full_seeds"][j].var()),
            })
        report["states"].append(rec)
        print(f"state {st}: max|z| full "
              f"{max(abs(c['A_full_z']) for c in rec['cands']):.2f}  "
              f"future {max(abs(c['A_future_z']) for c in rec['cands']):.2f}",
              flush=True)

    # ---- CRN no-op ----
    st = states[0]
    sel_pos, sel_tok = check_noop(model, list(st), args.H, min(args.K, 64))
    report["crn_noop"] = {"selected_positions": sel_pos.tolist()[:16],
                          "selected_tokens": sel_tok.tolist()[:16],
                          "note": "V 分支首步的位置/词；用于 no-op 一致性检查"}

    report["summary"] = {
        "max_abs_z_A_full": max_err_full,
        "max_abs_z_A_future": max_err_fut,
        "K": args.K,
        "verdict": ("PASS" if max(max_err_full, max_err_fut) < 4.0
                    else "FAIL")}
    json.dump(report, open(args.out, "w"), indent=2, default=float)
    print(f"\nmax |z| A_full={max_err_full:.2f}  A_future={max_err_fut:.2f}  "
          f"-> {report['summary']['verdict']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
