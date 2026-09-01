"""
Auditor-A 修正探针库（新增文件，不覆盖 probes2.py）。

修正三件事：
  F1  **分块惩罚**。probes2.fit_ridge_2block 把 [Zi, Zg, vec(Zi⊗Zg)] 塞进
      *同一个* 隐藏块，共享一个 gamma / 一个 alpha。64 个主效应列与 1024 个
      交互列被同一强度收缩，主效应想要的弱惩罚与交互想要的强惩罚无法同时
      满足 —— 这正是原项目 defect-4 的复刻。这里提供
      `fit_ridge_blocks`：每块一个独立的 gamma 网格，惩罚强度 = alpha/gamma^2。
      并且 **加性基线与 kron 模型来自同一次搜索**：gamma_int = 0 的切片就是
      加性模型，因此 kron 严格嵌套加性，val 上不可能比它差。
  F2  **gamma 网格上界**。probes2 的 gammas 上界 3.0；真实数据上 additive_pca
      选中的就是 3.0（撞到边界）。这里默认上界 100。
  F3  **torch 训练预算**。probes2._Runner 是 400 步全批 AdamW，且最优超参
      总落在网格边界 (lr=3e-3 最大, wd=1e-4 最小)。`Runner3` 默认 4000 步、
      更宽的 lr/wd 网格、可选记录 loss 曲线、可选 within-state 标定。

约定与 probes2 一致：返回 dict 含 pred_test / pred_val / val_score / hp。
"""
import numpy as np
import torch

from . import metrics as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"

ALPHAS_DEF = np.logspace(-3, 9, 37)
GAMMAS_MAIN = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0]
GAMMAS_INT = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _safe(v):
    return -1e18 if (v is None or not np.isfinite(v)) else float(v)


def _std_block(Xtr, *rest):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return [((X - mu) / sd).astype(np.float32) for X in (Xtr,) + rest]


def _ridge_path_gpu(Xg, yg, alphas):
    """Xg (n,p) float32 GPU, yg centered. 返回 list[w]（float32 GPU）。"""
    G = (Xg.T @ Xg).double()
    b = (Xg.T @ yg).double()
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    Vtb = V.T @ b
    return [(V @ (Vtb / (lam + float(a)))).float() for a in alphas]


def fit_ridge_blocks(blocks_tr, ytr, blocks_va, yva, blocks_te, selector,
                     gamma_grids, alphas=None, record_slices=None):
    """任意块数的岭回归，**每块独立缩放** → 每块独立惩罚。

    blocks_*: list of (n, p_b) arrays，第 0 块固定 gamma=1（参照块）。
    gamma_grids: list，长度与块数相同；第 0 块通常写 [1.0]。
    惩罚等价性：X = [g_0 B_0, g_1 B_1, ...]，罚 alpha*||w||^2
                ⇔ 对原尺度系数 u_b = g_b w_b 罚 (alpha/g_b^2)||u_b||^2。
    record_slices: dict name -> 一个 bool 列表，指示该子模型允许哪些块非零
        （用来在同一次搜索里取出嵌套子模型，例如 additive = 交互块 gamma=0）。
        返回 dict name -> best-model。
    """
    if alphas is None:
        alphas = ALPHAS_DEF
    nb = len(blocks_tr)
    Bt, Bv, Bs = [], [], []
    for b in range(nb):
        t, v, s = _std_block(blocks_tr[b], blocks_va[b], blocks_te[b])
        Bt.append(torch.as_tensor(t, device=DEV))
        Bv.append(torch.as_tensor(v, device=DEV))
        Bs.append(torch.as_tensor(s, device=DEV))
    ym = float(np.mean(ytr))
    yg = torch.as_tensor((np.asarray(ytr) - ym).astype(np.float32), device=DEV)

    if record_slices is None:
        record_slices = {"full": [True] * nb}
    best = {k: None for k in record_slices}

    import itertools
    for combo in itertools.product(*gamma_grids):
        active = [g > 0 for g in combo]
        if not any(active):
            continue
        cols = [b for b in range(nb) if active[b]]
        Xt = torch.cat([combo[b] * Bt[b] for b in cols], 1)
        Xv = torch.cat([combo[b] * Bv[b] for b in cols], 1)
        Xs = torch.cat([combo[b] * Bs[b] for b in cols], 1)
        ws = _ridge_path_gpu(Xt, yg, alphas)
        for w, a in zip(ws, alphas):
            pv = (Xv @ w + ym).cpu().numpy().astype(np.float64)
            sc = _safe(selector(yva, pv))
            for name, allow in record_slices.items():
                # 该 combo 是否落在这个子模型的可行域内
                if any(active[b] and not allow[b] for b in range(nb)):
                    continue
                cur = best[name]
                if cur is None or sc > cur["val_score"]:
                    best[name] = {
                        "val_score": sc, "pred_val": pv,
                        "pred_test": (Xs @ w + ym).cpu().numpy().astype(np.float64),
                        "hp": {"alpha": float(a),
                               "gammas": [float(g) for g in combo]},
                        "n_params": int(Xt.shape[1] + 1)}
    return best


# ------------------------------------------------------------------ torch ----
class Runner3:
    """probes2._Runner 的加预算 / 加诊断版本。"""

    def __init__(self, feats_tr, y_tr, sid_tr, feats_va, y_va, sid_va,
                 feats_te, loss_kind="mse", tau=1.0, calib="pooled"):
        self.loss_kind = loss_kind
        self.tau = tau
        self.calib_kind = calib
        to = lambda D: {k: torch.as_tensor(v, dtype=torch.float32, device=DEV)
                        for k, v in D.items()}
        self.ftr, self.fva, self.fte = to(feats_tr), to(feats_va), to(feats_te)
        self.ym = float(np.mean(y_tr)); self.ys = float(np.std(y_tr)) or 1.0
        self.ytr = torch.as_tensor((y_tr - self.ym) / self.ys,
                                   dtype=torch.float32, device=DEV)
        self.y_va = np.asarray(y_va, np.float64)
        self.sid_va = sid_va
        self.gidx_tr = self._group_tensor(sid_tr)
        _, self.gva = M.group_slices(sid_va)
        if loss_kind == "mse_wc":
            yc = self.ytr.clone(); G = self.gidx_tr
            gv = yc[G]; yc[G] = gv - gv.mean(1, keepdim=True); self.ytr = yc

    @staticmethod
    def _group_tensor(sid):
        uniq, groups = M.group_slices(sid)
        if len({len(g) for g in groups}) != 1:
            raise ValueError("ragged groups")
        return torch.as_tensor(np.stack(groups), dtype=torch.long, device=DEV)

    def _loss(self, s):
        if self.loss_kind in ("mse", "mse_wc"):
            return ((s - self.ytr) ** 2).mean()
        G = self.gidx_tr
        sg, yg = s[G], self.ytr[G]
        if self.loss_kind == "pairwise":
            ds = sg[:, :, None] - sg[:, None, :]
            dy = yg[:, :, None] - yg[:, None, :]
            iu = torch.triu_indices(sg.shape[1], sg.shape[1], offset=1)
            ds = ds[:, iu[0], iu[1]]; dy = dy[:, iu[0], iu[1]]
            m = dy.abs() > 1e-9
            if m.sum() == 0:
                return (s * 0).sum()
            return torch.nn.functional.softplus(-(ds[m] * torch.sign(dy[m]))).mean()
        if self.loss_kind == "listwise":
            ygc = yg - yg.mean(1, keepdim=True)          # within-state 标准化
            ygc = ygc / (ygc.std() + 1e-8)
            tgt = torch.softmax(ygc / self.tau, 1)
            return -(tgt * torch.log_softmax(sg, 1)).sum(1).mean()
        raise KeyError(self.loss_kind)

    def _calib(self, p, y):
        if self.loss_kind in ("mse", "mse_wc"):
            return self.ys, self.ym
        if np.std(p) < 1e-12:
            return 0.0, float(np.mean(y))
        if self.calib_kind == "within":
            # 在 within-state 中心化后的空间里拟合斜率（within_r2 的正确尺度）
            yc = np.asarray(y, np.float64).copy(); pc = np.asarray(p, np.float64).copy()
            for g in self.gva:
                yc[g] -= yc[g].mean(); pc[g] -= pc[g].mean()
            denom = float((pc ** 2).sum())
            a = float((pc * yc).sum() / denom) if denom > 1e-18 else 0.0
            return a, float(np.mean(y) - a * np.mean(p))
        a, b = np.polyfit(p, y, 1)
        return float(a), float(b)

    def run(self, make_model, selector, seeds=(0, 1, 2), epochs=4000,
            lrs=(3e-2, 1e-2, 3e-3, 1e-3), wds=(0.0, 1e-4, 1e-2),
            patience=600, eval_every=10, curves=False):
        best, curve_log = None, []
        for lr in lrs:
            for wd in wds:
                for sd in seeds:
                    torch.manual_seed(sd)
                    net = make_model().to(DEV)
                    opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                            weight_decay=wd)
                    local, bad, cv = None, 0, []
                    for ep in range(epochs):
                        net.train(); opt.zero_grad()
                        loss = self._loss(net(self.ftr))
                        if hasattr(net, "penalty"):
                            loss = loss + net.penalty()
                        loss.backward(); opt.step()
                        if (ep % eval_every) and ep != epochs - 1:
                            continue
                        net.eval()
                        with torch.no_grad():
                            pv = net(self.fva).cpu().numpy().astype(np.float64)
                        a, b = self._calib(pv, self.y_va)
                        sc = _safe(selector(self.y_va, a * pv + b))
                        if curves:
                            cv.append((ep, float(loss.item()), sc, float(a)))
                        if local is None or sc > local["val_score"]:
                            with torch.no_grad():
                                pt = net(self.fte).cpu().numpy().astype(np.float64)
                            local = {"val_score": sc, "pred_val": a * pv + b,
                                     "pred_test": a * pt + b,
                                     "calib": (float(a), float(b)),
                                     "best_epoch": ep,
                                     "train_loss": float(loss.item()),
                                     "n_params": int(sum(p.numel() for p in
                                                         net.parameters()))}
                            bad = 0
                        else:
                            bad += eval_every
                            if bad > patience:
                                break
                    if curves:
                        curve_log.append({"lr": lr, "wd": wd, "seed": sd,
                                          "curve": cv})
                    if local is None:
                        continue
                    local["hp"] = {"lr": lr, "wd": wd, "seed": sd,
                                   "epochs_budget": epochs}
                    if best is None or local["val_score"] > best["val_score"]:
                        best = local
        if best is not None and curves:
            best["curves"] = curve_log
        return best
