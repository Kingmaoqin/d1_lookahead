"""
Rescue 数据层：加载标签分片、构造特征块、文档级划分。

与旧 `src/dataset.py` 的区别（全部是本轮新增，理由见 REPORT_TO_CODE_AUDIT.md）：
  * `h_gm`  —— 同一 state 内其他候选 h_i 的**留一均值**。这是"掩码位置池化"的
              一个无需重跑骨干的估计，用来补上 S3（h_g 只有全位置池化一种）。
              必须留一，否则 h_i 会出现在自己的"全局"向量里，制造机械差异。
  * 关系块  —— [h_i, h_g, h_i−h_g, |h_i−h_g|, h_i⊙h_g] 等（P4）。
  * 动作块  —— proposed_token 的嵌入 / 反嵌入向量（P10）。
  * PCA     —— 只在 train 上拟合，用于把 768 维压到可训练的维度。
"""
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHEAP_KEYS = ("C1", "C2", "C3")


# --------------------------------------------------------------------- load --
def load_labels(tags, keys=None, root=None):
    root = root or ROOT
    files = []
    for t in tags:
        files += sorted(glob.glob(os.path.join(root, "data", f"labels_{t}",
                                               "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards for tags {tags}")
    parts = [np.load(f) for f in files]
    ks = set(parts[0].files)
    for p in parts:
        ks &= set(p.files)
    if keys is not None:
        ks &= set(keys)
    d = {k: np.concatenate([p[k] for p in parts], 0) for k in sorted(ks)}
    d["state_id"] = (d["prompt_row"].astype(np.int64) * 10_000
                     + d["step"].astype(np.int64))
    if "H_i" in d:
        d["n_layers"] = d["H_i"].shape[1]
    return d


def state_groups(state_id):
    """返回 (uniq_states, group_index_lists, group_id_per_row)."""
    uniq, inv = np.unique(state_id, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    bounds = np.flatnonzero(np.diff(sorted_inv)) + 1
    groups = np.split(order, bounds)
    return uniq, groups, inv


# ----------------------------------------------------------------- features --
def cheap_block(d):
    return np.concatenate([d[k] for k in CHEAP_KEYS], 1).astype(np.float32)


def h_i(d, layer):
    return d["H_i"][:, layer].astype(np.float32)


def h_g(d, layer):
    return d["H_g"][:, layer].astype(np.float32)


def h_gm(d, layer, groups=None):
    """同 state 内其他候选 h_i 的留一均值（掩码位池化的代理）。

    LOO 是必须的：若直接用组均值，则 h_i − mean 会含有 (1−1/n)·h_i，
    在 within-state 排序里会引入与 h_i 完全共线的成分。
    """
    X = h_i(d, layer)
    if groups is None:
        _, groups, _ = state_groups(d["state_id"])
    out = np.zeros_like(X)
    for g in groups:
        if len(g) == 1:
            out[g] = X[g]                       # 无邻居，退化为自身
            continue
        s = X[g].sum(0)
        out[g] = (s[None, :] - X[g]) / (len(g) - 1)
    return out


def center_within_state(y, state_id, groups=None):
    """y − mean_state(y)。"""
    if groups is None:
        _, groups, _ = state_groups(state_id)
    out = np.asarray(y, dtype=np.float64).copy()
    for g in groups:
        out[g] -= out[g].mean()
    return out


def relational_block(hi, hg):
    """P4 的显式关系特征。"""
    return np.concatenate([hi, hg, hi - hg, np.abs(hi - hg), hi * hg],
                          1).astype(np.float32)


# ---------------------------------------------------------------------- PCA --
class TrainPCA:
    """只在 train 行上拟合的 PCA（含中心化与可选白化）。"""

    def __init__(self, dim, whiten=False):
        self.dim = dim
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean(0, keepdims=True)
        Xc = X - self.mu_
        # 经济 SVD；n 通常 > d，直接用 Gram 更快
        G = Xc.T @ Xc
        w, V = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:self.dim]
        self.W_ = V[:, idx]
        self.lam_ = np.clip(w[idx], 1e-12, None)
        self.n_ = len(X)
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mu_) @ self.W_
        if self.whiten:
            Z = Z / np.sqrt(self.lam_ / max(self.n_ - 1, 1))[None, :]
        return Z.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# -------------------------------------------------------------------- split --
def doc_splits(d, seed=0, fracs=(0.6, 0.15, 0.25)):
    docs = np.unique(d["doc_id"])
    rng = np.random.default_rng(seed)
    docs = docs.copy()
    rng.shuffle(docs)
    n_tr, n_va = int(fracs[0] * len(docs)), int(fracs[1] * len(docs))
    sets = {"train": docs[:n_tr], "val": docs[n_tr:n_tr + n_va],
            "test": docs[n_tr + n_va:]}
    return {k: np.where(np.isin(d["doc_id"], v))[0] for k, v in sets.items()}


def check_split_disjoint(d, sp):
    a = set(d["doc_id"][sp["train"]].tolist())
    b = set(d["doc_id"][sp["val"]].tolist())
    c = set(d["doc_id"][sp["test"]].tolist())
    assert not (a & b) and not (a & c) and not (b & c), "document leakage"
    return True


# ------------------------------------------------------------ label helpers --
def noise_ceiling(seeds):
    ybar = seeds.mean(1)
    K = seeds.shape[1]
    noise = float((seeds.var(1, ddof=1) / K).mean())
    obs = float(ybar.var())
    sig = max(obs - noise, 0.0)
    return dict(ceiling=sig / max(obs, 1e-12), snr=sig / max(noise, 1e-12),
                noise_var=noise, obs_var=obs)


def within_state_noise_ceiling(seeds, state_id, groups=None):
    """within-state 目标的噪声天花板。

    组内中心化对每条 seed 副本分别做，然后按同一公式计算。
    """
    if groups is None:
        _, groups, _ = state_groups(state_id)
    S = seeds.astype(np.float64).copy()
    for g in groups:
        S[g] -= S[g].mean(0, keepdims=True)
    return noise_ceiling(S)
