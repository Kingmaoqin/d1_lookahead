# RESCUE 预注册

写于 2026-08-31，**在任何人读取 `data/labels_freshA/` 之前**。
本文件与 `rescue_audit/confirm_fresh.py` 一起 git commit；提交哈希即时间戳。

探索阶段的全部结果见 `rescue_audit/results/R1*.json`、`R2stats_*.json` 与
`rescue_audit/EXPERIMENT_REGISTRY.csv`。本文件只固定**一次**确认性检验。

---

## 0. 一句话

探索阶段**没有**找到 relational / 非线性的抢救路径。仍然存在的唯一效应，
是旧实验已经报告过的那个**状态级通道**。本预注册把这个效应——以及那个
关系型假设的否定——各锁一次，在全新数据上验证一次。

---

## 1. 冻结的对象

| 项 | 值 |
|---|---|
| backbone | `kuleshov-group/mdlm-owt`，全程冻结（`requires_grad_(False)`） |
| 骨干实现 | `src/mdlm_local.py`（Phase-S 已认证；本轮 toy 精确验证见 §5） |
| π_ref | ancestral，每步 1 次提交，top_k=50，temperature=1.0 |
| 任务 / 语料 | OpenWebText 窗口，L=256，prefix=64 |
| 标签 | `A_pertok` = Rao-Blackwell 化的 per-token Path-LL 优势 `A_full` |
| horizon H | 16 |
| K | 24（CRN 配对推演） |
| 记录点 | 扩散进度 0.10 / 0.20 / 0.30 / 0.40 / 0.50 / 0.60 |
| 候选定义 | 每 state 6 个：3 个 natural（掩码位均匀抽样）+ 3 个 informative（高置信×高不稳定）。动作 = `(i, argmax_v p(v|s_t))` |
| **新数据** | `data/labels_freshA/`，prompt offset **400–599**（历史所有采集只用 0–399，故完全未污染），200 prompts → **1,200 states / 7,200 candidate examples** |
| 与旧 test 的比较 | 旧 test = 600 states / 3,600 examples。新数据 = **2×** |

## 2. 冻结的探针

### 2.1 Primary probe — `additive_pca`

* **layer = 6**（在探索阶段用 a3/b3 的**验证集** within-state R² 选出，
  见 `R1deep_MDLM_anc_A_pertok_L*.json`；此后不再改）
* 控制块 C = `[C1(8) ; C2(12) ; C3(80)]`，按 train 均值/标准差标准化
* 隐藏块 H = `[PCA32(h_i) ; PCA32(h_g)]`，PCA **只在 train 行上拟合**，白化
* 模型：两块岭回归，`alphas = np.logspace(-2, 9, 34)`，
  `gammas = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]`（含 0 → 嵌套基线）
* 超参在**验证集 within-state R²** 上选
* 参数量 165

### 2.2 Secondary probe — `cheap+H`

同上，但隐藏块用全维 `[h_i(768) ; h_g(768)]`，不做 PCA。参数量 1637。
这是旧实验 G1/G2 的原始口径，用于直接可比。

### 2.3 Baseline（主对比的分母）

`best_cheap` ＝ 在**验证集**上从下面两个里选更好的那个：
1. cheap 单块岭回归（旧口径的基线）
2. `rank_cheap`：只用 cheap 的成对 logistic 排序 MLP（隐层 256，
   lr∈{3e-3,1e-3}×wd∈{1e-4,1e-2}×3 个初始化，早停按本配置自身最优）

> 为什么必须这样：探索阶段发现 **仅仅把目标函数从池化 MSE 换成排序损失**，
> 在不使用任何隐藏态的情况下就能拿到 Δtop1 +0.037（95% CI [+0.013,+0.061]）。
> 旧实验的 Δconcordance 是对着一个**目标函数不匹配**的基线量出来的。
> 公平的分母必须包含这个改进。

同时**并排报告**相对旧口径（cheap 岭回归）的 Δ，以便与旧数字直接对照。

## 3. 两个确认设计（都预先指定）

* **D1 · 冻结迁移（primary）**：探针在**旧数据 a3/b3** 上拟合完毕并冻结
  （train 用旧 train 划分，超参用旧 val 划分选定），然后原封不动地应用到
  **freshA 的全部 7,200 行**。fresh 数据上**不做任何拟合**。
  这是最干净的形式，也回答任务书 §30 的 "frozen probe transfer across dataset"。
* **D2 · 同协议重拟合（secondary）**：在 freshA 自己的文档级 60/15/25 划分上
  重拟合，在其 test 上评估。与探索阶段协议一致，用于检查 D1 的结论不是
  迁移带来的假象。

## 4. 端点与阈值

### 4.1 Primary endpoint

**Δ pairwise concordance（within-state）＝ `additive_pca` − `best_cheap`**

判定 rescue 成功需**同时**满足：
1. Δconcordance ≥ **+0.020**
2. 文档级聚类自助 95% CI 排除 0
3. state 级配对置换检验 p < 0.05（单侧 greater）

### 4.2 Secondary endpoints（无论主端点结果如何都报告）

| 端点 | 阈值 |
|---|---|
| Δ top-1 oracle-best 准确率 | ≥ +0.020 且 CI 排除 0 |
| Δ normalized regret（下降为正） | ≥ +0.010 且 CI 排除 0 |
| Δ within-state R² | ≥ +0.010 且 CI 排除 0 |
| Δ Kendall τ | 只报告，不设阈值 |

### 4.3 Relational endpoint（本轮 rescue 的核心假设）

**Δconcordance = `kron_pca` − `additive_pca`**，其中 `kron_pca` 在
`additive_pca` 的隐藏块上追加 `vec(PCA32(h_i) ⊗ PCA32(h_g))`（1024 维交互项），
其余完全相同。

* 判定"relational rescue 成立"：Δ ≥ **+0.010** 且 CI 排除 0。
* 附带报告低秩双线性 `bilinear`（rank 在 {2,4,8,16} 中按 val 选）相对
  `best_cheap` 的 Δconcordance。

### 4.4 必须归零的证伪对照

全部在 D2 设计上运行，每一个的 Δconcordance 必须落在 [−0.01, +0.01] 内：

1. `shuffle_hg` —— h_g 在 state 之间整块置换
2. `gauss_hg` —— h_g 换成同尺度高斯噪声
3. `gauss_hi` —— h_i 换成同尺度高斯噪声
4. `label_perm_within` —— 标签在 state 内置换

任何一个对照产生 ≥ +0.02 的"效应"，本次确认作废。

## 5. 已完成的前置验证（这些是本次确认可信的前提）

| 检查 | 结果 |
|---|---|
| 生产估计量 vs 精确 DP 真值（toy，K=2000） | 全部候选 max\|z\| = **2.13** → 标签定义与实现正确 |
| Rao-Blackwell 方差缩减（toy） | 4.1–9.9×（中位 6.2×） |
| 合成 A（线性） | 线性探针 within-R² +0.597 ✅ |
| 合成 B（双线性） | 线性 +0.0001 ❌ / **双线性 +0.553 ✅** / shuffled h_g −0.093 ✅ |
| 合成 D（仅排序） | pooled 准则下基线 within-R² **−1.005**，within 准则下 **+0.010** —— 证明选择准则本身值 1.0 个 R² |
| 合成 F（零假设） | 19 探针 × 2 准则，**零假阳性**（conc 全在 [0.489,0.516]） |
| 旧结果复现 | 78 格正格数、噪声天花板、污染恒等式**逐位复现** |

> 合成 B 是本次确认的关键前提：**如果关系型探针连植入的双线性信号都读不出来，
> 那么它在真实数据上的零结果毫无意义。** 它读得出来（+0.553），所以读不出来
> 是有信息的。

## 6. 探索阶段的观测值（用于对照，**不是**本次的证据）

MDLM_anc，layer 6，旧 test（600 states）：

| 对比 | Δconc | Δtop1 |
|---|---|---|
| cheap+H − cheap | +0.0107 | +0.0633 |
| cheap+h_i − cheap | +0.0003 | +0.0217 |
| rank(cheap+hidden) − rank(cheap) | **−0.0152** | −0.0283 |
| kron − additive | **−0.0137** | **−0.0600** |
| rank(cheap) − cheap（**无隐藏态**） | +0.0066 | +0.0367 |

## 7. 事先写下的预测（可被证伪）

1. **Primary endpoint 会失败**：Δconc(additive_pca − best_cheap) 预计落在
   **+0.000 ~ +0.008**，达不到 +0.020。
2. 相对旧口径（cheap 岭回归）的 Δconc 预计 **+0.008 ~ +0.014**，
   CI 排除 0 —— 即复现 G1 通过、G2 失败的旧格局。
3. **Δtop1 相对旧口径预计 +0.04 ~ +0.07 且 CI 排除 0**；但相对
   `best_cheap` 预计降到 +0.01 ~ +0.03。
4. **Relational endpoint 会失败且为负**：Δconc(kron − additive) 预计
   **−0.020 ~ −0.005**。
5. 四个证伪对照全部落在 [−0.01, +0.01]。

如果 1 与 4 的预测被推翻（即 Δconc ≥ +0.020，或交互项显著为正），
那才是真正的 REVIVE，并且是在完全未见过的数据上。

## 8. 判定规则

| 情形 | 判定 |
|---|---|
| Primary 三项全过 **或** relational endpoint 达标 | **REVIVE** |
| Primary 不过，但 ≥2 个 secondary endpoint 达标且方向一致 | **PARTIAL** |
| Primary 不过、relational 不过、secondary 至多 1 项达标 | **KILL CONFIRMED** |

无论落到哪一栏，全部数值原样写入 `FINAL_RESCUE_REPORT.md`。
**不得在看到结果后修改本文件**；如需修正，另开
`RESCUE_PREREGISTRATION_AMENDMENT.md` 并说明理由与时间。
