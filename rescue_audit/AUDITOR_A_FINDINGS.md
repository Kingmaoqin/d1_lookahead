# Auditor-A 复核报告（第二次）：对 ROUND-2 修正代码本身的对抗性审计

**范围**：**只审实现正确性**。统计设计 / 推断合理性由另一位审计员负责，本报告不介入。
**日期**：2026-08-31 · **环境**：`conda run -n llm`，`CUDA_VISIBLE_DEVICES=1`
**未触碰**：`data/labels_taskC/`、任何 `collect_task_labels.py` 进程（核实为 PID 2029132/2029208，分别占用 GPU1/GPU3，全程未干扰）。

**新增文件（不覆盖任何既有文件）**
`rescue_audit/auditA2_round2_recheck.py` —— 五个 stage：`nest` / `boot` / `capacity` / `arcsin` / `bootcal`
**新增结果**：`rescue_audit/results/auditA2_{nest,boot,bootcal,arcsin}.json`、
`auditA2_capacity_MDLM_anc_{real,B_bilinear,F_null}.json`
**新增日志**：`rescue_audit/logs/auditA2_cap_{real,B,F}.log`

---

## 0. 结论摘要

| # | 位置 | 严重度 | 对结论的影响 |
|---|---|---|---|
| A-1 | 自助 CI 脚本 `stat()`（`M.group_slices(sid_te[idx])`）+ `rlib/stats2.py:53-80` | **MAJOR** | CI 系统性过宽（覆盖率 100% vs 名义 95%）。修正后 MDLM within-informative **仍排除 0 且更强**；但 **FRESH holdout 由"含 0"翻为"排除 0"**，REVIEW_ROUND2 §4b 的措辞必须改 |
| A-2 | 产出 `stratified_concordance_ci.json` 的脚本**根本不在仓库里** | **MAJOR**（可复现性） | 提交的 CI 数字当时无法被任何人复算。现已由 `auditA2 --stage boot` 逐位复现并纳入仓库 |
| A-3 | `nonlinear_retest.py:73-76` 的"同容量 cheap-only 对照" | **MAJOR** | 参数量差 **7.3×**，"关系型 MLP 输给 cheap-only MLP"与"容量更大所以过拟合更重"完全混淆。做成真正等参数的对照后，真实数据上 Δconc = **+0.0213**（不是负值）。**总判定仍成立**（全部 MLP 变体都低于 cheap 岭回归） |
| A-4 | `rlib/screen.py:51`、`rlib/probes2.py:46`（`sd < 1e-8` 守卫漏掉 `sd = inf`）+ SEDD 分片 C3 数据 | **MAJOR** | SEDD 臂 100 维 cheap 控制块里 **64 维被静默置零**，实际只有 36 维。全项目所有 SEDD 对比（含 round-2 的 SEDD 复制臂）都是在残缺控制组上做的 |
| A-5 | `relational_recheck.py:126`、`nonlinear_retest.py:147`（`hash(sc)` 作随机种子） | MINOR | 植入信号的 ground-truth 结果**不可复现**（已实测三次进程给出三个不同种子） |
| A-6 | `relational_recheck.py:117-123`、`nonlinear_retest.py:142-143` | MINOR | synthetic 分支硬编码 layer 9，但输出 JSON 的 `config["layer"]` 记的是 6；`--arm` 被忽略却写进文件名。溯源信息误导 |
| A-7 | `nonlinear_retest.py` 的 "3000 步预算" + `probes3.Runner3.run` 早停 | MINOR | 39 个保存的拟合里 `best_epoch` 最大只有 280（最小 0），预算从未生效；且真实数据上 lr **仍撞网格边界**（7/13 = 54% 选中最小值 3e-3），F3 的"边界"症状只是换了一头 |
| A-8 | `probes3.py:13`、`relational_recheck.py:13` 文档字符串 | MINOR | 写"默认上界 100"，`GAMMAS_MAIN` 实际到 10000（代码对、注释错） |
| A-9 | `stratified_concordance.py:101` | MINOR | `sk = "A_full_seeds"` 硬编码，与 `--target` 不联动 |
| ✅ | `probes3.fit_ridge_blocks` 的嵌套断言 | **NOT-A-BUG** | 代码上成立、穷举参考实现上成立、17 个已存拟合上无一例外 |
| ✅ | 分层配对计数 / stratum 语义 / tie 处理 | **NOT-A-BUG** | 全部核实正确（见 §2） |
| ✅ | "自助抽样的 index 映射"（你最担心的那点） | **NOT-A-BUG** | `M.group_slices` 返回的是重抽样数组内的位置，而 `yt[idx]/pc[idx]/strat[idx]` 也都是同一个重抽样数组 —— 这一层映射**是对的**。真正的缺陷在别处（A-1） |
| ✅ | F2（gamma 上界 3.0）在 round-2 新代码里 | **NOT-A-BUG** | `probes2.fit_ridge_2block` 的默认上界确实仍是 3.0，但**不绑定**：选中 0.01–0.03，把网格放到 10000 结果逐位相同 |
| ✅ | 分半机制（不相交种子子集、无复用） | **NOT-A-BUG** | 正确；且 `A_pertok` 与 `mean(A_full_seeds)` 相差 ≤ 2.5e-8 |
| ✅ | arcsin 高斯近似 | **NOT-A-BUG**（误差已量化） | 在真实噪声结构下误差 −0.001…+0.004；在高斯噪声 + 经验边际下 ρ≈0.94 处系统性高估 0.007–0.009。天花板 0.8887 精度约 ±0.01，结论不变 |
| ✅ | 索引映射 / PCA 拟合行 / NaN 静默丢弃 | **NOT-A-BUG** | 逐处核实（见 §5） |

---

## 1. 嵌套断言（你的第 1 问）：**成立，且经三重独立验证**

### 1.1 代码层面

`rlib/probes3.py:82-105`：

```python
for combo in itertools.product(*gamma_grids):
    active = [g > 0 for g in combo]                     # 82-83
    ...
    for name, allow in record_slices.items():
        if any(active[b] and not allow[b] for b in range(nb)):   # 96
            continue
```

判定规则是"**该 combo 激活的每一块都必须被 allow**"。于是
`additive = [T,T,F]` 的可行域 = {γ_int = 0 的 combo}，
`kron = [T,T,T]` 的可行域 = 全部 combo，两者严格包含：
`F(additive) ⊂ F(kron)`。两个切片对**同一个** `sc`（`:93`）取 max，
因此 `kron.val_score ≥ additive.val_score` 按构造成立。

平局处理也是安全的：`sc > cur["val_score"]` 是严格大于，而 `itertools.product`
让最后一块（γ_int）变化最快，所以每个 γ_main 下 γ_int=0 的 combo **先**被遍历；
遇到平局时 kron 会退化到 additive 选中的同一个 combo，不会凭空造出 test 上的差异。

γ_main 网格里必须含 0.0 才能取到 `cheap_only` 切片 —— `relational_recheck.py:106`
（`gm = [0.0] + list(P3.GAMMAS_MAIN)`）确实这么做了。旁证：所有 6 个
`hk_L*` 条目的 `cheap_only` val 完全相同（0.550323），符合"该切片只有一个可行 combo"。

### 1.2 穷举参考实现（`--stage nest`）

用一份独立的 numpy float64 穷举实现（自己重做标准化 + 特征分解 + ridge path），
对三个切片各自的可行域取 max：

| 切片 | 实现 val | 穷举 val | \|diff\| | 选中 γ 是否合法 |
|---|---|---|---|---|
| cheap_only | −6.41542352 | −6.41542359 | 6.8e-08 | ✅ |
| additive | −1.94565726 | −1.94565722 | 3.8e-08 | ✅ |
| kron | −0.29741124 | −0.29741123 | 9.8e-09 | ✅ |

差异全部在 float32 GPU vs float64 CPU 的舍入量级。

**极端形状 / 极端 γ 的对抗测试**（`scratchpad/b4_extreme.py`，块尺寸取真实值 100/64/1024，
γ 网格用真实的 `GAMMAS_MAIN`（到 10000）× `GAMMAS_INT`）：
选中 γ_main = 10000 时，GPU-float32 与 CPU-float64 的 val 差 ≤ 3.1e-08，
`kron ≥ additive ≥ cheap_only` 依然成立。**Gram 矩阵在 float32 里累加**
（`probes3.py:45`）在这个尺度跨度下没有崩。

### 1.3 已存结果的实证扫描

`results/relational_recheck_*.json` 全部 6 个文件、17 个条目
（真实 MDLM_anc / MDLM_conf / SEDD_anc / FRESH，合成 B/F/A）：
**没有任何一处 `kron.val_score < additive.val_score`**，也没有
`additive < cheap_only`。断言在真实运行里零反例。

一点语义上的提醒（不是 bug）：`kron` 的可行域包含 γ_main=0（纯交互、无主效应），
`additive` 的可行域包含 γ_main=0（退化成 cheap_only）。合成 B_bilinear 上
`additive` 选中的正是 `[1.0, 0.0, 0.0]`，即字面上就是 cheap_only 模型 ——
这在植入的纯双线性信号下是**正确**行为，但读数字时要知道那一格的
"relational Δ" 实际是 kron − cheap_only。

---

## 2. 分层 concordance 的拆分（你的第 2 问）

### 2.1 stratum 语义 —— 正确

`src/collect.py:171-209 pick_candidates`：
`n_nat = cfg.n_cand − cfg.n_cand_conf = 6 − 3 = 3` 个 stratum **0 = natural**（掩码位均匀抽样），
`k_inf = 3` 个 stratum **1 = informative**（`conf + temporal_kl + 0.5·flip_count` 取 top-3）。
`stratified_concordance.py:16` 的注释与代码一致。

### 2.2 配对计数 —— 正确

实测（`scratchpad/a1.py`）：

| 臂 | 行数 | state 数 | 组大小分布 | (n_nat, n_inf) 分布 | y 打平配对 |
|---|---|---|---|---|---|
| MDLM_anc | 14400 | 2400 | {6: 2400} | {(3,3): 2400} | 0 / 36000 (0.00%) |
| SEDD_anc | 14400 | 2400 | {6: 2400} | {(3,3): 2400} | 0 / 36000 |
| FRESH | 7200 | 1200 | {6: 1200} | {(3,3): 1200} | 0 / 18000 |

**每个 state 严格 3 nat + 3 inf**，因此 3 / 3 / 9 的配对数结构成立。
test 集实测：MDLM/SEDD = 1800 / 1800 / 5400（600 state），FRESH = 900 / 900 / 2700（300 state）。
与 `results/stratified_concordance.json` 的 `n_pairs` 完全一致。

### 2.3 tie 处理 —— 与 `metrics.concordance` 逐位一致

`stratified_concordance.py:47-49` 的 `|dy| ≤ 1e-9 → 跳过`、`|dp| ≤ 1e-12 → 计 0.5`
与 `rlib/metrics.py:73-76` 完全同规则。实测 `conc_by_stratum(...)["all"]` vs
`M.concordance(...)`：

```
MDLM   cheap 0.7785555556 / 0.7785555556   +H 0.7888888889 / 0.7888888889
FRESH  cheap 0.7473333333 / 0.7473333333   +H 0.7606666667 / 0.7606666667
SEDD   cheap 0.8085555556 / 0.8085555556   +H 0.8118888889 / 0.8118888889
```

（`conc_by_stratum` 里 `n = len(g)` 与返回式推导中的 `n` 同名，但 Python 3 的
推导式有独立作用域，不会串。上表就是这一点的实证。）

### 2.4 **A-1（MAJOR）自助抽样的重复文档被合并成一个巨型 state**

你担心的那点 —— "`M.group_slices` 返回的是重抽样数组内的位置" —— **不是 bug**。
`stat(idx)` 里 `yt[idx] / pc[idx] / ph[idx] / strat[idx]` 与 `sid_te[idx]` 是同一个
重抽样数组，位置索引自洽。

真正的缺陷在下一层：`M.group_slices` 按 **state_id 的值**归组
（`metrics.py:14-19`，`np.unique + argsort`）。文档级自助抽样里同一个 document
可能被抽中 m 次，它的每个 state 就有 6m 行**同一个 state_id**，于是被合并成
**一个 6m 行的组**，而不是 m 个独立的 6 行组。

后果可以精确解出：设某 state 被抽中 m 次，
- 正确做法（每份拷贝是独立 state）：贡献 m × 15 个配对；
- 实际做法（合并成 6m 行）：同候选跨拷贝的配对 dy = 0 被过滤，
  异候选跨拷贝的配对每个 candidate-pair 出现 m² 次 → 贡献 **m² × 15** 个配对，
  且这 m² 份的 v 值完全相同（纯重复，零信息量）。

即：**该文档在比值统计量里被赋予 m² 而不是 m 的权重。**

**逐位验证**：我用解析权重 `w = m²`（原）/ `w = m`（修正）重写了自助，
与逐行重跑 `strat_ci.py` 的 `stat(idx)` 在 40 次抽样上比对，
**max |diff| = 0.000e+00**（完全相同，非近似）。同时它逐位复现了已提交的
`stratified_concordance_ci.json`（MDLM within_inf: mean 0.02580 / CI [0.003818, 0.050614]）。

**理论量级**：文档数 100 时 m 近似 Poisson(1)，
`sd(m)/E[m] = 1`，`sd(m²)/E[m²] = √11/2 = 1.658`。
**实测 sd 比值 1.55–1.68**，与理论吻合。方向是**保守**（CI 偏宽）。

**在已知真值上的覆盖率标定**（`--stage bootcal`，100 文档 × 6 state × 3 配对 ×
文档级随机效应，400 次重复）：

| 植入真值 Δ | orig（m²）覆盖率 | orig 平均 CI 宽 | fixed（m）覆盖率 | fixed 平均 CI 宽 |
|---|---|---|---|---|
| 0.000 | **1.000** | 0.0906 | 0.945 | 0.0554 |
| 0.026 | **0.998** | 0.0889 | 0.927 | 0.0544 |

名义 95% 的区间实际覆盖 100% —— **原实现严重过覆盖**，修正版接近名义水平。

#### 修正前 / 修正后（真实数据，L8，`--stage boot`，20000 次自助）

| 数据 | 子集 | 点估计 | **修正前** 95% CI | 排除 0 | **修正后** 95% CI | 排除 0 |
|---|---|---|---|---|---|---|
| MDLM_anc | within-natural | +0.0050 | [−0.0159, +0.0264] | ✗ | [−0.0089, +0.0189] | ✗ |
| MDLM_anc | **within-informative** | **+0.0256** | [+0.0038, +0.0506] | ✓ | **[+0.0117, +0.0400]** | **✓** |
| MDLM_anc | cross | +0.0070 | [−0.0049, +0.0186] | ✗ | [−0.0002, +0.0143] | ✗ |
| MDLM_anc | 全部 | +0.0103 | [+0.0009, +0.0198] | ✓ | [+0.0044, +0.0162] | ✓ |
| FRESH | within-natural | +0.0133 | [−0.0104, +0.0422] | ✗ | [−0.0033, +0.0289] | ✗ |
| FRESH | **within-informative** | **+0.0200** | [−0.0073, +0.0460] | **✗** | **[+0.0033, +0.0367]** | **✓ 翻转** |
| FRESH | cross | +0.0111 | [−0.0032, +0.0270] | ✗ | [+0.0019, +0.0207] | ✓ 翻转 |
| FRESH | 全部 | +0.0133 | [+0.0014, +0.0269] | ✓ | [+0.0053, +0.0216] | ✓ |
| SEDD | within-natural | +0.0011 | [−0.0231, +0.0211] | ✗ | [−0.0128, +0.0144] | ✗ |
| SEDD | within-informative | +0.0078 | [−0.0116, +0.0272] | ✗ | [−0.0033, +0.0189] | ✗ |
| SEDD | cross | +0.0026 | [−0.0084, +0.0152] | ✗ | [−0.0046, +0.0100] | ✗ |
| SEDD | 全部 | +0.0033 | [−0.0054, +0.0122] | ✗ | [−0.0021, +0.0089] | ✗ |

> **+0.0258 这个发现活下来了，而且更强。**（注：0.0258 是自助均值；数据上的点估计是 **+0.0256**，
> 修正后的自助均值 +0.0255 —— 三者一致，说明自助无系统偏移。）
>
> **但 REVIEW_ROUND2.md §4b 有一句必须改**：
> "在 fresh holdout 上**方向一致但 CI 含 0**" —— 这是过宽 CI 造成的假象。
> 用正确的聚类自助，FRESH 的 within-informative CI = **[+0.0033, +0.0367]，排除 0**。
> 也就是说这个 post-hoc 发现在 fresh holdout 上**复制成功了**，比原报告写的更有分量。
> （这不改变"它仍是 post-hoc 亚组、仍需要预注册确认性检验"这一点 —— 那是设计问题，归另一位审计员。）
>
> SEDD 臂不受影响：四个子集在两种权重下都含 0。

### 2.5 **A-2（MAJOR，可复现性）：CI 脚本不在仓库里**

`stratified_concordance_ci.json` 由 commit `e69dcd3` 提交，但**产生它的代码没有被提交**。
全仓库 grep 不到任何写这个文件的脚本；我最终在
`/tmp/.../scratchpad/strat_ci.py` 找到它。也就是说，交付出去的 CI 数字在当时
**没有任何人能复算**。现在 `auditA2_round2_recheck.py --stage boot` 同时给出
orig 与 corrected 两版，两版都已逐位对齐原脚本，缺口补上。

---

## 3. 标签可靠性天花板（你的第 3 问）

### 3.1 分半机制 —— 正确

`label_reliability_ceiling.py:49-60`：`perm = rng.permutation(K)`，
`a = perm[:m]`、`b = perm[m:2m]` —— **不相交，无复用**，`assert 2*m <= K` 守住边界。
`stratified_concordance.py:107-110` 同构。
`d["A_pertok"]` 与 `A_full_seeds.mean(1)` 的最大绝对差 ≤ **2.5e-08**（三个臂），
即分半分析和探针评估用的是同一个量。

### 3.2 噪声模型本身 —— 自洽

`rdata.within_state_noise_ceiling` 先对**每条 seed 列**做 within-state 中心化，
再算 `ceiling = (obs − noise) / obs`。组内中心化会把每条种子的噪声方差乘上
`(1 − 1/n)`，但**分子分母里出现的是同一个 `(1−1/n)σ²/K`**，所以估计的正是
"within-state 中心化后标签的可靠性 r(K)"，没有偏。`ratio = (1/r_K − 1)·K` 也随之正确。

**候选间的 CRN 相关**实测为 **+0.48**（同一 state 内 6 个候选的种子偏差相关系数均值），
这确实违反了"候选间独立"的朴素假设 —— 但由于估计量是在中心化之后做的，
它自动吸收了 CRN 效应。§3.3 的实证检验直接验证了这一点。

### 3.3 arcsin 这一步 —— 公平，误差已量化（`--stage arcsin`）

**(a) 用真实标签的经验边际 + 受控高斯噪声**：构造 `p = ρ·y_c + √(1−ρ²)·sd·z`
（z 独立标准正态，故真相关**精确等于 ρ**），比对实测 within-state concordance 与
`0.5 + arcsin(ρ)/π`：

| ρ | MDLM 实测 / 公式 / 误差 | SEDD | FRESH |
|---|---|---|---|
| 0.50 | 0.6661 / 0.6667 / −0.0005 | −0.0023 | +0.0041 |
| 0.70 | 0.7494 / 0.7468 / +0.0026 | +0.0002 | −0.0030 |
| 0.80 | 0.7867 / 0.7952 / −0.0085 | −0.0089 | −0.0049 |
| 0.90 | 0.8488 / 0.8564 / −0.0077 | −0.0092 | −0.0068 |
| **0.9395**（= √r(K)） | **0.8818 / 0.8887 / −0.0070** | −0.0085 | −0.0077 |
| 0.97 | 0.9154 / 0.9218 / −0.0065 | −0.0090 | −0.0029 |

**(b) 用真实的 MC 噪声结构**（自助重抽 K 条种子列 —— 保留异方差与候选间 CRN 相关，
构造"完美预测器 vs K 条种子标签"的配对，20 次重复）：

| 臂 | ρ | 实测 conc | arcsin 公式 | 误差 |
|---|---|---|---|---|
| MDLM | 0.9442 | 0.8924 | 0.8932 | **−0.0008** |
| SEDD | 0.9530 | 0.9007 | 0.9020 | **−0.0013** |
| FRESH | 0.9459 | 0.8985 | 0.8948 | **+0.0037** |

**结论**：高斯/arcsin 这一步对这批标签是**公平的**。在最相关的 ρ≈0.94 处，
纯高斯噪声版本系统性高估 0.007–0.009（n=2400 state 的 MC 噪声约 ±0.004，
所以这是真偏差不是抖动）；换成真实噪声结构后误差落到 ±0.004 以内。
**天花板 0.8887 的精度约 ±0.01**；最坏情况下 MDLM 的 "cheap 之上剩余空间"
从 +0.110 变成 +0.103，隐藏态吃掉的比例从 9.7% 变成 10.4%。**结论不动。**

（脚本本身已经报了 `curve_model_max_abs_error = 0.012–0.014`，
方向也一致：模型在小 m 处低估、大 m 处高估。这个自查是诚实的。）

### 3.4 口径一致性 —— 影响可忽略（NOT-A-BUG）

`label_reliability_ceiling.py` 的天花板在**全部行**上估计（`:82-93`、`:120`），
而 `probe_vs_full_label` 的 cheap / cheap+H 只在 **test 行**上评。我重算了
只用 test 行的天花板：

| 臂 | 全行 r(K) → 上限 | 仅 test → 上限 | 差 |
|---|---|---|---|
| MDLM | 0.8827 → 0.8887 | 0.8822 → 0.8885 | −0.0003 |
| SEDD | 0.9023 → 0.8988 | 0.9045 → 0.9000 | +0.0012 |
| FRESH | 0.8778 → 0.8863 | 0.8760 → 0.8854 | −0.0009 |

差异 ≤ 0.0012，不影响任何结论。

---

## 4. 新代码里的其它问题

### 4.1 **A-3（MAJOR）"同容量对照"名不副实**

`nonlinear_retest.py:71-76` 的注释与 `rlib/probes2.py:11` 的库级承诺
（"每个非线性/关系型探针都有一个同容量的 cheap-only 版本"）都声称容量匹配。
实际参数量（取自已提交的结果 JSON）：

| 探针 | n_params | 对照 | n_params | 比值 |
|---|---|---|---|---|
| relmlp_w128 | 94,977 | cheaponly_mlp_w128 | 13,057 | **7.27×** |
| relmlp_w512 | 379,905 | cheaponly_mlp_w512 | 52,225 | **7.27×** |

原因很直接：两者隐藏宽度相同，但输入维不同（`d_r = 5×128 = 640` vs `d_c = 100`）。
于是"关系型 MLP（0.7587）输给 cheap-only MLP（0.7788）"这个对比里，
**特征信息量**与**容量/过拟合程度**是完全混淆的 —— 这正是 commit message
里"same-family contrasts stay negative"所依赖的那条证据。

**修正**（`--stage capacity`）：换成**参数量逐位相同**的证伪对照，三者共用同一套
lr/wd/seed/epoch 网格：

- `real`：`rel = [h_i, h_g, h_i−h_g, |h_i−h_g|, h_i⊙h_g]`（PCA-白化后 128 维 → 640）
- `shuffle_hg`：同上，但 h_g **在 state 之间整体置换**（h_i 不动）。
  （已核实 h_g 在 state 内**严格恒定**：300 个 state 上的组内最大离差 = 0.0，
  所以 state 级置换是合法的。）
- `gauss`：整块 640 维换成同均值同方差的高斯噪声（纯容量地板）

| 数据 | 宽度 | n_params 相同？ | real | shuffle_hg | gauss | **Δconc(real−shufHg)** | Δconc(real−gauss) |
|---|---|---|---|---|---|---|---|
| **植入 B_bilinear** | 128 | ✅ 94,977 | 0.5628 | 0.5020 | 0.5064 | **+0.0608** | +0.0563 |
| **植入 B_bilinear** | 512 | ✅ 379,905 | 0.5703 | 0.5072 | 0.4904 | **+0.0631** | +0.0799 |
| **植入 F_null** | 128 | ✅ 94,977 | 0.5060 | 0.5088 | 0.5063 | **−0.0028** | −0.0003 |
| **植入 F_null** | 512 | ✅ 379,905 | 0.5019 | 0.5072 | 0.4968 | **−0.0053** | +0.0051 |
| **真实 MDLM_anc L8** | 128 | ✅ 94,977 | 0.7587 | 0.7373 | 0.7167 | **+0.0213** | +0.0420 |
| **真实 MDLM_anc L8** | 512 | ✅ 379,905 | 0.7603 | 0.7452 | 0.7099 | **+0.0151** | +0.0504 |

ground-truth harness 校准良好：植入信号上 +0.06，零假设上 Δ 绝对值 ≤ 0.005，
真实数据上 +0.015 ~ +0.021 —— **明显在零假设带之外**。

**这条推翻的是"证据"，不是"结论"**：
1. 原来的对比（relmlp 输给 cheaponly_mlp）**不能**用来论证"隐藏态不带信息"，因为容量不匹配；
2. 做成等参数对照后，隐藏态**确实**带可读出的排序信号（+0.021），与凸的 Kronecker 岭回归
   给出的 `hidden Δconc = +0.012` 同号同量级；
3. 但 `shuffle_hg` 同时摧毁了 h_g 的**主效应**和**交互**，所以 +0.021 是"h_g 的总贡献"，
   **不是**关系型贡献。关系型的干净检验是那个凸的、必然收敛的 Kronecker 岭回归
   （`relational_recheck`），它在 17 个真实条目上给出 `relational Δconc ∈ [−0.0031, +0.0002]`，
   而在植入 B_bilinear 上给出 **+0.0747** —— 这条**关系型阴性结论不受本条影响**；
4. **总判定仍然成立**：全部 6 个 MLP 变体（0.7099–0.7603）都**低于** cheap 岭回归基线
   （0.7786）与 cheaponly_mlp（0.7788）。整个 MLP 家族在真实数据上没有超过廉价特征。

### 4.2 **A-4（MAJOR）SEDD 的 cheap 控制块有 64/100 维被静默置零**

`rlib/screen.py:50-53` 与 `rlib/probes2.py:44-47` 的标准化守卫是

```python
sd = X.std(0, keepdims=True); sd[sd < 1e-8] = 1.0
```

只挡住 sd ≈ 0，**挡不住 sd = inf**。而 SEDD 分片（`s1`/`s2`）的 C3 块里，
64 个 JL 投影列（`C3_NAMES` 的 `proj_0..proj_63`，对应 cheap 索引 36–99）
是**常数 ≈ 1e29**：

| tag | C1 常数列 | C2 常数列 | C3 常数列 | C3 absmax |
|---|---|---|---|---|
| a3 / b3 / c3 / d3 / freshA | 0/8 | 0/12 | **0/80** | ~3.2e3 |
| **s1 / s2** | 0/8 | 0/12 | **64/80** | **3.48e+29** |

float32 下 `std` 溢出成 `inf`（就是那条 `numpy/_core/_methods.py:194
RuntimeWarning: overflow encountered in multiply`，已定位到 `screen.py:51`），
`sd < 1e-8` 判 False，于是 `(X − mu)/inf = 0` —— 这 64 列**变成恒 0**。
实测：`prep["pca"]["cheap"]` 与 `probes2._std` 输出均为 **finite=True、64 个恒零列、0 个 NaN**。
没有报错、没有 NaN、没有一行日志。

**影响**：SEDD 臂的 "cheap = C1+C2+C3 强控制组"实际上只有 36 个活列。
全项目所有 SEDD 对比（包括 round-2 的 SEDD 复制臂、`label_reliability` 的
`probe_vs_full_label`、`stratified_concordance` 的 SEDD 行）都是在残缺控制组上做的。

**方向说明**：控制组变弱应当**放大**隐藏态增益，而 SEDD 恰恰是三个臂里
Δhidden 最小的（+0.0033 池化），所以这条**不解释** SEDD 的零结果 ——
但"SEDD 复制失败"这个说法在修好 C3 之前不能算数，因为对照组根本不是同一个东西。

**最小修复建议**（未改动既有文件）：把两处守卫改成
`sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0`，并在 `load_labels` 里加一条
`assert np.isfinite(X).all() and X.std(0).max() < 1e12` 的分片健康检查；
C3 溢出的根因在 `src/collect.py::project_logprobs`（SEDD 的 `logits/lse` 语义
与 MDLM 不同）需要单独排查。

### 4.3 A-5（MINOR）植入信号不可复现

`relational_recheck.py:126` 与 `nonlinear_retest.py:147`：

```python
rng = np.random.default_rng(hash(sc) % (2 ** 31))
```

Python 的字符串 `hash()` 默认**每进程随机化**（未设 `PYTHONHASHSEED`）。
实测三次独立进程：`392392554 / 507354401 / 431998414`。
因此 `relational_recheck_synthetic_*.json` 与 `nonlinear_retest_synthetic_*.json`
里的植入信号**无法被重新生成**，ground-truth harness 的数字不可逐位复核。
（我自己的 `--stage capacity --scen ...` 用了显式 `syn_seed=12345`。）
`synthetic_tests.py:149` 有同样的问题。

### 4.4 A-6（MINOR）溯源信息误导

`relational_recheck.py:117-123` 与 `nonlinear_retest.py:142-143` 的 synthetic 分支
**硬编码 layer 9**（`RD.h_i(d, 9)`、`SC.prepare(d, sp, 9, ...)`），
但 `rep["config"] = vars(args)` 记录的是 `"layer": 6`；`--arm` 同样被忽略
（合成分支永远用 a3/b3），却被拼进输出文件名 `..._synthetic_MDLM_anc.json`。
数值自洽（层内一致），但归档的元数据是错的。

### 4.5 A-7（MINOR）F3 的"边界"症状没有真正解除

- **预算从未生效**：39 个保存的拟合里 `best_epoch` ∈ [0, 280]，最大 280；
  `patience=500`、`eval_every=10` 意味着最迟约 epoch 780 就早停了。
  "在 3000 步预算下重测"实际上是"在 ≤800 步内早停"。结论（加预算无用）成立，
  但措辞应改成"模型在数十步内就开始过拟合，加预算不可能有用"。
- **lr 换了一头撞边界**：真实数据 L8 的 13 个拟合里，选中 `lr = 3e-3`
  （新网格的**最小值**）的有 7 个 = **54%**（3e-2 两个、1e-2 四个）。F3 原本的症状是"撞最大值 3e-3（旧网格）"，
  现在是"撞最小值 3e-3（新网格）"。网格只往上扩了，没往下扩。
  由于关键对比（relmlp vs cheaponly / real vs shuffle_hg）双方共用同一网格，
  这不会翻转比较方向，但"边界已解除"的说法不成立。

### 4.6 A-8 / A-9（MINOR）

- `probes3.py:13` 与 `relational_recheck.py:13` 写"默认上界 100"，
  `GAMMAS_MAIN`（`probes3.py:28`）实际到 **10000**。代码对、注释错。
- `stratified_concordance.py:101` 硬编码 `sk = "A_full_seeds"`，
  与 `--target` 不联动；默认 `A_pertok` 时正确，改成 `A_future` 就会拿错种子矩阵。

---

## 5. 我核对过并确认**正确**的部分

| 检查项 | 结论 | 证据 |
|---|---|---|
| `record_slices` 嵌套逻辑 | ✅ | §1，三重验证 |
| float32 GPU ridge 在 γ=10000、1024 列交互块下的数值 | ✅ | 与 float64 CPU 差 ≤ 3.1e-08 |
| `_std_block` 用 train 的 mu/sd 变换 val/test | ✅ | `probes3.py:70` |
| `fit_ridge_blocks` 的 `n_params` 与所记 combo 对应 | ✅ | 记录发生在循环体内，`Xt` 即当前 combo |
| `_calib` "within" 模式 | ✅ | 在 val 的组内中心化空间里最小二乘拟合斜率（`probes3.py:166-172`），正是 `within_r2` 的正确尺度；截距是池化的、不影响 within_r2 与 concordance；(a,b) 拟合在 **val** 上、施加到 test —— 与超参选择同级，不构成 test 泄漏 |
| `Runner3._group_tensor` 的等组长要求 | ✅ | 全部 state 组大小恒为 6 |
| 早停计数器 `bad += eval_every` | ✅ | 逻辑正确（patience=500 → 50 次评估） |
| stratum 语义 / 3-3-9 配对结构 / tie 处理 | ✅ | §2.1–2.3 |
| 自助抽样中 `group_slices` 的位置索引映射（你的主要担心） | ✅ **不是 bug** | §2.4 第一段 |
| F2（γ 上界 3.0）在 `fit_ridge_2block` 默认网格 | ✅ 不绑定 | 三个臂选中 γ = 0.01–0.03；把网格扩到 10000 后 concordance **逐位相同**（0.7889 / 0.7607 / 0.8119） |
| 分半的种子子集不相交、无复用 | ✅ | `perm[:m]` / `perm[m:2m]`，`assert 2m ≤ K` |
| `A_pertok == mean(A_full_seeds)` | ✅ | 最大绝对差 ≤ 2.5e-08 |
| 天花板的 all-rows vs test-rows 口径 | ✅ 可忽略 | 差 ≤ 0.0012 |
| arcsin 近似 | ✅ 误差 ≤ 0.01 | §3.3 |
| **split-local ↔ global 行索引映射** | ✅ | `stratified_concordance.py:90-93`（`strat = d["stratum"][te]`、`g_te` 是 test 内位置）、`label_reliability_ceiling.py:162-191`（`seeds[te][:, b]` 与 `pred_test` 同为 test-local）、`relational_recheck.py:63-68`、`nonlinear_retest.py:167-168` —— 逐处核对无错位 |
| **PCA 是否拟合在正确的行上** | ✅ | `screen.py:44-48` 只用 `out["raw"][nm]["train"]` = `X[tr]` 拟合；`rdata.TrainPCA` 的 `mu_/W_/lam_` 全部来自 train |
| **h_gm 留一是否跨划分泄漏** | ✅ | 断言"每个 state 的所有行属于同一个 document"在三个臂上全部通过；state ⊂ document ⊂ split，留一均值不跨划分 |
| **静默丢弃的 NaN** | ✅ 无 NaN | 三个臂 `A_full_seeds` / `A_pertok` / C1 / C2 / C3 全部 finite；`prep` 与 `_std` 输出全部 finite。**但**见 A-4：SEDD 有 64 列被静默**置零**（不是 NaN，更难发现） |
| `make_controls` / `_shuffle_hg_states` 是否真的破坏了它声称要破坏的东西 | ✅ | h_g 组内严格恒定（组内最大离差 0.0），state 级置换合法；植入 B_bilinear 上 shuffle 使 Δconc 掉 0.061，F_null 上 Δ 绝对值 ≤ 0.005 —— 对照有效且无假阳性 |

---

## 6. 明确裁决：round-2 的数字站得住吗？

**总体：站得住，但三处必须改口，一处必须重跑。**

1. **关系型阴性结论（`relational_recheck`）—— 完全站得住。**
   嵌套断言在代码、穷举参考、极端 γ、17 个已存拟合上四重成立；
   数值精度 ≤ 3e-08；F1/F2 的修复是真的修好了。
   真实数据 `relational Δconc ∈ [−0.0031, +0.0002]`，同一套代码在植入
   B_bilinear 上给出 **+0.0747**。**这条结论我确认为可信。**

2. **within-informative 的 +0.0258 —— 站得住，而且比报告写的更强。**
   点估计（+0.0256）、配对计数（1800）、tie 处理、stratum 语义、
   split-local 索引全部核实无误；F2 的 γ 上界在这里不绑定。
   自助有一个 MAJOR 缺陷（重复文档权重 m² 而非 m），但方向是**保守**的
   （覆盖率 100% vs 名义 95%）。修正后 CI 从 [+0.0038, +0.0506] 收窄到
   **[+0.0117, +0.0400]**，仍排除 0。
   **必须改口 (a)**：FRESH holdout 的 within-informative CI 由 [−0.0073, +0.0460]
   变为 **[+0.0033, +0.0367]，排除 0** —— "fresh holdout 上 CI 含 0"这句话是错的，
   这个 post-hoc 发现在独立 holdout 上**复制成功**。

3. **标签天花板 0.886–0.899 —— 站得住。**
   分半机制正确、噪声模型自洽（CRN +0.48 被正确吸收）、arcsin 近似在真实噪声
   结构下误差 ≤ 0.004。天花板精度约 ±0.01，"cheap 之上还剩 0.09–0.14、冻结表征
   只吃掉 3–10%"这个论断不受影响。

4. **非线性/关系型 MLP 的阴性对比 —— 证据无效，需重跑。**
   **必须重跑**：`nonlinear_retest.py` 的 "同容量 cheap-only 对照"参数量差 7.3×，
   commit message 里"relational MLP vs cheap-only MLP 保持阴性"这条**不能用**。
   等参数对照下真实数据 Δconc = **+0.0213**（w128）/ **+0.0151**（w512），
   而零假设上 Δ 绝对值 ≤ 0.005、植入信号上 +0.06 —— 隐藏态**确实**带信号。
   **但总判定不翻**：全部 6 个 MLP 变体都低于 cheap 岭回归（0.7786）。
   **必须改口 (b)**：说法应从"关系型 MLP 输给同容量 cheap-only 对照"
   改成"关系型 MLP 在等参数证伪对照上确有 +0.02 的真实增益，但整个 MLP 家族
   连线性 cheap 岭回归都打不过"。

5. **SEDD 复制臂 —— 暂时不可用。**
   **必须改口 (c)**：SEDD 的 cheap 控制块有 64/100 维是常数 1e29 并被静默置零，
   它与 MDLM 的 cheap 控制**不是同一个对照组**。在 C3 修好并重采之前，
   "在第二骨干上很弱 / SEDD 复制失败"不能作为证据（无论方向）。

6. **可复现性欠账**：CI 脚本未入库（已由 `--stage boot` 补上）；
   合成场景用 `hash()` 作种子导致植入信号不可再生；synthetic 分支的
   `config["layer"]` 记录与实际不符。这三条不影响已有数字的**正确性**，
   但影响它们的**可核查性**，发表前必须清掉。

**一句话**：round-2 的核心数字（关系型阴性、天花板、informative 子集增益）经得起
实现层面的对抗审查；被打穿的是"非线性 MLP 家族的同容量对照"这一条证据链，
以及 SEDD 臂的控制组完整性 —— 前者需要换对照重跑，后者需要修数据。
MDLM/FRESH 上的核心判定不变。
