# Direction One：旧结果与最终口径对照

更新时间：2026-09-02。数字的唯一论文入口是
`DIRECTION1_ICLR_MASTER_RESULTS.csv`；本文件只解释冲突如何消解。

| 主题 | 旧口径 | 最终口径 | 裁决 |
|---|---:|---:|---|
| Nemotron 状态级 ΔR² | +0.1317 | **+0.1433**，95% CI [+0.0847,+0.2042] | 两者都有效。旧值是 `cheap+hidden`，新值是验证集选定的 `hidden-only`；新值为 headline，旧值为消融。 |
| Nemotron 状态级 ΔAUC | +0.0655 | **+0.0666**，95% CI [+0.0378,+0.0982] | 同上；不是复算矛盾。 |
| 时间稳定性 | 汇总正结果 | **6/6 时点 ΔR² CI 排除 0；5/6 时点 ΔAUC CI 排除 0** | 升级为强支持，不能因最后阶段样本减少而整体降格。 |
| Candidate Path-LL | 早期三划分小正或被总 KILL 覆盖 | **六个 backbone/policy/target 条件的 50-split Δwithin-R² 全为正，49–50/50 splits 为正，所有 NB-corrected CI 排除 0** | 最新 state-centered 估计优先；候选级 Path-LL 信号成立但效应小于状态级。 |
| Candidate task utility | Task C 探索性 +0.0381 | C+D bilinear-vs-cheap **+0.0143 [−0.0091,+0.0369]**；SVAMP E **+0.0133 [−0.0369,+0.0654]** | 五折和强基线后未确认；只否定稳定的候选级 task-reward action readout，不否定状态级 V 或 Path-LL A。 |
| Task C→D frozen transfer | 尚未完成 / 正在跑 | **−0.0356 [−0.0567,−0.0151]** vs cheap | 确认不迁移，作为边界结果。 |
| Task D 数据量 | 309/570、未完成 | **570/570；2,280 states；13,680 candidates** | 旧状态行已过期。 |
| Path-LL 与 task reward | 常被隐式当作同一效用 | A/B/C/E Pearson 分别 **+.0957/−.0174/+.0493/+.2246** | 两种 reward 相关但不等价；必须分开陈述。 |
| “within-prompt 为负，所以没有 V 信号” | 由 global probe 的 centered score 推断 | 直接对 centered target 拟合后 hidden ΔR²≈0，且 target signal variance=0.01285 | 这是一个真实的 within-prompt 边界；强状态级结果主要来自跨 prompt/trajectory 的未来效用差异。 |
| SmolLM2 AR→DLM | 可作为额外 backbone | 转换模型未通过生成/任务资格门 | 不进入主 claim。 |

## 为什么早期总 KILL 不再是最终结论

早期 KILL 混合了三个不同命题：状态价值 (V)、Path-LL 候选优势 (A)、任务奖励候选优势 (A_{task})。后续无泄漏拆分、验证集选层、直接 state-centered 拟合、50 次重复文档切分、Nadeau–Bengio 修正、聚类 bootstrap 和独立数据集扩展表明：前两个命题有正证据，第三个命题仍是边界/负结果。因此最终结论不是“全部成功”，也不是“全部失败”，而是一个更强、更精确的分层发现。

## 来源优先级

1. 原始 NPZ/JSON 的本次独立复算；
2. `data/v_audit.json`、`data/estimate_all.json`、五折/封存 JSON；
3. 最新 `docs/EXPERIMENT_LEDGER.md` 条目；
4. 旧报告与 Claude snapshot 只用于追溯，不覆盖更新后的可复算数字。
