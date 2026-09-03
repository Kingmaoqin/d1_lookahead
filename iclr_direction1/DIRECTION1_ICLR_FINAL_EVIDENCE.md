# Direction One — ICLR 2027 最终证据定稿

版本：2026-09-02。论文数字的唯一 authoritative source 是
[`DIRECTION1_ICLR_MASTER_RESULTS.csv`](DIRECTION1_ICLR_MASTER_RESULTS.csv)。

## 1. Executive conclusion

**Direction One 已经得到明确、可写入 ICLR 的正发现。** 冻结 diffusion language model 的隐藏表征编码了 exposed confidence 与廉价输出统计之外的 rollout-defined future utility。证据最强的是状态级 (V^pi(s))：Nemotron-Diffusion-3B 上 hidden-only 相对 cheap baseline 的增益为 **ΔR²=+0.1433 [ +0.0847,+0.2042 ]、ΔAUC=+0.0666 [ +0.0378,+0.0982 ]**，并在整个去噪轨迹保持稳定。最新 state-centered 分析还在 MDLM 与 SEDD、两种 policy、两个 Path-LL target 上一致检测到更小的候选级差分信号。

因此论文不应以“rescue failed”或旧 KILL 开场。准确主张是：**strong global future-utility representations coexist with smaller, reward-dependent local action signals**。任务奖励的候选级 advantage 尚未稳定确认，这是 claim 的边界，不是对 Direction One 的推翻。

## 2. Final scientific thesis

对固定参考解码策略 π、部分去噪状态 (s)、候选动作 (a) 与终局回报 (R)：

\[
V^\pi(s)=\mathbb E_{\tau\sim\pi(\cdot\mid s)}[R(\tau)],\qquad
Q^\pi(s,a)=\mathbb E_{\tau\sim\pi(\cdot\mid s,a)}[R(\tau)],
\]

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

最终数据支持两层结论：

- **强支持 (V)**：冻结 DLM state 对 rollout-defined task correctness 有超越 cheap statistics 的线性可读信息。
- **支持较小的 Path-LL (A)**：同一 state 内，候选 token/position 的未来 Path-LL 差异可由局部 hidden state 增量读出，跨 MDLM/SEDD 与 policy 重复。
- **不支持普遍的 task-reward (A)**：更严格的 candidate task-utility cross-fit 与 frozen transfer 未确认稳定增益。

这不是“隐藏状态通常比置信度好”的泛化陈述，而是对 rollout-defined (V/Q/A) 的可测分解。

## 3. Models and tasks

| 模型/数据 | 角色 | 规模与形式 | 是否 headline |
|---|---|---|---|
| MDLM | OpenWebText、ancestral/confidence policy、Path-LL candidate advantage | 169.6M，masked diffusion，非 instruction-tuned 实验骨干 | 是，candidate-level |
| SEDD | OpenWebText、ancestral policy、独立 diffusion parameterization | 169.6M，score-entropy discrete diffusion | 是，candidate-level replication |
| Nemotron-Diffusion-3B | GSM8K/SVAMP，state value、task reward、selective generation | 3B，block diffusion | 是，state-level headline 与 task boundary |
| SmolLM2-1.7B-Instruct 转换体 | AR→diffusion 可行性尝试 | 1.7B；转换后未通过生成与任务资格门 | 否；Do not cite as evidence |

任务 A/B/C/D 是互相区分的 GSM8K 批次；E 是 SVAMP。C 为 180 problems/720 states/4,320 candidates，D 为 570/2,280/13,680，E 为 200/800/4,800。`vreadA+B` 覆盖 800 个不重叠 GSM8K prompts 与 4,411 states。OpenWebText Path-LL 每个 arm 有 400 documents、2,400 states、14,400 candidates，(K=24,H=16)。本仓库没有通过资格门、可用于 Direction One 主张的 code-task 实验，不能暗示已有 code 泛化证据。

## 4. Evidence chronology

1. 初始 linear readout 显示隐藏表征对未来 rollout value 有增量信息。
2. 审计发现污染风险、索引缺陷、弱基线和 test-aware selection 风险；相关旧结论被降级。
3. 将 (V) 与 (A)、Path-LL 与 task utility 分开，避免一个负结果覆盖所有命题。
4. nonlinear rescue 显示信号不是完全线性假象，但复杂模型容易过拟合，故最终主表优先线性、可审计估计。
5. 直接 state-centered fitting 与验证集固定层后，candidate Path-LL 在 50 次文档切分中稳定为正。
6. Task C/D/E 扩样、五折 OOF 与 frozen transfer 关闭了 candidate task-utility 的夸大空间。
7. Nemotron 原始特征独立复算确认状态级 headline、六时点稳定性和 selective-generation 后果。

## 5. Headline results

### 5.1 状态级 future utility

| 指标 | Cheap | Hidden-only | Δ | 95% CI |
|---|---:|---:|---:|---:|
| R² | 0.3380 | 0.4812 | **+0.1433** | **[+0.0847,+0.2042]** |
| AUC | 0.8229 | 0.8896 | **+0.0666** | **[+0.0378,+0.0982]** |

`cheap+hidden` 的旧数值 ΔR²=+0.1317、ΔAUC=+0.0655 仍然有效，但只是 two-block ablation；hidden-only 是 validation-selected authoritative headline。

### 5.2 Candidate Path-LL 的 50-split state-centered 结果

| Backbone / policy | Target | Δwithin-R² | NB-corrected 95% CI | 正切分 |
|---|---|---:|---:|---:|
| MDLM / ancestral | A_pertok | **+.00553** | [.00331,.00775] | 50/50 |
| MDLM / ancestral | A_future | **+.00997** | [.00574,.01420] | 49/50 |
| MDLM / confidence | A_pertok | **+.01045** | [.00733,.01358] | 50/50 |
| MDLM / confidence | A_future | **+.01902** | [.01354,.02449] | 50/50 |
| SEDD / ancestral | A_pertok | **+.01143** | [.00801,.01486] | 50/50 |
| SEDD / ancestral | A_future | **+.02319** | [.01737,.02900] | 50/50 |

六个条件的 Δwithin-R² 区间全部排除 0；这足以自信写作“small but consistent candidate-level Path-LL differential”。Pairwise concordance 更弱且部分区间跨 0，必须作为次要指标。

## 6. State-level future utility

Nemotron headline 用 480/120/200 个 prompt 的 train/validation/test 文档级划分，层 18 与 pooling 在 validation 上确定，(K=8) completion rollouts 构造 (V_{reward})。隐藏 probe 达 R² .4812 和 AUC .8896，分别解释估计 noise ceiling 的 48.5%，并显著超过输出熵、置信度、mask/progress 等 cheap features。

跨实验的状态级 ΔAUC 综合估计为 fixed **+.0480 [.0268,.0692]**、random-effects **+.0463 [.0191,.0735]**（Nemotron 主实验 + 较小的 GSM8K-C 与 SVAMP-E readouts）。后两项单独的 NB-corrected CI 因样本量小而跨 0，但方向与 split sign consistency 一致；综合只作 supporting synthesis，不替代 3B 主实验。

直接将 features 与 target 都按 prompt 中心化后，hidden ΔR²≈0。该边界说明 headline 主要捕获 prompt/trajectory 间的未来成功差异，而不是同一 prompt 内每一细微阶段变化；这限定了表示几何，却不削弱 held-out prompt 上的预测事实。

## 7. Candidate-level future utility

Candidate 主检验比较同一 state 内 cheap-only 与 `cheap+h_local(3 validation-fixed layers)`，因此移除了 state-level 难度捷径。MDLM ancestral、MDLM confidence 和 SEDD ancestral 上，两个 Path-LL targets 的 50-split Δwithin-R² 均为正，并使用 Nadeau–Bengio 修正处理重复 train/test split 的依赖。

旧 pooled training 优先拟合占方差更大的 between-state 难度；即使它预测整体 target 很好，也可能忽略同一 state 的六个候选之间更小、真正与 action choice 有关的残差。修正后的做法在 train documents 内同时中心化 target 与 features，再直接优化 within-state estimand，所以它回答的是旧分析没有高功效回答的问题。这是 estimator 的纠正，不是从同一失败检验中挑一个好看的子群。

结论应定为“Path-LL action differential is linearly detectable”，而不是“所有 task-optimal action 都已可读”。在 task correctness reward 上，合并 GSM8K C+D 五折 OOF 的 bilinear-vs-cheap concordance 为 **+.0143 [−.0091,+.0369]**，SVAMP E 为 **+.0133 [−.0369,+.0654]**；均未排除 0。C→D frozen transfer 甚至为 **−.0356 [−.0567,−.0151]**。这表明 reward definition 与 domain transfer 是实质问题。

## 8. Temporal evidence

| 去噪进度 | n prompts | ΔR² [95% CI] | ΔAUC [95% CI] |
|---:|---:|---:|---:|
| 10.0% | 800 | +.1171 [.0267,.2170] | +.0643 [.0242,.1091] |
| 25.6% | 800 | +.0769 [.0045,.1547] | +.0506 [.0186,.0857] |
| 40.0% | 800 | +.1303 [.0554,.2090] | +.0511 [.0127,.0932] |
| 55.6% | 800 | +.1284 [.0679,.1891] | +.0538 [.0243,.0858] |
| 71.2% | 770 | +.0774 [.0108,.1418] | +.0300 [−.0057,.0668] |
| 85.6% | 441 | +.1320 [.0724,.1935] | +.0386 [.0092,.0747] |

ΔR² 为 **6/6** 时点区间排除 0；ΔAUC 为 **5/6**。信号不是单一 checkpoint 偶然命中。

## 9. Practical consequence

在 40% generation progress 做离线 abort simulation 时：

| Coverage（保留继续生成的 prompts） | Cheap acc. | Hidden acc. | 增益 [95% CI] | 账面 compute saved |
|---:|---:|---:|---:|---:|
| 70% | .7487 | .7723 | **+.0236 [.0025,.0423]** | 18% |
| 50% | .8078 | .8691 | **+.0612 [.0247,.0926]** | 30% |
| 30% | .8207 | .9001 | **+.0794 [.0299,.1368]** | 42% |

AURC 增益为 **+.0464 [.0171,.0812]**。Coverage 是“继续生成的比例”，accuracy 是被保留集合上的终局正确率；compute saved 使用 ((1-coverage)\times(1-0.4)) 的明确账面约定。它证明 practical potential，但仍是离线 simulation，不得称为已经部署的 wall-clock speedup。

## 10. Negative results that sharpen the paper

- Candidate effect 显著小于 state effect；concordance 也比 within-R² 不稳定。
- Candidate task utility：C+D +.0143 与 E +.0133 的 CI 均跨 0。
- Bilinear state-action interaction 没有稳定超过同容量 no-interaction baseline。
- C→D 零适配 frozen mapping 显著失败，说明表示到 decision rule 的迁移不是自动的。
- Nemotron prompt-centered state trajectory 的增量 hidden R²≈0。
- SmolLM2 conversion 未通过资格门；不存在可主张的第四合格 DLM 或 code-task 结果。

这些结果共同支持更精确的论文，而不是削弱主结果：模型有强 global future-value representation，但 local action extraction 更小、更依赖 reward 与 domain。

## 11. Final claim matrix

| Proposed claim | Evidence | Models/tasks | Confidence | Use in paper |
|---|---|---|---|---|
| Frozen DLM states expose rollout-defined utility beyond cheap statistics | ΔR² +.1433、ΔAUC +.0666，CI 均排除 0 | Nemotron-3B / GSM8K | **VERY HIGH** | Abstract, Figure 1 |
| State-value readout persists through denoising | 6/6 ΔR²、5/6 ΔAUC intervals exclude 0 | Nemotron-3B / GSM8K | **HIGH** | Figure 2 |
| State readout improves selective ranking | +.0236/+.0612/+.0794 retained-set accuracy at 70/50/30% coverage | Nemotron-3B / GSM8K | **HIGH**（offline） | Figure 4 |
| Candidate Path-LL differential is detectable within state | 六条件 Δwithin-R² 全正且 corrected CI 排除 0 | MDLM, SEDD / OWT | **HIGH** | Figure 3, Table 2 |
| Candidate task-correctness advantage is generally detectable | C+D/E intervals cross 0 | Nemotron-3B / GSM8K, SVAMP | **EXPLORATORY / unresolved** | Limitation |
| Frozen candidate rule transfers without adaptation | C→D −.0356 [−.0567,−.0151] | Nemotron-3B / GSM8K | **HIGH negative** | Boundary result |
| Hidden readout replaces explicit lookahead end-to-end | No online decoder experiment | — | **EXPLORATORY** | Do not claim |

## 12. ICLR paper positioning and updated prior art

截至 2026-09-02，最接近的工作分成三类：

1. **显式 lookahead/search**：POKE 以未来 Path-LL 指导 SMC 搜索并报告 reasoning gains；LookUM、AdaLook 与 Ripple-Pivot Search 也在 inference time 显式探索未来路径。它们证明 lookahead 有价值；Direction One 问的是这部分未来价值能否从 frozen hidden state 被摊销读出。[POKE](https://arxiv.org/abs/2602.03496)、[LookUM](https://arxiv.org/abs/2511.05563)、[AdaLook](https://arxiv.org/abs/2607.15655)、[Ripple-Pivot Search](https://arxiv.org/abs/2608.11742)。
2. **学习 unmasking/commitment policy**：Learning Unmasking Policies 从 confidence 学 policy；TraceLock 用 future stability 自监督训练 controller；SAS 用 pathwise log-likelihood bound 的 dense reward 和 GRPO 学 order policy。它们接近 practical endpoint，但 target/方法不是 frozen representation 中 rollout-defined (V/Q/A) 的受控 probing decomposition。[Learning Unmasking Policies](https://arxiv.org/abs/2512.09106)、[TraceLock](https://arxiv.org/abs/2605.24697)、[SAS](https://arxiv.org/abs/2606.23567)。
3. **hidden-state correctness/dependency probes**：ACL 2026 的 DLM functional-correctness probing 与 Unsure but Certain 已直接占据“hidden beats confidence for correctness”这一宽泛主张；pairwise-MI estimator 也已证明 frozen MDM hidden states 支持有用的一次 readout。[Probing Functional Correctness in DLMs](https://aclanthology.org/2026.acl-srw.15/)、[Unsure but Certain](https://arxiv.org/abs/2608.08791)、[Neural Pairwise MI Estimation](https://arxiv.org/abs/2605.20187)。

**没有发现精确碰撞。** 可守住的新颖性是：对 frozen DLM 做 rollout-defined (V/Q/A) 分解，证明 state utility 超过 exposed output statistics，并用严格 within-state controls 揭示更小的 candidate Path-LL differential；不是泛泛声称 hidden state 有 correctness information。

### Recommended title candidates

1. **Amortizing Lookahead: Frozen Diffusion Language Models Encode Rollout-Defined Future Utility**
2. **Do Diffusion Language Models Know Their Future Value? Probing State and Action Utility**
3. **Future Utility in Diffusion Language Model Representations**
4. **From Lookahead to Readout: State and Action Values in Frozen Diffusion Language Models**
5. **Beyond Confidence: Rollout-Defined Value Signals in Diffusion Language Models**

### Recommended one-sentence thesis

**Frozen diffusion language models linearly expose rollout-defined future state utility beyond output statistics and a smaller but consistent Path-LL action differential, creating a principled route to amortize explicit lookahead.**

### Recommended abstract-level contributions

- 提出 DLM decoding 的 rollout-defined V/Q/A probing framework，并将 global state utility 与 within-state action differential 分离。
- 在 Nemotron-Diffusion-3B 上建立强、无 test-selection 的状态级增量 readout，并证明六时点稳定性。
- 在 MDLM/SEDD、两种 policy 与两个 Path-LL targets 上建立 50-split state-centered candidate replication。
- 展示 selective-generation potential，并用 task-reward cross-fit/frozen-transfer negatives 明确适用边界。

### Figure and table plan

- **Figure 1**：V/Q/A 概念图 + Nemotron headline R²/AUC。
- **Figure 2**：六时点（主）与 layer profile（辅）的 state-value decodability。
- **Figure 3**：六个 candidate Path-LL 条件的 Δwithin-R² 与 corrected CI；concordance 放次 panel。
- **Figure 4**：risk/accuracy–coverage 曲线与显式 compute accounting。
- **Table 1**：model × task × target × policy × sample-size matrix。
- **Table 2**：所有主定量结果、统计方法、status；直接由 master CSV 生成。

## 13. What NOT to claim

- 不说“hidden states contain more than confidence”是首次发现；已有直接 prior art。
- 不说 candidate task-reward (A) 已稳定成立或 controller 已跨域迁移。
- 不把 Path-LL 等同于 downstream task correctness；原始复算相关性 A/B/C/E 为 +.0957/−.0174/+.0493/+.2246。
- 不把 selective simulation 写成真实 wall-clock 加速。
- 不说覆盖所有 DLM、所有规模、所有任务；目前 headline 是三种合格骨干、OpenWebText/GSM8K/SVAMP。
- 不将 repeated splits 当 50 个独立数据集；主表明确使用 NB correction。

## 14. Remaining ICLR-critical gaps

[ICLR 2027 官方征稿页](https://www.iclr.cc/Conferences/2027/CallForPapers)列出的 abstract/full-paper deadlines 是 2026-09-18 / 2026-09-25（AOE），因此只保留会实质改变 acceptance probability 的项目：

1. **MUST**：在事先锁定层、feature set、estimand 和统计方法后，用独立 prompts 或新合格 DLM 做一次 candidate Path-LL confirmatory replication；当前六条件很强，但来自同一批设计家族。
2. **MUST**：将 state-value probe 接入真实 decoder，报告 probe overhead、wall-clock、accuracy/compute Pareto；当前 selective result 是离线模拟。
3. **HIGH**：把 GSM8K-C 与 SVAMP-E 状态级样本扩至接近 800 prompts，使单数据集 corrected CI 有能力独立排除 0。
4. **HIGH**：预注册一种 task-grounded candidate reward 与强同容量 baseline；当前结果只能说明现有 (A_{task}) formulation 未确认。
5. **OPTIONAL**：增加 code task 或 instruction-tuned full-diffusion backbone，以拓宽而不是建立核心 claim。

## Superseded result ledger

| Old value/status | New value/status | Why changed | Current authoritative value |
|---|---|---|---|
| ΔR² +.1317 | ΔR² +.1433 | two-block `cheap+hidden` 与 validation-selected `hidden-only` 是不同 probe | **+.1433 [.0847,.2042]** headline；+.1317 为消融 |
| ΔAUC +.0655 | ΔAUC +.0666 | 同上 | **+.0666 [.0378,.0982]** headline |
| Candidate weak/non-significant | 六条件 Δwithin-R² 显著为正 | 旧 pooled objective 被 between-state variance 主导；新分析直接拟合 centered estimand | 六行 50-split NB-corrected 结果 |
| Task C +.0381 candidate signal | C+D +.0143，CI 跨 0 | 五折、扩样和强同容量基线推翻探索性大小 | **未确认** |
| Task D 309/570 未完成 | 570/570 完成 | 后续采集完成 | 570 problems / 2,280 states / 13,680 candidates |
| 早期总 KILL | 分层结论 | 旧门将 Path-LL A、task A、state V 混为一个命题 | V headline；Path-LL A strong support；task A boundary |

## Reproducibility and provenance

- 自动重建：`python iclr_direction1/build_iclr_evidence.py`
- Nemotron 独立审计日志：`VREAD_INDEPENDENT_RECOMPUTE.log`
- 原始 task/Path-LL 对齐复算：`TASK_REWARD_ALIGNMENT_RECOMPUTE.json`
- 全部候选池：`_ALL_RESULT_CANDIDATES.csv`
- 冲突裁决：`OLD_VS_NEW_EVIDENCE.md`
- 文件清单与 Git 起点：`FILE_INVENTORY.tsv`、`GIT_PROVENANCE.txt`
- Claude 历史报告只作审计快照：`CLAUDE_PRIOR_REPORT_SNAPSHOT.md`
