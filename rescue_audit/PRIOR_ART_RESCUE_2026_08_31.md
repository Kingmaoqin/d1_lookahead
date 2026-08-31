# 先行研究重搜（rescue 轮）— 2026-08-31

检索日期：2026-08-31。检索手段：Web 搜索（arXiv / OpenReview / HuggingFace papers）。
上一次重搜见 `docs/PRIOR_ART_UPDATE.md`（2026-08-19）；本文件只记录**新增**与
**对本轮 rescue 假设有直接影响**的条目。

---

## 1. 本研究要问的那句话，是否已被人做过？

> 是否已经有人直接研究 **frozen DLM hidden representation 中
> rollout-defined action advantage 的 nonlinear / relational decodability**？

**检索结论：没有找到直接撞车的工作。** 最接近的三类分别缺一块：

| 工作 | 做了什么 | 缺哪一块 |
|---|---|---|
| POKE / LookUM / AdaLook | 在推理时**实际做 lookahead** 来挑揭示顺序 | 不问"这个量是否已在表征里"，只问"搜出来有没有用" |
| TraceLock（The Path Matters） | 在 **frozen D-LLM** 的 hidden states + trace 上学一个 commitment 控制器 | 学的是**控制器**（有容量、有训练），不是"线性可读性"的表征结论；且目标是 commit/revisable 的二元决策，不是 rollout 定义的 A |
| Frozen VLA 探测研究（2605.28527） | 结构与本研究**高度同构**：decodability → matched-pair → online selection，冻结骨干 + 轻量探针 | (a) 机器人 VLA，不是 DLM；(b) **只用线性探针**；(c) 目标是 success/value，**不区分 state value 与 action advantage** |

因此本研究的两个定位仍然成立，并且经这一轮 rescue 后更锋利：

1. **V 与 A 的分离**是本研究的独有轴。Frozen-VLA 研究把 "encodes information
   about success" 当成整体结论；本研究证明了在 DLM 上这个整体结论会**误导**——
   可读的是 level（V / 题目难度），不可读的是 differential（A / 轨迹内波动）。
2. **relational / 非线性口径**是本轮新增。前人（含 frozen-VLA）都是线性探针，
   没有人排除过 `h_i^T W h_g` 这类状态调制型编码。

---

## 2. 新增条目（相对 08-19 那次重搜）

### 2.1 直接相关的解码方法

| 编号 | 标题 | 与本研究的关系 |
|---|---|---|
| arXiv 2602.03496 | *Lookahead Path Likelihood Optimization for Diffusion LLMs*（POKE / POKE-SMC） | 本研究代理奖励 Path-LL 的出处。POKE-SMC 与 LookUM 在同一 setting 下对比（44.3 → 45.3） |
| arXiv 2511.05563 / OpenReview `SVI1ZnmFmx` | *Lookahead Unmasking Elicits Accurate Decoding in DLMs*（LookUM） | 把采样重构为**路径选择**：path generator + verifier + importance sampling。其开销来自"每步评估 k 条候选路径" —— 正是本研究想问能否省掉的那部分 |
| arXiv 2607.15655 | *Adaptive Multi-Step Lookahead Decoding for DLMs*（AdaLook） | **新**。多步前瞻，明确承认"每步需要批量前向来评估假设"带来开销。若本研究的 A 可读性成立，AdaLook 的开销可被摊销；本轮结果**不支持**这一点 |
| arXiv 2512.09106 | *Learning Unmasking Policies for Diffusion Language Models* | 把掩码扩散采样形式化为 MDP，用轻量策略（单层 Transformer）做 RL。**注意**：它训练策略，属于本研究 kill 后被禁止的"加容量"路线 |
| arXiv 2605.24697 | *The Path Matters: Learning a Token-Commitment Policy for DLMs*（TraceLock） | **新**。frozen-D-LLM-only 设定，从 hidden states + trace 学 commitment 策略。本研究的 C2 轨迹稳定性控制特征即 TraceLock 式 |
| arXiv 2606.23567 | *Scheduling Thoughts: Learning the Order of Thought in DLMs*（SAS） | **新**。同样是"揭示顺序"这个自由度 |

### 2.2 探测方法学（本轮新增假设的先例）

| 编号 | 标题 | 关系 |
|---|---|---|
| arXiv 2605.28527 | *What Frozen VLAs Already Know About Success* | **最接近的方法学同构**。三段式（decodability / matched-pair / online selection）与本研究几乎一一对应。matched-pair 下 Pi0.5 达 ~92% 成对排序准确率、shuffled 对照在随机水平；在线选择在 push-plate 上 26.7%→44.3%。**只用线性探针，不区分 V 与 A** |
| arXiv 2606.02907 | *Linear Probes Detect Task Format, Not Reasoning Mode in LM Hidden States* | **对本研究 08-30 结论的独立佐证**：线性探针容易读出"任务表面属性"而非"过程状态"。与本研究"读到的是题目难度而非前瞻"高度一致 |
| arXiv 2509.21993 | *Bilinear representation mitigates reversal curse and enables consistent model editing* | **双线性表征的先例**。支持 P2 假设在方法学上不是杜撰 |
| arXiv 2406.13184 | *Locating and Extracting Relational Concepts in LLMs* | 关系型概念在 hidden state 中的定位；支持 P4 关系特征块 |
| arXiv 2509.25260 | *Internal Planning in LMs: Characterizing Horizon and Branch Awareness* | **直接相关**。问 LM 内部是否有 horizon / branch 意识 —— 与本研究 §12 的 horizon 扫描、§18 的 layer×timestep×horizon 三维分析同题 |
| arXiv 2606.00091 | *DLLM-JEPA: Joint Embedding Predictive Architectures for Masked DLMs* | **新**。JEPA 式联合嵌入预测；若 DLM 表征本就不编码未来 token 级差分，JEPA 训练目标是一条可能的补救路线（超出本研究 kill 后的容量禁令） |

### 2.3 检索到但判定**不相关**

- "Unsure but Certain"：检索只返回词典条目，**未找到同名论文**。任务书列出的
  这个条目可能是笔误或非公开工作。记录在案：**未能确认其存在**。
- "Probing Functional Correctness in Diffusion Language Models"：未检索到同名
  论文；返回的都是 HumanEval/MBPP pass@1 的常规评测描述。**未能确认其存在**。

---

## 3. 对本轮实验设计的影响

1. **AdaLook（2607.15655）的存在提高了本研究否定结果的价值**：又一篇工作在
   为"每步批量前向评估候选"付费。如果这笔钱本可以省掉，值得知道；如果省不掉，
   同样值得知道——后者正是本研究要给的答案。
2. **Frozen-VLA 论文（2605.28527）必须在最终报告里正面引用与对比**。它是
   "冻结表征里有 value-like 结构"的正面证据，而本研究给出的是同一框架下的
   **分解**结论：level 可读、differential 不可读。两者不矛盾，但本研究的分解
   使前者的"已经知道成功与否"这一说法在动作选择上不可直接套用。
3. **2606.02907（linear probes detect task format, not reasoning mode）**
   与本研究 08-30 的"读到的是题目难度而非前瞻"是同一现象在两个模态上的表现，
   应在讨论中并列。
4. 没有发现任何工作**否证**本轮 relational 假设，也没有工作**证实**它。
   因此 P2/P3/P4 仍然是有价值的探索，其零结果也是新的。
