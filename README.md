#  Cognitive-Nano-Qwen: System 2 慢思考推理架构

[![Model Size](https://img.shields.io/badge/Model-0.6B-blue?style=flat-square&logo=alibabacloud)](#)
[![Algorithm](https://img.shields.io/badge/RL-GRPO%20%7C%20MCTS-red?style=flat-square)](#)
[![Weights & Biases](https://img.shields.io/badge/Tracked_by-W&B-orange?style=flat-square&logo=weightsandbiases)](https://wandb.ai/yxf827206205-shandong-university/cognitive-nano-qwen/reports/Cognitive-Nano-Qwen-System-2---VmlldzoxNjA0NDI4OQ)

> **Scaling inference compute, not just model parameters** > 项目致力于探索极小规模 LLM 的推理极限：基于仅 **0.6B参数**的 Qwen_Base，从零构建 **System 2**推理架构。通过PRM、完全隔离的物理沙箱与MCTS的结合，让纳米级模型在数学推理任务上实现突破。
>
> 查看详细训练数据:[![Weights & Biases](https://img.shields.io/badge/Tracked_by-W&B-orange?style=flat-square&logo=weightsandbiases)](https://wandb.ai/yxf827206205-shandong-university/cognitive-nano-qwen/reports/Cognitive-Nano-Qwen-System-2---VmlldzoxNjA0NDI4OQ)

##  核心技术

在传统的 Scaling Law 框架下，0.6B 模型往往被认为缺乏进行复杂多步逻辑推理的容量。本项目实现在小模型上实现多步推导：
- **越级推理能力**：在完全依赖自身生成的代码沙箱反馈下，0.6B 模型在 GSM8K 测试集上实现了惊人的 **% Pass@1**。
- **System 2 算力换准确率**：引入 MCTS 搜索树与独立训练的 Value Head 后，**Pass@4 命中率飙升至 **，验证了“Test-time Compute”在极小模型上的可行性。
- **高鲁棒的判别大脑**：成功解决 PRM 训练中的冷坍缩难题，使价值头能够精准识别沙箱级错误。

---

##  核心架构与训练流水线 

### Phase 1 — SFT 工具学习
**目标**：为基座模型注入元认知能力与严格的工具调用协议。
**实现**：通过 Deepseek蒸馏出符合格式的数据, 在数据上进行LoRA微调，对 Qwen-0.6B 进行指令约束。强制模型学会在 `<think>` 与 `</think>` 标签内展开CoT，并严格遵循 `<|python_start|>` 和 `<|python_end|>` 的定界符格式来调用外部代码沙箱。

---

### Phase 2 — GRPO 强化学习 + 沙箱交互
**机制**：采用更为轻量高效的 **GRPO**。
引入了一个真实的、物理隔离的 Python 沙箱环境。当模型吐出代码后，外部拦截器会立刻执行代码，并将 `stdout`/`stderr` 的结果通过 `<|output_start|>` 标签喂回给模型。
训练全程采用**稀疏的结果导向奖励**，只要整条执行轨迹的最终答案与 Ground Truth 匹配，该轨迹的所有动作都将获得正向 Advantage。
**结果**：在此机制下，0.6B 模型的逻辑自洽性得到了提升，提高了模型的推理能力。

---

### Phase 3 — PRM / Value Head
这是构建 MCTS 裁判的关键阶段。通过一系列方法彻底解决了稀疏奖励下过程价值评估的难题：

#### 数据切片策略 
采用基于正则表达式的语义级切分。模型严格沿 `\n\n`（自然段落）或 `<|output_end|>`进行切分。保证了喂给 Value Head 的每一个 State都包含一个**完整的逻辑推理节点**，消除了截断带来的语义破坏。

#### Credit Assignment 设计 
在极长的推理链中，线性的奖励分配会导致早期状态的价值极度弥散。本项目在打分策略中引入了 $ratio^{0.5}$ 平滑策略。这意味着即使是开局走对的第一步（例如总步数10步中的第1步），也能获得 0.658 的强烈正向激励（而非线性的 0.55）。这极大地拔高了裁判大脑在推理早期的路径判别力。

#### 冷坍缩问题解决 
**痛点**：在 PRM 训练初期，模型极度悲观，倾向于把 80% 的中间状态打为低分，导致梯度消失与冷坍缩。
**解法**：在数据加载侧引入 `WeightedRandomSampler`，将连续的胜率 Label 强制切分为 4 个数据桶（0~0.25, 0.25~0.5, 0.5~0.75, 0.75~1.0），并在 DataLoader 中进行**强行重采样对齐（1:1:1:1 均衡分布）**。

---

### Phase 4 — System-2 推理搜索
**机制**：将 Phase 2 训练出的 Actor与 Phase 3 训练出的 Value_head进行缝合。在推理阶段（Inference），模型启动 MCTS。
裁判大脑实时对 Actor 生成的每一个候选步骤进行打分：遇到低置信度分支（如 `score < 0.1`）立即剪枝并回溯；遇到高置信度分支（如 `score > 0.8`）则继续深入调用沙箱。真正实现了用“Test-time Compute”换取准确率。

---

#  核心点


- **极小模型 System-2 推理验证**：在不足 1B 参数的微型模型上，完整的“思考-执行-判别-搜索”闭环是完全可行的。
- **Sandbox-Driven RL 架构**：验证了“代码沙箱执行反馈 + GRPO Outcome Reward”能够内生出高质量的复杂逻辑链，无需依赖昂贵的外部人工标注过程奖励。
- **推理算力换准确率范式**：在极小算力设备上复刻了 o1 模型的 Scaling Inference Compute 核心思想。
---

#  实验结果

在 GSM8K标准测试集上的评测结果如下：

| 模型架构 | 参数规模 | 训练阶段 | Pass@1 | Pass@4 (MCTS 搜索) |
| :--- | :---: | :--- | :---: | :---: |
| Qwen2.5-Base (Zero-Shot) | 0.5B | Pretrain | ~21% | - |
| Cognitive-Nano-Qwen (SFT) | 0.6B | Phase 1 | 42% | - |
| Cognitive-Nano-Qwen (GRPO) | 0.6B | Phase 2 | **63%** | - |
| **Cognitive-Nano-Qwen (System 2)** | **0.6B** | **Phase 1~4** | - | **95.2%** |


---

#  方法

本项目以极低的算力成本（单卡 4090）重现了前沿推理模型的演进路径。