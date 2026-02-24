"""
config.py — 所有超参数的单一数据源
"""
from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    # ── 路径 ──────────────────────────────────────────────────────────────
    base_model_path: str  = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
    sft_lora_path:   str  = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool——v3/final_lora"
    output_dir:      str  = "/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1"

    # ── GRPO 核心超参 ──────────────────────────────────────────────────────
    group_size:      int   = 6     # G：每个 Prompt 采样多少条回答
    kl_coef:         float = 0.04    # β：KL 惩罚系数
    clip_eps:        float = 0.2     # PPO 风格的 ratio clip 范围

    # ── 生成超参 ───────────────────────────────────────────────────────────
    # 注意：max_segment_tokens 是两次工具调用之间单段的最大长度
    # max_total_tokens 是整条轨迹（含注入文本）的硬上限
    max_segment_tokens: int   = 512
    max_total_tokens:   int   = 1024
    temperature:        float = 0.6
    top_p:              float = 0.9

    # ── 训练超参 ───────────────────────────────────────────────────────────
    learning_rate:   float = 5e-6
    num_epochs:      int   = 1
    batch_size:      int   = 4       # 每个 grad step 处理的 Prompt 数
    grad_accum:      int   = 2       # 梯度累积步数
    max_grad_norm:   float = 1.0
    warmup_ratio:    float = 0.05
    save_steps:      int   = 200

    # ── 奖励权重 ───────────────────────────────────────────────────────────
    reward_correct:  float = 1.0    # 最终答案正确
    reward_tool_use: float = 0.1    # 使用了工具
    reward_format:   float = 0.05   # 有 <think> 块

    # ── LoRA（叠加在合并后的 SFT 权重之上）──────────────────────────────
    lora_r:          int   = 16
    lora_alpha:      int   = 32
    lora_dropout:    float = 0.05

    # ── 杂项 ───────────────────────────────────────────────────────────────
    seed:               int = 42
    device:             str = "cuda"
    wandb_project:      str = "cognitive-nano-qwen"
    wandb_run_name:     str = "grpo-tool-v1"
    num_train_samples:  int = 1000

    # ── 特殊 Token 字面量（与 SFT 阶段一致）──────────────────────────────
    PYTHON_START: str = field(default="<|python_start|>", init=False, repr=False)
    PYTHON_END:   str = field(default="<|python_end|>",   init=False, repr=False)
    OUTPUT_START: str = field(default="<|output_start|>", init=False, repr=False)
    OUTPUT_END:   str = field(default="<|output_end|>",   init=False, repr=False)
