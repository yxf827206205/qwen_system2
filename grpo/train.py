"""
train.py — 入口脚本

用法示例：
  # 默认配置
  python train.py

  # 自定义关键参数
  python train.py \
    --base_model_path /path/to/Qwen3-0.6B-Base \
    --sft_lora_path   /path/to/sft_lora \
    --output_dir      /path/to/output \
    --group_size 8 \
    --kl_coef 0.02 \
    --num_train_samples 2000
"""
import argparse
import torch
from config  import GRPOConfig
from trainer import GRPOToolTrainer


def parse_args() -> GRPOConfig:
    defaults = GRPOConfig()
    parser   = argparse.ArgumentParser(description="GRPO Tool Training for Qwen")

    # 路径
    parser.add_argument("--base_model_path",    type=str,   default=defaults.base_model_path)
    parser.add_argument("--sft_lora_path",       type=str,   default=defaults.sft_lora_path)
    parser.add_argument("--output_dir",          type=str,   default=defaults.output_dir)

    # GRPO
    parser.add_argument("--group_size",          type=int,   default=defaults.group_size)
    parser.add_argument("--kl_coef",             type=float, default=defaults.kl_coef)
    parser.add_argument("--clip_eps",            type=float, default=defaults.clip_eps)

    # 生成
    parser.add_argument("--max_segment_tokens",  type=int,   default=defaults.max_segment_tokens)
    parser.add_argument("--max_total_tokens",    type=int,   default=defaults.max_total_tokens)
    parser.add_argument("--temperature",         type=float, default=defaults.temperature)
    parser.add_argument("--top_p",               type=float, default=defaults.top_p)

    # 训练
    parser.add_argument("--learning_rate",       type=float, default=defaults.learning_rate)
    parser.add_argument("--num_epochs",          type=int,   default=defaults.num_epochs)
    parser.add_argument("--batch_size",          type=int,   default=defaults.batch_size)
    parser.add_argument("--grad_accum",          type=int,   default=defaults.grad_accum)
    parser.add_argument("--max_grad_norm",       type=float, default=defaults.max_grad_norm)
    parser.add_argument("--save_steps",          type=int,   default=defaults.save_steps)

    # LoRA
    parser.add_argument("--lora_r",              type=int,   default=defaults.lora_r)
    parser.add_argument("--lora_alpha",          type=int,   default=defaults.lora_alpha)

    # 数据 / 杂项
    parser.add_argument("--num_train_samples",   type=int,   default=defaults.num_train_samples)
    parser.add_argument("--seed",                type=int,   default=defaults.seed)
    parser.add_argument("--wandb_project",       type=str,   default=defaults.wandb_project)
    parser.add_argument("--wandb_run_name",      type=str,   default=defaults.wandb_run_name)

    args = parser.parse_args()
    cfg  = GRPOConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 60)
    print("GRPO Tool Training — 配置摘要")
    print("=" * 60)
    for k, v in vars(cfg).items():
        if not k.startswith("_") and k.upper() != k:
            print(f"  {k:<25} = {v}")
    print("=" * 60)

    trainer = GRPOToolTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
