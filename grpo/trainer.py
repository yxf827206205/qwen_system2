"""
trainer.py — GRPO 训练器主体

训练循环流程：
  for batch in dataloader:
      rollouts = [generator.generate(prompt) × G]   ← on-policy 采样
      rewards  = [compute_reward(r) for r in rollouts]
      compute_grpo_loss_and_backward(...)            ← 逐序列 backward
      if grad_accum reached:
          clip_grad → optimizer.step → scheduler.step → zero_grad
"""
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

from config      import GRPOConfig
from model_utils import load_tokenizer, build_ref_and_policy, save_policy
from dataset     import GSM8KDataset, collate_fn
from generator   import ToolAwareGenerator, Rollout
from sandbox     import PythonSandbox
from reward      import compute_reward
from loss        import compute_grpo_loss_and_backward


class GRPOToolTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

        # ── Tokenizer & 模型 ──────────────────────────────────────────────
        print("\n========== 初始化模型 ==========")
        self.tokenizer = load_tokenizer(cfg)
        self.ref_model, self.policy = build_ref_and_policy(cfg, self.tokenizer)

        # ── 生成器 & 沙箱 ─────────────────────────────────────────────────
        self.sandbox   = PythonSandbox()
        self.generator = ToolAwareGenerator(
            model     = self.policy,   # on-policy 采样
            tokenizer = self.tokenizer,
            sandbox   = self.sandbox,
            cfg       = cfg,
        )

        # ── 数据集 ────────────────────────────────────────────────────────
        print("\n========== 加载数据集 ==========")
        dataset    = GSM8KDataset(self.tokenizer, cfg, split="train")
        self.loader = DataLoader(
            dataset,
            batch_size  = cfg.batch_size,
            shuffle     = True,
            collate_fn  = collate_fn,
            drop_last   = True,
        )

        # ── 优化器 & 调度器 ───────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=cfg.learning_rate, weight_decay=0.01
        )
        total_steps = len(self.loader) * cfg.num_epochs // cfg.grad_accum
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps  = max(1, int(total_steps * cfg.warmup_ratio)),
            num_training_steps = total_steps,
        )
        self.global_step = 0
        print("\n========== 初始化完成，准备训练 ==========\n")

    # ──────────────────────────────────────────────────────────────────────
    def _rollout_batch(
        self, batch: List[Dict[str, Any]]
    ):
        """
        对 batch 内每个 prompt 采样 G 条轨迹。

        Returns
        -------
        prompt_ids_list : List[Tensor]             长度 = num_prompts
        rollouts        : List[List[Rollout]]      [num_prompts][G]
        rewards         : List[List[float]]        [num_prompts][G]
        """
        prompt_ids_list: List[torch.Tensor]   = []
        rollouts:        List[List[Rollout]]  = []
        rewards:         List[List[float]]    = []

        self.policy.eval()   # 采样时关闭 dropout
        device = next(self.policy.parameters()).device

        with torch.no_grad():
            for item in batch:
                prompt_ids = item["prompt_ids"].to(device)   # [1, L]
                expected   = item["expected"]

                group_rollouts: List[Rollout] = []
                group_rewards:  List[float]   = []

                group_rollouts = self.generator.generate_group(prompt_ids, self.cfg.group_size)
                
                tqdm.write(f"\n[Debug] 轨迹末尾预览: {group_rollouts[0].text[-300:]}")
                group_rewards = [compute_reward(r.text, expected, self.cfg) for r in group_rollouts]

                prompt_ids_list.append(prompt_ids)
                rollouts.append(group_rollouts)
                rewards.append(group_rewards)

        self.policy.train()
        return prompt_ids_list, rollouts, rewards

    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        print("=== 开始 GRPO Tool 训练 ===\n")
        self.policy.train()
        self.optimizer.zero_grad()

        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch} / {self.cfg.num_epochs}")
            print(f"{'='*50}")
            epoch_stats = []

            for local_step, batch in enumerate(
                tqdm(self.loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
            ):
                # ① On-policy 采样（no_grad）
                prompt_ids_list, rollouts, rewards = self._rollout_batch(batch)

                # ② 计算 GRPO loss 并逐序列 backward（在 train mode 下）
                stats = compute_grpo_loss_and_backward(
                    policy_model    = self.policy,
                    ref_model       = self.ref_model,
                    prompt_ids_list = prompt_ids_list,
                    rollouts        = rollouts,
                    rewards         = rewards,
                    cfg             = self.cfg,
                )
                epoch_stats.append(stats)

                # ③ 梯度累积
                if (local_step + 1) % self.cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # ④ 日志
                    log_dict = {
                        "train/loss":         stats.loss,
                        "train/mean_kl":      stats.mean_kl,
                        "train/mean_reward":  stats.mean_reward,
                        "train/reward_std":   stats.reward_std,
                        "train/frac_masked":  stats.frac_masked,
                        "train/lr":           self.scheduler.get_last_lr()[0],
                        "train/epoch":        epoch,
                        "train/avg_gen_len":  stats.avg_gen_len,
                    }
                    wandb.log(log_dict, step=self.global_step)

                    tqdm.write(
                        f"  step={self.global_step:5d} | "
                        f"loss={stats.loss:.4f} | "
                        f"reward={stats.mean_reward:.3f}±{stats.reward_std:.3f} | "
                        f"kl={stats.mean_kl:.5f} | "
                        f"masked={stats.frac_masked:.1%} | "
                        f"len={stats.avg_gen_len:.0f}"
                    )

                    # ⑤ 定期保存
                    if self.global_step % self.cfg.save_steps == 0:
                        save_policy(
                            self.policy, self.tokenizer,
                            os.path.join(self.cfg.output_dir, f"step_{self.global_step}")
                        )

            # 每个 epoch 结束保存
            save_policy(
                self.policy, self.tokenizer,
                os.path.join(self.cfg.output_dir, f"epoch_{epoch}")
            )

        # 最终保存
        save_policy(
            self.policy, self.tokenizer,
            os.path.join(self.cfg.output_dir, "final")
        )
        wandb.finish()
        print("\n🎉 训练完成！")
