"""
loss.py — GRPO 损失计算（批量化重构版）

核心优化：
  原版：每条序列单独 forward，一个 batch 需要 num_prompts × G × 2 次 forward
        例：4 prompts × 6 rollouts × 2 models = 48 次串行 forward

  现版：同一 prompt 下的 G 条 rollout 打包成一个 padded batch，
        每个 prompt 只需 2 次 forward（ref + policy），
        总计 num_prompts × 2 = 8 次 forward

  理论加速：group_size 倍（group_size=6 时约 6×）

其余逻辑保持不变：
  - loss_mask 确保沙箱注入 Token 不产生梯度
  - 逐 prompt 组 backward()，及时释放计算图
  - PPO-clip + per-token KL 惩罚
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from config    import GRPOConfig
from generator import Rollout


@dataclass
class LossStats:
    loss:        float
    mean_kl:     float
    mean_reward: float
    reward_std:  float
    frac_masked: float
    avg_gen_len: float


# ──────────────────────────────────────────────────────────────────────────────
def _batch_seq_log_probs(
    model,
    prompt_ids:   torch.Tensor,    # [1, prompt_len]
    rollout_list: List[Rollout],   # G 条 rollout
    requires_grad: bool = False,
) -> List[torch.Tensor]:
    """
    将 G 条 rollout 打包成一个 padded batch，做 **1次** forward，
    返回每条序列 mask=1 位置的 log-prob 列表。

    padding 策略：右侧补 0，attention_mask 屏蔽 pad 位置。

    Returns
    -------
    lps_list : List[Tensor[n_valid_i]]   长度 = G
    """
    device     = prompt_ids.device
    prompt_len = prompt_ids.shape[1]
    G          = len(rollout_list)

    # ── 构造 padded batch ────────────────────────────────────────────────
    gen_lens = [r.gen_ids.shape[0] for r in rollout_list]
    max_gen  = max(gen_lens) if gen_lens else 0
    full_len = prompt_len + max_gen

    # 全部初始化为 0（pad_token），attention_mask 对应置 0
    input_ids      = torch.zeros(G, full_len, dtype=torch.long,  device=device)
    attention_mask = torch.zeros(G, full_len, dtype=torch.long,  device=device)

    for i, rollout in enumerate(rollout_list):
        gen_len = rollout.gen_ids.shape[0]
        seq_len = prompt_len + gen_len
        input_ids[i, :seq_len]      = torch.cat([prompt_ids.squeeze(0), rollout.gen_ids])
        attention_mask[i, :seq_len] = 1

    # ── 单次 forward ────────────────────────────────────────────────────
    with torch.set_grad_enabled(requires_grad):
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    # logits: [G, full_len, V]

    # ── 提取每条序列生成段的 valid log-prob ─────────────────────────────
    lps_list: List[torch.Tensor] = []
    for i, rollout in enumerate(rollout_list):
        gen_len = rollout.gen_ids.shape[0]
        if gen_len == 0:
            lps_list.append(torch.empty(0, device=device))
            continue

        # 生成段 logits：位置 [prompt_len-1, prompt_len+gen_len-1)
        gen_logits = outputs.logits[i, prompt_len - 1 : prompt_len + gen_len - 1, :]
        log_probs  = F.log_softmax(gen_logits, dim=-1)  # [gen_len, V]

        # 取实际 token 的 log-prob
        all_lps = log_probs.gather(
            1, rollout.gen_ids.unsqueeze(1)
        ).squeeze(1)  # [gen_len]

        # 用 loss_mask 过滤（mask=True 表示模型自主生成，纳入 loss）
        valid_lps = all_lps[rollout.loss_mask]  # [n_valid]
        lps_list.append(valid_lps)

    return lps_list


# ──────────────────────────────────────────────────────────────────────────────
def compute_grpo_loss_and_backward(
    policy_model, ref_model,
    prompt_ids_list, rollouts, rewards, cfg
) -> LossStats:
    num_prompts = len(prompt_ids_list)
    G = cfg.group_size
    total_seqs = num_prompts * G

    agg_loss = agg_kl = agg_reward = 0.0
    agg_masked = agg_total = 0
    all_rewards_flat = []

    for p_idx in range(num_prompts):
        prompt_ids     = prompt_ids_list[p_idx]
        group_rollouts = rollouts[p_idx]
        group_rewards  = rewards[p_idx]
        all_rewards_flat.extend(group_rewards)

        r_mean = sum(group_rewards) / G
        r_std  = math.sqrt(sum((r - r_mean) ** 2 for r in group_rewards) / G + 1e-8)
        advantages = [(r - r_mean) / (r_std + 1e-8) for r in group_rewards]

        for rollout in group_rollouts:
            n_tok = rollout.gen_ids.shape[0]
            agg_total  += n_tok
            agg_masked += n_tok - int(rollout.loss_mask.sum().item())

        valid_indices = [
            i for i, r in enumerate(group_rollouts)
            if r.gen_ids.shape[0] > 0 and r.loss_mask.sum().item() > 0
        ]
        if not valid_indices:
            continue

        valid_rollouts   = [group_rollouts[i] for i in valid_indices]
        valid_advantages = [advantages[i]     for i in valid_indices]
        valid_rewards    = [group_rewards[i]  for i in valid_indices]

        # ── ① ref forward（no_grad，批量）─────────────────────────────
        with torch.no_grad():
            ref_lps_list = _batch_seq_log_probs(
                ref_model, prompt_ids, valid_rollouts, requires_grad=False
            )

        # ── ② policy forward + loss 在同一个块内完成，立即 backward ────
        # 关键：每次只对【单条】rollout 做 forward，避免大 batch logits 驻留
        group_loss_sum = torch.tensor(0.0, device=prompt_ids.device)

        for k, (rollout, advantage, ref_lps) in enumerate(
            zip(valid_rollouts, valid_advantages, ref_lps_list)
        ):
            # 单条 forward，logits 形状 [1, seq_len, V]，显存可控
            policy_lps = _batch_seq_log_probs(
                policy_model, prompt_ids, [rollout], requires_grad=True
            )[0]

            n_valid = policy_lps.shape[0]
            if n_valid == 0:
                continue

            log_ratio  = policy_lps - ref_lps.detach()
            kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0
            ratio         = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            pg_term = -torch.min(ratio * advantage, clipped_ratio * advantage)

            seq_loss = (pg_term + cfg.kl_coef * kl_per_tok).mean()
            group_loss_sum = group_loss_sum + seq_loss

            agg_loss   += seq_loss.item()
            agg_kl     += kl_per_tok.mean().item()
            agg_reward += valid_rewards[k]

        n_valid_seqs = len(valid_indices)
        if n_valid_seqs > 0:
            (group_loss_sum / (total_seqs * cfg.grad_accum)).backward()

        del group_loss_sum, ref_lps_list

    # ── 汇总统计（保持不变）───────────────────────────────────────────
    n        = total_seqs
    r_all    = all_rewards_flat
    r_mean_t = sum(r_all) / max(n, 1)
    r_std_t  = math.sqrt(sum((r - r_mean_t) ** 2 for r in r_all) / max(n, 1))
    total_tokens = sum(r.gen_ids.shape[0] for group in rollouts for r in group)
    total_count  = sum(len(group) for group in rollouts)

    return LossStats(
        loss        = agg_loss   / max(n, 1),
        mean_kl     = agg_kl     / max(n, 1),
        mean_reward = r_mean_t,
        reward_std  = r_std_t,
        frac_masked = agg_masked / max(agg_total, 1),
        avg_gen_len = total_tokens / max(total_count, 1),
    )