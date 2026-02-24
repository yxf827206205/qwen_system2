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
    policy_model,
    ref_model,
    prompt_ids_list: List[torch.Tensor],   # len = num_prompts，每个 [1, L]
    rollouts:        List[List[Rollout]],  # [num_prompts][group_size]
    rewards:         List[List[float]],    # [num_prompts][group_size]
    cfg:             GRPOConfig,
) -> LossStats:
    """
    批量化版本的 GRPO loss 计算 + backward。

    每个 prompt 组：
      1. 一次 ref forward（no_grad）  → G 条序列的 ref log-probs
      2. 一次 policy forward（需梯度）→ G 条序列的 policy log-probs
      3. 逐序列计算 PPO-clip + KL，累加梯度后立即释放计算图

    相比原版串行 G×2 次 forward，显存峰值略有上升（batch 变大），
    但 CUDA kernel 启动次数大幅减少，端到端速度显著提升。
    """
    num_prompts = len(prompt_ids_list)
    G           = cfg.group_size
    total_seqs  = num_prompts * G

    agg_loss   = 0.0
    agg_kl     = 0.0
    agg_reward = 0.0
    agg_masked = 0
    agg_total  = 0
    all_rewards_flat: List[float] = []

    for p_idx in range(num_prompts):
        prompt_ids    = prompt_ids_list[p_idx]   # [1, L]
        group_rollouts = rollouts[p_idx]          # List[Rollout], len=G
        group_rewards  = rewards[p_idx]           # List[float],   len=G
        all_rewards_flat.extend(group_rewards)

        # ── Advantage 计算 ────────────────────────────────────────────────
        r_mean = sum(group_rewards) / G
        r_std  = math.sqrt(
            sum((r - r_mean) ** 2 for r in group_rewards) / G + 1e-8
        )
        advantages = [(r - r_mean) / (r_std + 1e-8) for r in group_rewards]

        # ── 统计 mask 信息 ────────────────────────────────────────────────
        for rollout in group_rollouts:
            n_tok = rollout.gen_ids.shape[0]
            agg_total  += n_tok
            agg_masked += n_tok - int(rollout.loss_mask.sum().item())

        # ── 过滤掉完全空的 rollout（极端边界情况）──────────────────────
        valid_indices = [
            i for i, r in enumerate(group_rollouts)
            if r.gen_ids.shape[0] > 0 and r.loss_mask.sum().item() > 0
        ]
        if not valid_indices:
            continue

        valid_rollouts  = [group_rollouts[i] for i in valid_indices]
        valid_advantages = [advantages[i]    for i in valid_indices]
        valid_rewards    = [group_rewards[i] for i in valid_indices]

        # ── ① 参考模型：批量 forward（no_grad）──────────────────────────
        with torch.no_grad():
            ref_lps_list = _batch_seq_log_probs(
                ref_model, prompt_ids, valid_rollouts, requires_grad=False
            )

        # ── ② 策略模型：批量 forward（需梯度）──────────────────────────
        # 注意：这里不用 no_grad，让 autograd 追踪计算图
        policy_lps_list = _batch_seq_log_probs(
            policy_model, prompt_ids, valid_rollouts, requires_grad=True
        )

        # ── ③ 逐序列计算 loss 并 backward ────────────────────────────────
        # 虽然两次 forward 是批量的，backward 仍逐序列进行，
        # 避免把整个 batch 的计算图同时保留在显存中。
        group_loss_sum = torch.tensor(0.0, device=prompt_ids.device)

        for k, (policy_lps, ref_lps, advantage) in enumerate(
            zip(policy_lps_list, ref_lps_list, valid_advantages)
        ):
            n_valid = policy_lps.shape[0]
            if n_valid == 0:
                continue

            # Per-token KL：exp(π - ref) - (π - ref) - 1  ≥ 0
            log_ratio = policy_lps - ref_lps.detach()
            kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0

            # PPO-clip 策略梯度
            ratio         = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            pg_term = -torch.min(
                ratio         * advantage,
                clipped_ratio * advantage,
            )  # [n_valid]

            seq_loss = (pg_term + cfg.kl_coef * kl_per_tok).mean()
            group_loss_sum = group_loss_sum + seq_loss

            # 统计（detach，不占显存）
            agg_loss   += seq_loss.item()
            agg_kl     += kl_per_tok.mean().item()
            agg_reward += valid_rewards[k]

        # ── 整组一次 backward，释放该 prompt 组的整个计算图 ────────────
        # 除以 total_seqs × grad_accum，保持梯度量级与原版一致
        n_valid_seqs = len(valid_indices)
        if n_valid_seqs > 0:
            (group_loss_sum / (total_seqs * cfg.grad_accum)).backward()

        # 显式释放
        del policy_lps_list, ref_lps_list, group_loss_sum

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    n        = total_seqs
    r_all    = all_rewards_flat
    r_mean_t = sum(r_all) / max(n, 1)
    r_std_t  = math.sqrt(
        sum((r - r_mean_t) ** 2 for r in r_all) / max(n, 1)
    )

    total_tokens = sum(
        r.gen_ids.shape[0]
        for group in rollouts
        for r in group
    )
    total_count = sum(len(group) for group in rollouts)

    return LossStats(
        loss        = agg_loss   / max(n, 1),
        mean_kl     = agg_kl     / max(n, 1),
        mean_reward = r_mean_t,
        reward_std  = r_std_t,
        frac_masked = agg_masked / max(agg_total, 1),
        avg_gen_len = total_tokens / max(total_count, 1),
    )