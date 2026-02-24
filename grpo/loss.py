"""
loss.py — GRPO 损失计算

修复 Bug 2：loss_mask 确保沙箱注入的 Token 不产生梯度
修复 Bug 3：逐序列 backward()，立即释放计算图，杜绝显存爆炸
"""
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config    import GRPOConfig
from generator import Rollout


@dataclass
class LossStats:
    loss:        float
    mean_kl:     float
    mean_reward: float
    reward_std:  float
    frac_masked: float   # 被 mask 掉的 token 比例（用于监控）
    avg_gen_len: float


# ──────────────────────────────────────────────────────────────────────────────
def _seq_log_probs(
    model,
    prompt_ids: torch.Tensor,   # [1, prompt_len]
    gen_ids:    torch.Tensor,   # [gen_len]
    loss_mask:  torch.Tensor,   # [gen_len] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    单次前向传播，返回被 mask 选中的有效 token 的 log-prob。

    Returns
    -------
    valid_lps  : [n_valid]  log p(token | context)  （只含 mask=1 的位置）
    valid_ids  : [n_valid]  对应的 token id
    """
    full_ids = torch.cat([prompt_ids.squeeze(0), gen_ids]).unsqueeze(0)  # [1, full_len]
    prompt_len = prompt_ids.shape[1]
    gen_len    = gen_ids.shape[0]

    outputs  = model(full_ids, use_cache=False, return_dict=True)
    # 预测位置 t 的 logits 在 t-1 处
    # 生成段对应 positions [prompt_len-1, prompt_len+gen_len-1)
    gen_logits = outputs.logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [gen_len, V]
    log_probs  = F.log_softmax(gen_logits, dim=-1)                                # [gen_len, V]

    # 只取每个位置的实际 token 对应的 log-prob
    all_lps = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)  # [gen_len]

    # 用 loss_mask 筛选有效位置
    valid_lps = all_lps[loss_mask]      # [n_valid]
    valid_ids = gen_ids[loss_mask]      # [n_valid]
    return valid_lps, valid_ids


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
    逐序列计算 GRPO loss 并立即 backward()，避免把所有计算图同时保留在显存中。

    调用方只需之后执行 optimizer.step() 即可。
    返回纯数值统计（LossStats），不含 Tensor，方便日志记录。

    GRPO 目标：
        loss_i = -A_i * min(ratio_i, clip(ratio_i))  +  β * KL_i
        A_i    = (r_i - mean_group) / (std_group + ε)
        KL_i   = exp(lp_policy - lp_ref) - (lp_policy - lp_ref) - 1   （per token 平均）
    """
    num_prompts = len(prompt_ids_list)
    G           = cfg.group_size
    total_seqs  = num_prompts * G       # 归一化分母

    agg_loss    = 0.0
    agg_kl      = 0.0
    agg_reward  = 0.0
    agg_masked  = 0
    agg_total   = 0
    all_rewards_flat: List[float] = []

    for p_idx in range(num_prompts):
        prompt_ids    = prompt_ids_list[p_idx]   # [1, L]
        group_rewards = rewards[p_idx]           # [G]
        all_rewards_flat.extend(group_rewards)

        # ── 计算该 group 的 advantage ────────────────────────────────────
        r_mean = sum(group_rewards) / G
        r_std  = math.sqrt(
            sum((r - r_mean) ** 2 for r in group_rewards) / G + 1e-8
        )
        advantages = [(r - r_mean) / (r_std + 1e-8) for r in group_rewards]

        for g_idx in range(G):
            rollout   = rollouts[p_idx][g_idx]
            advantage = advantages[g_idx]
            gen_ids   = rollout.gen_ids       # [gen_len]
            loss_mask = rollout.loss_mask     # [gen_len] bool

            n_valid = loss_mask.sum().item()
            agg_total  += gen_ids.shape[0]
            agg_masked += gen_ids.shape[0] - int(n_valid)

            if n_valid == 0:
                continue   # 整条轨迹全被 mask（理论上不会发生）

            # ── 参考模型：no_grad ────────────────────────────────────────
            with torch.no_grad():
                ref_lps, _ = _seq_log_probs(ref_model, prompt_ids, gen_ids, loss_mask)

            # ── 策略模型：需要梯度 ───────────────────────────────────────
            policy_lps, _ = _seq_log_probs(policy_model, prompt_ids, gen_ids, loss_mask)

            # ── Per-token KL：exp(π-ref) - (π-ref) - 1 ──────────────────
            log_ratio = policy_lps - ref_lps.detach()
            kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0   # ≥ 0

            # ── PPO-clip 策略梯度项 ──────────────────────────────────────
            ratio         = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            pg_term = -torch.min(
                ratio         * advantage,
                clipped_ratio * advantage,
            )  # [n_valid]

            # ── 序列 loss（对 valid token 平均）─────────────────────────
            seq_loss = (pg_term + cfg.kl_coef * kl_per_tok).mean()

            # ── 归一化后立即 backward，释放本条序列的计算图 ────────────
            # 关键：除以 total_seqs，等价于所有序列 loss 之和再除以 total_seqs
            (seq_loss / (total_seqs * cfg.grad_accum)).backward()

            # ── 纯数值统计（detach，不占显存）───────────────────────────
            agg_loss   += seq_loss.item()
            agg_kl     += kl_per_tok.mean().item()
            agg_reward += group_rewards[g_idx]

            # 显式释放本条序列的计算图相关临时变量
            del policy_lps, ref_lps, log_ratio, kl_per_tok, pg_term, seq_loss

    # ── 统计汇总 ─────────────────────────────────────────────────────────
    n = total_seqs
    r_all  = all_rewards_flat
    r_mean_total = sum(r_all) / max(n, 1)
    r_std_total  = math.sqrt(
        sum((r - r_mean_total) ** 2 for r in r_all) / max(n, 1)
    )
    total_tokens = 0
    total_count = 0
    for group in rollouts:
        for r in group:
            total_tokens += r.gen_ids.shape[0]
            total_count += 1
            
    avg_gen_len = total_tokens / max(total_count, 1)
    return LossStats(
        loss        = agg_loss   / max(n, 1),
        mean_kl     = agg_kl     / max(n, 1),
        mean_reward = r_mean_total,
        reward_std  = r_std_total,
        frac_masked = agg_masked / max(agg_total, 1),
        avg_gen_len = avg_gen_len,
    )
