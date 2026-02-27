"""


修复历史:
  v3 本版: 加权采样平衡标签 + 降低lr + EMA监控 → 收敛更平滑
    1. WeightedRandomSampler: 平衡 4 个标签桶，解决高标签数据占 54% 导致的梯度抖动
    2. batch_size 8→16，learning_rate 5e-5→1e-5，梯度更稳
    3. EMA loss (α=0.95) 上报 wandb，train/loss_ema 这条线才是真实收敛趋势
    4. 每 epoch 打印预测分布，监控模型是否坍缩到单一值
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_qwen import CognitiveQwen


# =====================================================================
# 1. 超参数配置
# =====================================================================
CONFIG = {
    # 模型路径
    "base_model_path": "/root/autodl-tmp/models/Qwen3-0.6B-Base",
    "lora_path":       "/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1/final",

    # 数据路径
    "value_data_path": "/root/autodl-tmp/data/value_data_full.jsonl",

    # 输出路径
    "output_head_path": "/root/autodl-tmp/checkpoints/value_head.pt",

    # 训练超参（v3 改动）
    "batch_size":                  16,   # v2:8  → v3:16，每批样本更多，梯度方向更稳
    "gradient_accumulation_steps": 4,    # 等效 batch = 64，和 v2 一致
    "epochs":                      5,
    "learning_rate":              3e-4,
    "weight_decay":                0.01,
    "max_length":                  1024,
    "warmup_ratio":                0.15,

    # EMA 平滑系数（用于 wandb 可视化趋势，不影响训练）
    "ema_alpha":                   0.95,

    # Wandb
    "wandb_project":  "cognitive-nano-qwen-value-head-training",
    "wandb_run_name": "train-value-head-v3-stable",
}


# =====================================================================
# 2. 数据集
# =====================================================================
class ValueDataset(Dataset):
    """
    读取 JSONL，每行: {"text": "...", "label": 0.875}
    label 是平滑后的胜率，范围 [0.0, 1.0]
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"找不到数据文件: {jsonl_path}")

        self.data = []
        print(f"  正在加载价值数据: {jsonl_path} ...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.tokenizer  = tokenizer
        self.max_length = max_length

        labels = [d["label"] for d in self.data]
        print(f"  成功加载 {len(self.data)} 条样本")
        print(f"  标签统计: min={min(labels):.3f}  max={max(labels):.3f}  "
              f"mean={sum(labels)/len(labels):.3f}")
        dist = {
            "[0.0, 0.25)":  sum(1 for l in labels if l < 0.25),
            "[0.25, 0.5)":  sum(1 for l in labels if 0.25 <= l < 0.5),
            "[0.5, 0.75)":  sum(1 for l in labels if 0.5  <= l < 0.75),
            "[0.75, 1.0]":  sum(1 for l in labels if l >= 0.75),
        }
        print(f"  标签分布: {dist}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item    = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          torch.tensor(float(item["label"]), dtype=torch.float32),
        }


# =====================================================================
# 3. 加权采样器（核心修复：平衡标签分布，解决梯度震荡）
# =====================================================================
def build_weighted_sampler(dataset: ValueDataset) -> WeightedRandomSampler:
    """
    将标签离散化为 4 个桶，按桶内样本数的倒数计算权重。
    高标签桶([0.75,1.0])占 54% → 权重最小；低标签桶权重最大。
    采样后每个 batch 里四种标签趋于均衡，梯度方向稳定。
    """
    def label_to_bucket(l: float) -> int:
        if l < 0.25:   return 0
        elif l < 0.5:  return 1
        elif l < 0.75: return 2
        else:          return 3

    labels     = [dataset.data[i]["label"] for i in range(len(dataset))]
    buckets    = [label_to_bucket(l) for l in labels]
    bucket_cnt = [buckets.count(b) for b in range(4)]
    bucket_w   = [1.0 / (c + 1e-8) for c in bucket_cnt]
    sample_w   = [bucket_w[b] for b in buckets]

    print(f"  桶内样本数: { {i: bucket_cnt[i] for i in range(4)} }")
    w_sum = sum(bucket_w)
    print(f"  桶权重占比: { {i: f'{bucket_w[i]/w_sum*100:.1f}%' for i in range(4)} }")

    return WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(sample_w),
        replacement=True,
    )


# =====================================================================
# 4. 主训练程序
# =====================================================================
def main():
    print("=" * 60)
    print("  Value Head 训练启动 (v3 稳定版)")
    print("=" * 60)

    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG,
    )

    # ── Step 1: Tokenizer ────────────────────────────────────────────
    print("\n[Step 1] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    new_tokens = ["<|python_start|>", "<|python_end|>",
                  "<|output_start|>", "<|output_end|>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    vocab_size = len(tokenizer)
    print(f"  词表大小: {vocab_size}")

    # ── Step 2: 模型 ─────────────────────────────────────────────────
    print("\n[Step 2] 组装 CognitiveQwen (MLP Value Head)...")
    model = CognitiveQwen(
        base_model_path=CONFIG["base_model_path"],
        lora_path=CONFIG["lora_path"],
        device="cuda",
        vocab_size=vocab_size,
    )

    # 冻结 backbone，只训练 value_head
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.value_head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / 总参数: {total:,} ({trainable/total*100:.4f}%)")

    # ── Step 3: 数据 + 加权采样器 ────────────────────────────────────
    print("\n[Step 3] 准备数据集与加权 DataLoader...")
    dataset    = ValueDataset(
        CONFIG["value_data_path"], tokenizer, max_length=CONFIG["max_length"]
    )
    sampler    = build_weighted_sampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,          # 用加权采样替换 shuffle=True
        num_workers=4,
        pin_memory=True,
    )
    print(f"  DataLoader: {len(dataloader)} steps/epoch  "
          f"(batch={CONFIG['batch_size']}, 加权采样已启用)")

    # ── Step 4: 优化器 + Cosine 调度器 ───────────────────────────────
    print("\n[Step 4] 初始化优化器与调度器...")
    optimizer = torch.optim.AdamW(
        model.value_head.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_update_steps = (
        len(dataloader) * CONFIG["epochs"]
        // CONFIG["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_update_steps * CONFIG["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    print(f"  总更新步数: {total_update_steps},  Warmup: {warmup_steps} steps")

    criterion = nn.MSELoss()

    # ── Step 5: 训练循环 ──────────────────────────────────────────────
    print("\n[Step 5] 开始训练...")
    model.train()
    global_step = 0
    ema_loss    = None
    alpha       = CONFIG["ema_alpha"]
    optimizer.zero_grad()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss_sum = 0.0
        epoch_preds    = []   # 收集本 epoch 所有预测值，用于分布统计

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels         = batch["label"].to("cuda")             # float32 (B,)

            # Backbone 完全冻结 → no_grad 省显存提速
            with torch.no_grad():
                outputs = model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden_state = outputs.hidden_states[-1].detach()  # (B, L, H)

            # 取每条序列最后一个有效 token 的向量
            seq_lengths   = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device="cuda")
            last_hidden   = last_hidden_state[batch_indices, seq_lengths]  # (B, H)

            # Value Head → sigmoid → float32
            # 注意：value_head 是 bfloat16，sigmoid 后 cast 到 float32 再算 MSE
            # 这样既保证了数值稳定，又避免了 dtype 不匹配报错
            value_logit = model.value_head(last_hidden).squeeze(-1)   # bfloat16 (B,)
            value_prob  = torch.sigmoid(value_logit).float()           # float32  (B,)

            loss        = criterion(value_prob, labels)                # float32 vs float32
            scaled_loss = loss / CONFIG["gradient_accumulation_steps"]
            scaled_loss.backward()

            epoch_loss_sum += loss.item()
            epoch_preds.extend(value_prob.detach().cpu().tolist())

            # EMA 更新（仅用于监控，不影响参数）
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = alpha * ema_loss + (1 - alpha) * loss.item()

            # 梯度累加 → 参数更新
            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0 \
                    or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    model.value_head.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({
                    "train/loss":        loss.item(),
                    "train/loss_ema":    ema_loss,   # ← 看这条线判断收敛，忽略单步噪声
                    "train/lr":          scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/epoch":       epoch + (step + 1) / len(dataloader),
                })

            pbar.set_postfix({
                "loss":     f"{loss.item():.4f}",
                "ema":      f"{ema_loss:.4f}",
                "lr":       f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # ── Epoch 结束统计 ────────────────────────────────────────────
        avg_loss = epoch_loss_sum / len(dataloader)
        pred_dist = {
            "pred[0.0,0.25)":  sum(1 for p in epoch_preds if p < 0.25),
            "pred[0.25,0.5)":  sum(1 for p in epoch_preds if 0.25 <= p < 0.5),
            "pred[0.5,0.75)":  sum(1 for p in epoch_preds if 0.5  <= p < 0.75),
            "pred[0.75,1.0]":  sum(1 for p in epoch_preds if p >= 0.75),
        }
        pred_mean = sum(epoch_preds) / len(epoch_preds) if epoch_preds else 0.0

        print(f"\n  ── Epoch {epoch+1} 完成 ──────────────────────────")
        print(f"    avg_loss = {avg_loss:.4f}   loss_ema = {ema_loss:.4f}")
        print(f"    预测均值 = {pred_mean:.3f}")
        print(f"    预测分布 = {pred_dist}")
        print(f"    (若所有预测堆在同一个桶，说明模型坍缩，需降低 lr)")
        print(f"  ─────────────────────────────────────────────────\n")

        wandb.log({
            "train/epoch_avg_loss": avg_loss,
            "train/pred_mean":      pred_mean,
            "epoch":                epoch + 1,
            **{f"train/{k}": v for k, v in pred_dist.items()},
        })

        # 每 epoch 保存一次 checkpoint
        # ckpt_path = CONFIG["output_head_path"].replace(".pt", f"_epoch{epoch+1}.pt")
        # os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        # torch.save(model.value_head.state_dict(), ckpt_path)
        # print(f"  Checkpoint 已保存: {ckpt_path}")

    # ── Step 6: 保存最终权重 ──────────────────────────────────────────
    print("\n[Step 6] 保存最终 Value Head 权重...")
    final_path = CONFIG["output_head_path"]
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.value_head.state_dict(), final_path)
    print(f"  最终权重已保存: {final_path}")

    wandb.finish()
    print("\n Done!")


# =====================================================================
# 5. 推理工具函数（供 MCTS 调用）
# =====================================================================
def load_trained_value_head(model: CognitiveQwen, head_path: str):
    """
    加载训练好的 Value Head 权重并切换到 eval 模式。

    MCTS 使用示例:
        model = CognitiveQwen(base_model_path, lora_path, vocab_size=N)
        load_trained_value_head(model, "/root/autodl-tmp/checkpoints/value_head.pt")
        # 推理时：
        prob = model.predict_value(input_ids, attention_mask)  # shape: (B,)，值域[0,1]
    """
    state_dict = torch.load(head_path, map_location="cuda")
    model.value_head.load_state_dict(state_dict)
    model.value_head.eval()
    print(f"  Value Head 权重加载完毕: {head_path}")


if __name__ == "__main__":
    main()