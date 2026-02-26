"""
select_value_data.py  —— Value Head 训练数据采集脚本（修复版）

修复点:
  - 标签平滑: 线性 ratio → 平方根曲线，让早期 snapshot 的标签更分散
  - 增加标签分布统计输出，方便验证数据质量
"""

import sys
import argparse
import json
import re
import random
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo.config import GRPOConfig
from grpo.sandbox import PythonSandbox
from grpo.generator import ToolAwareGenerator
import wandb


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def extract_final_answer(text: str):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-,]+)", text
    )
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def build_prompt(tokenizer, question: str) -> str:
    """与 GRPO 训练时保持完全一致的 Prompt 格式。"""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt + "<think>\n"


def slice_prefixes(full_text: str, prompt_text: str, n_snapshots: int) -> list:
    """
    将完整轨迹切成 n_snapshots 个前缀快照。
    每个快照对应推理过程中的一个"时间截面"。
    """
    gen_part = full_text[len(prompt_text):]

    # 按逻辑块（段落或工具调用边界）切分
    chunks = re.split(r'(\n\n|<\|output_end\|>)', gen_part)

    merged_chunks = []
    for i in range(0, len(chunks) - 1, 2):
        merged_chunks.append(chunks[i] + chunks[i + 1])
    if len(chunks) % 2 != 0 and chunks[-1]:
        merged_chunks.append(chunks[-1])

    total_chunks = len(merged_chunks)
    if total_chunks < 3:
        return [full_text]

    prefixes = []
    for i in range(1, n_snapshots + 1):
        cut_idx = max(1, int(total_chunks * i / n_snapshots))
        cut_text = "".join(merged_chunks[:cut_idx])
        prefix = prompt_text + cut_text
        if prefix not in prefixes:
            prefixes.append(prefix)

    if full_text not in prefixes:
        prefixes.append(full_text)

    return prefixes


def smooth_label(label: float, ratio: float) -> float:
    """
    修复版标签平滑：使用平方根曲线代替线性曲线。

    原版:  label_smoothed = 0.5 + (label - 0.5) * ratio
      → 早期 snapshot(ratio=0.25) 标签范围仅 [0.375, 0.625]，信号极弱

    修复:  ratio_curved = ratio ** 0.5
      → 早期 snapshot(ratio=0.25, curved=0.5) 标签范围变为 [0.25, 0.75]，
        信号强度翻倍，模型能学到更有效的特征

    对应关系 (n_snapshots=4):
      snapshot 1/4: ratio=0.25  线性→[0.375,0.625]  修复→[0.25,0.75]
      snapshot 2/4: ratio=0.50  线性→[0.25,0.75]    修复→[0.146,0.854]
      snapshot 3/4: ratio=0.75  线性→[0.125,0.875]  修复→[0.067,0.933]
      snapshot 4/4: ratio=1.00  线性→[0.0,1.0]      修复→[0.0,1.0]
    """
    ratio_curved = ratio ** 0.5
    return round(0.5 + (label - 0.5) * ratio_curved, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 主采集逻辑
# ──────────────────────────────────────────────────────────────────────────────

def collect(args):
    print("=" * 60)
    print(" Value Head 数据采集启动 (修复标签平滑版)")
    print("=" * 60)

    wandb.init(
        project="cognitive-nano-qwen-test1",
        name=f"eval-rl-chunk-{args.start_idx}",
        config=vars(args)
    )

    # ── 1. Tokenizer ──────────────────────────────────────────────
    print("\n1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    new_tokens = [
        "<|python_start|>", "<|python_end|>",
        "<|output_start|>", "<|output_end|>",
    ]
    tokenizer.add_tokens(new_tokens, special_tokens=True)

    # ── 2. 模型 ────────────────────────────────────────────────────
    print("2. 加载 Base Model + GRPO LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=False)
    model = model.merge_and_unload()
    model.eval()
    print("  模型加载完毕")

    # ── 3. 沙箱 + 生成器 ──────────────────────────────────────────
    cfg = GRPOConfig()
    cfg.max_segment_tokens = 512
    cfg.max_total_tokens = args.max_new_tokens
    cfg.temperature = 0.6

    sandbox = PythonSandbox()
    generator = ToolAwareGenerator(model, tokenizer, sandbox, cfg)

    # ── 4. 加载 GSM8K ─────────────────────────────────────────────
    print(f"\n3. 加载 GSM8K (索引 {args.start_idx} ~ {args.end_idx})...")
    raw = load_dataset("gsm8k", "main", split="train")
    raw = raw.shuffle(seed=42)
    actual_end = min(args.end_idx, len(raw))
    raw = raw.select(range(args.start_idx, actual_end))

    # ── 5. 采集 ───────────────────────────────────────────────────
    records = []
    stats = {
        "total_rollouts":  0,
        "correct_rollouts": 0,
        "no_answer":       0,
        "total_problems":  0,
        "passed_problems": 0,
    }

    print("\n4. 开始 Rollout + 切片...")
    for item in tqdm(raw, desc="采集中"):
        question = item["question"]
        expected = item["answer"].split("####")[-1].replace(",", "").strip()
        prompt_text = build_prompt(tokenizer, question)

        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")

        rollouts = generator.generate_group(prompt_ids, args.rollouts_per_problem)

        stats["total_problems"] += 1
        problem_any_correct = False

        for rollout in rollouts:
            stats["total_rollouts"] += 1
            full_text = prompt_text + rollout.text

            pred = extract_final_answer(full_text)
            if pred is None:
                stats["no_answer"] += 1
                label = 0.0
            else:
                is_correct = (pred == expected)
                label = 1.0 if is_correct else 0.0
                if is_correct:
                    stats["correct_rollouts"] += 1
                    problem_any_correct = True

            prefixes = slice_prefixes(full_text, prompt_text, args.snapshots_per_rollout)

            for i, prefix_text in enumerate(prefixes):
                ratio = (i + 1) / len(prefixes)
                # 修复: 使用平方根曲线平滑，让早期 snapshot 信号更强
                label_smoothed = smooth_label(label, ratio)
                records.append({
                    "text":  prefix_text,
                    "label": label_smoothed,
                })

        if problem_any_correct:
            stats["passed_problems"] += 1

    # ── 6. 保存 ───────────────────────────────────────────────────
    random.shuffle(records)

    output_path = args.output_path
    if output_path.startswith("root/"):
        output_path = "/" + output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 7. 统计 ───────────────────────────────────────────────────
    pass_1 = stats["correct_rollouts"] / max(stats["total_rollouts"], 1)
    pass_k = stats["passed_problems"]  / max(stats["total_problems"],  1)

    # 标签分布统计
    all_labels = [r["label"] for r in records]
    label_dist = {
        "[0.0,0.25)":  sum(1 for l in all_labels if l < 0.25),
        "[0.25,0.5)":  sum(1 for l in all_labels if 0.25 <= l < 0.5),
        "[0.5,0.75)":  sum(1 for l in all_labels if 0.5  <= l < 0.75),
        "[0.75,1.0]":  sum(1 for l in all_labels if l >= 0.75),
    }

    wandb.log({
        "eval/pass@1":                   pass_1,
        f"eval/pass@{args.rollouts_per_problem}": pass_k,
        "eval/total_problems":           stats["total_problems"],
        "eval/valid_rollouts":           stats["total_rollouts"] - stats["no_answer"],
        "eval/no_answer_rate":           stats["no_answer"] / max(stats["total_rollouts"], 1),
        "data/label_low":                label_dist["[0.0,0.25)"],
        "data/label_mid_low":            label_dist["[0.25,0.5)"],
        "data/label_mid_high":           label_dist["[0.5,0.75)"],
        "data/label_high":               label_dist["[0.75,1.0]"],
    })

    print("\n" + "=" * 60)
    print(f"  测试题目总数:  {stats['total_problems']} 题")
    print(f"  生成轨迹总数:  {stats['total_rollouts']} 条")
    print(f"  无答案轨迹:    {stats['no_answer']} 条")
    print(f"  Pass@1:        {pass_1:.2%}")
    print(f"  Pass@{args.rollouts_per_problem}:        {pass_k:.2%}")
    print(f"  标签分布:      {label_dist}")
    print(f"  Value 样本数:  {len(records)} 条")
    print(f"  已保存至:      {output_path}")
    print("=" * 60)

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path",       type=str,
                   default="/root/autodl-tmp/models/Qwen3-0.6B-Base")
    p.add_argument("--lora_path",             type=str,
                   default="/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1/final")
    p.add_argument("--output_path",           type=str,
                   default="/root/autodl-tmp/data/value_data.jsonl")
    p.add_argument("--start_idx",             type=int, default=0)
    p.add_argument("--end_idx",               type=int, default=500)
    p.add_argument("--rollouts_per_problem",  type=int, default=4)
    p.add_argument("--snapshots_per_rollout", type=int, default=4)
    p.add_argument("--max_new_tokens",        type=int, default=1024)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args)