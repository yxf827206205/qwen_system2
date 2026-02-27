"""
select_value_data.py  —— Value Head 训练数据采集脚本（Bug 修复版）

修复点:
  1. extract_final_answer: 
     - 使用 re.findall 取最后一个匹配（而非 re.search 取第一个）
     - 正则不再贪婪吃掉句号：[\d\.\-,]+ → 精确的数字/小数匹配
     - 额外 strip 掉尾部句号和空格，彻底防御
  2. 增加 pred vs expected 调试日志，出问题时能看到对比
  3. 标签平滑: 平方根曲线（保留原逻辑）
  4. 增加标签分布统计输出
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
    """
    从生成文本里提取最终答案。

    修复点：
    1. 使用 findall 取 **最后一个** 匹配，避免 <think> 内部的中间答案干扰
    2. 正则改为 [-\d,]+(?:\.\d+)? 精确匹配数字/小数，不贪婪吃句号
    3. 结果 strip 掉尾部句号，彻底防御 "36." != "36" 的坑
    """
    # 优先匹配 \boxed{...}（取最后一个）
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        return boxed_matches[-1].replace(",", "").strip().rstrip(".")

    # 其次匹配自然语言 "The final answer is ..."（取最后一个）
    nl_matches = re.findall(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([-\d,]+(?:\.\d+)?)",
        text
    )
    if nl_matches:
        return nl_matches[-1].replace(",", "").strip().rstrip(".")

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
    标签平滑：使用平方根曲线。

    对应关系 (n_snapshots=4):
      snapshot 1/4: ratio=0.25  → curved=0.50  正确:[0.25,0.75]
      snapshot 2/4: ratio=0.50  → curved=0.71  正确:[0.146,0.854]
      snapshot 3/4: ratio=0.75  → curved=0.87  正确:[0.067,0.933]
      snapshot 4/4: ratio=1.00  → curved=1.00  正确:[0.0,1.0]
    """
    ratio_curved = ratio ** 0.5
    return round(0.5 + (label - 0.5) * ratio_curved, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 主采集逻辑
# ──────────────────────────────────────────────────────────────────────────────

def collect(args):
    print("=" * 60)
    print(" Value Head 数据采集启动 (Bug 修复版)")
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
        "total_rollouts":   0,
        "correct_rollouts": 0,
        "no_answer":        0,
        "mislabel_fixed":   0,   # 🔥 新增：统计因修复正则而改变判定的数量
        "total_problems":   0,
        "passed_problems":  0,
    }

    # 🔥 调试：前 N 题打印 pred vs expected，方便验证修复效果
    DEBUG_PRINT_N = 10
    debug_count = 0

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

            # 🔥 调试输出：前几题打印比对结果，验证修复是否生效
            if debug_count < DEBUG_PRINT_N:
                tqdm.write(
                    f"  [Debug #{debug_count}] expected={repr(expected)}, "
                    f"pred={repr(pred)}, correct={pred == expected if pred else False}"
                )
                debug_count += 1

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

    all_labels = [r["label"] for r in records]
    label_dist = {
        "[0.0,0.25)":  sum(1 for l in all_labels if l < 0.25),
        "[0.25,0.5)":  sum(1 for l in all_labels if 0.25 <= l < 0.5),
        "[0.5,0.75)":  sum(1 for l in all_labels if 0.5  <= l < 0.75),
        "[0.75,1.0]":  sum(1 for l in all_labels if l >= 0.75),
    }

    # 🔥 新增：验证 terminal 样本的标签分布（正确 vs 错误）
    terminal_labels = []
    for rec in records:
        text = rec["text"]
        if "<|im_end|>" in text or "\\boxed{" in text:
            terminal_labels.append(rec["label"])

    terminal_correct = sum(1 for l in terminal_labels if l >= 0.9)
    terminal_wrong   = sum(1 for l in terminal_labels if l <= 0.1)
    terminal_ambig   = len(terminal_labels) - terminal_correct - terminal_wrong

    wandb.log({
        "eval/pass@1":                              pass_1,
        f"eval/pass@{args.rollouts_per_problem}":   pass_k,
        "eval/total_problems":                      stats["total_problems"],
        "eval/valid_rollouts":                      stats["total_rollouts"] - stats["no_answer"],
        "eval/no_answer_rate":                      stats["no_answer"] / max(stats["total_rollouts"], 1),
        "data/label_low":                           label_dist["[0.0,0.25)"],
        "data/label_mid_low":                       label_dist["[0.25,0.5)"],
        "data/label_mid_high":                      label_dist["[0.5,0.75)"],
        "data/label_high":                          label_dist["[0.75,1.0]"],
        "data/terminal_correct":                    terminal_correct,
        "data/terminal_wrong":                      terminal_wrong,
        "data/terminal_ambiguous":                  terminal_ambig,
    })

    print("\n" + "=" * 60)
    print(f"  测试题目总数:  {stats['total_problems']} 题")
    print(f"  生成轨迹总数:  {stats['total_rollouts']} 条")
    print(f"  无答案轨迹:    {stats['no_answer']} 条")
    print(f"  Pass@1:        {pass_1:.2%}")
    print(f"  Pass@{args.rollouts_per_problem}:        {pass_k:.2%}")
    print(f"  标签分布:      {label_dist}")
    print(f"  Value 样本数:  {len(records)} 条")
    print()
    print(f"  [Terminal 质量验证]")
    print(f"    terminal 正确样本 (label≥0.9): {terminal_correct}")
    print(f"    terminal 错误样本 (label≤0.1): {terminal_wrong}")
    print(f"    terminal 模糊样本 (中间值):     {terminal_ambig}  ← 应接近 0")
    if terminal_ambig > (terminal_correct + terminal_wrong) * 0.05:
        print(f"  ⚠️  警告：terminal 模糊样本比例过高，说明 extract_final_answer 仍有漏网")
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