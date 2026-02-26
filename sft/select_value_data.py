
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
    """和 GRPO 训练时保持完全一致的 Prompt 格式。"""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt + "<think>\n"

def slice_prefixes(full_text: str, prompt_text: str, n_snapshots: int) -> list[str]:
   
    gen_part = full_text[len(prompt_text):]
    
    # 使用正则表达式，按照 \n\n 或者 <|output_end|> 切分，并且保留分隔符
    # 这样能保证切下来的都是“一个完整的推导步骤”或“一次完整的工具调用”
    chunks = re.split(r'(\n\n|<\|output_end\|>)', gen_part)
    
    # 重新把文本和分隔符拼合成完整的逻辑块
    merged_chunks = []
    for i in range(0, len(chunks)-1, 2):
        merged_chunks.append(chunks[i] + chunks[i+1])
    if len(chunks) % 2 != 0 and chunks[-1]:
        merged_chunks.append(chunks[-1])
        
    total_chunks = len(merged_chunks)
    
    # 如果没分出几个逻辑块，直接返回完整的
    if total_chunks < 3:
        return [full_text]

    prefixes = []
    # 均匀抽样指定的帧数
    for i in range(1, n_snapshots + 1):
        cut_idx = int(total_chunks * i / n_snapshots)
        if cut_idx == 0:
            cut_idx = 1
            
        cut_text = "".join(merged_chunks[:cut_idx])
        prefix = prompt_text + cut_text
        
        if prefix not in prefixes:
            prefixes.append(prefix)
            
    # 确保最后一条绝对是全量文本
    if full_text not in prefixes:
        prefixes.append(full_text)
        
    return prefixes

# ──────────────────────────────────────────────────────────────────────────────
# 主采集逻辑
# ──────────────────────────────────────────────────────────────────────────────

def collect(args):
    print("=" * 60)
    print(" Value Head 数据采集与 RL 智商评测启动 (带 Python 沙箱)")
    print("=" * 60)
    wandb.init(
        project="cognitive-nano-qwen-test1",
        name=f"eval-rl-chunk-{args.start_idx}",
        config=vars(args)  # 自动记录所有命令行参数
    )

    # ── 1. 加载 Tokenizer ──────────────────────────────────────────
    print("\n1. 加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    new_tokens = ["<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)

    # ── 2. 加载模型 ────────────────────────────────────────────────
    print("2. 加载 Base Model 并挂载 GRPO LoRA ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=False)
    # 合并 LoRA 权重以获得极致推理速度
    model = model.merge_and_unload()   
    model.eval()
    print("    模型合并加载完毕")

    # ── 3. 初始化极其重要的环境 (沙箱 + 生成器) ───────────────────
    cfg = GRPOConfig()
    cfg.max_segment_tokens = 512
    cfg.max_total_tokens = args.max_new_tokens
    cfg.temperature = 0.6  # 稍微给点温度，探索不同的错误/正确路径
    
    sandbox = PythonSandbox()
    generator = ToolAwareGenerator(model, tokenizer, sandbox, cfg)

    # ── 4. 加载 GSM8K ─────────────────────────────────────────────
    print(f"3. 加载 GSM8K 数据集 (索引 {args.start_idx} 到 {args.end_idx}) ...")
    raw = load_dataset("gsm8k", "main", split="train")
    # 必须固定 seed=42，保证所有进程打乱后的全量题目顺序是一模一样的！
    raw = raw.shuffle(seed=42)
    # 核心：根据传入的索引去切分这个进程该做的题
    actual_end = min(args.end_idx, len(raw))
    raw = raw.select(range(args.start_idx, actual_end))

    # ── 5. 开始采集 ───────────────────────────────────────────────
    records = []
    
    stats = {
        "total_rollouts": 0,    # 总轨迹数
        "correct_rollouts": 0,  # 正确的轨迹数 (计算 pass@1)
        "no_answer": 0,         # 没提取出答案的残次品
        "total_problems": 0,    # 总题数
        "passed_problems": 0    # 至少有一条轨迹做对的题数 (计算 pass@k)
    }

    print("\n 开始并发 Rollout 并提取价值切片...")
    for item in tqdm(raw, desc="采集中"):
        question = item["question"]
        expected = item["answer"].split("####")[-1].replace(",", "").strip()
        prompt_text = build_prompt(tokenizer, question)

        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")

        # 并发执行带沙箱的完整推理！
        rollouts = generator.generate_group(prompt_ids, args.rollouts_per_problem)

        stats["total_problems"] += 1
        problem_any_correct = False  # 雷达：这道题有哪怕一次是对的吗？

        for rollout in rollouts:
            stats["total_rollouts"] += 1
            full_text = prompt_text + rollout.text
            
            # 判断答案是否正确
            pred = extract_final_answer(full_text)
            if pred is None:
                stats["no_answer"] += 1
                label = 0.0
            else:
                is_correct = (pred == expected)
                label = 1.0 if is_correct else 0.0
                if is_correct:
                    stats["correct_rollouts"] += 1
                    problem_any_correct = True # 触发容错通过！

            # 切割前缀，生成多个训练样本
            prefixes = slice_prefixes(full_text, prompt_text, args.snapshots_per_rollout)

            for i, prefix_text in enumerate(prefixes):
                ratio = (i + 1) / len(prefixes)
                # 极其聪明的平滑逻辑：开头部分即使最终错了，也不该直接打死(0.0)；最终对了，开局也不代表稳赢(1.0)
                label_smoothed = 0.5 + (label - 0.5) * ratio

                records.append({
                    "text": prefix_text,
                    "label": round(label_smoothed, 4),
                })

        # 更新这道题的整体容错胜率
        if problem_any_correct:
            stats["passed_problems"] += 1

    # ── 6. 物理防呆与保存 ─────────────────────────────────────────
    random.shuffle(records)
    
    # 修复路径缺少根目录斜杠的潜在 Bug，并自动创建所有父文件夹
    output_path = args.output_path
    if output_path.startswith("root/"):
        output_path = "/" + output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 7. 打印 ───────────────────────────────────────────
    pass_1 = stats["correct_rollouts"] / max(stats["total_rollouts"], 1)
    pass_k = stats["passed_problems"] / max(stats["total_problems"], 1)
    wandb.log({
        "eval/pass@1": pass_1,
        f"eval/pass@{args.rollouts_per_problem}": pass_k,
        "eval/total_problems": stats["total_problems"],
        "eval/valid_rollouts": stats["total_rollouts"] - stats["no_answer"],
        "eval/no_answer_rate": stats["no_answer"] / max(stats["total_rollouts"], 1)
    })

    print("\n" + "=" * 60)
    print(f" 完美带沙箱采集完成！并输出终极评测报告：")
    print(f"   测试题目总数: {stats['total_problems']} 题")
    print(f"   生成轨迹总数: {stats['total_rollouts']} 条 (每题 {args.rollouts_per_problem} 条)")
    print(f"   无答案轨迹:   {stats['no_answer']} 条")
    print("-" * 60)
    print(f"    Pass@1 (单次盲测准确率): {pass_1:.2%}  <-- 你的 RL 真实独立解题智商")
    print(f"    Pass@{args.rollouts_per_problem} (多次尝试准确率): {pass_k:.2%}  <-- 未来 MCTS 的潜力天花板！")
    print("-" * 60)
    print(f"   生成的 Value 数据样本数: {len(records)} 条")
    print(f"   已保存至: {output_path}")
    print("=" * 60)

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path",       type=str, default="/root/autodl-tmp/models/Qwen3-0.6B-Base")
    p.add_argument("--lora_path",             type=str, default="/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1/final") # 填入你最新的 GRPO output 目录
    p.add_argument("--output_path",           type=str, default="/root/autodl-tmp/data/value_data.jsonl") # 加了前缀斜杠
    p.add_argument("--num_problems",          type=int, default=500, help="从 GSM8K 取多少道题")

    p.add_argument("--start_idx",             type=int, default=0, help="处理的数据集起始索引")
    p.add_argument("--end_idx",               type=int, default=500, help="处理的数据集结束索引")

    p.add_argument("--rollouts_per_problem",  type=int, default=4, help="每道题并发采样几条轨迹")
    p.add_argument("--snapshots_per_rollout", type=int, default=4, help="每条轨迹切成几个前缀样本")
    p.add_argument("--max_new_tokens",        type=int, default=1024, help="最大长度")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args)