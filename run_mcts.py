import os
import re
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# 引入你的自定义模块
from cognitive_qwen import CognitiveQwen
from hf_mcts_v2 import HF_MCTS, MCTSConfig

# ================= 1. 配置区 =================
CONFIG = {
    "base_model_path": "/root/autodl-tmp/models/Qwen3-0.6B-Base",
    "lora_path": "/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1/final",
    "value_head_path": "/root/autodl-tmp/checkpoints/value_head.pt",
    
    "num_test_samples": 10,  # 先拿 100 道题试试水，全量是 1319 道
    "batch_size": 2,          # MCTS 树搜索极其消耗算力，通常 batch_size=1
    
    # MCTS 核心超参：你可以随意调整搜索的广度和深度
    "num_simulations": 64,    # 模拟次数（思考的努力程度，越大越准但也越慢）
    "branching_factor": 3,    # 每次扩展几个分支
    "max_depth":15           # 树的最大深度
}

# ================= 2. 工具函数 =================
def build_prompt(tokenizer, question: str) -> str:
    """和之前训练保持一致的 Prompt 格式"""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt + "<think>\n"

def extract_ground_truth(answer_str: str) -> str:
    """从 GSM8K 的 answer 字段提取纯数字"""
    return answer_str.split("####")[-1].replace(",", "").strip()

def extract_model_answer(text: str) -> str:
    """从模型生成的文本中提取数字"""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None

# ================= 3. 主评测流程 =================
def main():
    print("=" * 60)
    print("🚀 启动 GSM8K MCTS 终极评测 (System 2 慢思考)")
    print("=" * 60)

    print("\n1. 加载 Tokenizer 与词表对齐...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"], trust_remote_code=True)
    new_tokens = ["<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\n2. 加载 CognitiveQwen 与 裁判大脑 (Value Head)...")
    model = CognitiveQwen(
        base_model_path=CONFIG["base_model_path"],
        lora_path=CONFIG["lora_path"],
        device="cuda",
        vocab_size=len(tokenizer)
    )
    
    # 加载我们刚刚完美收敛的价值网络权重
    print(f"   📥 正在挂载裁判大脑: {CONFIG['value_head_path']}")
    model.value_head.load_state_dict(torch.load(CONFIG["value_head_path"], map_location="cuda"))
    model.value_head.to(torch.bfloat16) # 极其重要：确保精度对齐
    model.eval() # 开启推理模式，关闭 Dropout

    print("\n3. 初始化 MCTS 树搜索引警...")
    mcts_config = MCTSConfig(
        num_simulations=CONFIG["num_simulations"],
        branching_factor=CONFIG["branching_factor"],
        max_depth=CONFIG["max_depth"]
    )
    mcts_engine = HF_MCTS(model, tokenizer, mcts_config, device=torch.device("cuda"))

    print("\n4. 加载 GSM8K 测试集 (Test Split)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # 我们打乱并取前 N 道题进行测试
    dataset = dataset.shuffle(seed=42).select(range(min(CONFIG["num_test_samples"], len(dataset))))
    
    correct_count = 0
    total_count = 0
    results_log = []

    print("\n5. 开始树搜索解题 (这需要一些时间，请耐心等待裁判的深思熟虑)...")
    pbar = tqdm(dataset, desc="MCTS Eval")
    
    for item in pbar:
        question = item["question"]
        ground_truth = extract_ground_truth(item["answer"])
        prompt = build_prompt(tokenizer, question)
        
        # 传入树搜索，启动 System 2！
        # 这里的 expected_answer 可传可不传，如果你传了，代码里会优先选对的叶子节点，
        # 但如果是严谨的闭卷考试，你应该传 None，完全让 MCTS 靠裁判的 q_value 去选！
        best_response, best_score = mcts_engine.search(prompt_text=prompt, expected_answer=None) 
        
        # 提取模型答案
        model_answer = extract_model_answer(best_response)
        is_correct = (model_answer == ground_truth)
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        current_acc = correct_count / total_count
        pbar.set_postfix({"Acc": f"{current_acc:.1%}"})
        
        # 记录战报
        results_log.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "best_q_value": best_score,
            "full_response": best_response
        })

    print("\n" + "=" * 60)
    print(f"🎉 MCTS 评测完成！")
    print(f"   总题数: {total_count}")
    print(f"   正确数: {correct_count}")
    print(f"   🏆 最终 MCTS 准确率: {correct_count / total_count:.2%}")
    print("=" * 60)

    # 把详细战报存下来，供你欣赏模型是怎么一步步试错和剪枝的
    with open("mcts_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print("详细战报已保存至 mcts_eval_results.json")

if __name__ == "__main__":
    main()