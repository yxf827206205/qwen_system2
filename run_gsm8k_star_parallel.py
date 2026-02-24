import json
import time
import re
import torch
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from cognitive_qwen import CognitiveQwen
from hf_mcts_v2 import HF_MCTS, MCTSConfig

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool——v2/final_lora"
NUM_QUESTIONS = 600  

# ================= 辅助函数 =================
def extract_gsm8k_target(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].replace(',', '').strip()
    return answer_str.strip()

def extract_model_answer(text: str) -> str:
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).replace(',', '').strip()
    
    ans_match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-]+)', text)
    if ans_match:
        return ans_match.group(1).replace(',', '').strip()
    return None

# ================= 主程序 =================
def main():
    # 🔥 增加命令行参数解析
    parser = argparse.ArgumentParser(description="MCTS 并行数据生成器")
    parser.add_argument("--total_shards", type=int, default=1, help="总进程/分片数量")
    parser.add_argument("--shard_idx", type=int, default=0, help="当前进程的分片索引 (0 到 total_shards-1)")
    args = parser.parse_args()

    # 当前分片专属的输出文件
    os.makedirs("data", exist_ok=True)

    OUTPUT_FILE = f"data/gsm8k_star_shard_{args.shard_idx}_of_{args.total_shards}.jsonl"
    
    print(f"============== 🚀 启动 GSM8K STaR 收割机 [分片 {args.shard_idx}/{args.total_shards}] ==============")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    special_tokens = ['<|python_start|>', '<|python_end|>', '<|output_start|>', '<|output_end|>']
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # 2. 加载模型
    print(f"[分片 {args.shard_idx}] 加载模型到显存...")
    model = CognitiveQwen(
        base_model_path=BASE_MODEL_PATH, 
        lora_path=LORA_PATH, 
        device="cuda", 
        vocab_size=len(tokenizer)
    )
    model.eval()

    # 3. 配置 MCTS
    config = MCTSConfig(
        num_simulations=60,       
        branching_factor=2,      
        max_depth=15,            
        c_puct=1.414,            
        expansion_temperature=0.6, 
        expansion_top_k=20
    )
    mcts_engine = HF_MCTS(model, tokenizer, config, device=model.device)
    
    # 4. 加载并切分数据集
    print(f"[分片 {args.shard_idx}] 准备数据集...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # 保证所有进程打乱顺序一致，截取前 500 道
    dataset = dataset.shuffle(seed=42).select(range(NUM_QUESTIONS))
    
    # 🔥 核心切片逻辑：将 500 道题平均分给多个进程
    dataset = dataset.shard(num_shards=args.total_shards, index=args.shard_idx)
    print(f"✅ [分片 {args.shard_idx}] 分配到 {len(dataset)} 道题目！")

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            pass 

    success_count = 0
    start_time = time.time()
    
    for idx, item in enumerate(tqdm(dataset, desc=f"MCTS 进程 {args.shard_idx}", position=args.shard_idx)):
        question = item["question"]
        raw_answer = item["answer"]
        expected_answer = extract_gsm8k_target(raw_answer)
        
        messages = [
            {"role": "system", "content": "You are a meticulous math genius. You solve problems step-by-step."},
            {"role": "user", "content": question}
        ]

        raw_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        raw_prompt += "<think>\n"
        
        best_response, root_q_value = mcts_engine.search(raw_prompt, expected_answer=expected_answer)
        
        has_tool = '<|python_start|>' in best_response
        model_ans = extract_model_answer(best_response)
        is_correct = (model_ans is not None) and (model_ans == expected_answer)
        
        if has_tool and is_correct:
            messages.append({"role": "assistant", "content": "<think>\n" + best_response})
            record = {
                "messages": messages,
                "q_value": root_q_value,
                "is_correct": True
            }
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            success_count += 1
            tqdm.write(f"✅ [进程 {args.shard_idx} | 第 {idx+1} 题] 成功! 进度: {success_count}/{idx+1}")
        else:
            tqdm.write(f"❌ [进程 {args.shard_idx} | 第 {idx+1} 题] 丢弃。提取: {model_ans}, 期望: {expected_answer}")

    elapsed = (time.time() - start_time) / 3600 
    print(f"\n🎉 [分片 {args.shard_idx}] 结束！耗时: {elapsed:.2f} 小时，成功: {success_count}/{len(dataset)}")

if __name__ == "__main__":
    main()