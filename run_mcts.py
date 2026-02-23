import time
import torch
from transformers import AutoTokenizer

from cognitive_qwen import CognitiveQwen
from hf_mcts import HF_MCTS, MCTSConfig

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool——v2/final_lora"

def main():
    print("============== 认知纳米 (Cognitive Nano) MCTS 测试引擎 ==============")
    
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    special_tokens = ['<|python_start|>', '<|python_end|>', '<|output_start|>', '<|output_end|>']
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    model = CognitiveQwen(
        base_model_path=BASE_MODEL_PATH, 
        lora_path=LORA_PATH, 
        device="cuda", 
        vocab_size=len(tokenizer)
    )
    model.eval()

    # 🔥 加大预算：30 次探索足够它算出这道 4 步计算题了
    config = MCTSConfig(
        num_simulations=60,       
        branching_factor=2,      
        max_depth=15,            
        c_puct=1.414,            
        expansion_temperature=0.6, 
        expansion_top_k=20
    )
    
    mcts_engine = HF_MCTS(model, tokenizer, config, device=model.device)

    question = "A factory produces 1250 gadgets per day. They operate for 24 days a month. If each gadget is sold for $15 and the monthly operating cost is $120,500, what is the exact net profit for the month?"
    
    # 🔥 100% 对齐 SFT 训练格式的 One-Shot 模板
    messages = [
        {"role": "system", "content": "You are a meticulous math genius. You solve problems step-by-step."},
        {"role": "user", "content": "What is 15 multiplied by 8?"},
        {"role": "assistant", "content": "<think>\nWe need to compute the product of 15 and 8.\nLet's compute.\n<|python_start|>15 * 8<|python_end|>\n<|output_start|>120<|output_end|>\nSo 15 * 8 = 120.\nThus final answer: 120.\n</think>\n\nThe final answer is 120"},
        {"role": "user", "content": question}
    ]

    # 使用官方模板引擎组装，绝不差一个空格
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 引导模型进入思考空间
    raw_prompt += "<think>\n"


    print(f"\n[问题]: {question}")
    print(f"[配置]: 预算={config.num_simulations}次模拟 | 分支度={config.branching_factor}\n")
    print("正在启动 MCTS 树搜索，请关注终端中沙箱的疯狂拦截日志...\n")
    print("=" * 60)
    
    t0 = time.time()
    best_response, root_q_value = mcts_engine.search(raw_prompt)
    elapsed = time.time() - t0
    
    print("=" * 60)
    print(f"\n✅ MCTS 搜索完成! 耗时: {elapsed:.2f} 秒")
    print(f"根节点 Q 值 (胜率评估): {root_q_value:.4f}")
    
    print("\n👑 MCTS 找出的最佳推理路径：")
    print("-" * 50)
    print("<think>\n" + best_response)
    print("-" * 50)

if __name__ == "__main__":
    main()