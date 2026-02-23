import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_sft_mix/final_lora"
MAX_TEST_SAMPLES = 100  # 先测 100 条看看效果，全量测大约 1319 条

def load_model_and_tokenizer():
    print("1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    
    print("2. 加载基础模型并融合 LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # 调整 embedding 大小以容纳新 token
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 加载 LoRA 权重并将其与底座合并 (极大提升推理速度)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    
    return model, tokenizer

def execute_python_sandbox(code_str):
    """一个极简的 Python 沙箱，用于执行模型生成的数学表达式"""
    try:
        # 去除可能多余的空格或换行
        clean_code = code_str.strip()
        # 安全地计算数学表达式
        result = eval(clean_code, {"__builtins__": {}}, {})
        # 格式化为最多保留四位小数的浮点数或整数
        if isinstance(result, float):
            return f"{result:.4f}".rstrip('0').rstrip('.')
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def generate_with_tools(model, tokenizer, prompt):
    """带工具拦截器的生成循环"""
    messages = [{"role": "user", "content": prompt}]
    # 格式化 prompt
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    python_end_id = tokenizer.convert_tokens_to_ids('<|python_end|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>') # 获取 im_end 的 ID
    
    # 把所有可能代表“停止”的 ID 都放进列表里
    stop_ids = [tokenizer.eos_token_id, python_end_id]
    if im_end_id is not None:
        stop_ids.append(im_end_id)
    

    max_steps = 10 # 防止无限循环调用工具
    generated_text = ""
    
    for _ in range(max_steps):
        # 让模型生成，停在正常结束或 <|python_end|>
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=stop_ids,  # <-- 换成包含 im_end_id 的列表
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        
        # 截取新生成的部分
        new_tokens = outputs[0][len(input_ids[0]):]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        generated_text += new_text
        
        # 如果模型输出了 <|python_end|>，说明它发起了工具调用
        if new_tokens[-1].item() == python_end_id:
            # 提取 <|python_start|> 和 <|python_end|> 之间的代码
            match = re.search(r'<\|python_start\|>(.*?)<\|python_end\|>', generated_text, re.DOTALL)
            if match:
                code_to_run = match.group(1)
                # 执行真实计算
                calc_result = execute_python_sandbox(code_to_run)
                
                # 拼接执行结果
                tool_output_text = f"<|output_start|>{calc_result}<|output_end|>"
                generated_text += tool_output_text
                
                # 将拼接后的完整文本重新 tokenize，喂回给模型继续生成
                full_current_text = input_text + generated_text
                input_ids = tokenizer.encode(full_current_text, return_tensors="pt").to(model.device)
            else:
                # 解析失败，强行终止
                break
        else:
            # 正常生成结束
            break
            
    return generated_text

def extract_answer(text):
    """提取生成的答案和真实答案"""
    # 1. 尝试从 \boxed{} 中提取
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # 2. 尝试从 The final answer is 中提取 
    match = re.search(r'final answer is\s*([0-9.\-]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

def main():
    model, tokenizer = load_model_and_tokenizer()
    
    print("3. 加载 GSM8K 测试集...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_data = dataset.select(range(min(MAX_TEST_SAMPLES, len(dataset))))
    
    correct_count = 0
    total_count = len(test_data)
    
    print(f"\n开始评估，共 {total_count} 条数据...\n")
    
    for i, item in enumerate(tqdm(test_data)):
        question = item["question"]
        # GSM8K 的真实答案在 #### 后面
        ground_truth = item["answer"].split("####")[-1].strip()
        
        # 生成带工具调用的回复
        response = generate_with_tools(model, tokenizer, question)
        
        # 提取模型答案
        model_answer = extract_answer(response)
        
        # 简单比对（实际工程中可能需要更严谨的正则匹配，比如去除逗号等）
        is_correct = (model_answer == ground_truth)
        if is_correct:
            correct_count += 1
            
        if i < 3: # 打印前几个例子看看效果
            print(f"\n[{i+1}] Q: {question}")
            print(f"Model Output:\n{response}")
            print(f"Extracted: {model_answer} | Truth: {ground_truth} | Correct: {is_correct}\n")
            print("-" * 50)

    accuracy = correct_count / total_count * 100
    print(f"\n==== 测试完成 ====")
    print(f"总计: {total_count} | 正确: {correct_count} | 准确率 (Pass@1): {accuracy:.2f}%")

if __name__ == "__main__":
    main()