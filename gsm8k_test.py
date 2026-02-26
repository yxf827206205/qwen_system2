import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 配置区 =================
# 必须使用合并后的全新基座！
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1/final"
MAX_TEST_SAMPLES = 100  # 先测 100 条看看效果

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
    
    # 调整 embedding 大小以容纳新 token (如果 Base 模型已经扩充过，这步是安全冗余)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 加载 LoRA 权重并将其与底座合并 (极大提升推理速度)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    
    return model, tokenizer

def execute_python_sandbox(code_str):
    """一个极简的 Python 沙箱，用于执行模型生成的数学表达式"""
    try:
        clean_code = code_str.strip()
        result = eval(clean_code, {"__builtins__": {}}, {})
        if isinstance(result, float):
            return f"{result:.4f}".rstrip('0').rstrip('.')
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def generate_with_tools(model, tokenizer, prompt):
    """带工具拦截器的生成循环（修正 Tensor 拼接版）"""
    messages = [{"role": "user", "content": prompt}]
    
    # 格式化 prompt，并手动加上 <think>\n 引导模型进入慢思考状态
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_text += "<think>\n"
    
    # 首次编码
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    python_end_id = tokenizer.convert_tokens_to_ids('<|python_end|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    # 定义停止条件
    stop_ids = [tokenizer.eos_token_id, python_end_id]
    if im_end_id is not None:
        stop_ids.append(im_end_id)
        
    max_steps = 15  # 允许模型最多调用 15 次工具
    generated_text = "<think>\n"
    
    for step in range(max_steps):
        # 让模型生成，停在正常结束或 <|python_end|>
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=stop_ids,
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        
        # 截取新生成的部分 ID
        new_token_ids = outputs[0][len(input_ids[0]):]
        new_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
        generated_text += new_text
        
        # 将 outputs (包含 prompt + newly_generated) 设为下一轮的基座
        input_ids = outputs
        
        # 如果模型输出了 <|python_end|>，说明它发起了工具调用
        if len(new_token_ids) > 0 and new_token_ids[-1].item() == python_end_id:
            
            # 使用 rfind 查找最后一个 <|python_start|>，防止正则匹配到前面的旧代码
            start_idx = generated_text.rfind('<|python_start|>')
            end_idx = generated_text.rfind('<|python_end|>')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                code_to_run = generated_text[start_idx + len('<|python_start|>'):end_idx]
                
                # 真实计算
                calc_result = execute_python_sandbox(code_to_run)
                
                # 拼接执行结果
                tool_output_text = f"<|output_start|>{calc_result}<|output_end|>\n"
                generated_text += tool_output_text
                
                # 🔥 核心修复：把结果编码后直接 concat 到 Tensor 后面，绝对不重新 tokenize 全文！
                tool_output_ids = tokenizer.encode(tool_output_text, return_tensors="pt", add_special_tokens=False).to(model.device)
                input_ids = torch.cat([input_ids, tool_output_ids], dim=-1)
            else:
                break # 解析保护：找不到配对的标签就强行终止
        else:
            # 没有遇到 <|python_end|>，说明正常回答完毕（遇到了 EOS 或 <|im_end|>）
            break
            
    return generated_text

def extract_answer(text):
    """提取生成的答案和真实答案，兼容千分位逗号"""
    # 1. 尝试从 \boxed{} 中提取
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # 2. 尝试从 final answer is 中提取 
    match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-\,]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    return None

def main():
    model, tokenizer = load_model_and_tokenizer()
    
    print("3. 加载 GSM8K 测试集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    test_data = dataset.select(range(min(MAX_TEST_SAMPLES, len(dataset))))
    
    correct_count = 0
    total_count = len(test_data)
    
    print(f"\n开始评估，共 {total_count} 条数据...\n")
    
    for i, item in enumerate(tqdm(test_data)):
        question = item["question"]
        # GSM8K 的真实答案在 #### 后面，同时去除逗号防止比对失败
        ground_truth = item["answer"].split("####")[-1].replace(',', '').strip()
        
        # 生成带工具调用的回复
        response = generate_with_tools(model, tokenizer, question)
        
        # 提取模型答案
        model_answer = extract_answer(response)
        
        is_correct = (model_answer == ground_truth)
        if is_correct:
            correct_count += 1
            
        # 打印前 5 个例子，看看它到底在怎么思考
        if i < 5: 
            print(f"\n[{i+1}] Q: {question}")
            print(f"Model Output:\n{response}")
            print(f"Extracted: {model_answer} | Truth: {ground_truth} | Correct: {is_correct}\n")
            print("=" * 60)

    accuracy = correct_count / total_count * 100
    print(f"\n==== 测试完成 ====")
    print(f"总计: {total_count} | 正确: {correct_count} | 准确率 (Pass@1): {accuracy:.2f}%")

if __name__ == "__main__":
    main()