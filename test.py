import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_sft_mix/final_lora" # 确保路径对

def execute_python_sandbox(code_str):
    print(f"\n[🚀 触发沙箱拦截!] 正在执行代码: {code_str.strip()}")
    try:
        clean_code = code_str.strip()
        result = eval(clean_code, {"__builtins__": {}}, {})
        if isinstance(result, float):
            result_str = f"{result:.4f}".rstrip('0').rstrip('.')
        else:
            result_str = str(result)
        print(f"[✅ 沙箱返回结果]: {result_str}\n")
        return result_str
    except Exception as e:
        print(f"[❌ 沙箱执行报错]: {e}\n")
        return f"Error: {e}"

def test_forced_sandbox():
    print("1. 加载 Tokenizer 与模型...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    
    # 🔥 修复 1：显式重新注册 Token，确保绝对不会变成 unk
    special_tokens = ['<|python_start|>', '<|python_end|>', '<|output_start|>', '<|output_end|>']
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()

    raw_prompt = """<|im_start|>system
You are a math genius. You MUST use the python sandbox for calculations by writing pure python expressions (like 2+2) between <|python_start|> and <|python_end|>.<|im_end|>
<|im_start|>user
What is 15 multiplied by 8?<|im_end|>
<|im_start|>assistant
<think>
To find the product of 15 and 8, I will use the python sandbox.
<|python_start|>15 * 8<|python_end|><|output_start|>120<|output_end|>
The result is 120.
</think>

\\The final answer is 120<|im_end|>
<|im_start|>user
What is the exact result of 12345 multiplied by 98765?<|im_end|>
<|im_start|>assistant
<think>
To calculate this exactly, I will use the python sandbox:
<|python_start|>"""

    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    py_end_id = tokenizer.convert_tokens_to_ids('<|python_end|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    stop_ids = [tokenizer.eos_token_id, py_end_id, im_end_id]
    stop_ids = [sid for sid in stop_ids if sid is not None]
    
    print("正在等待模型补全算式...")
    generated_text = raw_prompt
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            eos_token_id=stop_ids,
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        
    new_tokens = outputs[0][len(input_ids[0]):]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    generated_text += new_text
    
    # 打印原始字符串看看它到底输出了什么鬼东西
    print("正在等待模型补全算式...")
    generated_text = raw_prompt
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            eos_token_id=stop_ids,
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        
    new_tokens = outputs[0][len(input_ids[0]):]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    print(f"模型输出的算式部分 (Raw): {repr(new_text)}")
    
    # 🔥 终极土匪逻辑：只要它输出了文本，不管是带 python_end 还是 im_end，
    # 我们通通把特殊符号剥离，强行扔进沙箱！
    if len(new_tokens) > 0:
        # 清理掉所有可能捣乱的结尾符
        code_to_run = new_text.replace("<|python_end|>", "").replace("<|im_end|>", "").strip()
        
        # 执行沙箱
        calc_result = execute_python_sandbox(code_to_run)
        
        # 帮模型擦屁股：哪怕它刚才输出了 im_end，我们在拼接历史时也把它纠正为 python_end
        tool_output_text = f"{code_to_run}<|python_end|><|output_start|>{calc_result}<|output_end|>\nSo the exact result is "
        
        generated_text += tool_output_text
        print(f"向模型注入结果: {tool_output_text}")
        
        full_inputs = tokenizer(generated_text, return_tensors="pt").to(model.device)
        print("正在等待模型根据沙箱结果得出最终结论...")
        outputs2 = model.generate(
            full_inputs.input_ids,
            attention_mask=full_inputs.attention_mask,
            max_new_tokens=100,
            eos_token_id=stop_ids,
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        final_tokens = outputs2[0][len(full_inputs.input_ids[0]):]
        final_text = tokenizer.decode(final_tokens, skip_special_tokens=False)
        generated_text += final_text

    print("\n" + "="*50)
    print("最终的完整推导过程：")
    print("="*50)
    final_output_display = generated_text.split("What is the exact result of 12345 multiplied by 98765?<|im_end|>\n<|im_start|>assistant\n")[-1]
    print(final_output_display)

if __name__ == "__main__":
    test_forced_sandbox()