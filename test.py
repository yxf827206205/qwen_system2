import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
# ⚠️ 确保这里指向你最新微调出来的 LoRA 路径！
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool——v2/final_lora" 

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

def test_natural_sandbox():
    print("1. 加载 Tokenizer 与模型...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    
    # 显式重新注册 Token
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

    messages = [
        {"role": "system", "content": "You are a meticulous math genius. You solve problems step-by-step."},
        {"role": "user", "content": "What is 15 multiplied by 8?"},
        {"role": "assistant", "content": "<think>\nWe need to compute the product of 15 and 8.\nLet's compute.\n<|python_start|>15 * 8<|python_end|>\n<|output_start|>120<|output_end|>\nSo 15 * 8 = 120.\nThus final answer: 120.\n</think>\n\nThe final answer is 120"},
        {"role": "user", "content": "What is the exact result of 12345 multiplied by 987?"}
    ]

    # 🔥 100% 对齐你的 SFT 训练逻辑！
    # 注意这里 add_generation_prompt=True，它会自动帮你补上最后一条 <|im_start|>assistant\n
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    raw_prompt += "<think>\n"

    print("-" * 50)
    print("现在喂给模型的，是跟它在 SFT 训练槽里泡着时一模一样的原汤格式：")
    print(raw_prompt)
    print("-" * 50)


    py_end_id = tokenizer.convert_tokens_to_ids('<|python_end|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    stop_ids = [tokenizer.eos_token_id, py_end_id, im_end_id] 
    stop_ids = [sid for sid in stop_ids if sid is not None]
    
    generated_text = raw_prompt
    
    # 🔥 Agent 循环：最多允许它连续思考和调用沙箱 5 次
    max_steps = 5
    
    for step in range(max_steps):
        print(f"\n--- [第 {step+1} 步思考中...] ---")
        inputs = tokenizer(generated_text, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                eos_token_id=stop_ids,
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            
        new_tokens = outputs[0][len(inputs.input_ids[0]):]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        print(f"模型原始输出:\n{new_text}")
        
        # 🔥 幻觉截断器：如果模型输出了 python_end，立刻把后面抢答的废话剪掉！
        if "<|python_end|>" in new_text:
            # 以 python_end 为界限切割
            split_parts = new_text.split("<|python_end|>")
            # 只保留 python_end 以及它前面的干净代码
            clean_new_text = split_parts[0] + "<|python_end|>"
            
            # 把干净的文本拼接到总上下文里
            generated_text += clean_new_text
            
            # 🔥 修复正则 Bug：找出所有的代码块，但只取最后一次生成的那个！
            matches = re.findall(r'<\|python_start\|>(.*?)<\|python_end\|>', generated_text, re.DOTALL)
            if matches:
                code_to_run = matches[-1].strip() # <--- 关键！取最后一个！
                calc_result = execute_python_sandbox(code_to_run)
                
                # 注入真实的沙箱结果
                tool_output_text = f"\n<|output_start|>{calc_result}<|output_end|>\n"
                generated_text += tool_output_text
                print(f"向模型脑海注入沙箱结果: {tool_output_text.strip()}")
            else:
                print("[⚠️ 警告] 提取代码失败。")
                break
                
        elif "<|im_end|>" in new_text:
            generated_text += new_text
            print("\n🏁 模型输出 <|im_end|>，思考与解答结束。")
            break
        else:
            generated_text += new_text
            print("\n[⚠️ 提示] 模型生成达到了长度限制，或者格式未闭环。")

    print("\n" + "="*50)
    print("最终的完整推导过程：")
    print("="*50)
    final_output_display = generated_text.split("<|im_start|>assistant\n")[-1]
    print(final_output_display)

# def main():
#     print("1. 加载 Tokenizer 与模型...")
#     tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    
#     # 拿到我们特殊 Token 的专属身份证号 (ID)
#     py_start_id = tokenizer.convert_tokens_to_ids('<|python_start|>')
#     py_end_id = tokenizer.convert_tokens_to_ids('<|python_end|>')
#     print(f"\n[目标 ID] python_start: {py_start_id}, python_end: {py_end_id}")

#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     base_model.resize_token_embeddings(len(tokenizer))
#     model = PeftModel.from_pretrained(base_model, LORA_PATH)
#     model = model.merge_and_unload()
#     model.eval()

#     # 构造一个肯定需要大数计算的题目，逼它用工具
#     messages = [
#         {"role": "system", "content": "You are a meticulous math genius. You solve problems step-by-step."},
#         {"role": "user", "content": "What is the exact result of 987654 multiplied by 1234?please use python to calculate and give me the answer."}
#     ]
#     raw_prompt = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     ) + "<think>\n"

#     inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)

#     print("\n2. 模型正在纯裸跑推理 (没有沙箱拦截，只看它吐出的内容)...")
#     with torch.inference_mode():
#         outputs = model.generate(
#             inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             max_new_tokens=256, # 只需要生成前面几步，看它会不会出刀
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id
#         )

#     # 剥离 Prompt，只看模型新生成的 Token
#     new_tokens = outputs[0][len(inputs.input_ids[0]):]
#     new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
#     print("\n" + "="*50)
#     print("模型生成的文本内容:")
#     print(new_text)
#     print("="*50)

#     print("\n🔍 显微镜视角 (生成的原始 Token ID):")
#     token_list = new_tokens.tolist()
#     print(token_list)

#     # 判决时刻！
#     print("\n=== 🎯 最终判决 ===")
#     if py_start_id in token_list:
#         print("✅ 完美！模型直接输出了单一的 <|python_start|> Token！它真正掌握了工具召唤术！")
#     else:
#         if "<|python_start|>" in new_text:
#             print("❌ 失败！文本里有标签，但 ID 列表里没有专属 ID。它还在像拼单词一样拼写标签！")
#         else:
#             print("⚠️ 没触发工具，模型可能选择了纯心算，请调整 Prompt 或给个 One-Shot 引导一下。")

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    test_natural_sandbox()