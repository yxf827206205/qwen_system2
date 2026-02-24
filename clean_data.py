import json
import glob
import os

# 自动匹配你当前目录下所有的分片文件 (如果你后来合并了，也可以直接改成具体的文件名)
INPUT_PATTERN = "/root/autodl-tmp/data/gsm8k_star.jsonl" 
OUTPUT_FILE = "/root/autodl-tmp/data/clean_sft_data.jsonl"

def main():
    print("============== 🧹 启动 STaR 数据清洗与合并程序 ==============")
    
    # 找到所有匹配的文件
    input_files = glob.glob(INPUT_PATTERN)
    
    # 如果没找到分片文件，尝试找合并后的名字
    if not input_files:
        if os.path.exists("final_gsm8k_star_500.jsonl"):
            input_files = ["final_gsm8k_star_500.jsonl"]
        else:
            print("❌ 找不到输入文件！请检查你的 jsonl 文件名是否正确。")
            return

    print(f"找到 {len(input_files)} 个数据文件，准备开始清洗...")
    
    cleaned_count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for file_path in input_files:
            print(f"正在清洗文件: {file_path}")
            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                        
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    
                    # 1. 过滤掉 System Prompt，只保留 user 和 assistant
                    clean_messages = [m for m in messages if m["role"] != "system"]
                    
                    # 2. 找到 assistant 的回答，暴力切除末尾的 <|im_end|>
                    for m in clean_messages:
                        if m["role"] == "assistant":
                            # 替换掉 <|im_end|> 并去掉末尾多余的空格或换行
                            m["content"] = m["content"].replace("<|im_end|>", "").strip()
                            
                    # 3. 组装成纯净的数据格式（扔掉 q_value 和 is_correct）
                    new_data = {"messages": clean_messages}
                    
                    fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                    cleaned_count += 1

    print(f"\n✅ 清洗合并彻底完成！共处理 {cleaned_count} 条极品轨迹。")
    print(f"💾 纯净数据已保存至: {OUTPUT_FILE}")
    print("\n🚀 下一步：")
    print(f"打开你的 sft_qwen.py，把 DATASETS 的路径改成 '{OUTPUT_FILE}'，然后直接启动微调吧！")

if __name__ == "__main__":
    main()