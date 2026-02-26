import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from cognitive_qwen import CognitiveQwen  # 必须引入你自定义的模型类

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
LORA_PATH = "/root/autodl-tmp/checkpoints/qwen_sft_star_v3/final_lora" 
VALUE_DATA_PATH = "value_data.jsonl" # 你的状态价值数据集
OUTPUT_HEAD_PATH = "/root/autodl-tmp/checkpoints/value_head.pt" 

BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-4 # 只有一个线性层，学习率可以稍微大一点点

# ================= 1. 数据集定义 =================
class ValueDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = float(item["label"]) # 1.0 为通往正确的节点，0.0 为死胡同节点

        encoded = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)
        }

# ================= 2. 主训练程序 =================
def main():
    print("============== 启动 Value Head 价值直觉训练 ==============")
    
    print("1. 加载 Tokenizer 与模型骨架...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载你的 CognitiveQwen 架构
    model = CognitiveQwen(
        base_model_path=BASE_MODEL_PATH, 
        lora_path=LORA_PATH, 
        device="cuda", 
        vocab_size=len(tokenizer)
    )
    
    print("2. 🥶 实施局部冰冻手术 (Freezing Backbone)...")
    # 把模型里所有的参数都冻结（不需要计算梯度，极其省显存！）
    for param in model.parameters():
        param.requires_grad = False
        
    # 🔥 唯独把 Value Head 的封印解开！
    for param in model.value_head.parameters():
        param.requires_grad = True

    # 打印确认
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"当前可训练参数量: {trainable_params} (应该非常小，通常只有几千/几万个参数)")

    print("3. 准备数据与优化器...")
    # 假设你已经准备好了 value_data.jsonl
    # 如果还没有，你需要先跑个脚本，从过去的 MCTS 失败/成功树里提炼出 text 和 1.0/0.0 label
    dataset = ValueDataset(VALUE_DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=LR)
    
    # 损失函数：因为你的 CognitiveQwen 里 value 是原始 logit，MCTS 里用了 sigmoid
    # 所以我们用 BCEWithLogitsLoss，它是最完美的二分类/概率回归损失函数
    criterion = nn.BCEWithLogitsLoss()

    print("4. 开始爆炒价值头！(Training Value Head)...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["label"].to(model.device)

            optimizer.zero_grad()

            # 前向传播 (极其快，因为底层被冻结了)
            # 输出的 outputs[1] 就是 Value Head 的打分 (shape: batch, seq_len, 1)
            outputs = model.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            value_logits = model.value_head(hidden_states).squeeze(-1) # shape: (batch, seq_len)

            # 我们只需要评估最后一个 Token 的价值（因为它代表了当前步骤的最终局势）
            # 找到实际序列长度的最后一个有效 Token (跳过 padding)
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=model.device)
            last_token_values = value_logits[batch_indices, seq_lengths]

            # 计算损失并反向传播
            loss = criterion(last_token_values, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")

    print("5. 训练完成，剥离并保存 Value Head...")
    # 我们不需要保存几 GB 的大模型，只需要把价值头的权重保存下来
    torch.save(model.value_head.state_dict(), OUTPUT_HEAD_PATH)
    print(f"✅ Value Head 权重已单独保存至: {OUTPUT_HEAD_PATH}")

if __name__ == "__main__":
    main()