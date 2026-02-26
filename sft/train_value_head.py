
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))               
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from  cognitive_qwen import CognitiveQwen  



# ================= 1. 超参数与配置区 =================
CONFIG = {
    "base_model_path": "/root/autodl-tmp/models/Qwen3-0.6B-Base",
    "lora_path": "/root/autodl-tmp/checkpoints/qwen_grpo_tool_v1/final", 
    "value_data_path": "/root/autodl-tmp/data/value_data_full.jsonl",    
    "output_head_path": "/root/autodl-tmp/checkpoints/value_head.pt",
    
    "batch_size": 8,                  
    "gradient_accumulation_steps": 8, 
    "epochs": 3,
    "learning_rate": 3e-4,         
    "max_length": 1024,              
    
    "wandb_project": "cognitive-nano-qwen-value-head-training",
    "wandb_run_name": "train-value-head-final1"
}

# ================= 2. 数据集定义 =================
class ValueDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.data = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"找不到数据文件: {jsonl_path}，请检查合并脚本是否成功！")
            
        print(f"正在加载价值数据: {jsonl_path} ...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f" 成功加载 {len(self.data)} 条价值评估数据！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = float(item["label"])

        # 编码并强制截断/填充到统一长度
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

# ================= 3. 主训练程序 =================
def main():
    print("==============  启动 Value Head 训练 ==============")
    
    # 初始化 Wandb
    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG
    )
    
    print("1. 加载 Tokenizer 与词表对齐...")
    # 从 Base 拿字典，手动补齐沙箱魔法咒语！
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"], trust_remote_code=True)
    new_tokens = ["<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("2. 组装 CognitiveQwen 并实施局部冰冻手术 (Freezing Backbone)...")
    # 加载你的 CognitiveQwen 架构
    model = CognitiveQwen(
        base_model_path=CONFIG["base_model_path"], 
        lora_path=CONFIG["lora_path"], 
        device="cuda", 
        vocab_size=len(tokenizer) # 传递正确的 151673 词表大小
    )
    
    # 把大模型里所有的参数都冻结（不需要计算梯度，极其省显存！）
    for param in model.parameters():
        param.requires_grad = False
        
    #  唯独把顶端 Value Head 的封印解开！
    for param in model.value_head.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型锁定完毕！当前可训练参数量: {trainable_params} (防 OOM 状态激活)")

    print("3. 准备数据与优化器...")
    dataset = ValueDataset(CONFIG["value_data_path"], tokenizer, max_length=CONFIG["max_length"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=CONFIG["learning_rate"])
    
    # 二分类/概率回归损失函数 (极其适合评估 0.0 ~ 1.0 的胜率)
    criterion = nn.BCEWithLogitsLoss()

    print("4. (Training Value Head)...")
    model.train()
    
    global_step = 0
    total_steps = len(dataloader) * CONFIG["epochs"]
    
    for epoch in range(CONFIG["epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["label"].to(model.device)

            # 前向传播 (极其快，因为底层被冻结了)
            outputs = model.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            value_logits = model.value_head(hidden_states).squeeze(-1) # shape: (batch, seq_len)

            # 找到实际序列长度的最后一个有效 Token (代表了当前推理步骤的最终局势)
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=model.device)
            last_token_values = value_logits[batch_indices, seq_lengths]

            # 计算损失并除以累加步数
            loss = criterion(last_token_values, labels)
            loss_to_backward = loss / CONFIG["gradient_accumulation_steps"]
            loss_to_backward.backward()

            # 梯度累加更新逻辑
            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                

                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch + (step / len(dataloader)),
                    "train/learning_rate": CONFIG["learning_rate"],
                    "train/global_step": global_step
                })
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")

    print("5. 训练完成，剥离并保存 Value Head...")
    # 确保保存目录存在
    os.makedirs(os.path.dirname(CONFIG["output_head_path"]), exist_ok=True)
    torch.save(model.value_head.state_dict(), CONFIG["output_head_path"])
    print(f"Value Head 裁判权重已独立保存至: {CONFIG['output_head_path']}")
    
    wandb.finish()

if __name__ == "__main__":
    main()