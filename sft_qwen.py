import os
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
# 🔥 修改 1：导入 SFTConfig
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model

# ================= 配置 =================

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"

DATASETS = {
    "general": "/root/autodl-tmp/data/cold_start_lite_tool_use.jsonl",
    "math1": "/root/autodl-tmp/data/gsm8k_cot_sft_tool_use.jsonl",
    "math2": "/root/autodl-tmp/data/gsm8k_distill_clean.jsonl",
    "math3": "/root/autodl-tmp/data/gsm8k_distill_clean.jsonl",
    "math4": "/root/autodl-tmp/data/gsm8k_distill_clean.jsonl"
}
OUTPUT_DIR = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool"

WANDB_PROJECT = "cognitive-nano-qwen"
WANDB_RUN_NAME = "qwen-use-tool-sft"

# ================= 数据加载 =================

def load_and_mix_datasets():
    datasets = []
    for name, path in DATASETS.items():
        ds = load_dataset("json", data_files=path, split="train")
        print(f"{name} size:", len(ds))
        datasets.append(ds)

    # 直接 concat
    mixed = concatenate_datasets(datasets)
    print("Total dataset size:", len(mixed))
    return mixed

# ================= 主程序 =================

def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    print("1. Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加入 4 个特殊 token（逻辑完美）
    special_tokens = [
        '<|python_start|>', '<|python_end|>',
        '<|output_start|>', '<|output_end|>'
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    print("2. Model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # 4090 起飞关键
    )

    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads() # 配合 LoRA 稳定梯度

    print("3. LoRA")
    peft_config = LoraConfig(
        r=32,                
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # 保存新 token 的嵌入矩阵（逻辑完美）
        modules_to_save=["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("4. Dataset 预处理")
    dataset = load_and_mix_datasets()

    # 🔥 修改 2：使用 map 提前将 messages 转换为拼接好的 text 字段
    # 这比在 SFTTrainer 里动态 formatting_func 更稳，特别是搭配 packing=True 时
    def process_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    print("正在应用 Chat Template 预处理数据...")
    dataset = dataset.map(process_chat_template, num_proc=4, desc="Applying chat template")

    print("5. SFTConfig")
    # 🔥 修改 3：将 TrainingArguments 替换为 SFTConfig，并将独有参数移入
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",       # 告诉 Trainer 直接读取 "text" 列
        max_length=1024,             # 移到这里
        packing=True,                    # 移到这里（显著提升 4090 吞吐量）
        
        per_device_train_batch_size=6,   
        gradient_accumulation_steps=2,
        learning_rate=1e-4,              
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        optim="adamw_torch_fused",       # 配合 flash_attn 更快
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    print("=== Training ===")
    trainer.train()

    print("Save model")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()