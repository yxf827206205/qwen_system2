import os
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AddedToken  
)
from peft import LoraConfig, get_peft_model

# ================= 配置 =================

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B-Base"
COLD_START_LORA = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool——v2/final_lora"
DATASETS = {    
    # "math2": "/root/autodl-tmp/data/gsm8k_distill_clean.jsonl",
    # "math3": "/root/autodl-tmp/data/gsm8k_distill_clean.jsonl",
    "math4": "/root/autodl-tmp/data/clean_sft_data.jsonl"
}
OUTPUT_DIR = "/root/autodl-tmp/checkpoints/qwen_sft_use_tool——v3"

WANDB_PROJECT = "cognitive-nano-qwen"
WANDB_RUN_NAME = "qwen-use-tool-sft2-continue"
from peft import PeftModel
# ================= 数据加载 =================

def continue_sft():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    print("1. Tokenizer 加载 (从冷启动 LoRA 目录读取)")
    tokenizer = AutoTokenizer.from_pretrained(COLD_START_LORA, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens = [
        AddedToken('<|python_start|>', rstrip=False, lstrip=False, special=True),
        AddedToken('<|python_end|>', rstrip=False, lstrip=False, special=True),
        AddedToken('<|output_start|>', rstrip=False, lstrip=False, special=True),
        AddedToken('<|output_end|>', rstrip=False, lstrip=False, special=True)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    print("2. 加载原版基座模型")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    # 必须给原版基座扩充词表
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.enable_input_require_grads()

    print("3. 加载冷启动 LoRA 并开启继续训练 (Continue Training)")
    # 注意：is_trainable=True 是核心！它会激活这个 LoRA 和词表层的梯度
    model = PeftModel.from_pretrained(base_model, COLD_START_LORA, is_trainable=True)
    model.print_trainable_parameters()

    print("4. Dataset 预处理")
    dataset = load_and_mix_datasets()

    def process_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": text}
    
    dataset = dataset.map(process_chat_template, num_proc=4, desc="Applying chat template")

    print("5. SFTConfig (针对 300 条黄金数据优化超参数)")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",       
        max_length=1024,             
        packing=True,                    
        per_device_train_batch_size=2,   
        gradient_accumulation_steps=4,   # Effective Batch Size = 8
        learning_rate=1e-4,              # 学习率比冷启动稍低，进行平滑微调
        num_train_epochs=5,              # 跑 5 轮
        logging_steps=5,
        save_steps=600,          # 每轮保存一次
        bf16=True,
        optim="adamw_torch_fused",       
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
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
        processing_class=tokenizer, 
    )

    print("=== 开始传承训练 (Continue Training) ===")
    trainer.train()

    print("Save model")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    wandb.finish()
    print("Done!")

def load_and_mix_datasets():
    datasets = []
    for name, path in DATASETS.items():
        ds = load_dataset("json", data_files=path, split="train")
        print(f"{name} size:", len(ds))
        datasets.append(ds)

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


    special_tokens = [
        AddedToken('<|python_start|>', rstrip=False, lstrip=False, special=True),
        AddedToken('<|python_end|>', rstrip=False, lstrip=False, special=True),
        AddedToken('<|output_start|>', rstrip=False, lstrip=False, special=True),
        AddedToken('<|output_end|>', rstrip=False, lstrip=False, special=True)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    print("2. Model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    # 扩充词表：这一步会让这 4 个词在 Embedding 矩阵里拥有专属的行
    # model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()

    print("3. LoRA")
    peft_config = LoraConfig(
        r=32,                
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # 保存新 Token 的权重，极其正确
        # modules_to_save=["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("4. Dataset 预处理")
    dataset = load_and_mix_datasets()

    def process_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": text}
    
    print("正在应用 Chat Template 预处理数据...")
    dataset = dataset.map(process_chat_template, num_proc=4, desc="Applying chat template")

    # test_str = "<|python_start|>1+1<|python_end|>"
    # test_tokens = tokenizer.encode(test_str, add_special_tokens=False)
    # print(f"\n[验证 Tokenizer] '{test_str}' 编码后的 ID: {test_tokens}")


    print("5. SFTConfig")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",       
        max_length=1024,             
        packing=True,                    
        per_device_train_batch_size=6,   
        gradient_accumulation_steps=2,
        learning_rate=1e-4,              
        num_train_epochs=4,
        logging_steps=10,
        save_steps=600,
        bf16=True,
        optim="adamw_torch_fused",       
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
        processing_class=tokenizer, 
    )

    print("=== Training ===")
    trainer.train()

    print("Save model")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    continue_sft()