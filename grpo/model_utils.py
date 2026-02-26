"""
model_utils.py — 安全的模型加载工具 (动态挂载接力版)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken
from peft import PeftModel
from grpo.config import GRPOConfig

def load_tokenizer(cfg: GRPOConfig) -> AutoTokenizer:
    # 必须从 SFT LoRA 路径加载 Tokenizer，确保包含 4 个沙箱词！
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_lora_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def build_ref_and_policy(cfg: GRPOConfig, tokenizer: AutoTokenizer):
    vocab_size = len(tokenizer)

    # ── 1. 参考模型 (Reference Model) ──
    print("\n[build_models] 加载参考模型 (Reference Model) ...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # 关键步骤：扩充词表以匹配 Tokenizer
    ref_base.resize_token_embeddings(vocab_size)
    
    # 挂载 SFT LoRA 并彻底冻结
    ref_model = PeftModel.from_pretrained(ref_base, cfg.sft_lora_path, is_trainable=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    print("  ✅ 参考模型加载完毕 (Base + SFT_LoRA, 已冻结)")


    # ── 2. 策略模型 (Policy Model) ──
    print("\n[build_models] 加载策略模型 (Policy Model) ...")
    policy_base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    policy_base.resize_token_embeddings(vocab_size)
    
    # 🔥 核心魔法：挂载相同的 SFT LoRA，但开启训练模式！(接力训练)
    policy_model = PeftModel.from_pretrained(policy_base, cfg.sft_lora_path, is_trainable=True)

    # 确保只有 LoRA 参数可训练，防止炸显存
    for name, param in policy_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    policy_model.print_trainable_parameters()
    print("  ✅ 策略模型加载完毕 (继承 SFT 记忆，开启 RL 继续训练)")

    # (可选) 开启 PyTorch C++ 底层编译提速，如果报错会自动忽略
    try:
        print("  -> 正在使用 torch.compile 编译计算图以提速...")
        policy_model = torch.compile(policy_model)
        ref_model = torch.compile(ref_model)
    except Exception as e:
        print(f"  ⚠️ torch.compile 启动失败，回退到普通模式: {e}")

    return ref_model, policy_model

def save_policy(policy_model: nn.Module, tokenizer: AutoTokenizer, path: str):
    policy_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  💾 已保存 → {path}")