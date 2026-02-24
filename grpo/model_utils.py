"""
model_utils.py — 安全的模型加载工具

修复 Bug 4：彻底规避 merge_and_unload() 的新词表权重丢失问题。

问题根源：
  SFT 阶段把 4 个新 Token 加入了词表（embed_tokens 矩阵被 resize），
  但 LoRA 配置里 modules_to_save 被注释掉了，因此新增行的权重
  并没有保存进 LoRA adapter_model.safetensors。

  merge_and_unload() 在合并时只处理已保存的 LoRA 矩阵，
  新 Token 对应的 embed 行被静默地保留为 base 模型的随机初始化值
  → 新 Token 被随机编码 → 生成乱码。

正确做法：
  1. 先加载 base 模型。
  2. resize_token_embeddings()。            ← 必须在 PeftModel 之前！
  3. 通过 PeftModel.from_pretrained() 挂载 LoRA。
  4. 不 merge，而是保留 PeftModel 格式。
  5. 参考模型与策略模型都从同一份已挂载 LoRA 的权重出发，
     参考模型冻结所有参数；策略模型再叠加一层新的可训练 LoRA。
"""
import copy
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken
from peft import PeftModel, LoraConfig, get_peft_model

from config import GRPOConfig


# ──────────────────────────────────────────────────────────────────────────────
def load_tokenizer(cfg: GRPOConfig) -> AutoTokenizer:
    """从 SFT LoRA 目录加载 Tokenizer（含新 Token 定义）。"""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.sft_lora_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 保证特殊 Token 已注册（幂等操作，若已存在则跳过）
    new_tokens = [
        AddedToken(cfg.PYTHON_START, rstrip=False, lstrip=False, special=True),
        AddedToken(cfg.PYTHON_END,   rstrip=False, lstrip=False, special=True),
        AddedToken(cfg.OUTPUT_START, rstrip=False, lstrip=False, special=True),
        AddedToken(cfg.OUTPUT_END,   rstrip=False, lstrip=False, special=True),
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return tokenizer


# ──────────────────────────────────────────────────────────────────────────────
def _load_sft_peft_model(cfg: GRPOConfig, vocab_size: int) -> PeftModel:
    """
    加载 base 模型 → resize → 挂载 SFT LoRA。
    不执行 merge_and_unload()，保留 PeftModel 格式。
    """
    print("  [model_utils] 加载 base 模型 …")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # ★ 先 resize，再挂 LoRA —— 顺序至关重要！
    print(f"  [model_utils] resize_token_embeddings → {vocab_size}")
    base.resize_token_embeddings(vocab_size)

    print(f"  [model_utils] 挂载 SFT LoRA: {cfg.sft_lora_path}")
    peft_model = PeftModel.from_pretrained(
        base,
        cfg.sft_lora_path,
        is_trainable=False,   # 参考模型/基础模型阶段先冻结
    )
    return peft_model


# ──────────────────────────────────────────────────────────────────────────────
"""
model_utils.py — 安全的模型加载工具 (完美合并基座版)
"""
import copy
from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import GRPOConfig

def load_tokenizer(cfg: GRPOConfig) -> AutoTokenizer:
    # 词表已经完美保存在基座里了，直接加载即可
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def build_ref_and_policy(cfg: GRPOConfig, tokenizer: AutoTokenizer) -> Tuple[nn.Module, nn.Module]:
    print("\n[build_models] 加载完美基座...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # ── 1. 参考模型（深拷贝并彻底冻结）────────────────────────────────────────
    print("  -> 构建参考模型 (Reference Model)")
    ref_model = copy.deepcopy(base_model)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()

    # ── 2. 策略模型（挂载全新的 GRPO LoRA）──────────────────────────────────
    print("  -> 构建策略模型 (Policy Model)")
    grpo_lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    policy_model = get_peft_model(base_model, grpo_lora_cfg)
    policy_model.print_trainable_parameters()

    return ref_model, policy_model


# def build_ref_and_policy(cfg: GRPOConfig, tokenizer: AutoTokenizer) -> Tuple[nn.Module, nn.Module]:
#     vocab_size = len(tokenizer)

#     # ===============================================================
#     # 1. 构建参考模型 (Reference Model) - 冻结的 V3 完全体
#     # ===============================================================
#     print("\n[build_models] 加载参考模型基座 (Qwen3-0.6B-Base) ...")
#     ref_base = AutoModelForCausalLM.from_pretrained(
#         cfg.base_model_path,
#         torch_dtype=torch.bfloat16,
#         device_map=cfg.device,
#         trust_remote_code=True,
#         attn_implementation="flash_attention_2",
#     )
#     ref_base.resize_token_embeddings(vocab_size)
    
#     print("  [build_models] 挂载 V3 SFT 记忆 (只读模式) ...")
#     ref_model = PeftModel.from_pretrained(
#         ref_base, 
#         cfg.sft_lora_path, 
#         is_trainable=False
#     )
#     ref_model.eval()
#     for param in ref_model.parameters():
#         param.requires_grad_(False)
#     print("  ✅ 参考模型已完美冻结")

#     # ===============================================================
#     # 2. 构建策略模型 (Policy Model) - V3 之上再叠一层 GRPO LoRA
#     # ===============================================================
#     print("\n[build_models] 加载策略模型基座 (Qwen3-0.6B-Base) ...")
#     policy_base = AutoModelForCausalLM.from_pretrained(
#         cfg.base_model_path,
#         torch_dtype=torch.bfloat16,
#         device_map=cfg.device,
#         trust_remote_code=True,
#         attn_implementation="flash_attention_2",
#     )
#     policy_base.resize_token_embeddings(vocab_size)
    
#     print("  [build_models] 挂载 V3 SFT 记忆作为底层插件 (adapter_name='sft_v3') ...")
#     policy_model = PeftModel.from_pretrained(
#         policy_base, 
#         cfg.sft_lora_path, 
#         adapter_name="sft_v3",
#         is_trainable=False # 底层 SFT 记忆不可篡改
#     )
    
#     print("  [build_models] 叠加 GRPO 专属顶层插件 (adapter_name='grpo_rl') ...")
#     grpo_lora_cfg = LoraConfig(
#         r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
#         bias="none", task_type="CAUSAL_LM",
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
#     )
    
#     # 🔥 核心魔法：使用 add_adapter 新增一层 LoRA，并使用 set_adapter 激活它
#     policy_model.add_adapter("grpo_rl", peft_config=grpo_lora_cfg)
#     policy_model.set_adapter("grpo_rl") 
    
#     policy_model.print_trainable_parameters()
#     print("  ✅ 策略模型已就绪 (现在它拥有双层 LoRA，且只训练顶层)")

#     return ref_model, policy_model

# ──────────────────────────────────────────────────────────────────────────────
def save_policy(policy_model: nn.Module, tokenizer: AutoTokenizer, path: str):
    """保存策略模型的 LoRA 权重和 Tokenizer。"""
    policy_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  💾 已保存 → {path}")
