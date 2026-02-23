import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class CognitiveQwen(nn.Module):
    """
    cognitiveQwen专属双头模型 (Dual-Head Architecture)
    共享底层的 Transformer Backbone，同时输出:
    1. Language Head (LM Logits): 用于生成下一个 Token
    2. Value Head (Scalar): 用于 MCTS 节点胜率预测 (0~1)
    """
    def __init__(self, base_model_path, lora_path=None, device="cuda", vocab_size=None):
        super().__init__()
        self.device = device
        
        print("1. 加载 Qwen 基座配置与模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
        
        # 💥【极度关键】：扩充词表，否则 LoRA 加载会报维度错误
        if vocab_size is not None:
            self.base_model.resize_token_embeddings(vocab_size)
            
        # 2. 融合 SFT LoRA 权重
        if lora_path:
            print(f"2. 融合 SFT LoRA 权重 ({lora_path})...")
            from peft import PeftModel
            self.base_model = PeftModel.from_pretrained(self.base_model, lora_path)
            self.base_model = self.base_model.merge_and_unload()
            
        # 3. 构造价值头 (Value Head)
        print("3. 初始化 Value Head (价值头)...")
        config = self.base_model.config
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False).to(
            self.device, 
            dtype=torch.bfloat16
        )
        nn.init.zeros_(self.value_head.weight) 

    def forward(self, input_ids, attention_mask=None, past_key_values=None, return_value=False, use_cache=True):
        """
        前向传播引擎
        :param return_value: 如果为 True，额外返回当前状态的价值评估
        """
        # 1. 运行 Qwen 原始的前向传播
        # 注意我们要拿到 hidden_states 以便喂给 Value Head
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True
        )
        
        logits = outputs.logits
        new_past_key_values = outputs.past_key_values
        
        if return_value:
            # 取最后一层的隐状态: [batch_size, seq_len, hidden_size]
            last_hidden_state = outputs.hidden_states[-1]
            
            # 取出序列最后一个 Token 的向量进行价值评估: [batch_size, hidden_size]
            last_token_hidden = last_hidden_state[:, -1, :] 
            
            # 通过 Value Head 预测胜率分数
            value = self.value_head(last_token_hidden)
            
            # 返回: logits, value, 以及用于 MCTS 树状复用的新 KV Cache
            return logits, value, new_past_key_values
            
        # 正常生成模式
        return logits, new_past_key_values