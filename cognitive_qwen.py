import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class CognitiveQwen(nn.Module):
    """
    CognitiveQwen 双头模型 (Dual-Head Architecture) - 修复版
    共享底层的 Transformer Backbone，同时输出:
    1. Language Head (LM Logits): 用于生成下一个 Token
    2. Value Head (Scalar 0~1): 用于 MCTS 节点胜率预测

    修复点:
    - Value Head 从单线性层升级为 MLP (Linear->GELU->Linear)，表达能力更强
    - 使用 Xavier 初始化替代零初始化，避免训练早期梯度异常
    - forward() 中 return_value=True 时直接返回 sigmoid 后的概率值 (0~1)
    """

    def __init__(self, base_model_path: str, lora_path: str = None,
                 device: str = "cuda", vocab_size: int = None):
        super().__init__()
        self.device = device

        # ── 1. 加载基座模型 ──────────────────────────────────────────
        print("  [CognitiveQwen] 1. 加载 Qwen 基座模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        # 扩充词表（必须在加载 LoRA 之前）
        if vocab_size is not None:
            self.base_model.resize_token_embeddings(vocab_size)

        # ── 2. 融合 LoRA 权重 ────────────────────────────────────────
        if lora_path:
            print(f"  [CognitiveQwen] 2. 融合 LoRA 权重 ({lora_path})...")
            from peft import PeftModel
            self.base_model = PeftModel.from_pretrained(self.base_model, lora_path)
            self.base_model = self.base_model.merge_and_unload()
            print("  [CognitiveQwen]    LoRA 融合完毕")

        # ── 3. 构造 Value Head (MLP) ────────────────
        print("  [CognitiveQwen] 3. 初始化 Value Head (MLP)...")
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1, bias=True)
        ).to(self.device, dtype=torch.bfloat16)

        # Xavier 初始化
        nn.init.xavier_uniform_(self.value_head[0].weight)
        nn.init.zeros_(self.value_head[0].bias)
        nn.init.xavier_uniform_(self.value_head[3].weight)
        nn.init.zeros_(self.value_head[3].bias)

        # self.value_head.to(torch.bfloat16)

        print(f"  [CognitiveQwen]    hidden_size={hidden_size}, Value Head 参数量="
              f"{sum(p.numel() for p in self.value_head.parameters())}")

    # ────────────────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask=None,
                past_key_values=None, return_value: bool = False,
                use_cache: bool = True):
        """
        前向传播
        :param return_value: True → 额外返回 sigmoid 后的胜率 (0~1 float)
        :return:
            return_value=False: (logits, past_key_values)
            return_value=True:  (logits, value_prob, past_key_values)
                value_prob shape: (batch_size, 1)，值域 [0, 1]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits
        new_pkv = outputs.past_key_values

        if return_value:
            # 取最后一层 hidden state 的最后一个有效 token
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, H)
            # MLP → logit → sigmoid → 概率
            value_prob = torch.sigmoid(self.value_head(last_hidden))  # (B, 1)
            return logits, value_prob, new_pkv

        return logits, new_pkv

    # ────────────────────────────────────────────────────────────────
    def predict_value(self, input_ids, attention_mask=None) -> torch.Tensor:
        """
        便捷接口：只需要胜率时调用，返回 shape=(batch,) 的概率值
        MCTS rollout 评估时使用此方法
        """
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            value_prob = torch.sigmoid(self.value_head(last_hidden))
        return value_prob.squeeze(-1)  # (B,)