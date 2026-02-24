# """
# generator.py — 带 KV Cache 的工具感知生成器

# 修复 Bug 1：彻底使用 past_key_values，O(N) 生成而非 O(N²)
# 修复 Bug 2：维护 loss_mask，注入的工具输出 Token 不纳入梯度计算
# """
# import re
# from dataclasses import dataclass
# from typing import Optional, Tuple, List

# import torch
# import torch.nn.functional as F
# from transformers import PreTrainedTokenizerBase

# from config  import GRPOConfig
# from sandbox import PythonSandbox


# @dataclass
# class Rollout:
#     """
#     一条完整的生成轨迹，包含所有需要的信息。

#     gen_ids    : 所有生成/注入的 token id（prompt 之后的部分）
#     loss_mask  : 1 = 模型自主生成（纳入 loss），0 = 沙箱注入（不纳入 loss）
#     text       : 解码后的完整生成文本（不含 prompt）
#     """
#     gen_ids:   torch.Tensor   # [gen_len]  LongTensor
#     loss_mask: torch.Tensor   # [gen_len]  BoolTensor
#     text:      str


# class ToolAwareGenerator:
#     """
#     驱动模型生成，遇到 <|python_end|> 时暂停、执行沙箱、注入结果，再继续。

#     核心设计：
#     ┌─────────────────────────────────────────────────────┐
#     │  Prefill(prompt) → past_kv₀, logits₀               │
#     │                                                     │
#     │  loop:                                              │
#     │    _decode(logits_i, past_kv_i)                     │
#     │      ↳ seg_ids, past_kv_{i+1}                       │
#     │    if hit <|python_end|>:                           │
#     │      inject_ids → Prefill(inject_ids, past_kv_{i+1})│
#     │                  → past_kv_{i+2}, logits_{i+2}     │
#     │      loss_mask[inject_positions] = 0               │
#     │    else: break                                      │
#     └─────────────────────────────────────────────────────┘
#     """

#     def __init__(
#         self,
#         model,
#         tokenizer: PreTrainedTokenizerBase,
#         sandbox:   PythonSandbox,
#         cfg:       GRPOConfig,
#     ):
#         self.model     = model
#         self.tokenizer = tokenizer
#         self.sandbox   = sandbox
#         self.cfg       = cfg

#         self._py_end_id = tokenizer.convert_tokens_to_ids(cfg.PYTHON_END)
#         self._eos_id    = tokenizer.eos_token_id

#     # ──────────────────────────────────────────────────────────────────────
#     @torch.no_grad()
#     def generate(self, prompt_ids: torch.Tensor) -> Rollout:
#         """
#         prompt_ids : [1, prompt_len]
#         Returns    : Rollout（详见 dataclass 定义）
#         """
#         device = prompt_ids.device

#         # ① Prefill：把整个 prompt 喂给模型，建立初始 KV Cache
#         out     = self.model(prompt_ids, use_cache=True, return_dict=True)
#         past_kv = out.past_key_values
#         logits  = out.logits[:, -1, :]   # [1, vocab]，用于采样第一个 token

#         all_gen_ids: List[int]   = []
#         all_mask:    List[bool]  = []
#         total_gen = 0

#         while total_gen < self.cfg.max_total_tokens:
#             remaining = self.cfg.max_total_tokens - total_gen
#             seg_max   = min(self.cfg.max_segment_tokens, remaining)

#             # ② 从当前 logits 和 KV cache 解码一段文字
#             seg_ids, past_kv, hit_tool = self._decode(
#                 logits, past_kv, seg_max,
#                 self.cfg.temperature, self.cfg.top_p, device,
#             )

#             # seg_ids 是模型自主生成的，全部 mask=1
#             all_gen_ids.extend(seg_ids.tolist())
#             all_mask.extend([True] * len(seg_ids))
#             total_gen += len(seg_ids)

#             if not hit_tool:
#                 break   # EOS，生成完毕

#             # ③ 沙箱执行
#             full_text_so_far = self.tokenizer.decode(all_gen_ids, skip_special_tokens=False)
#             code   = self._extract_last_code(full_text_so_far)
#             result = self.sandbox.execute(code) if code else "# (no code found)"

#             injection_text = f"{self.cfg.OUTPUT_START}{result}{self.cfg.OUTPUT_END}"
#             inj_ids = self.tokenizer(
#                 injection_text, return_tensors="pt", add_special_tokens=False
#             ).input_ids.to(device)   # [1, inj_len]

#             # ④ 把注入 token 喂给模型，延伸 KV Cache（相当于 teacher-force）
#             #    同时拿到下一段解码用的 logits
#             out     = self.model(inj_ids, past_key_values=past_kv, use_cache=True, return_dict=True)
#             past_kv = out.past_key_values
#             logits  = out.logits[:, -1, :]   # [1, vocab]

#             # 注入 token 全部 mask=0（不参与 loss）
#             all_gen_ids.extend(inj_ids.squeeze(0).tolist())
#             all_mask.extend([False] * inj_ids.shape[1])
#             total_gen += inj_ids.shape[1]

#         gen_ids_t  = torch.tensor(all_gen_ids, dtype=torch.long,  device=device)
#         loss_mask_t = torch.tensor(all_mask,   dtype=torch.bool,  device=device)
#         text = self.tokenizer.decode(all_gen_ids, skip_special_tokens=False)

#         return Rollout(gen_ids=gen_ids_t, loss_mask=loss_mask_t, text=text)

#     # ──────────────────────────────────────────────────────────────────────
#     def _decode(
#         self,
#         logits:   torch.Tensor,           # [1, vocab]  当前可用 logits
#         past_kv,                          # KV cache
#         max_new:  int,
#         temp:     float,
#         top_p:    float,
#         device:   torch.device,
#     ) -> Tuple[torch.Tensor, object, bool]:
#         """
#         从给定 logits/past_kv 开始自回归解码。
#         每个 step：采样 → forward（得到新 logits + 新 past_kv）→ 继续。

#         Returns: (seg_ids, updated_past_kv, hit_tool_end)
#         """
#         seg_ids: List[int] = []
#         hit_tool = False

#         for _ in range(max_new):
#             # 采样
#             tid, next_tok = self._sample(logits, temp, top_p)
#             seg_ids.append(tid)

#             # 无论是否结束，都用该 token 推进一步（更新 KV Cache）
#             out     = self.model(next_tok, past_key_values=past_kv,
#                                  use_cache=True, return_dict=True)
#             past_kv = out.past_key_values
#             logits  = out.logits[:, -1, :]   # 为下次采样准备

#             if tid == self._py_end_id:
#                 hit_tool = True
#                 break
#             if tid == self._eos_id:
#                 break

#         seg_ids_t = torch.tensor(seg_ids, dtype=torch.long, device=device)
#         return seg_ids_t, past_kv, hit_tool

#     # ──────────────────────────────────────────────────────────────────────
#     @staticmethod
#     def _sample(
#         logits: torch.Tensor,   # [1, vocab]
#         temp:   float,
#         top_p:  float,
#     ) -> Tuple[int, torch.Tensor]:
#         """Top-p 采样，返回 (token_id, next_tok_tensor [1,1])"""
#         logits = logits / max(temp, 1e-6)

#         if top_p < 1.0:
#             sorted_logits, sorted_idx = torch.sort(logits, descending=True)
#             cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#             remove    = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
#             sorted_logits[remove] = -float("inf")
#             logits.scatter_(1, sorted_idx, sorted_logits)

#         probs    = F.softmax(logits, dim=-1)
#         next_tok = torch.multinomial(probs, num_samples=1)   # [1, 1]
#         return next_tok.item(), next_tok

#     # ──────────────────────────────────────────────────────────────────────
#     @staticmethod
#     def _extract_last_code(text: str) -> Optional[str]:
#         matches = re.findall(r"<\|python_start\|>(.*?)<\|python_end\|>", text, re.DOTALL)
#         return matches[-1].strip() if matches else None
"""
generator.py — 终极压榨显存的 Batch 并发生成器
"""
import re
from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import PreTrainedTokenizerBase

from config  import GRPOConfig
from sandbox import PythonSandbox

@dataclass
class Rollout:
    gen_ids:   torch.Tensor
    loss_mask: torch.Tensor
    text:      str

class ToolAwareGenerator:
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
        sandbox:   PythonSandbox,
        cfg:       GRPOConfig,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.sandbox   = sandbox
        self.cfg       = cfg

        self._py_end_id = tokenizer.convert_tokens_to_ids(cfg.PYTHON_END)
        self._eos_id    = tokenizer.eos_token_id
        
        self._stop_ids = []
        
        # 1. 安全提取默认的 eos_token_id (兼容 int 或 list 格式)
        if isinstance(self._eos_id, int):
            self._stop_ids.append(self._eos_id)
        elif isinstance(self._eos_id, list):
            self._stop_ids.extend(self._eos_id)
            
        # 2. 安全提取 python_end
        if isinstance(self._py_end_id, int):
            self._stop_ids.append(self._py_end_id)
            
        # 3. 强行加入 Qwen 系列最常见的对话结束符（多一层安全保障）
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int):
            self._stop_ids.append(im_end_id)
            
        # 🔥 终极过滤：把所有意外的 None 全部踢掉，确保列表里只有纯净的整数！
        self._stop_ids = [x for x in self._stop_ids if x is not None]

    @torch.no_grad()
    def generate_group(self, prompt_ids: torch.Tensor, group_size: int) -> List[Rollout]:
        """
        核心大杀器：将 group_size 个序列打包成 Tensor，在 GPU 上并发生成！
        速度提升数十倍，瞬间榨干显存！
        """
        device = prompt_ids.device
        G = group_size
        
        # 初始化 G 条并行序列的状态
        seqs = [prompt_ids.squeeze(0)] * G
        is_done = [False] * G
        chunk_history = [[] for _ in range(G)]
        total_gen_tokens = [0] * G
        
        while not all(is_done):
            # 1. 过滤出还需要生成的活跃序列
            active_idx = [i for i, done in enumerate(is_done) if not done]
            if not active_idx:
                break
                
            active_seqs = [seqs[i] for i in active_idx]
            max_len = max(len(s) for s in active_seqs)
            
            # 2. 动态左侧填充 (Left Padding)，构造超级 Batch 矩阵
            pad_id = self.tokenizer.pad_token_id
            padded_input = torch.full((len(active_seqs), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(padded_input)
            
            for r, s in enumerate(active_seqs):
                padded_input[r, max_len - len(s):] = s
                attention_mask[r, max_len - len(s):] = 1
            
            # 3. 呼叫底层的 C++ 并发生成引擎！
            outputs = self.model.generate(
                input_ids=padded_input,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.max_segment_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                eos_token_id=self._stop_ids,
                pad_token_id=pad_id,
                use_cache=True, 
            )
            
            # 4. 解析并行输出结果，并分别处理沙箱调用
            for r, orig_i in enumerate(active_idx):
                input_len = padded_input.shape[1]
                new_tokens = outputs[r][input_len:]
                
                # 剥离右侧多余的 padding
                valid_len = len(new_tokens)
                for k in range(len(new_tokens)):
                    if new_tokens[k].item() in self._stop_ids:
                        valid_len = k + 1
                        break
                new_tokens = new_tokens[:valid_len]
                
                chunk_history[orig_i].append(('gen', new_tokens))
                seqs[orig_i] = torch.cat([seqs[orig_i], new_tokens])
                total_gen_tokens[orig_i] += len(new_tokens)
                
                if len(new_tokens) == 0 or total_gen_tokens[orig_i] >= self.cfg.max_total_tokens:
                    is_done[orig_i] = True
                    continue
                    
                last_tok = new_tokens[-1].item()
                
                # 如果这个序列触发了沙箱，立刻执行并把结果注入进去！
                if last_tok == self._py_end_id:
                    full_text = self.tokenizer.decode(seqs[orig_i], skip_special_tokens=False)
                    code = self._extract_last_code(full_text)
                    result = self.sandbox.execute(code) if code else "# (no code)"
                    
                    inj_text = f"{self.cfg.OUTPUT_START}{result}{self.cfg.OUTPUT_END}"
                    inj_ids = self.tokenizer(inj_text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0).to(device)
                    
                    chunk_history[orig_i].append(('tool', inj_ids))
                    seqs[orig_i] = torch.cat([seqs[orig_i], inj_ids])
                    total_gen_tokens[orig_i] += len(inj_ids)
                else:
                    is_done[orig_i] = True # 遇到 EOS，这条序列光荣退休

        # 5. 精准拼装 Rollout 和 Mask
        rollouts = []
        for i in range(G):
            gen_id_list, mask_list = [], []
            for ctype, c_ids in chunk_history[i]:
                gen_id_list.append(c_ids)
                mask_bool = (ctype == 'gen') # 模型生成的算 loss，工具注入的不算
                mask_list.append(torch.full_like(c_ids, mask_bool, dtype=torch.bool))
            
            if gen_id_list:
                final_gen_ids = torch.cat(gen_id_list)
                final_mask = torch.cat(mask_list)
            else:
                final_gen_ids = torch.empty(0, dtype=torch.long, device=device)
                final_mask = torch.empty(0, dtype=torch.bool, device=device)
                
            text = self.tokenizer.decode(final_gen_ids, skip_special_tokens=False)
            rollouts.append(Rollout(gen_ids=final_gen_ids, loss_mask=final_mask, text=text))
            
        return rollouts

    @staticmethod
    def _extract_last_code(text: str) -> Optional[str]:
        matches = re.findall(r"<\|python_start\|>(.*?)<\|python_end\|>", text, re.DOTALL)
        return matches[-1].strip() if matches else None