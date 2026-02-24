"""
generator.py — 终极防崩溃并发生成器 (基于原生 Generate)
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
        
        # 1. 极其严密的结束符提取
        self._stop_ids = []
        if isinstance(tokenizer.eos_token_id, int): self._stop_ids.append(tokenizer.eos_token_id)
        elif isinstance(tokenizer.eos_token_id, list): self._stop_ids.extend(tokenizer.eos_token_id)
        
        self._stop_ids.extend([
            151645,  # <|im_end|>
            151672,  # <|endoftext|>
            151670,
            self._py_end_id
        ])
        self._stop_ids = list(set([x for x in self._stop_ids if x is not None]))

    @torch.no_grad()
    def generate_group(self, prompt_ids: torch.Tensor, group_size: int) -> List[Rollout]:
        device = prompt_ids.device
        G = group_size
        
        seqs = [prompt_ids.squeeze(0)] * G
        is_done = [False] * G
        chunk_history = [[] for _ in range(G)]
        total_gen_tokens = [0] * G
        sandbox_states = [{} for _ in range(G)] # 持久化沙箱记忆
        
        while not all(is_done):
            active_idx = [i for i, done in enumerate(is_done) if not done]
            if not active_idx:
                break
                
            active_seqs = [seqs[i] for i in active_idx]
            max_len = max(len(s) for s in active_seqs)
            
            # 2. 严密的左填充构造 (防止 Qwen 崩溃)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self._stop_ids[0]
            padded_input = torch.full((len(active_seqs), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(padded_input)
            
            for r, s in enumerate(active_seqs):
                padded_input[r, max_len - len(s):] = s
                attention_mask[r, max_len - len(s):] = 1
            
            # 3. 呼叫 HF 官方底层的 C++ 生成引擎 (RoPE 和位置编码绝对安全！)
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
            
            # 4. 拆解分析生成结果
            for r, orig_i in enumerate(active_idx):
                input_len = padded_input.shape[1]
                new_tokens = outputs[r][input_len:]
                
                # 精准剔除右侧可能存在的多余 Padding
                valid_len = len(new_tokens)
                hit_eos_or_tool = False
                for k in range(len(new_tokens)):
                    if new_tokens[k].item() in self._stop_ids:
                        valid_len = k + 1
                        hit_eos_or_tool = True
                        break
                new_tokens = new_tokens[:valid_len]
                
                chunk_history[orig_i].append(('gen', new_tokens))
                seqs[orig_i] = torch.cat([seqs[orig_i], new_tokens])
                total_gen_tokens[orig_i] += len(new_tokens)
                
                last_tok = new_tokens[-1].item() if len(new_tokens) > 0 else None
                
                # ① 超出全局字数限制 -> 掐死
                if total_gen_tokens[orig_i] >= self.cfg.max_total_tokens:
                    is_done[orig_i] = True
                    continue
                    
                # ② 触发沙箱 -> 执行并注入
                if last_tok == self._py_end_id:
                    full_text = self.tokenizer.decode(seqs[orig_i], skip_special_tokens=False)
                    code = self._extract_last_code(full_text)
                    result = self.sandbox.execute(code, state=sandbox_states[orig_i]) if code else "# (no code)"
                    
                    inj_text = f"{self.cfg.OUTPUT_START}{result}{self.cfg.OUTPUT_END}"
                    inj_ids = self.tokenizer(inj_text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0).to(device)
                    
                    chunk_history[orig_i].append(('tool', inj_ids))
                    seqs[orig_i] = torch.cat([seqs[orig_i], inj_ids])
                    total_gen_tokens[orig_i] += len(inj_ids)
                    
                # ③ 正常遇到对话结束符 -> 光荣退休
                elif hit_eos_or_tool:
                    is_done[orig_i] = True
                    
                # ④ 🔥 没超全局限制，没用工具，没遇到 EOS (说明只是跑满了单次 max_segment_tokens)
                # -> 留活口！下一轮大循环会自动把它最新的文本重新丢给 GPU 继续无缝生成！
                else:
                    pass

        # 5. 精确组装 Loss 计算所需的张量
        rollouts = []
        for i in range(G):
            gen_id_list, mask_list = [], []
            for ctype, c_ids in chunk_history[i]:
                gen_id_list.append(c_ids)
                mask_list.append(torch.full_like(c_ids, ctype == 'gen', dtype=torch.bool))
            
            final_gen_ids = torch.cat(gen_id_list) if gen_id_list else torch.empty(0, dtype=torch.long, device=device)
            final_mask = torch.cat(mask_list) if mask_list else torch.empty(0, dtype=torch.bool, device=device)
                
            text = self.tokenizer.decode(final_gen_ids, skip_special_tokens=False)
            rollouts.append(Rollout(gen_ids=final_gen_ids, loss_mask=final_mask, text=text))
            
        return rollouts

    @staticmethod
    def _extract_last_code(text: str) -> Optional[str]:
        matches = re.findall(r"<\|python_start\|>(.*?)<\|python_end\|>", text, re.DOTALL)
        return matches[-1].strip() if matches else None