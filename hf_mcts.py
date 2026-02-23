import math
import re
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class MCTSConfig:
    num_simulations: int = 64
    branching_factor: int = 3
    max_depth: int = 40
    c_puct: float = 1.414
    expansion_temperature: float = 0.8
    expansion_top_k: int = 20

def execute_python_sandbox(code_str):
    print(f" [🚀 沙箱拦截] 执行: {code_str.strip()}")
    try:
        clean_code = code_str.strip()
        result = eval(clean_code, {"__builtins__": {}}, {})
        result_str = f"{result:.4f}".rstrip('0').rstrip('.') if isinstance(result, float) else str(result)
        print(f" [✅ 沙箱返回]: {result_str}")
        return result_str
    except Exception as e:
        print(f" [❌ 沙箱报错]: {e}")
        return f"Error: {e}"

class MCTSNode:
    __slots__ = (
        "tokens", "parent", "children", "visit_count", "total_value",
        "past_key_values", "last_logits", "state_value", "is_terminal", "text_so_far", "depth"
    )

    def __init__(self, tokens: List[int], parent: Optional["MCTSNode"] = None, 
                 past_key_values=None, last_logits=None, state_value=0.5,
                 is_terminal: bool = False, text_so_far: str = ""):
        self.tokens = tokens
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_value = 0.0
        
        self.past_key_values = past_key_values
        self.last_logits = last_logits  # 缓存下一步的概率分布
        self.state_value = state_value  # 缓存当前节点的胜率评估
        self.is_terminal = is_terminal
        self.depth = 0 if parent is None else parent.depth + 1
        self.text_so_far = text_so_far

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def uct_score(self, c_puct: float) -> float:
        if self.visit_count == 0: return float("inf")
        parent_n = self.parent.visit_count if self.parent else 1
        exploration = c_puct * math.sqrt(math.log(parent_n) / self.visit_count)
        return self.q_value + exploration

    def is_fully_expanded(self, branching_factor: int) -> bool:
        return len(self.children) >= branching_factor

    def is_leaf(self) -> bool:
        return len(self.children) == 0

class HF_MCTS:
    def __init__(self, model, tokenizer, config: MCTSConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.eos_token_id = tokenizer.eos_token_id
        self.im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.py_start_id = tokenizer.convert_tokens_to_ids('<|python_start|>')
        self.py_end_id = tokenizer.convert_tokens_to_ids('<|python_end|>')
        
        self.stop_tokens = {self.eos_token_id, self.im_end_id}
        if None in self.stop_tokens: self.stop_tokens.remove(None)

        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(42)

    @torch.inference_mode()
    def search(self, prompt_text: str) -> Tuple[str, float]:
        prompt_tokens = self.tokenizer.encode(prompt_text)
        
        # 1. 初始化根节点，同时获取下一步分布和初始价值
        ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        outputs = self.model(ids, use_cache=True, return_value=True)
        
        root_logits = outputs[0][:, -1, :]
        root_val = torch.sigmoid(outputs[1]).item()
        root_pkv = outputs[2]
        
        root = MCTSNode(
            tokens=prompt_tokens, parent=None,
            past_key_values=root_pkv, last_logits=root_logits, state_value=root_val,
            text_so_far=prompt_text
        )

        for _ in range(self.config.num_simulations):
            node = self._select(root)
            if not node.is_terminal and node.depth < self.config.max_depth:
                children = self._expand(node)
                eval_node = children[0] if children else node
            else:
                eval_node = node
                
            reward = self._evaluate_with_value_head(eval_node, prompt_text)
            self._backpropagate(eval_node, reward)

        best_node = self._get_best_child(root, use_visit_count=True)
        best_tokens = best_node.tokens
        response_text = self.tokenizer.decode(best_tokens[len(prompt_tokens):], skip_special_tokens=False)
        return response_text, root.q_value

    def _select(self, node: MCTSNode) -> MCTSNode:
        cfg = self.config
        while not node.is_leaf() and not node.is_terminal and node.is_fully_expanded(cfg.branching_factor):
            node = max(node.children, key=lambda c: c.uct_score(cfg.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        cfg = self.config
        new_children = []
        
        candidate_tokens = self._sample_diverse_tokens(
            node.last_logits, k=cfg.branching_factor, 
            temperature=cfg.expansion_temperature, top_k=cfg.expansion_top_k
        )
        
        for start_token_id in candidate_tokens:
            current_id = start_token_id
            child_tokens = list(node.tokens)
            current_pkv = node.past_key_values
            
            step_count = 0
            is_terminal = False
            state_value = 0.5
            next_logits = None
            
            # 完美的 KV Cache 前进逻辑
            while step_count < 64:
                child_tokens.append(current_id)
                token_str = self.tokenizer.decode([current_id])
                
                # 传入最新的 token，获取下一步的预测
                ids = torch.tensor([[current_id]], dtype=torch.long, device=self.device)
                outputs = self.model(ids, past_key_values=current_pkv, use_cache=True, return_value=True)
                
                next_logits = outputs[0][:, -1, :]
                state_value = torch.sigmoid(outputs[1]).item()
                current_pkv = outputs[2]
                
                # 如果遇到换行符或停止符，这一个推理块 (Chunk) 就完成了
                if current_id in self.stop_tokens or current_id == self.py_end_id or '\n' in token_str:
                    is_terminal = (current_id in self.stop_tokens)
                    break
                    
                current_id = torch.argmax(next_logits, dim=-1).item()
                step_count += 1
                
            child_text = self.tokenizer.decode(child_tokens, skip_special_tokens=False)
            
            # 沙箱拦截与 KV 重新注入
            if child_tokens[-1] == self.py_end_id:
                matches = re.findall(r'<\|python_start\|>(.*?)<\|python_end\|>', child_text, re.DOTALL)
                if matches:
                    code = matches[-1]
                    result = execute_python_sandbox(code)
                    inject_text = f"<|output_start|>{result}<|output_end|>\n"
                    inject_tokens = self.tokenizer.encode(inject_text, add_special_tokens=False)
                    
                    ids = torch.tensor([inject_tokens], dtype=torch.long, device=self.device)
                    inject_outputs = self.model(ids, past_key_values=current_pkv, use_cache=True, return_value=True)
                    
                    next_logits = inject_outputs[0][:, -1, :]
                    state_value = torch.sigmoid(inject_outputs[1]).item()
                    current_pkv = inject_outputs[2]
                    
                    child_tokens.extend(inject_tokens)
                    child_text += inject_text

            child = MCTSNode(
                tokens=child_tokens, parent=node,
                past_key_values=current_pkv, last_logits=next_logits, state_value=state_value,
                is_terminal=is_terminal, text_so_far=child_text
            )
            node.children.append(child)
            new_children.append(child)
            
        return new_children

    def _evaluate_with_value_head(self, node: MCTSNode, prompt_text: str) -> float:
        """评估节点：不再读取历史作弊，并且完全依赖 MCTSNode 缓存的 state_value，速度极快"""
        if node.is_terminal:
            # 🔥 修复奖励作弊：只提取大模型在此次搜索中“真正生成”的内容，切断前缀！
            generated_part = node.text_so_far.replace(prompt_text, "")
            return 1.0 if "The final answer is" in generated_part else 0.0
            
        # 直接读取 _expand 阶段顺便计算出的价值，省去一次极度耗时的前向传播！
        return node.state_value

    def _backpropagate(self, node: MCTSNode, reward: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent

    def _get_best_child(self, node: MCTSNode, use_visit_count: bool = True) -> MCTSNode:
        if node.is_leaf() or node.is_terminal: return node
        best = max(node.children, key=lambda c: c.visit_count if use_visit_count else c.q_value)
        return self._get_best_child(best, use_visit_count)

    def _sample_diverse_tokens(self, logits: torch.Tensor, k: int, temperature: float, top_k: int) -> List[int]:
        vocab = logits.size(-1)
        effective_top_k = min(top_k, vocab) if top_k > 0 else vocab
        if effective_top_k < vocab:
            topk_vals, _ = torch.topk(logits, effective_top_k)
            threshold = topk_vals[:, -1:].expand_as(logits)
            logits = logits.masked_fill(logits < threshold, float("-inf"))
            
        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=min(k, effective_top_k), replacement=False, generator=self._rng)
        return sampled[0].tolist()