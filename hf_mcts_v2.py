import math
import re
import copy
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class MCTSConfig:
    num_simulations: int = 64         # 狂暴算力：探索 64 次
    branching_factor: int = 3         # 每次发散 3 种思路
    max_depth: int = 15               # 搜索深度控制在 15 步左右
    c_puct: float = 2.0               # 🔥 提高探索因子，逼迫模型怀疑高分，多去看看旁边的分支
    expansion_temperature: float = 0.8
    expansion_top_k: int = 20
    min_chunk_tokens: int = 8
    max_chunk_tokens: int = 120       # 🔥 单步思考允许生成更长，足够写完一段完整的逻辑或代码


def execute_python_sandbox(code_str):
    """
    🔥 升级版沙箱：加入基础运算能力，防止连 sum() 都报错
    """
    try:
        clean_code = code_str.strip()
        safe_env = {
            "math": math, "sum": sum, "max": max, 
            "min": min, "abs": abs, "float": float, "int": int
        }
        result = eval(clean_code, {"__builtins__": {}}, safe_env)
        result_str = f"{result:.4f}".rstrip('0').rstrip('.') if isinstance(result, float) else str(result)
        return result_str
    except Exception as e:
        return f"Error: {e}"


class MCTSNode:
    __slots__ = (
        "tokens", "parent", "children", "visit_count", "total_value",
        "past_key_values", "last_logits", "state_value", "is_terminal",
        "text_so_far", "depth"
    )

    def __init__(self, tokens: List[int], parent: Optional["MCTSNode"] = None,
                 past_key_values=None, last_logits=None, state_value=0.5,
                 is_terminal: bool = False, text_so_far: str = ""):
        self.tokens = tokens
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_value = 0.0

        # 在 V2 版本中，past_key_values 绝大部分时间都应该是 None，防止 OOM
        self.past_key_values = past_key_values 
        self.last_logits = last_logits
        self.state_value = state_value
        self.is_terminal = is_terminal
        self.depth = 0 if parent is None else parent.depth + 1
        self.text_so_far = text_so_far

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def uct_score(self, c_puct: float) -> float:
        parent_n = max(self.parent.visit_count if self.parent else 1, 1)
        
        if self.visit_count == 0:
            # 引入 AlphaGo 的 PUCT (Predictor + UCB) 思想
            # 既然 _expand 时裁判已经给了 state_value，直接把它作为先验 Q 值
            exploration = c_puct * math.sqrt(parent_n) / 1.0
            return self.state_value + exploration
            
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
        self.stop_tokens.discard(None)

        nl_tokens = set()
        for s in ['\n', '\n\n', ' \n']:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and tid != tokenizer.unk_token_id:
                nl_tokens.add(tid)
        self._newline_token_ids = nl_tokens

        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(42)

    @torch.inference_mode()
    def search(self, prompt_text: str, expected_answer: str = None) -> Tuple[str, float]:
        prompt_tokens = self.tokenizer.encode(prompt_text)

        ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        outputs = self.model(ids, use_cache=True, return_value=True)

        root_logits = outputs[0][:, -1, :]
        root_val = torch.sigmoid(outputs[1]).item()
        
        # 根节点不保存 KV Cache
        root = MCTSNode(
            tokens=prompt_tokens, parent=None,
            past_key_values=None, last_logits=root_logits, state_value=root_val,
            text_so_far=prompt_text
        )

        for sim_idx in range(self.config.num_simulations):
            node = self._select(root)

            if not node.is_terminal and node.depth < self.config.max_depth:
                children = self._expand(node)
                eval_node = children[0] if children else node
            else:
                eval_node = node

            reward = self._evaluate(eval_node, prompt_text)
            self._backpropagate(eval_node, reward)

        best_response, best_score = self._extract_best_response(root, prompt_tokens, expected_answer)
        return best_response, best_score

    def _select(self, node: MCTSNode) -> MCTSNode:
        cfg = self.config
        while (not node.is_leaf()
               and not node.is_terminal
               and node.is_fully_expanded(cfg.branching_factor)):
            node = max(node.children, key=lambda c: c.uct_score(cfg.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> List["MCTSNode"]:
        cfg = self.config
        new_children = []

        candidate_tokens = self._sample_diverse_tokens(
            node.last_logits, k=cfg.branching_factor,
            temperature=cfg.expansion_temperature, top_k=cfg.expansion_top_k
        )

        for start_token_id in candidate_tokens:
            current_id = start_token_id
            child_tokens = list(node.tokens)
            
            # 空间换时间，扩展时重算 KV Cache
            ids = torch.tensor([node.tokens], dtype=torch.long, device=self.device)
            out = self.model(ids, use_cache=True, return_value=True)
            current_pkv = out[2]

            step_count = 0
            is_terminal = False
            state_value = 0.5
            next_logits = None
            consecutive_newlines = 0

            while step_count < cfg.max_chunk_tokens:
                child_tokens.append(current_id)
                token_str = self.tokenizer.decode([current_id])

                ids = torch.tensor([[current_id]], dtype=torch.long, device=self.device)
                out = self.model(ids, past_key_values=current_pkv,
                                 use_cache=True, return_value=True)

                next_logits = out[0][:, -1, :]
                state_value = torch.sigmoid(out[1]).item()
                current_pkv = out[2]
                step_count += 1

                if current_id in self.stop_tokens:
                    is_terminal = True
                    break

                if current_id == self.py_end_id:
                    is_terminal = False
                    break

                if '\n' in token_str:
                    consecutive_newlines += 1
                else:
                    consecutive_newlines = 0

                if step_count >= cfg.min_chunk_tokens and consecutive_newlines >= 2:
                    break
                current_id = torch.argmax(next_logits, dim=-1).item()
            
            child_text = self.tokenizer.decode(child_tokens, skip_special_tokens=False)

            if child_tokens and child_tokens[-1] == self.py_end_id:
                matches = re.findall(
                    r'<\|python_start\|>(.*?)<\|python_end\|>', child_text, re.DOTALL
                )
                if matches:
                    code = matches[-1]
                    result = execute_python_sandbox(code)
                    inject_text = f"<|output_start|>{result}<|output_end|>\n"
                    inject_tokens = self.tokenizer.encode(inject_text, add_special_tokens=False)

                    ids = torch.tensor([inject_tokens], dtype=torch.long, device=self.device)
                    inj_out = self.model(ids, past_key_values=current_pkv,
                                        use_cache=True, return_value=True)

                    next_logits = inj_out[0][:, -1, :]
                    state_value = torch.sigmoid(inj_out[1]).item()

                    child_tokens.extend(inject_tokens)
                    child_text += inject_text

            child = MCTSNode(
                tokens=child_tokens, parent=node,
                past_key_values=None, 
                last_logits=next_logits,
                state_value=state_value, is_terminal=is_terminal,
                text_so_far=child_text
            )
            node.children.append(child)
            new_children.append(child)

        return new_children

    def _evaluate(self, node: MCTSNode, prompt_text: str) -> float:
        """
        🔥 彻底拆除作弊漏洞！100% 相信裁判大脑！
        """
        generated_part = node.text_so_far[len(prompt_text):]
        base = node.state_value
        
        # 1. 阈值熔断：垃圾思路直接判死刑，停止探索
        if base < 0.15:
            node.is_terminal = True
            return 0.0

        # 2. 如果是结尾节点，直接返回裁判的最终判决！绝不用正则表达式强加分数！
        if node.is_terminal:
            return base

        # 3. 探索过程中的微小鼓励，防止数值过于枯燥
        bonus = 0.0
        if '<|output_start|>' in generated_part:
            bonus += 0.02 

        return min(base + bonus, 1.0)

    def _backpropagate(self, node: MCTSNode, reward: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent

    def _extract_best_response(self, root: MCTSNode, prompt_tokens: List[int], expected_answer: str = None) -> Tuple[str, float]:
        if root.is_leaf():
            return "", root.q_value
    
        all_leaves = []
    
        def _dfs(node: MCTSNode):
            if node.visit_count > 0 and node.is_leaf():
                all_leaves.append(node)
            for child in node.children:
                _dfs(child)
    
        _dfs(root)
    
        if not all_leaves:
            return "", root.q_value
            
        def _extract_ans(text: str):
            match = re.search(r'\\boxed\{([^}]+)\}', text)
            if match: return match.group(1).replace(',', '').strip()
            match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-]+)', text)
            if match: return match.group(1).replace(',', '').strip()
            return None
    
        def _key(node: MCTSNode):
            text = self.tokenizer.decode(
                node.tokens[len(prompt_tokens):], skip_special_tokens=False
            )
            has_tool = 1 if '<|output_start|>' in text else 0
            is_terminal = 1 if node.is_terminal else 0
            
            # 有答案参考时，对的优先级最高
            if expected_answer is not None:
                model_ans = _extract_ans(text)
                is_correct = 1 if (model_ans == expected_answer) else 0
                return (is_correct, node.q_value, is_terminal, has_tool, -node.depth)
            
            # 🔥 闭卷考试时：裁判的 q_value 具有最高统治力！
            return (node.q_value, is_terminal, has_tool, -node.depth)
    
        best = max(all_leaves, key=_key)
        response_text = self.tokenizer.decode(
            best.tokens[len(prompt_tokens):], skip_special_tokens=False
        )
        return response_text, root.q_value

    def _sample_diverse_tokens(self, logits: torch.Tensor, k: int,
                                temperature: float, top_k: int) -> List[int]:
        vocab = logits.size(-1)
        effective_top_k = min(top_k, vocab) if top_k > 0 else vocab

        if self._newline_token_ids:
            nl_mask = torch.zeros(vocab, dtype=torch.bool, device=logits.device)
            for tid in self._newline_token_ids:
                if tid < vocab:
                    nl_mask[tid] = True
            logits = logits.masked_fill(nl_mask, float("-inf"))

        if effective_top_k < vocab:
            topk_vals, _ = torch.topk(logits, effective_top_k)
            threshold = topk_vals[:, -1:].expand_as(logits)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = torch.softmax(logits / temperature, dim=-1)

        if probs.sum() == 0:
            probs = torch.ones_like(probs) / vocab

        num_samples = min(k, int((probs > 0).sum().item()))
        num_samples = max(num_samples, 1)

        sampled = torch.multinomial(
            probs, num_samples=num_samples, replacement=False, generator=self._rng
        )
        return sampled[0].tolist()