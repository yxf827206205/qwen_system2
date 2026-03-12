import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    branching_factor: int = 3
    max_depth: int = 18
    c_puct: float = 2.0
    expansion_temperature: float = 0.8
    expansion_top_k: int = 20
    min_chunk_tokens: int = 8
    max_chunk_tokens: int = 120
    correction_low_score_threshold: float = 0.3
    hard_prune_correction_threshold: int = 2


def execute_python_sandbox(code_str):
    try:
        clean_code = code_str.strip()
        safe_env = {
            "math": math,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "float": float,
            "int": int,
        }
        result = eval(clean_code, {"__builtins__": {}}, safe_env)
        if isinstance(result, float):
            return f"{result:.4f}".rstrip("0").rstrip(".")
        return str(result)
    except Exception as e:
        return f"Error: {e}"


class MCTSNode:
    __slots__ = (
        "tokens",
        "parent",
        "children",
        "visit_count",
        "total_value",
        "past_key_values",
        "last_logits",
        "state_value",
        "is_terminal",
        "text_so_far",
        "depth",
        "is_correction",
        "correction_count",
        "is_pruned",
        "triggered_tool_event",
    )

    def __init__(
        self,
        tokens: List[int],
        parent: Optional["MCTSNode"] = None,
        past_key_values=None,
        last_logits=None,
        state_value: float = 0.5,
        is_terminal: bool = False,
        text_so_far: str = "",
        is_correction: bool = False,
        correction_count: int = 0,
        is_pruned: bool = False,
        triggered_tool_event: bool = False,
    ):
        self.tokens = tokens
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_value = 0.0

        # 显存防线：节点对象中不长期保存 KV cache
        self.past_key_values = None
        self.last_logits = last_logits
        self.state_value = state_value
        self.is_terminal = is_terminal
        self.depth = 0 if parent is None else parent.depth + 1
        self.text_so_far = text_so_far
        self.is_correction = is_correction
        self.correction_count = correction_count
        self.is_pruned = is_pruned
        self.triggered_tool_event = triggered_tool_event

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def uct_score(self, c_puct: float) -> float:
        if self.is_pruned:
            return -float("inf")

        parent_n = max(self.parent.visit_count if self.parent else 1, 1)
        if self.visit_count == 0:
            exploration = c_puct * math.sqrt(parent_n)
            return self.state_value + exploration
        exploration = c_puct * math.sqrt(math.log(parent_n) / self.visit_count)
        return self.q_value + exploration

    def is_fully_expanded(self, branching_factor: int) -> bool:
        live_children = [c for c in self.children if not c.is_pruned]
        return len(live_children) >= branching_factor

    def is_leaf(self) -> bool:
        return len([c for c in self.children if not c.is_pruned]) == 0


class HF_MCTS:
    def __init__(self, model, tokenizer, config: MCTSConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.py_start_id = tokenizer.convert_tokens_to_ids("<|python_start|>")
        self.py_end_id = tokenizer.convert_tokens_to_ids("<|python_end|>")

        self.stop_tokens = {self.eos_token_id, self.im_end_id}
        self.stop_tokens.discard(None)

        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(42)

    @torch.inference_mode()
    def search(self, prompt_text: str, expected_answer: str = None) -> Tuple[str, float]:
        prompt_tokens = self.tokenizer.encode(prompt_text)

        ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        outputs = self.model(ids, use_cache=False, return_value=True)

        root_logits = outputs[0][:, -1, :]
        root_val = torch.sigmoid(outputs[1]).item()

        root = MCTSNode(
            tokens=prompt_tokens,
            parent=None,
            past_key_values=None,
            last_logits=root_logits,
            state_value=root_val,
            text_so_far=prompt_text,
        )

        for _ in range(self.config.num_simulations):
            node = self._select(root)

            if not node.is_terminal and not node.is_pruned and node.depth < self.config.max_depth:
                children = self._expand(node)
                eval_node = children[0] if children else node
            else:
                eval_node = node

            reward = self._evaluate(eval_node, prompt_text)
            self._backpropagate(eval_node, reward)

        return self._extract_best_response(root, prompt_tokens, expected_answer)

    def _select(self, node: MCTSNode) -> MCTSNode:
        cfg = self.config
        while (
            not node.is_leaf()
            and not node.is_terminal
            and not node.is_pruned
            and node.is_fully_expanded(cfg.branching_factor)
        ):
            live = [c for c in node.children if not c.is_pruned]
            if not live:
                break
            node = max(live, key=lambda c: c.uct_score(cfg.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        cfg = self.config
        if node.last_logits is None:
            return []

        candidate_tokens = self._sample_diverse_tokens(
            node.last_logits,
            k=cfg.branching_factor,
            temperature=cfg.expansion_temperature,
            top_k=cfg.expansion_top_k,
        )
        if not candidate_tokens:
            return []

        rollout_tokens = [list(node.tokens) + [tok] for tok in candidate_tokens]
        active = [True for _ in rollout_tokens]
        is_terminal = [False for _ in rollout_tokens]
        last_logits = [None for _ in rollout_tokens]
        state_values = [node.state_value for _ in rollout_tokens]
        generated_counts = [1 for _ in rollout_tokens]

        for _ in range(cfg.max_chunk_tokens):
            if not any(active):
                break

            batch_indices = [i for i, flag in enumerate(active) if flag]
            batch_sequences = [rollout_tokens[i] for i in batch_indices]
            ids, attention_mask = self._left_pad_batch(batch_sequences)
            out = self.model(ids, attention_mask=attention_mask, use_cache=False, return_value=True)

            seq_lens = attention_mask.sum(dim=1)
            next_logits = out[0][torch.arange(len(batch_indices), device=self.device), seq_lens - 1, :]
            next_values = torch.sigmoid(out[1]).view(-1)

            for local_idx, branch_idx in enumerate(batch_indices):
                branch_tokens = rollout_tokens[branch_idx]
                current_text = self.tokenizer.decode(branch_tokens, skip_special_tokens=False)
                current_token_id = branch_tokens[-1]

                last_logits[branch_idx] = next_logits[local_idx].unsqueeze(0)
                state_values[branch_idx] = next_values[local_idx].item()

                if current_token_id in self.stop_tokens:
                    is_terminal[branch_idx] = True
                    active[branch_idx] = False
                    continue

                if self._is_semantic_break(current_text, current_token_id, generated_counts[branch_idx]):
                    active[branch_idx] = False
                    continue

                next_id = torch.argmax(next_logits[local_idx], dim=-1).item()
                rollout_tokens[branch_idx].append(next_id)
                generated_counts[branch_idx] += 1

        new_children: List[MCTSNode] = []
        for idx, tokens in enumerate(rollout_tokens):
            child_tokens = list(tokens)
            child_text = self.tokenizer.decode(child_tokens, skip_special_tokens=False)
            is_correction = node.is_correction
            correction_count = node.correction_count
            triggered_tool_event = False

            if child_tokens and child_tokens[-1] == self.py_end_id:
                triggered_tool_event = True
                child_tokens, child_text, is_correction, correction_count = self._inject_tool_or_correction(
                    child_tokens,
                    child_text,
                    node,
                )

            child = MCTSNode(
                tokens=child_tokens,
                parent=node,
                past_key_values=None,
                last_logits=last_logits[idx],
                state_value=state_values[idx],
                is_terminal=is_terminal[idx],
                text_so_far=child_text,
                is_correction=is_correction,
                correction_count=correction_count,
                triggered_tool_event=triggered_tool_event,
            )
            node.children.append(child)
            new_children.append(child)

        return new_children

    def _inject_tool_or_correction(
        self,
        child_tokens: List[int],
        child_text: str,
        parent: MCTSNode,
    ) -> Tuple[List[int], str, bool, int]:
        matches = re.findall(r"<\|python_start\|>(.*?)<\|python_end\|>", child_text, re.DOTALL)
        if not matches:
            return child_tokens, child_text, parent.is_correction, parent.correction_count

        code = matches[-1]
        result = execute_python_sandbox(code)
        if result.startswith("Error:"):
            error_trace = result[len("Error:") :].strip()
            inject_text = (
                f"<|error_start|>{error_trace}<|error_end|>\n"
                "Please fix the error.\n"
            )
            is_correction = True
            correction_count = parent.correction_count + 1
        else:
            inject_text = f"<|output_start|>{result}<|output_end|>\n"
            is_correction = False
            correction_count = parent.correction_count

        inject_tokens = self.tokenizer.encode(inject_text, add_special_tokens=False)
        child_tokens.extend(inject_tokens)
        child_text += inject_text
        return child_tokens, child_text, is_correction, correction_count

    def _evaluate(self, node: MCTSNode, prompt_text: str) -> float:
        generated_part = node.text_so_far[len(prompt_text) :]
        base = float(node.state_value)

        if node.triggered_tool_event:
            base = min(base + 0.05, 1.0)

        if node.is_correction:
            # 相对衰减：连续纠错会逐步降低价值
            decay = 0.8 ** node.correction_count
            base *= decay

        if (
            node.correction_count >= self.config.hard_prune_correction_threshold
            and base < self.config.correction_low_score_threshold
        ):
            self._hard_prune(node)
            return 0.0

        if node.is_terminal:
            return max(0.0, min(base, 1.0))

        if "<|output_start|>" in generated_part:
            base += 0.03
        if "<|error_start|>" in generated_part:
            base -= 0.04

        return max(0.0, min(base, 1.0))

    def _hard_prune(self, node: MCTSNode):
        node.is_pruned = True
        node.is_terminal = True

        parent = node.parent
        if parent is None:
            return

        siblings = [s for s in parent.children if s is not node and not s.is_pruned]
        if not siblings:
            return

        best = max(siblings, key=lambda c: c.q_value)
        transfer = max(best.q_value, 0.5)
        best.visit_count += 1
        best.total_value += transfer

    def _backpropagate(self, node: MCTSNode, reward: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent

    def _extract_best_response(
        self,
        root: MCTSNode,
        prompt_tokens: List[int],
        expected_answer: str = None,
    ) -> Tuple[str, float]:
        if root.is_leaf():
            return "", root.q_value

        all_leaves: List[MCTSNode] = []

        def _dfs(node: MCTSNode):
            live_children = [c for c in node.children if not c.is_pruned]
            if node.visit_count > 0 and len(live_children) == 0 and not node.is_pruned:
                all_leaves.append(node)
            for child in live_children:
                _dfs(child)

        _dfs(root)
        if not all_leaves:
            return "", root.q_value

        def _extract_ans(text: str):
            match = re.search(r"\\boxed\{([^}]+)\}", text)
            if match:
                return match.group(1).replace(",", "").strip()
            match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-]+)", text)
            if match:
                return match.group(1).replace(",", "").strip()
            return None

        def _key(n: MCTSNode):
            text = self.tokenizer.decode(n.tokens[len(prompt_tokens) :], skip_special_tokens=False)
            has_tool = 1 if "<|output_start|>" in text else 0
            has_correction = 1 if "<|error_start|>" in text else 0
            if expected_answer is not None:
                model_ans = _extract_ans(text)
                is_correct = 1 if model_ans == expected_answer else 0
                return (is_correct, n.q_value, has_tool, has_correction, -n.depth)
            return (n.q_value, has_tool, has_correction, -n.depth)

        best = max(all_leaves, key=_key)
        response_text = self.tokenizer.decode(best.tokens[len(prompt_tokens) :], skip_special_tokens=False)
        return response_text, root.q_value

    def _sample_diverse_tokens(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float,
        top_k: int,
    ) -> List[int]:
        vocab = logits.size(-1)
        effective_top_k = min(top_k, vocab) if top_k > 0 else vocab

        if effective_top_k < vocab:
            topk_vals, _ = torch.topk(logits, effective_top_k)
            threshold = topk_vals[:, -1:].expand_as(logits)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if probs.sum() == 0:
            probs = torch.ones_like(probs) / vocab

        num_samples = max(1, min(k, int((probs > 0).sum().item())))
        sampled = torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=False,
            generator=self._rng,
        )
        return sampled[0].tolist()

    def _left_pad_batch(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            input_ids[i, -seq_len:] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention_mask[i, -seq_len:] = 1

        return input_ids, attention_mask

    def _is_semantic_break(self, text: str, token_id: int, generated_tokens: int) -> bool:
        if token_id in self.stop_tokens:
            return True
        if token_id == self.py_end_id:
            return True
        if generated_tokens >= self.config.min_chunk_tokens and "\n\n" in text[-12:]:
            return True
        return False
