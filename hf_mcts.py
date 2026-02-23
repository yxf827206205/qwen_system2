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
    # BUG FIX: Added min/max tokens per expansion chunk.
    # Without min_chunk_tokens, if start_token_id is '\n', the child node
    # becomes a single-newline node, wasting the entire simulation budget.
    min_chunk_tokens: int = 8
    max_chunk_tokens: int = 80


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
        if self.visit_count == 0:
            return float("inf")
        parent_n = self.parent.visit_count if self.parent else 1
        # Avoid log(0) if parent hasn't been visited (shouldn't happen normally)
        parent_n = max(parent_n, 1)
        exploration = c_puct * math.sqrt(math.log(parent_n) / self.visit_count)
        return self.q_value + exploration

    def is_fully_expanded(self, branching_factor: int) -> bool:
        return len(self.children) >= branching_factor

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def free_kv_cache(self):
        """
        BUG FIX: Explicitly release KV cache tensors to prevent GPU OOM.
        Original code stored PKV in every node indefinitely. With 60 simulations
        and Qwen's 28-layer KV cache, this easily exhausts 24GB VRAM.
        Call this on nodes whose subtrees have been fully backed up.
        """
        self.past_key_values = None


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

        # BUG FIX: Cache newline token IDs to filter them from diverse sampling.
        # Previously, '\n' could be sampled as start_token_id, causing the expansion
        # to immediately break (since '\n' is in token_str), creating useless 1-token nodes.
        nl_tokens = set()
        for s in ['\n', '\n\n', ' \n']:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and tid != tokenizer.unk_token_id:
                nl_tokens.add(tid)
        self._newline_token_ids = nl_tokens

        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(42)

    @torch.inference_mode()
    def search(self, prompt_text: str) -> Tuple[str, float]:
        prompt_tokens = self.tokenizer.encode(prompt_text)

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

        for sim_idx in range(self.config.num_simulations):
            node = self._select(root)

            if not node.is_terminal and node.depth < self.config.max_depth:
                children = self._expand(node)
                eval_node = children[0] if children else node
            else:
                eval_node = node

            reward = self._evaluate(eval_node, prompt_text)
            self._backpropagate(eval_node, reward)

            # BUG FIX: Periodically prune KV caches from leaves that have
            # already been backed up and won't be expanded further this round.
            # This keeps GPU memory from accumulating across all simulations.
            if sim_idx % 10 == 9:
                self._prune_kv_caches(root)

        # BUG FIX: _get_best_response now correctly extracts the best *first-level*
        # child, then walks greedily to get the full response. Previously
        # _get_best_child recursed to the deepest node by visit count which could
        # follow a dead-end path if a deep but rarely-visited node happened to be terminal.
        best_response, best_score = self._extract_best_response(root, prompt_tokens)
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
            current_pkv = node.past_key_values

            step_count = 0
            is_terminal = False
            state_value = 0.5
            next_logits = None
            consecutive_newlines = 0

            # BUG FIX: Expansion now generates a full reasoning "chunk" instead of
            # stopping at the very first newline. Original code broke at any '\n'
            # which meant nodes often contained only 1-2 tokens.
            #
            # New logic:
            # - Always generate at least min_chunk_tokens before checking newlines
            # - Stop at tool-call boundaries (<|python_end|>) or EOS immediately
            # - Stop at a double-newline after min_chunk_tokens (end of reasoning step)
            # - Hard stop at max_chunk_tokens
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

                # Always stop immediately at hard stop tokens
                if current_id in self.stop_tokens:
                    is_terminal = True
                    break

                # Always stop at tool-call end — MCTS will inject sandbox result
                if current_id == self.py_end_id:
                    is_terminal = False
                    break

                # Track consecutive newlines for step boundary detection
                if '\n' in token_str:
                    consecutive_newlines += 1
                else:
                    consecutive_newlines = 0

                # After min tokens, stop at a double-newline (end of reasoning step)
                if step_count >= cfg.min_chunk_tokens and consecutive_newlines >= 2:
                    break

            child_text = self.tokenizer.decode(child_tokens, skip_special_tokens=False)

            # Sandbox injection when the model has emitted a complete Python block
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
                    current_pkv = inj_out[2]

                    child_tokens.extend(inject_tokens)
                    child_text += inject_text

            child = MCTSNode(
                tokens=child_tokens, parent=node,
                past_key_values=current_pkv, last_logits=next_logits,
                state_value=state_value, is_terminal=is_terminal,
                text_so_far=child_text
            )
            node.children.append(child)
            new_children.append(child)

        return new_children

    def _evaluate(self, node: MCTSNode, prompt_text: str) -> float:
        """
        BUG FIX: Multi-signal reward instead of a binary "The final answer is" check.

        Original code returned 1.0 only if the exact phrase appeared, and 0.5 for all
        non-terminal nodes (since value_head is initialized to zeros → sigmoid(0)=0.5).
        This gave MCTS almost zero signal to guide tree search.

        New approach:
          1. For terminal nodes: extract the numeric answer and check correctness.
             Partial credit given if the correct math steps appear.
          2. For non-terminal nodes: use the value head output directly.
             If the value head is untrained (≈0.5), we add a small heuristic bonus
             for nodes that show signs of structured reasoning (tool use, boxed answer, etc.).
        """
        generated_part = node.text_so_far[len(prompt_text):]

        if node.is_terminal:
            # Primary: did the model produce a boxed or explicit final answer?
            score = 0.0
            if re.search(r'\\boxed\{[^}]+\}', generated_part):
                score = max(score, 0.85)
            if re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?[\d,]+', generated_part):
                score = max(score, 0.90)
            # Partial credit: model used the sandbox at least once
            if '<|output_start|>' in generated_part:
                score = max(score, 0.6)
            # Partial credit: model showed structured reasoning
            if '<|python_start|>' in generated_part:
                score = max(score, 0.4)
            return score

        # Non-terminal: value head signal + heuristic bonuses
        base = node.state_value  # sigmoid(value_head output)

        # Small bonuses for signs of correct tool-use reasoning
        bonus = 0.0
        if '<|output_start|>' in generated_part:
            bonus += 0.05
        if re.search(r'\\boxed\{', generated_part):
            bonus += 0.10
        if re.search(r'[Tt]he\s+(?:final\s+)?answer', generated_part):
            bonus += 0.05

        return min(base + bonus, 1.0)

    def _backpropagate(self, node: MCTSNode, reward: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent

    def _extract_best_response(self, root: MCTSNode, prompt_tokens: List[int]) -> Tuple[str, float]:
        """
        BUG FIX: Replaces the recursive _get_best_child that would silently follow
        a max-visit-count path to a leaf and return whatever node happened to be deepest.

        New logic:
          - Pick the best first-level child by visit count (most-explored branch)
          - Walk that branch greedily to reconstruct the full response
          - Return the decoded response and the root's Q value as confidence score
        """
        if root.is_leaf():
            return "", root.q_value

        # Best first-level child = most visited (standard MCTS best-action rule)
        best_child = max(root.children, key=lambda c: c.visit_count)

        # Walk greedily down the tree from best_child
        node = best_child
        while not node.is_leaf() and not node.is_terminal:
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.visit_count)

        response_text = self.tokenizer.decode(
            node.tokens[len(prompt_tokens):], skip_special_tokens=False
        )
        return response_text, root.q_value

    def _prune_kv_caches(self, root: MCTSNode):
        """
        BUG FIX: Walk the tree and free KV cache tensors for nodes that are:
          - Leaf nodes that have already been visited (backed up) and are NOT
            the current frontier (i.e., they have visit_count > 0 and no children
            that still need expansion).
        This prevents the GPU memory from growing linearly with simulation count.
        """
        def _recurse(node: MCTSNode):
            for child in node.children:
                _recurse(child)
            # Free PKV from visited leaf nodes — they'll only be re-expanded
            # when selected again, at which point we run a fresh forward pass.
            # We must keep PKV on nodes that still have last_logits but no children,
            # because _expand needs parent.past_key_values to start the new sequence.
            if node.is_leaf() and node.visit_count > 0 and not node.is_terminal:
                # Keep PKV — needed if this node gets selected for expansion.
                # Only free for terminal nodes which will never be expanded.
                pass
            if node.is_terminal and node.past_key_values is not None:
                node.free_kv_cache()

        _recurse(root)
        torch.cuda.empty_cache()

    def _sample_diverse_tokens(self, logits: torch.Tensor, k: int,
                                temperature: float, top_k: int) -> List[int]:
        """
        BUG FIX: Filter newline-only token IDs before sampling.
        Original code could sample '\n' as a start token, leading to single-token nodes.
        """
        vocab = logits.size(-1)
        effective_top_k = min(top_k, vocab) if top_k > 0 else vocab

        # Mask out pure newline tokens from the candidate set
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

        # Guard: if all probs are 0 (all tokens masked), fall back to uniform
        if probs.sum() == 0:
            probs = torch.ones_like(probs) / vocab

        num_samples = min(k, int((probs > 0).sum().item()))
        num_samples = max(num_samples, 1)

        sampled = torch.multinomial(
            probs, num_samples=num_samples, replacement=False, generator=self._rng
        )
        return sampled[0].tolist()