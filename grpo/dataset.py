"""
dataset.py — GSM8K 数据集封装
"""
from typing import Dict, Any
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from config import GRPOConfig


SYSTEM_PROMPT = (
    "You are a meticulous math genius. "
    "Solve problems step-by-step inside <think>…</think>. "
    "You may call Python to compute intermediate results:\n"
    "  <|python_start|>\n  your_code_here\n  <|python_end|>\n"
    "The interpreter will inject the output between "
    "<|output_start|> and <|output_end|>. "
    "Always end with your final answer inside \\boxed{}."
)


class GSM8KDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        cfg:       GRPOConfig,
        split:     str = "train",
    ):
        raw = load_dataset("gsm8k", "main", split=split)
        raw = raw.shuffle(seed=cfg.seed).select(
            range(min(cfg.num_train_samples, len(raw)))
        )
        self.examples  = list(raw)
        self.tokenizer = tokenizer
        self.cfg       = cfg

    def __len__(self) -> int:
        return len(self.examples)

    # def __getitem__(self, idx: int) -> Dict[str, Any]:
    #     item     = self.examples[idx]
    #     expected = item["answer"].split("####")[-1].replace(",", "").strip()

    #     messages = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user",   "content": item["question"]},
    #     ]
    #     # 在 prompt 末尾预置 <think>\n，引导模型进入推理模式
    #     prompt_text = (
    #         self.tokenizer.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )
    #         + "<think>\n"
    #     )
    #     prompt_ids = self.tokenizer(
    #         prompt_text, return_tensors="pt", add_special_tokens=False
    #     ).input_ids  # [1, L]

    #     return {
    #         "prompt_ids": prompt_ids,
    #         "expected":   expected,
    #         "question":   item["question"],
    #     }
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item     = self.examples[idx]
        expected = item["answer"].split("####")[-1].replace(",", "").strip()

        messages = [
            {"role": "user", "content": item["question"]},
        ]
        
        # 在 prompt 末尾预置 <think>\n，引导模型进入推理模式 (和生成测试时完全一样)
        prompt_text = (
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + "<think>\n"
        )
        
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids  # [1, L]

        return {
            "prompt_ids": prompt_ids,
            "expected":   expected,
            "question":   item["question"],
        }


def collate_fn(batch):
    """不做 padding，保持列表形式传入 trainer。"""
    return batch
