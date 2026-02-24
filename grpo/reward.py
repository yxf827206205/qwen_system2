"""
reward.py — 奖励函数
"""
import re
from typing import Optional
from config import GRPOConfig


def extract_final_answer(text: str) -> Optional[str]:
    """从生成文本里提取最终答案。优先 \\boxed{}，其次自然语言描述。"""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).replace(",", "").strip()

    m = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*\$?([\d\.\-,]+)", text
    )
    if m:
        return m.group(1).replace(",", "").strip()

    return None


def compute_reward(generated_text: str, expected_answer: str, cfg: GRPOConfig) -> float:
    """
    返回 [0, reward_correct + reward_tool_use + reward_format] 范围内的标量奖励。

    奖励结构（稀疏 + 密集混合）：
      1. 正确性（最重要）：答案与期望值匹配
      2. 工具使用：成功调用了沙箱
      3. 格式规范：包含 <think> 推理块
    """
    reward = 0.0
    clean_expected = expected_answer.replace(",", "").strip()

    # 1. 正确性
    pred = extract_final_answer(generated_text)
    if pred is not None and pred == clean_expected:
        reward += cfg.reward_correct

    # 2. 工具使用加分
    if cfg.PYTHON_START in generated_text and cfg.PYTHON_END in generated_text:
        reward += cfg.reward_tool_use

    # 3. 格式加分
    if "<think>" in generated_text and "</think>" in generated_text:
        reward += cfg.reward_format

    return reward
