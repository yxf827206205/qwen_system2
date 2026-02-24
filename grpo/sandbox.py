"""
sandbox.py — 受限的 Python 代码执行沙箱
用 SIGALRM 实现超时（Linux only）。生产环境请替换为 Docker 子进程。
"""
import re
import signal
import contextlib
import builtins
from io import StringIO
from typing import Optional


class PythonSandbox:
    """
    安全（尽力而为）的 exec() 沙箱。
    - 通过 SIGALRM 限制执行时间（仅 Linux）
    - 拦截危险内置函数
    - 捕获 stdout 及最后一个表达式的值
    """

    TIMEOUT_SEC    = 5
    MAX_OUTPUT_LEN = 600  # 超出截断，防止撑爆 context

    _BLOCKED_BUILTINS = {
        "open", "exec", "eval", "compile",
        "__import__", "breakpoint", "input",
    }

    def __init__(self):
        self._safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if name not in self._BLOCKED_BUILTINS and not name.startswith("__")
        }

    # ──────────────────────────────────────────────────────────────────────
    def execute(self, code: str) -> str:
        """执行 code，返回 stdout + 最后表达式值，结果长度不超过 MAX_OUTPUT_LEN。"""
        buf = StringIO()

        def _alarm(signum, frame):
            raise TimeoutError(f"执行超时（>{self.TIMEOUT_SEC}s）")

        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(self.TIMEOUT_SEC)

        try:
            local_ns = {}
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": self._safe_builtins}, local_ns)  # noqa: S102

            stdout_out = buf.getvalue()
            expr_out   = self._eval_last_expr(code, local_ns)
            combined   = (stdout_out + expr_out).strip() or "# (no output)"
            return combined[: self.MAX_OUTPUT_LEN]

        except TimeoutError as e:
            return f"# TimeoutError: {e}"
        except Exception as e:
            return f"# {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    # ──────────────────────────────────────────────────────────────────────
    def _eval_last_expr(self, code: str, local_ns: dict) -> str:
        """尝试对最后一行进行 eval，捕获非 None 返回值。"""
        lines = [ln.strip() for ln in code.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        last = lines[-1]
        skip_prefixes = ("import", "from", "def", "class", "if", "for",
                         "while", "try", "with", "#", "print", "return")
        if any(last.startswith(p) for p in skip_prefixes):
            return ""
        try:
            val = eval(last, {"__builtins__": self._safe_builtins}, local_ns)  # noqa: S307
            return str(val) if val is not None else ""
        except Exception:
            return ""
