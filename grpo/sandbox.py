"""
sandbox.py — 安全且极速的内存 Python 沙盒 (支持持久化记忆)
"""
import ast
import contextlib
import signal
from io import StringIO
from typing import Any, Dict, Optional

class PythonSandbox:
    TIMEOUT_SEC = 0.2  
    MAX_OUTPUT_LEN = 1000

    def __init__(self):
        # 限制可用的内置函数，防止恶意代码破坏系统
        self._safe_builtins = {
            "print": print, "range": range, "len": len,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "sum": sum, "max": max, "min": min, "abs": abs, "round": round,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "sorted": sorted, "any": any, "all": all,
        }

    def execute(self, code: str, state: Optional[dict] = None) -> str:
        """执行 Python 代码，支持传入 state 字典实现上下文记忆。"""
        buf = StringIO()

        def _alarm(signum, frame):
            raise TimeoutError(f"执行超时（>{self.TIMEOUT_SEC}s）")

        # 设置超时闹钟
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, self.TIMEOUT_SEC) # 使用更精确的浮点数定时器

        try:
            # 🔥 核心：如果传入了 state（记忆），就使用它；否则创建一个空的全新环境
            local_ns = state if state is not None else {}
            
            with contextlib.redirect_stdout(buf):
                # 在内存中极速编译并执行字节码，共享 local_ns
                exec(code, {"__builtins__": self._safe_builtins}, local_ns)

            stdout_out = buf.getvalue()
            # 捕获最后一行表达式的输出（类似于 Jupyter 的行为）
            expr_out   = self._eval_last_expr(code, local_ns)
            
            combined   = (stdout_out + expr_out).strip() or "# (no output)"
            return combined[: self.MAX_OUTPUT_LEN]

        except TimeoutError as e:
            return f"TimeoutError: {e}"
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        finally:
            # 必须关闭闹钟
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    def _eval_last_expr(self, code: str, local_ns: dict) -> str:
        """尝试计算代码块最后一行表达式的值。"""
        try:
            tree = ast.parse(code)
            if not tree.body:
                return ""
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                # 编译最后一行表达式
                expr_code = compile(ast.Expression(last_node.value), "<ast>", "eval")
                # 🔥 使用刚才 exec 执行后的同一个 local_ns 计算表达式
                val = eval(expr_code, {"__builtins__": self._safe_builtins}, local_ns)
                if val is not None:
                    return str(val) + "\n"
        except Exception:
            pass
        return ""