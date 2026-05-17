"""AI translation layer for the calculus drawing agent.

This module only talks to the DeepSeek/OpenAI-compatible API and sanitizes the
model output into a single SymPy-friendly expression string.
"""
from __future__ import annotations

import re
from typing import Optional

from openai import OpenAI


_CODE_FENCE_RE = re.compile(r"```(?:python|py|text)?|```", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^(?:[zfgy]\s*=|f\s*\([^)]*\)\s*=|z\s*\([^)]*\)\s*=)", re.IGNORECASE)
_ALLOWED_CHARS_RE = re.compile(r"[^0-9a-zA-Z_+\-*/^().,| ]")


class MathAgent:
    """Translate natural-language function descriptions into SymPy expressions."""

    def __init__(self, api_key: str | None, base_url: str = "https://api.deepseek.com"):
        if not api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY。请在环境变量或 Streamlit secrets 中配置。")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _clean_model_output(text: str) -> str:
        """Keep only the first plausible expression line and normalize syntax."""
        text = _CODE_FENCE_RE.sub("", text or "").strip()

        # 有些模型会输出解释 + 公式，这里优先取第一行非空内容
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = lines[0] if lines else text

        text = _PREFIX_RE.sub("", text.strip())
        text = text.replace("，", ",").replace("（", "(").replace("）", ")")
        text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
        text = re.sub(r"\s+", "", text)
        text = text.replace("ln(", "log(")

        return _ALLOWED_CHARS_RE.sub("", text)

    def chat_to_formula(self, user_query: str, is_3d: bool = False) -> Optional[str]:
        """Return one pure expression such as ``sin(x)+x**2`` or ``x**2+y**2``."""
        vars_info = "x 和 y" if is_3d else "x"

        examples = (
            "例：'x的平方加sin x' -> x**2 + sin(x)\n"
            "例：'e的负x平方' -> exp(-x**2)\n"
            "例：'x和y的平方和' -> x**2 + y**2\n"
            "例：'x的绝对值' -> Abs(x)"
        )

        system_content = f"""
你是一个严格的数学表达式翻译器。把用户描述翻译成关于变量 {vars_info} 的 SymPy/Python 表达式。
必须遵守：
1. 只输出一个表达式，不要解释，不要 Markdown，不要写 f(x)=、y=、z=。
2. 允许函数：sin, cos, tan, asin, acos, atan, exp, log, sqrt, Abs, sinh, cosh。
3. 幂运算必须用 **，不能用 ^。
4. 乘法要明确写 *，例如 2*x。
5. 常数函数直接输出数字，例如 5。
6. 如果是二元函数，只能使用 x 和 y；如果是一元函数，只能使用 x。
{examples}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.0,
                max_tokens=80,
            )

            return self._clean_model_output(response.choices[0].message.content or "")

        except Exception:
            return None
