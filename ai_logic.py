import re
from openai import OpenAI


class MathAgent:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def chat_to_formula(self, user_query):
        system_content = (
            "你是一个数学公式转换器。将用户的描述转换为纯粹的 Python 数学表达式。\n"
            "【绝对禁止】：禁止输出 \\frac, \\sin, \\times 等 LaTeX 格式！\n"
            "【绝对禁止】：禁止输出 y=, f(x)=, 或者任何解释性文字。\n"
            "【正确示例】：如果用户说'正弦x除以x'，你只输出: sin(x)/x\n"
            "【正确示例】：如果用户说'x的平方'，你只输出: x**2"
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0
            )
            res = response.choices[0].message.content.strip()

            # 暴力清洗：如果 AI 还是不听话输出了 LaTeX 或代码块，强制替换
            res = re.sub(r'```python|```|y\s*=|f\(x\)\s*=|[\s]', '', res)
            res = res.replace(r'\frac', '').replace('{', '(').replace('}', ')').replace('^', '**')
            # 针对 sin(x)/x 的特殊清洗
            if '\\sin' in res:
                res = res.replace('\\', '')

            return res
        except Exception as e:
            print(f"AI解析失败: {e}")
            return None
