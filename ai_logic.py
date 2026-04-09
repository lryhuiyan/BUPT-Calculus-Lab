import re
from openai import OpenAI


class MathAgent:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def chat_to_formula(self, user_query):
        """
        将自然语言解析为纯净的数学表达式
        示例：'x的平方' -> 'x**2'
        """
        system_content = (
            "你是一个专业的数学符号转换接口。\n"
            "【规则】：\n"
            "1. 只输出关于变量 x 的 Python 表达式。\n"
            "2. 严禁输出 'y=', 'f(x)=' 等前缀，严禁输出任何中文说明。\n"
            "3. 严禁输出 Markdown 代码块标签（```）。\n"
            "4. 幂运算必须用 **，乘法必须带 *。\n"
            "5. 反三角函数必须使用 asin, acos, atan 格式。"
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

            # 【双重清洗】：防止 AI 偶尔“不听话”输出代码块或多余等号
            # 移除所有空白字符、代码框标记、以及类似 y= 的前缀
            clean_res = re.sub(r'```python|```|y\s*=|f\(x\)\s*=|[\s]', '', res)

            # 兼容性处理：把 ^ 替换为 ** (双重保险)
            clean_res = clean_res.replace('^', '**')

            return clean_res

        except Exception as e:
            # 如果 AI 解析挂了，至少在终端打印一下方便排查
            print(f"DeepSeek API 交互异常: {e}")
            return None