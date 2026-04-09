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
            "你是一个数学符号转换接口。只输出 Python 数学表达式。\n"
            "禁止 LaTeX，禁止 y=，禁止文字说明。幂运算使用 **。"
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
            # 强效清洗
            res = re.sub(r'```python|```|y\s*=|f\(x\)\s*=|[\s]', '', res)
            return res.replace('^', '**')
        except:
            return None
