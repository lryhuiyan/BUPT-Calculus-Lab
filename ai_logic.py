import re
from openai import OpenAI

class MathAgent:
    """与大模型 API 通信的核心代理类"""
    def __init__(self, api_key):
        # 初始化客户端，这里接入的是兼容 OpenAI 格式的 DeepSeek 接口
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def chat_to_formula(self, user_query, is_3d=False):
        """
        将自然语言转化为纯 Python/SymPy 可识别的数学公式。
        采用 Zero-shot Prompting 策略，严格限制输出格式以防止系统崩溃。
        """
        vars_info = "x 和 y" if is_3d else "变量 x"
        
        # 预设 System Prompt，下达格式铁律
        system_content = (
            f"你是一个数学翻译接口。将描述转为关于 {vars_info} 的 Python 表达式。\n"
            "1. 只输出纯表达式，不包含 y= 或 f(x)=\n"
            "2. 禁止任何库前缀 (如 np. 或 math.)\n"
            "3. 保持幂运算为 **\n"
            "4. 如果用户输入的是常数或常值函数（例如 5 或 10），请直接输出该常数数字。\n"
            "5. 绝不输出任何解释性文字"
        )
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": system_content}, 
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0 # 设为 0.0，保证同一输入绝对产生相同结果，减少幻觉
            )
            
            # 获取大模型返回内容
            res = response.choices[0].message.content.strip()
            
            # 强力正则清洗：剔除 Markdown 语法块、等号前缀以及多余空格
            res = re.sub(r'```python|```|[yz]\s*=|f\(x\)\s*=|[\s]', '', res)
            
            # 兼容处理：将普通数学上推的 ^ 替换为 Python 的幂运算 **
            return res.replace('^', '**')
        except: 
            return None
