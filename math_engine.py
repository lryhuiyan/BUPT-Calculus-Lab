import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class MathEngine:
    def __init__(self):
        self.x = sp.symbols('x')

    def parse_expression(self, expr_str):
        clean_str = expr_str.replace('\\', '').replace('cdot', '*')
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            return parse_expr(clean_str, transformations=transformations, global_dict=sp.__dict__)
        except:
            return sp.sympify(clean_str)

    def smart_pow_fix(self, expr):
        """
        核心兼容逻辑：
        1. 如果幂次的分母是偶数 (如 1/2)，保持原样 (sqrt(x) 负半轴无定义)。
        2. 如果幂次的分母是奇数 (如 2/3)，使用 Abs 补丁确保实数域图像完整。
        """
        def fix_node(node):
            if node.is_Pow:
                base, exp = node.as_base_exp()
                if exp.is_Rational:
                    # 获取分母
                    denom = exp.q 
                    if denom % 2 != 0:
                        # 分母为奇数 (如 1/3, 2/3)，负数有实数根
                        return sp.Pow(sp.Abs(base), exp) if exp.p % 2 == 0 else sp.sign(base) * sp.Pow(sp.Abs(base), exp)
            return node

        return expr.replace(sp.Pow, fix_node)

    def get_analysis(self, expr):
        deriv = sp.diff(expr, self.x)
        integral = sp.integrate(expr, self.x)
        return deriv, sp.simplify(integral)

    def generate_plotly_plot(self, expr_list):
        fig = go.Figure()
        x_vals = np.linspace(-15, 15, 3000)
        zero_idx = np.abs(x_vals).argmin()
        
        for expr, label, color in expr_list:
            try:
                # 应用智能幂次修复
                fixed_expr = self.smart_pow_fix(expr)
                f_np = sp.lambdify(self.x, fixed_expr, modules=['numpy'])
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # 计算时使用 complex128 确保中间过程不溢出
                    y_raw = f_np(x_vals.astype(complex))
                    
                    # 过滤逻辑：
                    # 如果原表达式里有 sqrt 或分母为偶数的幂，原本就会在 y_raw 产生虚部
                    # 我们的智能修复已经处理了奇数分母，所以剩下的虚部一定是真正的“无定义”
                    y_plot = np.where(np.abs(np.imag(y_raw)) < 1e-9, np.real(y_raw), np.nan)
                    
                    # 修复 sin(x)/x 零点
                    if np.isnan(y_plot[zero_idx]) or np.isinf(y_plot[zero_idx]):
                        try:
                            lim = float(sp.limit(expr, self.x, 0))
                            y_plot[zero_idx] = lim
                        except: pass

                    y_plot[np.abs(y_plot) > 100] = np.nan 

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_plot, mode='lines', name=label,
                    line=dict(color=color, width=3),
                    connectgaps=False
                ))
            except:
                continue

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(zeroline=True, zerolinewidth=2, range=[-7, 7]),
            yaxis=dict(zeroline=True, zerolinewidth=2, range=[-5, 5]),
            dragmode='pan'
        )
        return fig
