import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class MathEngine:
    def __init__(self):
        self.x = sp.symbols('x')

    def parse_expression(self, expr_str):
        # 清洗 LaTeX 遗留字符
        clean_str = expr_str.replace('\\', '').replace('cdot', '*')
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            return parse_expr(clean_str, transformations=transformations, global_dict=sp.__dict__)
        except:
            return sp.sympify(clean_str)

    def get_analysis(self, expr):
        deriv = sp.diff(expr, self.x)
        integral = sp.integrate(expr, self.x)
        return deriv, sp.simplify(integral)

    def generate_plotly_plot(self, expr_list):
        fig = go.Figure()
        # 高采样率确保曲线平滑
        x_vals = np.linspace(-15, 15, 3000)
        
        for expr, label, color in expr_list:
            try:
                # 计院严谨性：不再使用 sp.Abs 强制转换，保留原始数学特性
                # 使用 lambdify 生成 numpy 函数
                f_np = sp.lambdify(self.x, expr, modules=['numpy'])
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # 关键：计算原始值，允许产生复数(Complex)
                    y_raw = f_np(x_vals.astype(complex))
                    
                    # 1. 处理定义域：如果虚部绝对值 > 1e-9，说明在实数域无定义，设为 NaN
                    y_plot = np.where(np.abs(np.imag(y_raw)) < 1e-9, np.real(y_raw), np.nan)
                    
                    # 2. 处理 sin(x)/x 等 0/0 型间断点
                    # 找到 x 极接近 0 的索引
                    zero_idx = np.abs(x_vals).argmin()
                    if np.isnan(y_plot[zero_idx]) or np.isinf(y_plot[zero_idx]):
                        # 使用 SymPy 计算精确极限
                        try:
                            lim = float(sp.limit(expr, self.x, 0))
                            y_plot[zero_idx] = lim
                        except:
                            pass

                    # 3. 过滤垂直渐近线带来的视觉污染
                    y_plot[np.abs(y_plot) > 100] = np.nan 

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_plot, mode='lines', name=label,
                    line=dict(color=color, width=3),
                    connectgaps=False # 严格不连接 NaN 断点，确保 sqrt(x) 定义域视觉正确
                ))
            except:
                continue

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(title="x", zeroline=True, zerolinewidth=2, range=[-7, 7]),
            yaxis=dict(title="y", zeroline=True, zerolinewidth=2, range=[-5, 5]),
            dragmode='pan' # 默认开启拖动
        )
        return fig
