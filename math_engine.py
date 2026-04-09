import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


class MathEngine:
    def __init__(self):
        self.x = sp.symbols('x')

    def parse_expression(self, expr_str):
        """支持隐式乘法（如2x）并解决命名空间冲突"""
        clean_str = expr_str.replace('^', '**')
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            # 显式注入 sympy 字典，确保 asin, acos 等函数在云端被正确解析
            return parse_expr(clean_str, transformations=transformations, global_dict=sp.__dict__)
        except Exception:
            return sp.sympify(clean_str)

    def get_analysis(self, expr):
        """计算导数和最简原函数"""
        deriv = sp.diff(expr, self.x)
        # 求积分并强制简化，方便作业展示
        integral = sp.integrate(expr, self.x)
        return deriv, sp.simplify(integral)

    def generate_plotly_plot(self, expr_list):
        fig = go.Figure()
        x_vals = np.linspace(-15, 15, 1500)

        for expr, label, color in expr_list:
            try:
                # 【全象限绘图补丁】：手动替换幂运算，确保负数分数次方（如x**2/3）正常显示
                fixed_expr = expr.replace(
                    lambda e: e.is_Pow,
                    lambda e: sp.Pow(sp.Abs(e.base), e.exp)
                )

                f_np = sp.lambdify(self.x, fixed_expr, modules=['numpy'])
                y_plot = np.real(f_np(x_vals))

                # 过滤异常值，防止 tan(x) 等函数拉断坐标轴
                y_plot[np.abs(y_plot) > 100] = np.nan

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_plot,
                    mode='lines', name=label,
                    line=dict(color=color, width=2.5),
                    connectgaps=False
                ))
            except Exception:
                continue

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(title="x", zeroline=True, zerolinewidth=2, range=[-6, 6]),
            yaxis=dict(title="y", zeroline=True, zerolinewidth=2, range=[-6, 6]),
            dragmode='pan',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig