import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class MathEngine:
    def __init__(self):
        self.x = sp.symbols('x')

    def parse_expression(self, expr_str):
        clean_str = expr_str.replace('\\', '').replace('math.', '').replace('cdot', '*')
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            return parse_expr(clean_str, transformations=transformations, global_dict=sp.__dict__)
        except:
            return sp.sympify(clean_str)

    def generate_plotly_plot(self, expr_list):
        fig = go.Figure()
        # 1. 计院精准采样：在 0 附近手动避开，防止 inf 连线
        # 我们分两段采样，中间留一个 1e-5 的极小缝隙
        x_left = np.linspace(-15, -1e-5, 1000)
        x_right = np.linspace(1e-5, 15, 1000)
        # 在两段之间插入一个真正的 NaN，强制 Plotly 提笔
        x_vals = np.concatenate([x_left, [0], x_right])
        
        for expr, label, color in expr_list:
            try:
                # --- 幂函数兼容逻辑 ---
                pows = expr.atoms(sp.Pow)
                fixed_expr = expr
                for p in pows:
                    base, exponent = p.as_base_exp()
                    if exponent.is_Rational:
                        p_val, q_val = exponent.p, exponent.q
                        if q_val % 2 != 0:
                            if p_val % 2 == 0:
                                fixed_expr = fixed_expr.subs(p, sp.Abs(base)**exponent)
                            else:
                                fixed_expr = fixed_expr.subs(p, sp.sign(base) * sp.Abs(base)**exponent)
                
                f_np = sp.lambdify(self.x, fixed_expr, modules=['numpy'])
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # 2. 计算 y 值
                    y_plot = f_np(x_vals.astype(float))
                    
                    # 3. 处理无效值和异常跳变
                    y_plot = np.where(np.isfinite(y_plot), y_plot, np.nan)
                    
                    # 强制在 x=0 的索引位（即我们插入的那个点）处理特殊情况
                    zero_idx = 1000 # 因为左边有 1000 个点，索引 1000 正好是 [0]
                    
                    # 如果是 sin(x)/x，我们在 0 处填入极限值
                    if "sin" in str(expr) and "x" in str(expr):
                        try:
                            lim = float(sp.limit(expr, self.x, 0))
                            y_plot[zero_idx] = lim
                        except: pass
                    else:
                        # 否则（比如导函数），确保 0 处是 NaN，强行断开
                        y_plot[zero_idx] = np.nan

                    # 裁切视觉溢出
                    y_plot = np.where(np.abs(y_plot) < 100, y_plot, np.nan)

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_plot, 
                    mode='lines', 
                    name=label,
                    line=dict(color=color, width=3),
                    connectgaps=False 
                ))
            except Exception as e:
                print(f"Plot Error: {e}")
                continue

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(zeroline=True, zerolinewidth=2, range=[-7, 7]),
            yaxis=dict(zeroline=True, zerolinewidth=2, range=[-5, 5]),
            dragmode='pan'
        )
        return fig

    def get_analysis(self, expr):
        deriv = sp.diff(expr, self.x)
        integral = sp.integrate(expr, self.x)
        return deriv, sp.simplify(integral)
