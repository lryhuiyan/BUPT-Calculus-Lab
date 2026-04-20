import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re

class MathEngine:
    def __init__(self):
        # 强制在实数域计算，防止绝对值求导报错
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Symbol('y', real=True)

    def parse_expression(self, formula_str):
        """解析字符串为 SymPy 表达式"""
        try:
            f_str = formula_str.replace('^', '**')
            f_str = re.sub(r'\|([^|]+)\|', r'Abs(\1)', f_str)
            f_str = re.sub(r'\babs\(', 'Abs(', f_str)
            # 🚀 致命修复：参数名必须是 local_dict
            return sp.parse_expr(f_str, local_dict={'abs': sp.Abs})
        except Exception as e:
            raise ValueError(f"公式解析失败: {e}")

    def get_analysis_2d(self, expr):
        deriv = sp.diff(expr, self.x).doit()
        integral = sp.integrate(expr, self.x).doit()
        
        try:
            deriv2 = sp.diff(deriv, self.x).doit()
            # 🚀 致命修复1：绝对不能加 sp.simplify，否则复杂函数会直接卡死你的电脑
            curvature = sp.Abs(deriv2) / ((1 + deriv**2)**1.5)
        except Exception:
            curvature = sp.Integer(0)
            
        return deriv, integral, curvature

    def _eval_2d_safe(self, expr, x_vals):
        # 🚀 致命修复2：拦截无法求出解析式的积分/导数，防止 Numpy 崩溃
        if expr.has(sp.Integral) or expr.has(sp.Derivative):
            return np.full_like(x_vals, np.nan) 
            
        f = sp.lambdify(self.x, expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        try:
            y_raw = f(x_vals)
            if np.isscalar(y_raw):
                y_raw = np.full_like(x_vals, float(y_raw))
                
            # 清理复数 (如 ln(负数) 产生的虚部)
            if np.iscomplexobj(y_raw):
                mask = np.abs(np.imag(y_raw)) > 1e-7
                y_raw = np.real(y_raw)
                y_raw[mask] = np.nan
                
            y_clean = np.array(y_raw, dtype=float)
            
            # 🚀 致命修复3：只切断斜率极其夸张的垂直跳跃（间断点），不再用 NaN 大面积污染数据
            dy = np.abs(np.diff(y_clean))
            dx = np.abs(np.diff(x_vals))
            slopes = np.append(dy / dx, 0)
            
            # 当斜率大于 500 且跳跃跨度大于 1 时，认定为间断跳跃，直接切断
            cut_mask = (slopes > 500) & (np.append(dy, 0) > 1.0)
            y_clean[cut_mask] = np.nan
            
            return y_clean
        except Exception:
            return np.full_like(x_vals, np.nan)

    def generate_2d_plot(self, items):
        fig = go.Figure()
        x_vals = np.linspace(-15, 15, 3000) # 3000 采样率保证曲线平滑
        has_valid_trace = False

        for expr, name, color in items:
            y_clean = self._eval_2d_safe(expr, x_vals)
            if np.all(np.isnan(y_clean)): 
                continue # 如果数据全是空，跳过这个图层

            has_valid_trace = True
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_clean, mode='lines', 
                name=name, line=dict(color=color, width=2.5),
                connectgaps=False # 核心：遇到断点绝对不连线
            ))

        if not has_valid_trace: return None

        fig.update_layout(
            xaxis_title="x", yaxis_title="y", hovermode="x unified",
            # 🚀 用 Plotly 原生功能限制坐标系高度，完美剪裁无穷大渐近线
            yaxis=dict(range=[-20, 20]), 
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    def generate_3d_plot(self, expr):
        if expr.has(sp.Integral) or expr.has(sp.Derivative):
            return None
            
        fig = go.Figure()
        x_vals = np.linspace(-10, 10, 150)
        y_vals = np.linspace(-10, 10, 150)
        X, Y = np.meshgrid(x_vals, y_vals)

        f = sp.lambdify((self.x, self.y), expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        try:
            Z = f(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, float(Z))
                
            if np.iscomplexobj(Z):
                mask = np.abs(np.imag(Z)) > 1e-7
                Z = np.real(Z)
                Z[mask] = np.nan
                
            if np.all(np.isnan(Z)): return None
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, colorscale='Blues', showscale=False
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='f(x, y)',
                    zaxis=dict(range=[-25, 25]) # 🚀 强行切平 3D 无穷大尖刺
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            return fig
        except Exception:
            return None
