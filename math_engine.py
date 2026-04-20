import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re

class MathEngine:
    def __init__(self):
        # 必须声明 real=True，确保引擎在实数域内解析，避免绝对值求导崩溃
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Symbol('y', real=True)

    def parse_expression(self, formula_str):
        """解析字符串为 SymPy 表达式"""
        try:
            f_str = formula_str.replace('^', '**')
            f_str = re.sub(r'\|([^|]+)\|', r'Abs(\1)', f_str)
            f_str = re.sub(r'\babs\(', 'Abs(', f_str)
            return sp.parse_expr(f_str, locals={'abs': sp.Abs})
        except Exception as e:
            raise ValueError(f"公式解析失败: {e}")

    def get_analysis_2d(self, expr):
        """获取 2D 函数的导数、积分和精确曲率"""
        deriv = sp.diff(expr, self.x).doit()
        integral = sp.integrate(expr, self.x).doit()
        
        # 🚀 纯符号推导曲率：|y''| / (1 + y'^2)^(3/2)
        try:
            deriv2 = sp.diff(deriv, self.x).doit()
            curvature = sp.Abs(deriv2) / (1 + deriv**2)**sp.Rational(3, 2)
            curvature = sp.simplify(curvature)
        except Exception:
            curvature = sp.Integer(0)
            
        return deriv, integral, curvature

    def _eval_and_clean_2d(self, expr, x_vals):
        """🚀 统一的可视化数据清洗中心"""
        f = sp.lambdify(self.x, expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        
        try:
            y_raw = f(x_vals)
            if np.isscalar(y_raw):
                y_raw = np.full_like(x_vals, float(y_raw))
                
            # 1. 拦截复数 (解决 ln(sin(x)) 连线问题)
            y_cplx = np.array(y_raw, dtype=complex)
            with np.errstate(invalid='ignore'):
                mask_invalid = np.abs(np.imag(y_cplx)) > 1e-7
            y_clean = np.real(y_cplx)
            y_clean[mask_invalid] = np.nan
            
            # 2. 拦截撑爆坐标轴的极值
            y_clean[np.abs(y_clean) > 200] = np.nan
            y_clean[np.isinf(y_clean)] = np.nan
            
            # 3. 切断垂直突变 (解决绝对值导数、cot(x) 连线问题)
            dy = np.diff(y_clean)
            dx = np.diff(x_vals)
            with np.errstate(divide='ignore', invalid='ignore'):
                slopes = np.abs(dy / dx)
                
            for i in range(len(slopes)):
                # 如果垂直跳跃高度大于 0.5 且斜率极大，认定为间断点
                if slopes[i] > 300 and np.abs(dy[i]) > 0.5:
                    y_clean[i] = np.nan
                    y_clean[i+1] = np.nan
                    
            return y_clean
        except Exception:
            return np.full_like(x_vals, np.nan)

    def _eval_and_clean_3d(self, expr, X, Y):
        """3D 极值与虚数清洗"""
        f = sp.lambdify((self.x, self.y), expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        try:
            Z_raw = f(X, Y)
            if np.isscalar(Z_raw):
                Z_raw = np.full_like(X, float(Z_raw))
                
            Z_cplx = np.array(Z_raw, dtype=complex)
            with np.errstate(invalid='ignore'):
                mask_invalid = np.abs(np.imag(Z_cplx)) > 1e-7
            Z_clean = np.real(Z_cplx)
            Z_clean[mask_invalid] = np.nan
            
            Z_clean[np.isinf(Z_clean)] = np.nan
            Z_clean[np.abs(Z_clean) > 50] = np.nan # 防破面
            return Z_clean
        except Exception:
            return np.full_like(X, np.nan)

    def generate_2d_plot(self, items):
        """生成 2D 图像"""
        fig = go.Figure()
        x_vals = np.linspace(-15, 15, 3000) # 高精度采样

        for expr, name, color in items:
            y_clean = self._eval_and_clean_2d(expr, x_vals)
            
            # 屏蔽全是 NaN 的空图层
            if np.all(np.isnan(y_clean)): continue

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_clean, mode='lines', 
                name=name, line=dict(color=color, width=2.5)
            ))

        fig.update_layout(
            xaxis_title="x", yaxis_title="y",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    def generate_3d_plot(self, expr):
        """生成 3D 图像"""
        fig = go.Figure()
        x_vals = np.linspace(-10, 10, 150)
        y_vals = np.linspace(-10, 10, 150)
        X, Y = np.meshgrid(x_vals, y_vals)

        Z_clean = self._eval_and_clean_3d(expr, X, Y)
        
        if np.all(np.isnan(Z_clean)): return None

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_clean,
            colorscale='Blues', showscale=False
        ))

        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(x, y)'),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        return fig
