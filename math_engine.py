import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re

class MathEngine:
    def __init__(self):
        # 强制定义在实数域
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Symbol('y', real=True)

    def parse_expression(self, formula_str):
        """解析字符串为 SymPy 表达式"""
        try:
            f_str = formula_str.replace('^', '**')
            # 兼容处理绝对值符号 |x| 和 abs(x)
            f_str = re.sub(r'\|([^|]+)\|', r'Abs(\1)', f_str)
            f_str = re.sub(r'\babs\(', 'Abs(', f_str)
            return sp.parse_expr(f_str, locals={'abs': sp.Abs})
        except Exception as e:
            raise ValueError(f"公式解析失败: {e}")

    def get_analysis_2d(self, expr):
        """获取 2D 函数的导数和积分"""
        deriv = sp.diff(expr, self.x).doit()
        integral = sp.integrate(expr, self.x).doit()
        return deriv, integral

    def _clean_discontinuities_2d(self, x_vals, y_vals):
        """
        🚀 终极断线算法：暴力切断所有的渐近线与突变
        """
        y_clean = np.copy(y_vals)
        
        # 1. 拦截暴走数值，防止坐标轴被撑爆
        y_clean[np.abs(y_clean) > 200] = np.nan
        y_clean[np.isinf(y_clean)] = np.nan
        
        # 2. 暴力切断垂直跳变（对付 cot(x) 等）
        for i in range(len(y_clean) - 1):
            y1, y2 = y_clean[i], y_clean[i+1]
            if np.isnan(y1) or np.isnan(y2):
                continue
                
            dy = np.abs(y2 - y1)
            dx = np.abs(x_vals[i+1] - x_vals[i])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = dy / dx
            
            # 突变判定：Y轴跨度大于10且极其陡峭，一律当做渐近线切断
            if dy > 10.0 and slope > 500:
                y_clean[i] = np.nan
                y_clean[i+1] = np.nan
                
        return y_clean

    def _clean_discontinuities_3d(self, z_vals, threshold=25):
        """3D 防破面算法：切除无穷大奇点尖刺"""
        z_clean = np.copy(z_vals)
        z_clean[np.isinf(z_clean)] = np.nan
        z_clean[np.abs(z_clean) > threshold] = np.nan
        return z_clean

    def generate_2d_plot(self, items):
        """生成 2D 图像"""
        fig = go.Figure()
        # 提高分辨率，让切断更精细
        x_vals = np.linspace(-20, 20, 4000) 

        for expr, name, color in items:
            f_lambdified = sp.lambdify(self.x, expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
            
            try:
                y_raw = f_lambdified(x_vals)
                if np.isscalar(y_raw):
                    y_raw = np.full_like(x_vals, y_raw)
                    
                # 🚀 致命修复：强制复数拦截！
                # 把求出的值转为复数数组，一旦发现有虚部，说明超出了实数定义域，强行注入 NaN
                y_cplx = np.array(y_raw, dtype=complex)
                with np.errstate(invalid='ignore'):
                    mask_invalid = np.abs(np.imag(y_cplx)) > 1e-7
                    
                y_clean = np.real(y_cplx)
                y_clean[mask_invalid] = np.nan
                    
            except Exception:
                continue 
            
            # 再过一遍断线逻辑
            y_clean = self._clean_discontinuities_2d(x_vals, y_clean)

            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=y_clean, 
                mode='lines', 
                name=name,
                line=dict(color=color, width=2.5)
            ))

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
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

        f_lambdified = sp.lambdify((self.x, self.y), expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        
        try:
            Z_raw = f_lambdified(X, Y)
            if np.isscalar(Z_raw):
                Z_raw = np.full_like(X, Z_raw)
                
            # 🚀 3D 同理拦截复数，解决 x^(-2/3) 的负半轴问题
            Z_cplx = np.array(Z_raw, dtype=complex)
            with np.errstate(invalid='ignore'):
                mask_invalid = np.abs(np.imag(Z_cplx)) > 1e-7
                
            Z_clean = np.real(Z_cplx)
            Z_clean[mask_invalid] = np.nan
                
        except Exception:
            return None

        Z_final = self._clean_discontinuities_3d(Z_clean)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_final,
            colorscale='Blues',
            showscale=False
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='f(x, y)'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        return fig
