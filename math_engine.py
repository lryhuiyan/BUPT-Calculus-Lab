import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re

class MathEngine:
    def __init__(self):
        # 🚀 致命修复 1：必须声明 real=True，强迫引擎在实数域内解析绝对值
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
        🚀 终极断线算法：强制切断 sign(x) 在 0 处的红线连线
        """
        y_clean = np.array(y_vals, dtype=float)
        
        # 强行截断超出合理绘图范围的极端值
        y_clean[np.abs(y_clean) > 50] = np.nan
        y_clean[np.isinf(y_clean)] = np.nan
        
        dx = np.diff(x_vals)
        dy = np.diff(y_clean)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.abs(dy / dx)
            
        median_slope = np.nanmedian(slopes)
        if np.isnan(median_slope): median_slope = 0
        
        # 寻找阶跃点：针对 sign(0)=0 导致的两步跳跃，降低 dy 阈值进行精准切割
        for i in range(len(slopes)):
            if slopes[i] > 40 and np.abs(dy[i]) > 0.1:
                if median_slope == 0 or slopes[i] > 10 * median_slope:
                    y_clean[i] = np.nan
                    y_clean[i+1] = np.nan
                
        return y_clean

    def _clean_discontinuities_3d(self, z_vals, threshold=25):
        """3D 防破面算法：切除无穷大奇点尖刺"""
        z_clean = np.array(z_vals, dtype=float)
        z_clean[np.isinf(z_clean)] = np.nan
        z_clean[np.abs(z_clean) > threshold] = np.nan
        return z_clean

    def generate_2d_plot(self, items):
        """生成 2D 图像"""
        fig = go.Figure()
        x_vals = np.linspace(-10, 10, 2000) 

        for expr, name, color in items:
            f_lambdified = sp.lambdify(self.x, expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
            
            try:
                y_vals = f_lambdified(x_vals)
                if np.isscalar(y_vals):
                    y_vals = np.full_like(x_vals, y_vals)
                    
                # 🚀 致命修复 2：如果产生虚数，强行提取实部并把纯虚数置空
                if np.iscomplexobj(y_vals):
                    y_vals[np.iscomplex(y_vals)] = np.nan
                    y_vals = np.real(y_vals)
                    
            except Exception:
                continue 
            
            y_clean = self._clean_discontinuities_2d(x_vals, y_vals)

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
            Z = f_lambdified(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
                
            # 🚀 致命修复 3：3D 模式过滤虚数点（解决 x**(-2/3) 负数轴不显示的问题）
            if np.iscomplexobj(Z):
                Z[np.iscomplex(Z)] = np.nan
                Z = np.real(Z)
                
        except Exception:
            return None

        Z_clean = self._clean_discontinuities_3d(Z)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_clean,
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
