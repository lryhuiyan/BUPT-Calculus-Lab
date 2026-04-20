import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re

class MathEngine:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')

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
        🚀 2D 终极断线算法：基于中位数斜率的异常值检测
        专门对付绝对值导数的垂直红线，以及 1/x 的垂直渐近线。
        """
        y_clean = np.array(y_vals, dtype=float)
        
        # 1. 强行截断超出合理绘图范围的极端值（防止撑爆坐标轴）
        y_clean[np.abs(y_clean) > 50] = np.nan
        y_clean[np.isinf(y_clean)] = np.nan
        
        # 2. 计算斜率
        dx = np.diff(x_vals)
        dy = np.diff(y_clean)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.abs(dy / dx)
            
        # 获取整条曲线的基准斜率
        median_slope = np.nanmedian(slopes)
        if np.isnan(median_slope): median_slope = 0
        
        # 3. 寻找阶跃点：局部斜率极大，且远远超出整条曲线的平均水平
        for i in range(len(slopes)):
            # 条件：斜率 > 50，且达到中位数的 10 倍以上，且实际 y 跳跃明显
            if slopes[i] > 50 and slopes[i] > 10 * median_slope and np.abs(dy[i]) > 0.5:
                # 在阶跃点强制注入 NaN，Plotly 遇到 NaN 会直接停止连线
                y_clean[i] = np.nan
                y_clean[i+1] = np.nan
                
        return y_clean

    def _clean_discontinuities_3d(self, z_vals, threshold=25):
        """
        🚀 3D 防破面算法：切除奇点尖刺
        """
        z_clean = np.array(z_vals, dtype=float)
        z_clean[np.isinf(z_clean)] = np.nan
        # 你的 3D 图像 Z 轴大概到 25，超过这个值的数据直接挖空，不要拉尖刺
        z_clean[np.abs(z_clean) > threshold] = np.nan
        return z_clean

    def generate_2d_plot(self, items):
        """生成 2D 图像"""
        fig = go.Figure()
        # 采样点增加到 2000，让阶跃点的捕捉更精准
        x_vals = np.linspace(-10, 10, 2000) 

        for expr, name, color in items:
            f_lambdified = sp.lambdify(self.x, expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
            
            try:
                y_vals = f_lambdified(x_vals)
                if np.isscalar(y_vals):
                    y_vals = np.full_like(x_vals, y_vals)
            except Exception:
                continue 
            
            # 🔪 在喂给 Plotly 之前，强行切断间断点
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
        x_vals = np.linspace(-10, 10, 150) # 稍微提高 3D 精度
        y_vals = np.linspace(-10, 10, 150)
        X, Y = np.meshgrid(x_vals, y_vals)

        f_lambdified = sp.lambdify((self.x, self.y), expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        
        try:
            Z = f_lambdified(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
        except Exception:
            return None

        # 🔪 切除 3D 无穷大尖刺
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
