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
            # 将绝对值转换为 SymPy 标准格式
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
        🚀 核心黑科技：智能间断点切断算法
        用于防止 Plotly 把分段函数（如 sign(x)）或渐近线（如 1/x）连起来。
        """
        y_clean = np.array(y_vals, dtype=float)
        
        # 1. 过滤掉本身就是无穷大的点
        y_clean[np.isinf(y_clean)] = np.nan
        
        # 计算相邻点的差值(dy)和坐标跨度(dx)
        dy = np.diff(y_clean)
        dx = np.diff(x_vals)
        
        # 2. 遍历寻找“阶跃”突变点
        for i in range(1, len(dy) - 1):
            # 计算当前点和前后的局部斜率
            slope_current = np.abs(dy[i] / dx[i])
            slope_prev = np.abs(dy[i-1] / dx[i-1])
            slope_next = np.abs(dy[i+1] / dx[i+1])
            
            # 判定逻辑：
            # (1) 绝对跳跃量 > 0.5 (过滤掉微小的数值抖动)
            # (2) 局部斜率 > 50 (说明线非常陡峭)
            if np.abs(dy[i]) > 0.5 and slope_current > 50:
                # (3) 核心防误伤逻辑：只有当当前斜率远大于前后斜率时（比如大于 5 倍），
                # 才认为是间断跳跃。这样不会误伤 x^3 或 e^x 这种本身就连续且陡峭的函数。
                if slope_current > 5 * slope_prev or slope_current > 5 * slope_next:
                    y_clean[i] = np.nan    # 插入 NaN，Plotly 遇到 NaN 会自动断开连线
                    y_clean[i+1] = np.nan
                    
        return y_clean

    def _clean_discontinuities_3d(self, z_vals, threshold=100):
        """3D 渐近线切断：防止曲面拉出极长的尖刺"""
        z_clean = np.array(z_vals, dtype=float)
        z_clean[np.isinf(z_clean)] = np.nan
        # 超过正常显示范围的极端值直接切断，防止曲面破面
        z_clean[np.abs(z_clean) > threshold] = np.nan
        return z_clean

    def generate_2d_plot(self, items):
        """生成 2D 图像"""
        fig = go.Figure()
        # 提高采样率，让断点的边缘更精准
        x_vals = np.linspace(-10, 10, 2000) 

        for expr, name, color in items:
            # 传入 numpy 的 sign 和 abs 函数，供 lambdify 使用
            f_lambdified = sp.lambdify(self.x, expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
            
            try:
                y_vals = f_lambdified(x_vals)
                if np.isscalar(y_vals):
                    y_vals = np.full_like(x_vals, y_vals)
            except Exception:
                continue 
            
            # 🚀 在丢给 Plotly 画图前，先经过间断点清洗
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
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    def generate_3d_plot(self, expr):
        """生成 3D 图像"""
        fig = go.Figure()
        x_vals = np.linspace(-10, 10, 100)
        y_vals = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        f_lambdified = sp.lambdify((self.x, self.y), expr, modules=['numpy', {'sign': np.sign, 'Abs': np.abs}])
        
        try:
            Z = f_lambdified(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
        except Exception:
            return None

        # 🚀 3D 极值清洗
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
