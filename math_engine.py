import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
from functools import lru_cache

# ==========================================
# 🚀 性能加速层：全局编译缓存池
# 利用 lru_cache 避免对重复输入的公式进行耗时的解析与求导
# ==========================================
@lru_cache(maxsize=32)
def cached_parse(expr_str):
    """将字符串安全解析为 SymPy 的 AST (抽象语法树)"""
    # 清洗大模型可能残留的 Python 库前缀
    clean_str = re.sub(r'(math|np|numpy|sp|sympy)\.', '', expr_str)
    clean_str = clean_str.replace('\\', '').replace('cdot', '*')
    
    # 允许隐式乘法，如将 '2x' 解析为 '2*x'
    transformations = standard_transformations + (implicit_multiplication_application,)
    x, y = sp.symbols('x y', real=True)
    try:
        return parse_expr(clean_str, transformations=transformations, global_dict={'x': x, 'y': y, 'Abs': sp.Abs, **sp.__dict__})
    except:
        return sp.sympify(clean_str)

@lru_cache(maxsize=32)
def cached_compile_2d(expr):
    """缓存一元函数的符号求导树，解决实时输入时的严重卡顿问题"""
    x = sp.symbols('x', real=True)
    f_p = sp.diff(expr, x).doit()      # 一阶导
    f_pp = sp.diff(f_p, x).doit()      # 二阶导
    return f_p, f_pp

class MathEngine:
    """核心数学与物理渲染引擎"""
    def __init__(self):
        # 强制声明为实数域，从底层切断 SymPy 产生无关复数解的可能
        self.x, self.y = sp.symbols('x y', real=True)

    def parse_expression(self, expr_str):
        return cached_parse(expr_str)

    def _fix_real_roots(self, expr):
        """
        [黑科技] 符号层复数根拦截器：
        解决像 x**(2/3) 在 x<0 时产生 0+0j 导致图像丢失的问题。
        通过符号替换，将偶分子/奇分母的分数次幂强制转化为带绝对值和符号函数的实数映射。
        """
        if hasattr(expr, 'replace'):
            # 处理对数函数，自适应 1 个参数(ln)或 2 个参数(带底数)，给真数套绝对值
            expr = expr.replace(sp.log, lambda *args: sp.log(sp.Abs(args[0]), *args[1:]))
            
            # 处理幂函数
            for p in expr.atoms(sp.Pow):
                base, exp = p.as_base_exp()
                if exp.is_Rational and exp.q % 2 != 0: # 如果分母是奇数
                    if exp.p % 2 == 0:                 # 分子是偶数 (如 2/3) -> 恒正
                        expr = expr.subs(p, sp.Abs(base)**exp)
                    else:                              # 分子是奇数 (如 1/3) -> 保留底数符号
                        expr = expr.subs(p, sp.sign(base) * sp.Abs(base)**exp)
        return expr

    def _broadcast_scalar(self, val, target_array):
        """标量广播补丁：防止常数函数(如 y=5 或 y'=0)运算后变成0维标量导致报错"""
        if np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0):
            return np.full_like(target_array, float(val))
        return val

    def generate_2d_plot(self, expr_list):
        """生成 2D 交互式图像"""
        fig = go.Figure()
        
        # 混合采样：1000个常规点 + 原点附近200个纳米级探针点（精准捕捉趋于无穷的趋势）
        x_main = np.linspace(-20, 20, 1001)
        x_micro = np.linspace(-1e-4, 1e-4, 201) 
        x_vals = np.sort(np.unique(np.concatenate([x_main, x_micro])))
        
        for expr, label, color in expr_list:
            # 1. 渲染主函数 y = f(x)
            try:
                fixed_expr = self._fix_real_roots(expr)
                
                # 常数特判
                if fixed_expr.is_constant():
                    f_np = lambda x: float(fixed_expr)
                else:
                    f_np = sp.lambdify(self.x, fixed_expr, 'numpy')
                
                with np.errstate(all='ignore'): # 忽略被除数为0等警告
                    y_p = f_np(x_vals)
                
                y_p = self._broadcast_scalar(y_p, x_vals)
                # 智能物理断路：如果算出无穷大(inf)，将其转为 NaN 以切断连线伪影
                y_p[~np.isfinite(y_p)] = np.nan 
            except Exception:
                continue # 主图层彻底崩溃时放弃该图层
            
            # 2. 计算实时曲率 κ = |y''| / (1 + y'^2)^1.5
            try:
                f_p, f_pp = cached_compile_2d(expr)
                
                fixed_fp = self._fix_real_roots(f_p)
                fixed_fpp = self._fix_real_roots(f_pp)
                
                # 将一阶和二阶导转化为 Numpy 数组
                yp_vals = self._broadcast_scalar(
                    float(fixed_fp) if fixed_fp.is_constant() else sp.lambdify(self.x, fixed_fp, 'numpy')(x_vals), 
                    x_vals
                )
                ypp_vals = self._broadcast_scalar(
                    float(fixed_fpp) if fixed_fpp.is_constant() else sp.lambdify(self.x, fixed_fpp, 'numpy')(x_vals), 
                    x_vals
                )
                
                with np.errstate(all='ignore'):
                    # 在 Numpy 级别拼接曲率，规避 SymPy 树爆炸引发的死锁
                    k_vals = np.abs(ypp_vals) / (1 + yp_vals**2)**1.5
                    k_vals[~np.isfinite(k_vals)] = 0 # 处理除以 0 的情况
            except Exception:
                k_vals = np.zeros_like(x_vals)

            # 3. 将计算好的数据推入 Plotly 图层
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_p, mode='lines', name=label, 
                line=dict(color=color, width=2.5),
                customdata=k_vals, # 将曲率数据绑定到图层，供悬停显示
                hovertemplate="<b>%{name}</b><br>x: %{x:.4f}<br>y: %{y:.4f}<br>κ: %{customdata:.4f}<extra></extra>",
                connectgaps=False # 遇到 NaN 必须断开画笔
            ))
        
        # 锁定白底背景、坐标范围和统一悬停交互
        fig.update_layout(
            template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
            xaxis=dict(range=[-20, 20], zeroline=True, gridcolor='#f0f0f0'),
            yaxis=dict(range=[-10, 25], zeroline=True, gridcolor='#f0f0f0'), # y轴拉高，保留"冲天"感
            dragmode='pan', hovermode="x unified"
        )
        return fig

    def generate_3d_plot(self, expr):
        """生成 3D 交互式曲面"""
        # 混合采样：保留中心网格密度以捕捉十字撕裂带，同时控制总体积防止 WebSocket 断流
        t_main = np.linspace(-20, 20, 81)
        t_micro = np.linspace(-0.1, 0.1, 21)
        t = np.sort(np.unique(np.concatenate([t_main, t_micro])))
        X, Y = np.meshgrid(t, t)
        
        fixed_expr = self._fix_real_roots(expr)
        
        if fixed_expr.is_constant():
            f_np = lambda x, y: float(fixed_expr)
        else:
            f_np = sp.lambdify((self.x, self.y), fixed_expr, 'numpy')
        
        try:
            with np.errstate(all='ignore'):
                Z = f_np(X, Y)
                Z = self._broadcast_scalar(Z, X)
                # 3D 采用纯数值梯度计算偏导数，极速且不易崩溃
                GY, GX = np.gradient(Z, t, t)

            # 放宽数据拦截线，允许 Z 算到 ±800，保留向天花板冲刺的数据支撑
            Z_plot = np.where(np.abs(Z) < 800, Z, np.nan)

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z_plot,
                customdata=np.stack((GX, GY), axis=-1),
                colorscale=[[0, 'rgb(180, 220, 255)'], [1, 'rgb(50, 130, 210)']],
                opacity=0.85,
                cmin=-15, cmax=25, # 匹配新的天花板视口
                contours=dict(
                    z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="white"),
                    # 深色高对比度网格，凸显曲面形变张力
                    x=dict(show=True, color="rgba(0,0,0,0.6)", width=1.5),
                    y=dict(show=True, color="rgba(0,0,0,0.6)", width=1.5)
                ),
                hovertemplate="<b>X</b>: %{x:.2f} <b>Y</b>: %{y:.2f} <b>Z</b>: %{z:.2f}<br><b>梯度</b>: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>"
            )])
            
            # 锁定正方体空间，逼迫渲染引擎产生物理截断感
            fig.update_layout(
                paper_bgcolor='white',
                scene=dict(
                    xaxis=dict(range=[-20, 20]),
                    yaxis=dict(range=[-20, 20]),
                    zaxis=dict(range=[-10, 25]), # 锁定 25 视口，实现冲向云端的视觉
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            return fig
        except: return None

    def get_analysis_2d(self, expr):
        """符号层解析几何：原函数的符号一阶导数与不定积分"""
        return sp.diff(expr, self.x).doit(), sp.integrate(expr, self.x).doit()
