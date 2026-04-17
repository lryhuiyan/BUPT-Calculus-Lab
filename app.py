import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 核心初始化与状态管理
# ==========================================
DEFAULT_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"

def get_api_key():
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except: pass
    return DEFAULT_KEY

# 初始化缩放状态
if 'zoom_val' not in st.session_state:
    st.session_state.zoom_val = 1.0

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

st.set_page_config(page_title="微积分绘图实验室", layout="wide")

# ==========================================
# 👈 侧边栏：按键控制中心
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    # 模式切换
    mode = st.radio("维度选择", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    st.markdown("---")
    
    # 🚀 核心改动：缩放按键控制
    st.subheader("🔍 视图缩放")
    col_in, col_out, col_reset = st.columns(3)
    with col_in:
        if st.button("➕", help="放大"):
            st.session_state.zoom_val *= 0.8  # 范围缩小 = 视角放大
    with col_out:
        if st.button("➖", help="缩小"):
            st.session_state.zoom_val *= 1.25 # 范围扩大 = 视角缩小
    with col_reset:
        if st.button("🏠", help="重置"):
            st.session_state.zoom_val = 1.0

    st.markdown("---")
    user_input = st.text_input("输入函数内容", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        show_f = st.checkbox("原函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("不定积分 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 微积分绘图实验室")
st.caption("📱 操作指南：使用侧边栏的 [+] [-] 按钮控制缩放，单指滑动图像进行平移或旋转。")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            # ✅ 禁用手势缩放，由按键统一控制
            config = {
                'scrollZoom': False,       # 彻底关闭滚轮和捏合缩放
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'modeBarButtonsToRemove': ['autoScale2d', 'autoscale', 'zoom2d', 'zoom3d']
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    # 🚀 通过控制 eye 参数实现 3D 按键缩放
                    z_factor = st.session_state.zoom_val
                    fig.update_layout(
                        scene=dict(
                            dragmode='orbit',
                            camera=dict(
                                eye=dict(x=1.25*z_factor, y=1.25*z_factor, z=1.25*z_factor)
                            ),
                            # 保持坐标轴范围固定，只动相机
                            xaxis=dict(range=[-20, 20]),
                            yaxis=dict(range=[-20, 20]),
                            zaxis=dict(range=[-10, 25])
                        ),
                        height=600, margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)
                
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")

            else:
                deriv, integral = engine.get_analysis_2d(expr)
                items = []
                if show_f: items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))

                fig = engine.generate_2d_plot(items)
                if fig:
                    # 🚀 通过修改 range 实现 2D 按键缩放
                    z_factor = st.session_state.zoom_val
                    fig.update_layout(
                        dragmode='pan', 
                        height=500,
                        xaxis=dict(range=[-20*z_factor, 20*z_factor]),
                        yaxis=dict(range=[-10*z_factor, 25*z_factor])
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                col1, col2 = st.columns(2)
                with col1: st.latex(rf"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(rf"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析出错: {e}")
