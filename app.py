import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 核心初始化（保持你的所有原始设置）
# ==========================================
DEFAULT_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"

def get_api_key():
    try:
        if "DEEPSEEK_API_KEY" in st.secrets: return st.secrets["DEEPSEEK_API_KEY"]
    except: pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 缓存数学解析，提速响应
@st.cache_data(show_spinner=False)
def cached_parse_expression(_engine, formula):
    return _engine.parse_expression(formula)

# 维护交互状态
if 'zoom_val' not in st.session_state: st.session_state.zoom_val = 1.0
if 'drag_mode' not in st.session_state: st.session_state.drag_mode = 'turntable'
if 'needs_camera_sync' not in st.session_state: st.session_state.needs_camera_sync = False

# ⚡ 严格还原你的原始命名 ⚡
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置（还原你的文字）
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")

    if st.button("🔄 物理刷新 (清除异常缓存)"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.zoom_val = 1.0
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")

    st.markdown("### ✍️ 函数输入")
    default_val = "x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)"
    user_input = st.text_input(
        "描述或输入函数:",
        value=default_val
    )

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面：图像与控制区
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# 还原你的提示
with st.expander("💡 快速使用指南 (点击展开/收起)", expanded=True):
    st.markdown("""
    * **视角控制**：使用下方的 **[ ➕ ] [ ➖ ]** 缩放，点击 **[ 🔄 ]** 切换旋转/平移。
    * **切换模式**：切换时**视角不会重置**，系统会自动记忆你转好的角度。
    * **一键重置**：如果图像找不到了，点击 **[ 🏠 ]** 恢复初始视角。
    """)

@st.fragment
def render_vis(expr, is_3d):
    # --- 🕹️ 自制控制键：加号、减号、切换、复位 ---
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        if st.button("➕", use_container_width=True):
            st.session_state.zoom_val *= 0.7
            st.session_state.needs_camera_sync = True # 标记：需要动相机
            st.rerun()
    with c2:
        if st.button("➖", use_container_width=True):
            st.session_state.zoom_val *= 1.4
            st.session_state.needs_camera_sync = True # 标记：需要动相机
            st.rerun()
    with c3:
        if is_3d:
            cur = "旋转" if st.session_state.drag_mode == 'turntable' else "平移"
            if st.button(f"🔄 切换到：{'平移' if cur=='旋转' else '旋转'}", use_container_width=True):
                st.session_state.drag_mode = 'pan' if st.session_state.drag_mode == 'turntable' else 'turntable'
                st.session_state.needs_camera_sync = False # 🚀 核心：切换模式不准动相机
                st.rerun()
        else:
            st.button("📍 2D模式(默认平移)", disabled=True, use_container_width=True)
    with c4:
        if st.button("🏠", use_container_width=True):
            st.session_state.zoom_val = 1.0
            st.session_state.drag_mode = 'turntable' if is_3d else 'pan'
            st.session_state.needs_camera_sync = True
            st.rerun()

    # --- 🎨 绘图核心 ---
    config = {'displayModeBar': False, 'scrollZoom': True}
    z = st.session_state.zoom_val

    if is_3d:
        fig = engine.generate_3d_plot(expr)
        if fig:
            # 基础 Layout
            layout_dict = {
                "uirevision": "constant", # 锁定视角的核心
                "height": 700,
                "margin": dict(l=0, r=0, b=0, t=0),
                "scene": {"dragmode": st.session_state.drag_mode}
            }
            # 🚀 只有当点击加减号时，才注入相机参数，否则不写，Plotly 就会保留原有位置
            if st.session_state.needs_camera_sync:
                layout_dict["scene"]["camera"] = dict(eye=dict(x=1.5*z, y=1.5*z, z=1.5*z))
                st.session_state.needs_camera_sync = False # 用完即止

            fig.update_layout(**layout_dict)
            st.plotly_chart(fig, use_container_width=True, config=config, key="plot3d")
    else:
        deriv, integral = engine.get_analysis_2d(expr)
        items = [(expr, "f(x)", "#1f77b4"), (deriv, "f'(x)", "#d62728"), (integral, "F(x)", "#ff7f0e")]
        fig = engine.generate_2d_plot(items)
        if fig:
            layout_2d = {"uirevision": "constant", "dragmode": "pan", "height": 550}
            if st.session_state.needs_camera_sync:
                layout_2d["xaxis"] = dict(range=[-10*z, 10*z])
                layout_2d["yaxis"] = dict(range=[-10*z, 15*z])
                st.session_state.needs_camera_sync = False
            fig.update_layout(**layout_2d)
            st.plotly_chart(fig, use_container_width=True, config=config, key="plot2d")

# --- 主执行区 ---
if user_input:
    formula = agent.chat_to_formula(user_input, is_3d)
    if formula:
        try:
            expr = cached_parse_expression(engine, formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")
            render_vis(expr, is_3d) # 调用局部刷新

            if is_3d:
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")
            else:
                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(engine.get_analysis_2d(expr)[0])} \quad F(x) = {sp.latex(engine.get_analysis_2d(expr)[1])}")
        except Exception as e:
            st.error(f"渲染出错: {e}")
