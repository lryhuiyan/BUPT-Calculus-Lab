import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 核心初始化
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

# 缓存 AI 翻译
@st.cache_data(show_spinner=False)
def cached_chat_to_formula(_agent, input_str, is_3d):
    return _agent.chat_to_formula(input_str, is_3d=is_3d)

# 初始化视图状态
if 'zoom_level' not in st.session_state: st.session_state.zoom_level = 1.0
if 'drag_mode' not in st.session_state: st.session_state.drag_mode = 'turntable'

# 还原原始项目名称
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")
    if st.button("🔄 物理刷新"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.zoom_level = 1.0
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    st.markdown("### ✍️ 函数输入")
    user_input = st.text_input("描述或输入函数:", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面：局部刷新片段
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# ✅ 针对用户的 Tips
with st.expander("💡 快速使用指南", expanded=True):
    st.markdown("""
    * **缩放控制**：点击 **[ ➕ ]** 放大，**[ ➖ ]** 缩小。
    * **视角锁定**：切换旋转/平移模式时，**画面位置绝对不会复位**。
    * **一键归位**：如果图像找不到了，点击 **[ 🏠 ]** 恢复初始状态。
    """)

@st.fragment
def render_visualization(expr, is_3d):
    # --- 🎮 自定义控制面板 ---
    st.markdown("##### 🕹️ 视图交互控制")
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    
    with c1:
        if st.button("➕", use_container_width=True):
            st.session_state.zoom_level *= 0.8  # 减小系数 = 放大
            st.rerun()
    with c2:
        if st.button("➖", use_container_width=True):
            st.session_state.zoom_level *= 1.25 # 增大系数 = 缩小
            st.rerun()
    with c3:
        if is_3d:
            current_label = "旋转" if st.session_state.drag_mode == 'turntable' else "平移"
            if st.button(f"🔄 切换到：{'平移' if current_label=='旋转' else '旋转'}", use_container_width=True):
                st.session_state.drag_mode = 'pan' if st.session_state.drag_mode == 'turntable' else 'turntable'
                st.rerun()
        else:
            st.button("📍 2D默认平移", disabled=True, use_container_width=True)
    with c4:
        if st.button("🏠", use_container_width=True):
            st.session_state.zoom_level = 1.0
            st.session_state.drag_mode = 'turntable' if is_3d else 'pan'
            st.rerun()

    # --- 🎨 绘图核心逻辑 ---
    config = {'displayModeBar': False, 'scrollZoom': True}
    z = st.session_state.zoom_level

    if is_3d:
        fig = engine.generate_3d_plot(expr)
        if fig:
            fig.update_layout(
                # 🚀 核心 1：uirevision 设为固定值，切换 dragmode 时绝对不重置视角
                uirevision='keep_the_damn_view',
                height=700, margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    dragmode=st.session_state.drag_mode,
                    # 🚀 核心 2：根据 zoom_level 动态计算相机位置
                    camera=dict(eye=dict(x=1.5*z, y=1.5*z, z=1.5*z))
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=config, key="stable_3d")
    else:
        deriv, integral = engine.get_analysis_2d(expr)
        items = []
        if show_f: items.append((expr, "f(x)", "#1f77b4"))
        if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
        if show_integral: items.append((integral, "F(x)", "#ff7f0e"))
        fig = engine.generate_2d_plot(items)
        if fig:
            fig.update_layout(
                uirevision='keep_the_damn_view',
                dragmode='pan',
                height=550,
                # 🚀 2D 缩放逻辑
                xaxis=dict(range=[-10*z, 10*z]),
                yaxis=dict(range=[-10*z, 15*z])
            )
            st.plotly_chart(fig, use_container_width=True, config=config, key="stable_2d")

# --- 主逻辑执行 ---
if user_input:
    formula = cached_chat_to_formula(agent, user_input, is_3d)
    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")
            
            # 执行可视化
            render_visualization(expr, is_3d)

            # 解析报告
            if is_3d:
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")
            else:
                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
