import streamlit as st
import re  # 🚀 新增正则库，用于处理绝对值
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 核心初始化与缓存
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

@st.cache_data(show_spinner=False)
def cached_chat_to_formula(_agent, input_str, is_3d):
    return _agent.chat_to_formula(input_str, is_3d=is_3d)

@st.cache_data(show_spinner=False)
def cached_parse_expression(_engine, formula):
    return _engine.parse_expression(formula)

# 🚀 专门对付绝对值的清洗函数
def sanitize_formula(f_str):
    if not f_str: return f_str
    # 1. 把数学上的 |x| 强转为 SymPy 认的 Abs(x)
    # 用正则匹配，解决 |x| + |y| 这种多绝对值情况
    f_str = re.sub(r'\|([^|]+)\|', r'Abs(\1)', f_str)
    # 2. 把小写的 abs() 强转为大写的 Abs()
    f_str = re.sub(r'\babs\(', 'Abs(', f_str)
    return f_str

# 状态初始化
if 'zoom_val' not in st.session_state: st.session_state.zoom_val = 1.0
if 'drag_mode' not in st.session_state: st.session_state.drag_mode = 'turntable'
if 'needs_camera_sync' not in st.session_state: st.session_state.needs_camera_sync = False

st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置
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
    # 默认值顺便也改得更直观一点
    default_val = "Abs(x) + Abs(y)" if is_3d else "Abs(x)" 
    user_input = st.text_input(
        "描述或输入函数:",
        value=default_val,
        help="支持自然语言（如：x的绝对值）或标准公式。"
    )

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面逻辑
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

with st.expander("💡 快速使用指南 (点击展开/收起)", expanded=True):
    st.markdown("""
    * **视角控制**：使用下方的 **[ ➕ ] [ ➖ ]** 缩放，点击 **[ 🔄 ]** 切换旋转/平移。
    * **切换模式**：切换时**视角不会重置**，系统会自动记忆你转好的角度。
    * **支持绝对值**：你可以直接输入 `|x|`，或者用文字描述“x的绝对值”。
    * **一键重置**：如果图像找不到了，点击 **[ 🏠 ]** 恢复初始视角。
    """)

@st.fragment
def render_vis(expr, is_3d):
    # 控制面板
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        if st.button("➕", use_container_width=True):
            st.session_state.zoom_val *= 0.7
            st.session_state.needs_camera_sync = True
            st.rerun()
    with c2:
        if st.button("➖", use_container_width=True):
            st.session_state.zoom_val *= 1.4
            st.session_state.needs_camera_sync = True
            st.rerun()
    with c3:
        if is_3d:
            cur = "旋转" if st.session_state.drag_mode == 'turntable' else "平移"
            if st.button(f"🔄 切换到：{'平移' if cur=='旋转' else '旋转'}", use_container_width=True):
                st.session_state.drag_mode = 'pan' if st.session_state.drag_mode == 'turntable' else 'turntable'
                st.session_state.needs_camera_sync = False
                st.rerun()
        else:
            st.button("📍 2D模式(默认平移)", disabled=True, use_container_width=True)
    with c4:
        if st.button("🏠", use_container_width=True):
            st.session_state.zoom_val = 1.0
            st.session_state.drag_mode = 'turntable' if is_3d else 'pan'
            st.session_state.needs_camera_sync = True
            st.rerun()

    # 绘图逻辑
    config = {'displayModeBar': False, 'scrollZoom': True}
    z = st.session_state.zoom_val

    if is_3d:
        fig = engine.generate_3d_plot(expr)
        if fig:
            layout_dict = {
                "uirevision": "constant", 
                "height": 700,
                "margin": dict(l=0, r=0, b=0, t=0),
                "scene": {"dragmode": st.session_state.drag_mode}
            }
            if st.session_state.needs_camera_sync:
                layout_dict["scene"]["camera"] = dict(eye=dict(x=1.5*z, y=1.5*z, z=1.5*z))
                st.session_state.needs_camera_sync = False

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
    # 1. 拿取原始公式
    raw_formula = agent.chat_to_formula(user_input, is_3d)
    
    if raw_formula:
        # 🚀 2. 核心补丁：在进入数学引擎前，强行清洗绝对值符号
        safe_formula = sanitize_formula(raw_formula)
        
        try:
            expr = cached_parse_expression(engine, safe_formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")
            
            render_vis(expr, is_3d)

            if is_3d:
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")
            else:
                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(engine.get_analysis_2d(expr)[0])} \quad F(x) = {sp.latex(engine.get_analysis_2d(expr)[1])}")
        
        except Exception as e:
            st.error(f"渲染出错，可能是函数不收敛或存在不可导奇点。系统报错信息: {e}")
