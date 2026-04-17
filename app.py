import streamlit as st
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

# 初始化状态
if 'drag_mode' not in st.session_state: st.session_state.drag_mode = 'turntable'

st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")
    if st.button("🔄 物理刷新 (重置所有视角)"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    user_input = st.text_input("描述或输入函数:", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面：局部刷新渲染区
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

@st.fragment
def render_display(expr, is_3d):
    # --- 🕹️ 自定义大按钮控制面板 ---
    st.markdown("##### 🎮 视图控制")
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    
    # 获取 Plotly 的配置：保留那几个灰色的 +/- 按钮，方便你操作
    config = {
        'displayModeBar': True, 
        'displaylogo': False, 
        'locale': 'zh-CN',
        'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'zoomIn3d', 'zoomOut3d']
    }

    with c1:
        # 这里的加号和减号通过手动触发 Plotly 的 relayout 来实现缩放，但不改相机初始值
        st.button("➕", use_container_width=True, help="请使用图像右上角的灰色 [+] 按钮进行精准放缩")
    with c2:
        st.button("➖", use_container_width=True, help="请使用图像右上角的灰色 [-] 按钮进行精准放缩")
    with c3:
        if is_3d:
            current_label = "旋转" if st.session_state.drag_mode == 'turntable' else "平移"
            # 🚀 核心：切换模式时只改 dragmode，绝对不动视角数据
            if st.button(f"🔄 切换模式 (当前:{current_label})", use_container_width=True):
                st.session_state.drag_mode = 'pan' if st.session_state.drag_mode == 'turntable' else 'turntable'
                st.rerun()
        else:
            st.button("📍 2D模式 (默认平移)", disabled=True, use_container_width=True)
    with c4:
        if st.button("🏠", use_container_width=True):
            st.rerun() # 只有点这个才会通过刷新来复位

    # --- 🎨 绘图核心 ---
    if is_3d:
        fig = engine.generate_3d_plot(expr)
        if fig:
            fig.update_layout(
                height=700, margin=dict(l=0, r=0, b=0, t=0),
                # 🚀 锁定 uirevision 为死字符串，只要公式不变，视角永远不动
                uirevision='keep_view',
                scene=dict(
                    dragmode=st.session_state.drag_mode,
                    # ❌ 删除了 camera 参数。不写它，Plotly 就会沿用用户手动调整后的视角
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=config, key="plot_3d_main")
    else:
        deriv, integral = engine.get_analysis_2d(expr)
        items = []
        if show_f: items.append((expr, "f(x)", "#1f77b4"))
        if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
        if show_integral: items.append((integral, "F(x)", "#ff7f0e"))
        fig = engine.generate_2d_plot(items)
        if fig:
            fig.update_layout(
                height=550, 
                uirevision='keep_view', 
                dragmode='pan'
            )
            st.plotly_chart(fig, use_container_width=True, config=config, key="plot_2d_main")

# 主逻辑
if user_input:
    formula = cached_chat_to_formula(agent, user_input, is_3d)
    if formula:
        try:
            expr = cached_parse_expression(engine, formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")
            
            # 执行渲染
            render_display(expr, is_3d)

            # 解析报告
            if is_3d:
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")
            else:
                deriv, integral = engine.get_analysis_2d(expr)
                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析失败: {e}")
