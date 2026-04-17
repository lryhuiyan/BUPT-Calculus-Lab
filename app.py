import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp
import time

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

# 初始化全局状态
if 'drag_mode' not in st.session_state: st.session_state.drag_mode = 'turntable'
if 'view_id' not in st.session_state: st.session_state.view_id = str(time.time())
if 'last_formula' not in st.session_state: st.session_state.last_formula = ""

st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")
    if st.button("🔄 物理刷新"):
        st.cache_resource.clear()
        st.session_state.view_id = str(time.time())
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
# 📊 主页面逻辑
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

if user_input:
    # 1. 获取公式
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)
    
    # 🚀 逻辑检查：如果公式变了，才允许重置视角；否则 view_id 保持不变
    if formula != st.session_state.last_formula:
        st.session_state.view_id = str(time.time())
        st.session_state.last_formula = formula

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")
            
            # 2. 🕹️ 自定义控制面板
            st.markdown("##### 🎮 视图控制")
            c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
            
            with c1:
                # 放大按键（通过控制条调用缩放）
                st.button("➕", use_container_width=True, help="请使用图像右上角灰色 [+] 按钮")
            with c2:
                # 缩小按键
                st.button("➖", use_container_width=True, help="请使用图像右上角灰色 [-] 按钮")
            with c3:
                if is_3d:
                    label = "当前：转动" if st.session_state.drag_mode == 'turntable' else "当前：平移"
                    if st.button(f"🔄 模式切换 ({label})", use_container_width=True):
                        # 🚀 关键：切换模式时，view_id 不变，图像绝对不复位
                        st.session_state.drag_mode = 'pan' if st.session_state.drag_mode == 'turntable' else 'turntable'
                        st.rerun()
                else:
                    st.button("📍 2D模式(默认平移)", disabled=True, use_container_width=True)
            with c4:
                if st.button("🏠 复位", use_container_width=True):
                    st.session_state.view_id = str(time.time()) # 只有点复位，才更新ID触发复位
                    st.rerun()

            # 3. 🎨 绘图核心
            # 配置：显示所有灰色按钮，包含 +/-
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'zoomIn3d', 'zoomOut3d']
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 焊死视角的核心 1：uirevision 绑定 view_id
                        uirevision=st.session_state.view_id,
                        scene=dict(
                            dragmode=st.session_state.drag_mode, # 模式在这里动态切换
                            aspectmode='cube'
                        ),
                        height=700, margin=dict(l=0, r=0, b=0, t=0)
                    )
                    # 🚀 焊死视角的核心 2：key 绑定 view_id
                    st.plotly_chart(fig, use_container_width=True, config=config, key=f"plot_{st.session_state.view_id}")
            else:
                # 2D 逻辑
                deriv, integral = engine.get_analysis_2d(expr)
                items = [(expr, "f(x)", "#1f77b4"), (deriv, "f'(x)", "#d62728"), (integral, "F(x)", "#ff7f0e")]
                fig = engine.generate_2d_plot(items)
                if fig:
                    fig.update_layout(
                        uirevision=st.session_state.view_id,
                        dragmode='pan', 
                        height=550
                    )
                    st.plotly_chart(fig, use_container_width=True, config=config, key=f"plot_{st.session_state.view_id}")

            # 4. 解析报告
            if is_3d:
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")
            else:
                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析失败: {e}")
