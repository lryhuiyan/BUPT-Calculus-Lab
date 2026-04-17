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

# 初始化交互状态（确保切换时不复位）
if 'zoom_factor' not in st.session_state: st.session_state.zoom_factor = 1.0
if 'drag_mode' not in st.session_state: st.session_state.drag_mode = 'turntable' 

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 保持原始项目名称
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")
    if st.button("🔄 物理刷新 (清除异常缓存)"):
        st.cache_resource.clear()
        st.session_state.zoom_factor = 1.0
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    st.markdown("### ✍️ 函数输入")
    default_val = "x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)"
    user_input = st.text_input(
        "描述或输入函数:",
        value=default_val,
        help="支持自然语言（如：x的平方）或标准公式。"
    )

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面：操作面板 + 绘图区
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# ✅ 针对用户的 Tips
with st.expander("💡 快速使用指南", expanded=True):
    st.markdown("""
    * **缩放控制**：点击下方的 **[ ➕ ]** 放大，**[ ➖ ]** 缩小。
    * **交互切换**：3D模式下可点击 **[ 🔄 切换模式 ]** 在“旋转”和“平移”之间切换。
    * **视角复位**：如果画面找不到了，点击 **[ 🏠 ]** 即可。
    """)

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ------------------------------------------
            # 🕹️ 自定义控制按键 (图像上方)
            # ------------------------------------------
            st.markdown("##### 🎮 视图控制面板")
            c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
            with c1:
                if st.button("➕", use_container_width=True): st.session_state.zoom_factor *= 0.7
            with c2:
                if st.button("➖", use_container_width=True): st.session_state.zoom_factor *= 1.4
            with c3:
                if is_3d:
                    current_label = "旋转" if st.session_state.drag_mode == 'turntable' else "平移"
                    if st.button(f"🔄 切换到：{'平移' if current_label=='旋转' else '旋转'}", use_container_width=True):
                        st.session_state.drag_mode = 'pan' if st.session_state.drag_mode == 'turntable' else 'turntable'
                else:
                    st.button("📍 2D模式(默认平移)", disabled=True, use_container_width=True)
            with c4:
                if st.button("🏠", use_container_width=True):
                    st.session_state.zoom_factor = 1.0
                    st.session_state.drag_mode = 'turntable' if is_3d else 'pan'

            # ------------------------------------------
            # 🎨 绘图核心 (锁定 uirevision)
            # ------------------------------------------
            # 隐藏原生不准的工具栏
            config = {'displayModeBar': False, 'scrollZoom': True, 'locale': 'zh-CN'}
            z = st.session_state.zoom_factor

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 核心：只要 uirevision 不变，手动旋转的角度就不会复位
                        uirevision='constant', 
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene=dict(
                            dragmode=st.session_state.drag_mode,
                            camera=dict(eye=dict(x=1.5*z, y=1.5*z, z=1.5*z)), # 按键控制缩放
                            aspectmode='cube'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True, config=config, key="3d_canvas")
                
                # 3D 偏导分析
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")

            else:
                deriv, integral = engine.get_analysis_2d(expr)
                items = [(expr, "f(x)", "#1f77b4"), (deriv, "f'(x)", "#d62728"), (integral, "F(x)", "#ff7f0e")]
                
                fig = engine.generate_2d_plot(items)
                if fig:
                    fig.update_layout(
                        uirevision='constant',
                        dragmode='pan', 
                        height=550,
                        # 2D 缩放通过坐标轴 Range 联动
                        xaxis=dict(range=[-10*z, 10*z]),
                        yaxis=dict(range=[-10*z, 15*z])
                    )
                    st.plotly_chart(fig, use_container_width=True, config=config, key="2d_canvas")

                # 2D 解析报告
                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染组件故障: {e}")
