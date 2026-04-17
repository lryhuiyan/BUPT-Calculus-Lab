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
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except: pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 保持项目名称
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具控制")
    
    if st.button("🔄 物理刷新 (重置所有状态)"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    st.markdown("### ✍️ 函数输入")
    user_input = st.text_input(
        "描述或输入函数:", 
        value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)"
    )

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("原函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("不定积分 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# ✅ 针对用户的 Tips：简单易懂
with st.expander("💡 快速操作指南", expanded=True):
    st.markdown("""
    * **视角控制**：使用图像右上角的灰色按钮。
    * **旋转与平移**：点击 **[转动图标]** 进入旋转模式，点击 **[十字箭头]** 进入平移模式。
    * **缩放画面**：点击右上角 **[+] [-] 按钮**，或滚动鼠标滑轮。
    * **不复位功能**：切换平移/旋转模式或调整参数时，**当前缩放和角度会被锁定**，不会自动重置。
    * **一键归位**：如果画面找不到了，点击 **[小房子]** 图标恢复初始状态。
    """)

st.markdown("---")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ✅ 灰色按钮栏配置：在 3D 和 2D 都显式加入放大缩小按钮
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                # 强行为 2D 和 3D 添加放大缩小按钮
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'zoomInGeom', 'zoomOutGeom'] 
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 默认使用 turntable 转动，缩放更平稳
                        scene=dict(dragmode='turntable'),
                        # 🚀 核心：uirevision=True 保证切换按钮不复位
                        uirevision=True, 
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, config=config, key="3d_main_plot")

                # 3D 分析
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
                    fig.update_layout(
                        # 🚀 2D 默认平移，设置 uirevision
                        uirevision=True,
                        dragmode='pan',
                        height=550
                    )
                    st.plotly_chart(fig, use_container_width=True, config=config, key="2d_main_plot")

                # 2D 报告
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染组件故障: {e}")
