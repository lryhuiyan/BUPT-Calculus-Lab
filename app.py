import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 核心初始化 (PC 性能优化版)
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
    # 初始化数学引擎和 AI 逻辑
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 页面配置
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：控制面板
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")
    
    # 唯一的强制复位手段
    if st.button("🔄 物理刷新 (重置所有视角)"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("维度选择:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    st.markdown("### ✍️ 函数输入")
    user_input = st.text_input(
        "输入公式或描述:", 
        value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)"
    )

    if not is_3d:
        st.subheader("🖼️ 图层开关")
        show_f = st.checkbox("f(x)", value=True)
        show_deriv = st.checkbox("f'(x)", value=True)
        show_integral = st.checkbox("F(x)", value=True)

# ==========================================
# 📊 主页面：图像展示区
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# ✅ 电脑版专用的极简 Tips
st.markdown("""
    > **🖥️ 电脑版操作指南**：
    > * **缩放**：滚动鼠标滑轮。
    > * **控制**：使用图像右上角灰色按钮切换 **[旋转]** 或 **[十字平移]**。
    > * **状态锁定**：切换模式时视角自动锁定，**绝不复位**。
""")

if user_input:
    # 1. AI 翻译公式
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            # 2. 引擎解析
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ✅ 灰色按钮栏配置 (显示所有必备工具)
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d'] if not is_3d else []
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 关键：uirevision=True 告诉 Plotly 保持当前用户调整的相机视角
                        uirevision=True, 
                        scene=dict(dragmode='orbit'),
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0)
                        # ❌ 绝对不要在这里写 scene_camera 或 xaxis_range，否则会强制复位
                    )
                    # 🚀 关键：给 3D 图表一个死 key
                    st.plotly_chart(fig, use_container_width=True, config=config, key="pc_3d_plot")

                # 3D 偏导展示
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
                        # 🚀 关键：2D 同样锁定视角
                        uirevision=True,
                        dragmode='pan',
                        height=550
                    )
                    # 🚀 关键：给 2D 图表一个死 key
                    st.plotly_chart(fig, use_container_width=True, config=config, key="pc_2d_plot")

                # 2D 报告展示
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染组件故障: {e}")
