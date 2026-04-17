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

st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")
    if st.button("🔄 物理刷新"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    user_input = st.text_input("输入函数内容", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# 简单明了的 Tips
st.markdown("""
    > **视角锁定已开启**：点击右上角 **[十字/旋转]** 切换模式，或使用 **[+] [-]** 缩放，图像均**不会自动复位**。
""")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ✅ 灰色按钮配置：全部保留
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
                        # 🚀 核心 1：uirevision=True 配合去掉所有 range 限制
                        uirevision=True,
                        scene=dict(
                            dragmode='orbit',
                            aspectmode='cube'
                            # ❌ 这里绝对不要写 xaxis=dict(range=...)
                        ),
                        height=600,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    # 🚀 核心 2：固定 key，确保组件不被销毁
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="3d_plot_stable")

                st.markdown("### 📝 偏导数分析")
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
                        # 🚀 2D 同样锁定视角，去掉 range 限制
                        uirevision=True,
                        dragmode='pan', 
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="2d_plot_stable")

                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(deriv)} \quad F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
