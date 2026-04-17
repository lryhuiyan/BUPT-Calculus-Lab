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
    except:
        pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 页面配置
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# 优化 CSS：提升触屏手势响应
st.markdown("""
    <style>
    .js-plotly-plot { touch-action: pan-x pan-y pinch-zoom !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 👈 侧边栏：工具配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")

    if st.button("🔄 物理刷新 (清除异常缓存)"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")

    st.markdown("### ✍️ 函数输入")
    default_val = "x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)"
    user_input = st.text_input(
        "描述或输入函数:",
        value=default_val,
        help="支持直接描述（如：x的绝对值）或标准数学公式。"
    )

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# ✅ 针对用户的 Tips：简单有效
with st.expander("💡 快速使用指南", expanded=True):
    st.markdown("""
    * **AI 绘图**：直接输入口头描述或数学公式，系统自动识别。
    * **视角控制**：
        * **切换平移/转动**：点击图像右上角灰色按钮栏的 **[十字箭头]** 切换平移，点击 **[旋转图标]** 切换转动。
        * **缩放画面**：使用右上角 **[+] [-] 按钮**，或者直接在图像上双指捏合（电脑端滚动滑轮）。
    * **视角锁定**：系统会自动记住你的缩放和旋转角度。切换平移/转动模式时，**图像位置不会复位**。
    * **恢复初始**：如果画面调乱了，点击右上角 **[小房子]** 图标。
    """)

st.markdown("---")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.markdown("### 🧮 当前解析函数")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # 工具栏配置：强制中文、显示按钮
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
                        # 🚀 核心修复 1：将 uirevision 锁定在公式上，数据不变，视角绝对不重置
                        uirevision=str(expr),
                        scene=dict(dragmode='orbit'),
                        height=600,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    # 🚀 核心修复 2：固定 key，防止 Streamlit 重新创建组件导致状态丢失
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key=f"plot_3d_{formula}")

                # 3D 偏导分析
                st.markdown("### 📝 偏导数分析")
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                c1, c2 = st.columns(2)
                with c1: st.latex(rf"f_x = {sp.latex(fx)}")
                with c2: st.latex(rf"f_y = {sp.latex(fy)}")

            else:
                deriv, integral = engine.get_analysis_2d(expr)
                items = []
                if show_f: items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))

                fig = engine.generate_2d_plot(items)
                if fig:
                    fig.update_layout(
                        # 🚀 2D 同样锁定视角，切换按钮不复位
                        uirevision=str(expr),
                        dragmode='pan', 
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key=f"plot_2d_{formula}")

                # 2D 解析报告
                st.markdown("### 📝 解析推导报告")
                col1, col2 = st.columns(2)
                with col1: st.latex(f"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(f"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
