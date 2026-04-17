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

# 保持原始项目名称
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ✅ 优化 CSS：改善触屏手感
st.markdown("""
    <style>
    .js-plotly-plot { 
        touch-action: pan-x pan-y pinch-zoom !important; 
        user-select: none;
    }
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
        help="支持自然语言（如：x的平方）或标准公式。"
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

# ✅ 针对用户的 Tips：简单、直观、有效
with st.expander("💡 快速使用指南 (点击展开/收起)", expanded=True):
    st.markdown("""
    * **视角控制**：使用图像右上角的灰色按钮栏进行操作。
    * **切换模式**：点击 **[转动图标]** 旋转视角，点击 **[十字箭头]** 平移画面。切换时**视角不会重置**。
    * **画面缩放**：点击右上角 **[+] [-] 按钮**，或滚动鼠标滑轮。
    * **一键复位**：如果图像找不到了，点击右上角的 **[小房子]** 图标恢复初始视角。
    """)

st.markdown("---")

if user_input:
    # 调用 AI 逻辑
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.markdown("### 🧮 当前解析函数")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ✅ 灰色按钮栏配置：为 3D 显式添加 zoomInGeom 和 zoomOutGeom
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d'] if not is_3d else ['zoomInGeom', 'zoomOutGeom']
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 默认 turntable 转动
                        scene=dict(dragmode='turntable'),
                        # 🚀 切换按键不复位的核心：只要公式不变，uirevision 就不变
                        uirevision=formula,
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="plot_3d")

                # 3D 分析
                st.markdown("### 📝 偏导数")
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                c1, c2 = st.columns(2)
                with c1: st.latex(f"f_x = {sp.latex(fx)}")
                with c2: st.latex(f"f_y = {sp.latex(fy)}")

            else:
                # 2D 逻辑
                deriv, integral = engine.get_analysis_2d(expr)
                items = []
                if show_f: items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))

                fig = engine.generate_2d_plot(items)
                if fig:
                    fig.update_layout(
                        # 🚀 2D 默认平移且不复位
                        uirevision=formula,
                        dragmode='pan', 
                        height=550
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="plot_2d")

                # 2D 报告
                st.markdown("### 📝 解析推导报告")
                col1, col2 = st.columns(2)
                with col1: st.latex(f"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(f"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
