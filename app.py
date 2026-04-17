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

# ✅ 优化 CSS
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
st.title("🚀 基于DeepSeek V3的微积分绘论工具")

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
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.markdown("### 🧮 当前解析函数")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ✅ 这里的 config 是关键：显式开启 3D 的 zoomIn3d 和 zoomOut3d
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                # 2D 用 zoomIn2d, 3D 必须用 zoomIn3d 才会出现那两个灰色的 +/- 按钮
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d'] if not is_3d else ['zoomIn3d', 'zoomOut3d']
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 1. 默认 turntable 转动
                        scene=dict(dragmode='turntable'),
                        # 🚀 2. 切换按钮不复位的核心：只要 uirevision 不变，视角就不动
                        uirevision='constant',
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    # 🚀 3. 固定 key，防止重新加载
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="stable_3d_plot")

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
                        uirevision='constant',
                        dragmode='pan', 
                        height=550
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="stable_2d_plot")

                # 2D 报告
                st.markdown("### 📝 解析推导报告")
                col1, col2 = st.columns(2)
                with col1: st.latex(f"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(f"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
