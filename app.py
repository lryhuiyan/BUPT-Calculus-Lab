import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 核心初始化与缓存 (提速响应)
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

# 缓存数学解析，避免按钮操作时重复计算
@st.cache_data(show_spinner=False)
def cached_parse_expression(_engine, formula):
    return _engine.parse_expression(formula)

# 还原你的原始项目名称
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：工具配置 (严格还原你的文字)
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")

    if st.button("🔄 物理刷新 (清除异常缓存)"):
        st.cache_resource.clear()
        st.cache_data.clear()
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
# 📊 主页面：还原提示与图像展示
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# 还原你的提示文案
with st.expander("💡 快速使用指南 (点击展开/收起)", expanded=True):
    st.markdown("""
    * **AI 绘图**：直接在左侧输入函数的口头描述或公式，AI 会自动识别。
    * **如何缩放**：
        * **电脑端**：滚动鼠标滑轮。
        * **手机端**：使用图像右上角的 **灰色 [+] [-] 按钮**。
    * **如何移动**：
        * **2D 模式**：单指滑动或左键拖拽。
        * **3D 模式**：单指滑动旋转，点击右上角 **[十字箭头]** 图标切换到平移。
    * **一键重置**：如果图像找不到了，点击右上角的 **[小房子]** 图标。
    """)

st.markdown("---")

if user_input:
    # 调用 AI 逻辑
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = cached_parse_expression(engine, formula)
            st.markdown("### 🧮 当前解析函数")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            # ✅ 灰色按钮栏配置：全量开启自带按钮，包含缩放、平移、旋转
            config = {
                'scrollZoom': True,
                'displayModeBar': True, # 开启自带工具栏
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                # 显式添加 2D 和 3D 的放大缩小按钮
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'zoomIn3d', 'zoomOut3d']
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 3D 默认转动模式
                        scene=dict(dragmode='turntable'),
                        # 🚀 切换按钮不复位的核心：锁定 uirevision
                        uirevision='constant',
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="plot_3d_final")

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
                        uirevision='constant',
                        dragmode='pan', 
                        height=550
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="plot_2d_final")

                st.markdown("### 📝 解析推导报告")
                col1, col2 = st.columns(2)
                with col1: st.latex(f"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(f"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
