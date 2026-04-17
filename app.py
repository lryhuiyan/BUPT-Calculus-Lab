import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 初始化与配置 (安全与兼容性处理)
# ==========================================
DEFAULT_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"

def get_api_key():
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 页面基础配置
st.set_page_config(page_title="DeepSeek 微积分绘图实验室", layout="wide")

# ==========================================
# 👈 侧边栏：实验室控制面板
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具控制台")

    if st.button("🔄 强制刷新内核", help="清除缓存并重置数学引擎"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择坐标系:", ["一元函数 (2D平面)", "二元函数 (3D空间)"])
    is_3d = (mode == "二元函数 (3D空间)")

    st.markdown("### ✍️ 数学输入")
    default_val = "x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)"
    user_input = st.text_input(
        "描述或输入函数表达式:",
        value=default_val,
        help="支持中文描述或标准公式。"
    )

    if not is_3d:
        st.subheader("🖼️ 显示图层")
        show_f = st.checkbox("原函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("不定积分 F(x)", value=True)

# ==========================================
# 📊 主页面：图像与解析展示
# ==========================================
st.title("🚀 基于 DeepSeek V3 的微积分绘图工具")

# ✅ 用户交互引导 (针对手机端交互重点说明)
with st.expander("💡 触屏与鼠标操作指南", expanded=True):
    col_pc, col_phone = st.columns(2)
    with col_pc:
        st.markdown("""
        **💻 电脑端 (鼠标)**
        * **放缩**：滚动滑轮
        * **平移**：右键拖拽
        * **旋转**：左键拖拽 (仅3D)
        """)
    with col_phone:
        st.markdown("""
        **📱 手机端 (触屏)**
        * **放缩**：**双指张合** (捏合屏幕)
        * **平移/旋转**：单指滑动
        * **小技巧**：点击图像右上角的“十字箭头”可切换平移模式
        """)

st.markdown("---")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.markdown("### 🧮 实时解析公式")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            # --- 核心配置：定义中文按钮标签 ---
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',  # ✅ 强制中文语言环境
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'config': {'responsive': True}
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    # ✅ 优化 3D 平动：默认增加平移工具
                    fig.update_layout(
                        scene=dict(dragmode='turntable'), # 手机端旋转更稳
                        margin=dict(l=0, r=0, b=0, t=0),
                        height=600 # 手机端拉高视野
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                st.markdown("### 📝 偏导数分析")
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                c1, c2 = st.columns(2)
                with c1: st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(fx)}")
                with c2: st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(fy)}")

            else:
                deriv, integral = engine.get_analysis_2d(expr)
                items = []
                if show_f: items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))

                fig = engine.generate_2d_plot(items)
                if fig:
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                st.markdown("### 📝 微积分解析报告")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**一阶导数 f'(x)**")
                    st.latex(sp.latex(deriv))
                with col2:
                    st.markdown("**不定积分 F(x) + C**")
                    st.latex(sp.latex(integral))

        except Exception as e:
            st.error(f"渲染异常: {e}")
