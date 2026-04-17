import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 初始化与配置 (安全与兼容性处理)
# ==========================================
# 兜底 Key，确保在没有任何配置的情况下也能运行
DEFAULT_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"

def get_api_key():
    """
    自适应获取 API Key：
    1. 优先尝试从 Streamlit Cloud 网页后台获取 Secrets
    2. 如果抛出异常（本地运行且无配置文件），则使用 DEFAULT_KEY
    """
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        # 捕获本地运行找不到 secrets.toml 的报错
        pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    """全局单例模式初始化引擎，避免重复加载导致内存溢出"""
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

# 页面基础配置
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：实验室控制面板
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")

    if st.button("🔄 物理刷新 (清除异常缓存)", help="当图像卡死或代码更新不生效时，点击此按钮重置内核。"):
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
        help="支持自然语言或标准 Python 公式。"
    )

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面：图像与解析展示
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# 用户交互引导
# ✅ 针对全设备优化的用户交互提示语
with st.expander("💡 快速上手指南 (User Guide)", expanded=True):
    st.columns(2)
    col_pc, col_phone = st.columns(2)

    with col_pc:
        st.markdown("""
        **💻 电脑端操作 (Mouse)**
        * **缩放**：滚动鼠标滑轮。
        * **平移/旋转**：左键按住拖拽。
        * **精确查看**：鼠标悬停在曲线上，可查看 $(x, y)$ 坐标及实时曲率 $\kappa$。
        * **快捷复位**：点击右上角“小房子”图标恢复初始视角。
        """)

    with col_phone:
        st.markdown("""
        **📱 手机/触屏操作 (Touch)**
        * **缩放**：双指张合（捏合）图像。
        * **旋转/平移**：单指滑动。
        * **查看数值**：手指长按并滑动。
        * **进阶建议**：切换至**横屏模式**可获得更大的绘图视野！
        """)

    st.info("🤖 **AI 提示**：你可以直接输入“x的平方”或“x加y的绝对值”，后台 DeepSeek 会自动为你精准翻译。")

st.markdown("---")

if user_input:
    # 1. 语言模型翻译
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            # 2. 数学引擎解析
            expr = engine.parse_expression(formula)

            st.markdown("### 🧮 当前解析函数")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")

            # 3. 渲染与分析逻辑
            if is_3d:
                # --- 3D 渲染 ---
                fig = engine.generate_3d_plot(expr)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, theme=None, config={
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'displaylogo': False
                    })

                # 3D 偏导分析
                st.markdown("### 📝 偏导数分析")
                fx = sp.diff(expr, engine.x).doit()
                fy = sp.diff(expr, engine.y).doit()
                c1, c2 = st.columns(2)
                with c1: st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(fx)}")
                with c2: st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(fy)}")

            else:
                # --- 2D 渲染 ---
                deriv, integral = engine.get_analysis_2d(expr)
                items = []
                if show_f: items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))

                fig = engine.generate_2d_plot(items)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, theme=None, config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'displaylogo': False
                    })

                # 2D 报告展示
                st.markdown("### 📝 微积分解析报告")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**一阶导数 f'(x)**")
                    st.latex(sp.latex(deriv))
                with col2:
                    st.markdown("**不定积分 F(x) + C**")
                    st.latex(sp.latex(integral))

        except Exception as e:
            st.error(f"渲染失败: {e}")
            st.info("提示：请检查函数输入或点击‘物理刷新’。")
