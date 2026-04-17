import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 初始化与配置
# ==========================================
# 确保这里的 API_KEY 是有效的
MY_API_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"


@st.cache_resource
def init_resources():
    """全局单例模式初始化引擎，避免每次刷新页面重复创建对象"""
    return MathEngine(), MathAgent(MY_API_KEY)


engine, agent = init_resources()

# 配置页面标题和布局宽度
st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏：实验室控制面板
# ==========================================
with st.sidebar:
    st.header("⚙️ 工具配置")

    # 增加 tooltip (help) 解释按钮作用
    if st.button("🔄 物理刷新 (清除异常缓存)", help="当图像卡死或渲染出现残影时，点击此按钮强制重置内核。"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    mode = st.radio("选择模式:", ["一元函数", "二元函数"])
    is_3d = (mode == "二元函数")

    # 用户输入区
    st.markdown("### ✍️ 函数输入")
    user_input = st.text_input(
        "描述或输入函数:",
        value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)",
        help="支持自然语言（如：x的平方加y的平方）或 Python 公式（如：x**2 + y**2）。"
    )

    # 2D 专属：图层开关
    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)

# ==========================================
# 📊 主页面：图像与解析展示
# ==========================================
st.title("🚀 基于DeepSeek V3的微积分绘图工具")

# ✅ 新增：友好的用户交互提示语
with st.expander("💡 点击查看使用小贴士 (Tips)", expanded=False):
    st.markdown("""
    * **自然语言驱动**：你可以直接对它说 _“求以x为底2的对数”_ 或 _“x的绝对值乘以sin(x)”_，AI 会自动帮你翻译成标准数学公式。
    * **极限视觉呈现**：针对类似 $x^{-2/3}$ 这种在 $0$ 处发散的函数，本工具做了底层的物理断路处理，完美还原“直冲云霄”的数学直觉。
    * **视图操作**：
        * **2D 模式**：鼠标左键按住拖拽平移，滚轮缩放，鼠标悬停可查看精确的 $(x, y)$ 坐标与实时曲率 $\kappa$。
        * **3D 模式**：鼠标左键旋转视角，右键平移，滚轮缩放。
    """)

st.markdown("---")

# 主逻辑执行区
if user_input:
    # 1. 语言大模型翻译公式
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            # 2. 数学引擎解析公式
            expr = engine.parse_expression(formula)

            # 展示当前解析出的标准数学公式 (LaTeX格式)
            st.markdown("### 🧮 当前解析函数")
            if is_3d:
                st.latex(rf"f(x, y) = {sp.latex(expr)}")
            else:
                st.latex(rf"f(x) = {sp.latex(expr)}")

            st.markdown("---")

            # 3. 渲染 3D 或 2D 图像
            if is_3d:
                # 生成 3D 图形对象
                fig = engine.generate_3d_plot(expr)
                if fig:
                    st.plotly_chart(fig, width='stretch', theme=None)

                # 计算并渲染偏导数 LaTeX
                st.markdown("### 📝 偏导数")
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                c1, c2 = st.columns(2)
                with c1:
                    st.latex(f"f_x = {sp.latex(fx)}")
                with c2:
                    st.latex(f"f_y = {sp.latex(fy)}")

            else:
                # 执行 2D 解析几何运算
                deriv, integral = engine.get_analysis_2d(expr)
                items = []
                # 根据用户侧边栏勾选状态，动态推入图层
                if show_f: items.append((expr, "f(x)", "#1f77b4"))  # 蓝色：原函数
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))  # 红色：导函数
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))  # 橙色：积分函数

                # 生成 2D 图形对象
                fig = engine.generate_2d_plot(items)
                if fig:
                    st.plotly_chart(fig, width='stretch', theme=None, config={'scrollZoom': True})

                # 渲染解析推导报告 LaTeX
                st.markdown("### 📝 解析推导报告")
                col1, col2 = st.columns(2)
                with col1:
                    st.latex(f"f'(x) = {sp.latex(deriv)}")
                with col2:
                    st.latex(f"F(x) = {sp.latex(integral)}")

        except Exception as e:
            # 捕获并展示引擎深层报错，防止页面直接白屏
            st.error(f"渲染组件故障: {e}")
