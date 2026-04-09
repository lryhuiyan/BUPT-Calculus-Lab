import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# 1. 基础配置
MY_API_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"


@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)


engine, agent = init_resources()

st.set_page_config(page_title="基于DeepSeek的一元函数微积分运算处理器", layout="wide", initial_sidebar_state="expanded")

# 自定义 CSS 样式，更有质感
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 基于DeepSeek的一元函数微积分运算处理器")
st.caption("由 DeepSeek-V3 与 SymPy 驱动的符号运算实验室")

# 2. 侧边栏：加入快捷案例
with st.sidebar:
    st.header("📌 实验控制台")

    # 案例库：方便演示
    st.subheader("内置案例")
    preset = st.selectbox("选择典型函数：",
                          ["手动输入", "经典星形线 (x**2/3)", "高斯分布 (exp(-x**2))", "震荡衰减 (sin(x)/x)"])

    preset_map = {
        "经典星形线 (x**2/3)": "x**(2/3)",
        "高斯分布 (exp(-x**2))": "exp(-x**2)",
        "震荡衰减 (sin(x)/x)": "sin(x)/x"
    }

    default_val = preset_map.get(preset, "x**2")
    user_input = st.text_input("当前函数 f(x):", value=default_val)

    st.divider()
    st.subheader("视图选项")
    show_f = st.checkbox("绘制 f(x)", value=True)
    show_deriv = st.checkbox("绘制 f'(x)", value=True)
    show_integral = st.checkbox("绘制 F(x)", value=False)

    submit = st.button("运行实验")

# 3. 运行逻辑
if user_input:
    formula_str = agent.chat_to_formula(user_input)
    if formula_str:
        try:
            expr = engine.parse_expression(formula_str)
            deriv, integral = engine.get_analysis(expr)

            # 绘图数据
            plot_items = []
            if show_f: plot_items.append((expr, "函数 f(x)", "#1f77b4"))
            if show_deriv: plot_items.append((deriv, "导函数 f'(x)", "#d62728"))
            if show_integral: plot_items.append((integral, "最简原函数 F(x)", "#ff7f0e"))

            # 渲染图表
            fig = engine.generate_plotly_plot(plot_items)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # 结果展示区
            st.markdown("### 📝 数学推导报告")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**解析式 f(x)**")
                st.latex(sp.latex(expr))
            with col2:
                st.success("**一阶导数 f'(x)**")
                st.latex(sp.latex(deriv))
            with col3:
                st.warning("**不定积分 F(x)**")
                st.latex(f"{sp.latex(integral)} + C")

            # 下载功能：作业提交加分项
            st.divider()
            html_data = fig.to_html()
            st.download_button(
                label="📥 下载本次实验交互式图像 (HTML)",
                data=html_data,
                file_name="bupt_math_report.html",
                mime="text/html"
            )

        except Exception as e:
            st.error(f"分析失败，请检查输入格式。错误详情: {e}")