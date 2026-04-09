import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# 1. 基础配置
# 本地测试直接填 Key，云端部署时建议换回 st.secrets
MY_API_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"


@st.cache_resource
def init_resources():
    # 捕获可能的初始化异常
    try:
        return MathEngine(), MathAgent(MY_API_KEY)
    except Exception as e:
        st.error(f"初始化引擎失败: {e}")
        return None, None


engine, agent = init_resources()

# 严格保留你的页面配置
st.set_page_config(
    page_title="基于DeepSeek的一元函数微积分运算处理器",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 严格保留你的自定义 CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 严格保留你的标题
st.title("🎓 基于DeepSeek的一元函数微积分运算处理器")
st.caption("由 DeepSeek-V3 与 SymPy 驱动的符号运算实验室")

# 2. 侧边栏：按照你的要求去掉了典型函数选择
with st.sidebar:
    st.header("📌 实验控制台")

    # 修改点：默认值设为 sin(x)/x，方便你直接测试修复效果
    user_input = st.text_input("当前函数 f(x):", value="sin(x)/x")

    st.divider()
    st.subheader("视图选项")
    show_f = st.checkbox("绘制 f(x)", value=True)
    show_deriv = st.checkbox("绘制 f'(x)", value=True)
    show_integral = st.checkbox("绘制 F(x)", value=False)

    submit = st.button("运行实验")

# 3. 运行逻辑
if user_input:
    # 调用 AI 将输入转为公式
    formula_str = agent.chat_to_formula(user_input)

    if formula_str:
        try:
            # 这里的 engine 会自动处理字符串中的反斜杠补丁
            expr = engine.parse_expression(formula_str)
            deriv, integral = engine.get_analysis(expr)

            # 准备绘图数据
            plot_items = []
            if show_f: plot_items.append((expr, "函数 f(x)", "#1f77b4"))
            if show_deriv: plot_items.append((deriv, "导函数 f'(x)", "#d62728"))
            if show_integral: plot_items.append((integral, "最简原函数 F(x)", "#ff7f0e"))

            # 渲染图表
            fig = engine.generate_plotly_plot(plot_items)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # 严格保留你的结果展示区文案
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
                # 即使积分算不出来，也会优雅显示
                st.latex(f"{sp.latex(integral)} + C")

            # 严格保留你的下载功能
            st.divider()
            html_data = fig.to_html()
            st.download_button(
                label="📥 下载本次实验交互式图像 (HTML)",
                data=html_data,
                file_name="bupt_math_report.html",
                mime="text/html"
            )

        except Exception as e:
            # 如果解析失败，把 AI 识别的结果打印出来，方便调试
            st.error(f"数学引擎解析失败。AI 识别结果: `{formula_str}`")
            st.warning(f"错误详情: {e}")
    else:
        st.error("AI 接口未返回结果，请检查网络或 API Key。")

# 底部页脚
st.write("---")
st.caption("BUPT Computer Science | Calculus Lab v2.0")
