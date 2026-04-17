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
    except Exception: pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

st.set_page_config(page_title="DeepSeek 数学实验室", layout="wide")

# ✅ 核心 CSS：允许双指捏合缩放(pinch-zoom)，禁止网页整体缩放，防止手势干扰
st.markdown("""
    <style>
    .js-plotly-plot { 
        touch-action: pinch-zoom !important; 
        user-select: none;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 👈 侧边栏
# ==========================================
with st.sidebar:
    st.title("⚙️ 设置")
    if st.button("🔄 刷新内核"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    mode = st.radio("模式", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = (mode == "二元函数 (3D)")
    
    user_input = st.text_input("输入函数 (支持中文)", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        show_f = st.checkbox("原函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("积分 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 微积分绘图实验室")

# 简单明了的提示
st.markdown("""
    > **操作贴士**：📱 手机端直接用**双指捏合/张开**即可放缩。双击图像可重置视角。
""")

st.markdown("---")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            # 全局配置
            config = {
                'scrollZoom': True,        # 开启捏合/滚动缩放
                'displayModeBar': True,    # 显示工具栏（含平移/缩放切换）
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset'     # 双击复位，救命功能
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 解决缩放乱套的关键：使用 turntable 而非 orbit
                        # 这样缩放中心会稳定在屏幕中心，不会随手指乱飘
                        scene=dict(dragmode='turntable'), 
                        height=600, 
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)
                
                # 偏导分析
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
                    # 🚀 2D 模式回归默认，双指张合即为放缩
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                # 解析报告
                col1, col2 = st.columns(2)
                with col1: st.latex(rf"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(rf"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析失败: {e}")
