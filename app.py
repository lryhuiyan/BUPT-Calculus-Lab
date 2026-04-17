import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 初始化与配置
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

st.set_page_config(page_title="DeepSeek 数学实验室", layout="wide")

# ✅ 强力 CSS：锁定手势，防止捏合时网页乱跑
st.markdown("""
    <style>
    .js-plotly-plot { touch-action: none !important; }
    .stExpander { border: none !important; box-shadow: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 👈 侧边栏
# ==========================================
with st.sidebar:
    st.title("⚙️ 设置")
    if st.button("🔄 强制刷新内核"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    mode = st.radio("模式", ["2D 平面", "3D 空间"])
    is_3d = (mode == "3D 空间")
    
    user_input = st.text_input("输入函数 (支持中文)", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        show_f = st.checkbox("f(x)", value=True)
        show_deriv = st.checkbox("导数 f'(x)", value=True)
        show_integral = st.checkbox("积分 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 微积分绘图实验室")

# ✅ 极简 Tips：用图标和极短文字解决战斗
st.markdown("""
    > **操作指南** > 🖱️ **电脑**：滚轮缩放 | 左键旋转 | 右键平移  
    > 📱 **手机**：双指捏合缩放 | 单指拖动平移  
    > ✨ **技巧**：双击图像可快速复位视角
""")

st.markdown("---")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset'
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        scene=dict(dragmode='pan'), # 🚀 3D 默认设为平移，手机操作最稳
                        height=600, margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)
                
                # 偏导分析
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
                    # 🚀 2D 默认设为缩放模式，手机单指划框就能放大
                    fig.update_layout(dragmode='zoom', height=500)
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                # 解析报告
                col1, col2 = st.columns(2)
                with col1: st.latex(rf"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(rf"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析失败: {e}")
