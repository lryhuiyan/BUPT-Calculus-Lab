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

# ✅ 【黑科技 CSS】彻底拦截浏览器干扰，实现原生 App 级手势感应
st.markdown("""
    <style>
    .js-plotly-plot { 
        touch-action: none !important; 
        -webkit-tap-highlight-color: transparent;
        user-select: none;
    }
    /* 隐藏不必要的工具栏，让界面干净得像原生 App */
    .modebar-container { right: 10px !important; top: 10px !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 👈 侧边栏 (极简设计)
# ==========================================
with st.sidebar:
    st.title("⚙️ 控制台")
    mode = st.radio("维度", ["2D 平面", "3D 空间"])
    is_3d = (mode == "3D 空间")
    user_input = st.text_input("输入函数", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")
    if st.button("🔄 重置引擎"):
        st.cache_resource.clear()
        st.rerun()

# ==========================================
# 📊 智能交互主界面
# ==========================================
st.title("🚀 微积分绘图实验室")

# 极简提示，不再让用户做选择题
st.caption("📱 手机端：单指滑动旋转/平移，双指捏合缩放。双击重置。")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            # ✅ 智能交互配置：开启全自动化感应
            config = {
                'scrollZoom': True,        # 允许双指/滚轮缩放
                'displayModeBar': False,   # 隐藏工具栏，因为手势已经够智能了
                'locale': 'zh-CN',
                'doubleClick': 'reset',    # 双击复位
                'showAxisDragHandles': False # 禁止拖动轴，防止手势冲突
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 智能感应锁定：默认旋转，双指自动缩放
                        scene=dict(
                            dragmode='orbit', 
                            hovermode=False, # 手机端关闭 hover 避免遮挡手势
                            aspectmode='cube'
                        ),
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)
                
                # 偏导分析
                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.latex(rf"f_x = {sp.latex(fx)} \quad f_y = {sp.latex(fy)}")

            else:
                deriv, integral = engine.get_analysis_2d(expr)
                items = [(expr, "f(x)", "#1f77b4"), (deriv, "f'(x)", "#d62728"), (integral, "F(x)", "#ff7f0e")]

                fig = engine.generate_2d_plot(items)
                if fig:
                    # 🚀 2D 智能感应：默认平移，双指自动缩放
                    fig.update_layout(
                        dragmode='pan', 
                        height=550,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                st.latex(rf"f'(x) = {sp.latex(deriv)}")
                st.latex(rf"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析失败: {e}")
