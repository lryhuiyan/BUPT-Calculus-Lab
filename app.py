import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 基础初始化
# ==========================================
DEFAULT_KEY = "sk-c262ed499b0643d6bbc979f93b00ee5e"

def get_api_key():
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except: pass
    return DEFAULT_KEY

MY_API_KEY = get_api_key()

@st.cache_resource
def init_resources():
    return MathEngine(), MathAgent(MY_API_KEY)

engine, agent = init_resources()

st.set_page_config(page_title="微积分绘图工具", layout="wide")

# ==========================================
# 👈 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制面板")
    if st.button("🔄 刷新引擎"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    mode = st.radio("维度选择", ["2D 平面", "3D 空间"])
    is_3d = (mode == "3D 空间")
    
    user_input = st.text_input("输入函数内容", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        show_f = st.checkbox("原函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("不定积分 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 微积分绘图实验室")

# 极简提示，只说最核心的
st.info("💡 **操作指南**：单指滑动旋转/平移，双指张合缩放图像。双击图像可复位。")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            # ✅ 核心配置：精简工具栏，删除所有“自动缩放”和“框选”按钮
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                'modeBarButtonsToRemove': [
                    'autoScale2d', 'autoscale', 'zoom2d', 'zoom3d', 
                    'lasso2d', 'select2d', 'toggleHover', 'hoverClosestCartesian'
                ]
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        scene=dict(dragmode='orbit'), # 3D 保持最标准的旋转模式
                        height=600, 
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)
                
                # 3D 偏导
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
                    # 🚀 2D 彻底固定为平移模式 (pan)，禁止拉框缩放
                    fig.update_layout(dragmode='pan', height=500)
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                # 2D 解析
                col1, col2 = st.columns(2)
                with col1: st.latex(rf"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(rf"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析出错: {e}")
