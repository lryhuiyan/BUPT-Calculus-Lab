import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

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

# 🚀 提速核心：AI 缓存
@st.cache_data(show_spinner=False)
def cached_chat_to_formula(_agent, input_str, is_3d):
    return _agent.chat_to_formula(input_str, is_3d=is_3d)

# 🚀 提速核心：解析缓存
@st.cache_data(show_spinner=False)
def cached_parse_expression(_engine, formula):
    return _engine.parse_expression(formula)

st.set_page_config(page_title="基于DeepSeek V3的微积分绘图工具", layout="wide")

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
    user_input = st.text_input("描述或输入函数:", value=default_val)

    if not is_3d:
        st.subheader("🖼️ 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("最简原函数 F(x)", value=True)
        show_curvature = st.checkbox("曲率 κ", value=True) 

st.title("🚀 基于DeepSeek V3的微积分绘图工具")

with st.expander("💡 快速使用指南", expanded=True):
    st.markdown("""
    * **视角控制**：使用图像右上角的自带灰色工具栏进行缩放、平移。
    * **切换不复位**：系统已锁定视角，切换模式时绝不会重置。
    """)

if user_input:
    formula = cached_chat_to_formula(agent, user_input, is_3d)

    if formula:
        try:
            expr = cached_parse_expression(engine, formula)
            st.markdown("### 🧮 当前解析函数")
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")

            config = {
                'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False,
                'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'zoomIn3d', 'zoomOut3d']
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        scene=dict(dragmode='turntable'),
                        uirevision='constant', height=700, margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="3d_final")
                else:
                    st.warning("⚠️ 该 3D 函数在实数域内无有效图像，或存在不可解析的数学对象。")

                fx, fy = sp.diff(expr, engine.x).doit(), sp.diff(expr, engine.y).doit()
                st.markdown("### 📝 偏导数")
                c1, c2 = st.columns(2)
                with c1: st.latex(f"f_x = {sp.latex(fx)}")
                with c2: st.latex(f"f_y = {sp.latex(fy)}")

            else:
                deriv, integral, curvature = engine.get_analysis_2d(expr)
                
                items = []
                if show_f: items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv: items.append((deriv, "f'(x)", "#d62728"))
                if show_integral: items.append((integral, "F(x)", "#ff7f0e"))
                if show_curvature: items.append((curvature, "曲率 κ", "#2ca02c")) 

                fig = engine.generate_2d_plot(items)
                if fig:
                    fig.update_layout(
                        uirevision='constant', dragmode='pan', height=550
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key="2d_final")
                else:
                    st.warning("⚠️ 渲染失败。可能是该函数的积分无闭式解，且主函数超出了实数域。")

                st.markdown("### 📝 解析推导报告")
                st.latex(rf"f'(x) = {sp.latex(deriv)}")
                
                if integral.has(sp.Integral):
                    st.latex(r"F(x) \text{ 无初等函数解}")
                else:
                    st.latex(rf"F(x) = {sp.latex(integral)}")
                    
                st.latex(rf"\kappa = {sp.latex(curvature)}")

        except Exception as e:
            st.error(f"渲染出错: {e}")
