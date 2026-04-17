import streamlit as st
from math_engine import MathEngine
from ai_logic import MathAgent
import sympy as sp

# ==========================================
# ⚙️ 初始化
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

st.set_page_config(page_title="BUPT 微积分实验室", layout="wide")

# ✅ 强力 CSS：禁止所有浏览器默认手势，把控制权 100% 交给 Plotly 代码
# 这样能解决缩放方向“反了”以及捏合时网页乱跑的问题
st.markdown("""
    <style>
    .js-plotly-plot { 
        touch-action: none !important; 
        -webkit-user-select: none; 
        user-select: none;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 👈 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制面板")
    if st.button("🔄 刷新引擎"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    mode = st.radio("模式选择", ["2D 平面", "3D 空间"])
    is_3d = (mode == "3D 空间")
    
    user_input = st.text_input("输入函数", value="x**(-2/3)+y**(-2/3)" if is_3d else "x**(2/3)")

    if not is_3d:
        show_f = st.checkbox("f(x)", value=True)
        show_deriv = st.checkbox("导数 f'(x)", value=True)
        show_integral = st.checkbox("积分 F(x)", value=True)

# ==========================================
# 📊 主页面
# ==========================================
st.title("🚀 微积分绘图实验室")

# 提示语也同步简化
st.caption("📱 手机端：单指拖动平移/旋转，双指捏合缩放。双击重置视角。")

if user_input:
    formula = agent.chat_to_formula(user_input, is_3d=is_3d)

    if formula:
        try:
            expr = engine.parse_expression(formula)
            st.latex(rf"f({'x, y' if is_3d else 'x'}) = {sp.latex(expr)}")
            
            # ✅ 核心配置：砍掉所有“乱套”的自动化按钮
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'locale': 'zh-CN',
                'doubleClick': 'reset',
                # 彻底删除 Autoscale 和各种框选工具
                'modeBarButtonsToRemove': [
                    'autoScale2d', 'autoscale', 'zoom2d', 'zoom3d', 
                    'lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
                ]
            }

            if is_3d:
                fig = engine.generate_3d_plot(expr)
                if fig:
                    fig.update_layout(
                        # 🚀 3D 视角固定：单指旋转，缩放交由双指捏合
                        scene=dict(
                            dragmode='orbit', 
                            xaxis_fixedrange=False,
                            yaxis_fixedrange=False,
                            zaxis_fixedrange=False
                        ),
                        height=600, 
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)
                
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
                    # 🚀 2D 彻底禁用框选：默认就是平移 (pan)
                    fig.update_layout(
                        dragmode='pan', 
                        height=500,
                        xaxis=dict(fixedrange=False), 
                        yaxis=dict(fixedrange=False)
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None, config=config)

                col1, col2 = st.columns(2)
                with col1: st.latex(rf"f'(x) = {sp.latex(deriv)}")
                with col2: st.latex(rf"F(x) = {sp.latex(integral)}")

        except Exception as e:
            st.error(f"解析失败: {e}")
