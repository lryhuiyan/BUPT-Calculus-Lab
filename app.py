"""Streamlit app for a DeepSeek-powered calculus and plotting agent."""
from __future__ import annotations

import os
import re

import streamlit as st
import sympy as sp

from ai_logic import MathAgent
from math_engine import MathEngine, normalize_formula

st.set_page_config(page_title="基于 DeepSeek V3 的微积分绘图 Agent", layout="wide")


def get_api_key() -> str | None:
    """优先从 Streamlit secrets 读取 API Key，其次从环境变量读取。"""
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("DEEPSEEK_API_KEY")


@st.cache_resource
def init_engine() -> MathEngine:
    return MathEngine()


@st.cache_resource
def init_agent(api_key: str | None) -> MathAgent | None:
    if not api_key:
        return None
    return MathAgent(api_key)


@st.cache_data(show_spinner=False)
def cached_ai_translate(input_str: str, is_3d: bool, api_key_marker: str) -> str | None:
    # api_key_marker 只用于让缓存知道“有没有 key”，不会保存真实 key
    agent = init_agent(get_api_key())
    if agent is None:
        return None
    return agent.chat_to_formula(input_str, is_3d=is_3d)


@st.cache_data(show_spinner=False)
def cached_parse(formula: str):
    return init_engine().parse_expression(formula)


def looks_like_formula(text: str) -> bool:
    """
    判断用户输入是否像公式。
    如果像公式，就先直接解析；如果不像，再交给 DeepSeek 翻译。
    """
    return bool(
        re.search(
            r"[xypi]|sin|cos|tan|log|ln|sqrt|exp|Abs|abs|\d",
            text,
            flags=re.I,
        )
    )


def parse_or_translate(user_input: str, is_3d: bool) -> tuple[sp.Expr, str, str]:
    """
    优先直接解析标准公式；如果失败，再调用 DeepSeek 翻译自然语言。
    返回：SymPy 表达式、内部公式字符串、解析来源。
    """
    engine = init_engine()
    candidates: list[tuple[str, str]] = []

    if looks_like_formula(user_input):
        candidates.append(("直接解析", normalize_formula(user_input)))

    key = get_api_key()
    translated = cached_ai_translate(
        user_input,
        is_3d,
        "has-key" if key else "no-key",
    ) if key else None

    if translated:
        candidates.append(("DeepSeek 翻译", normalize_formula(translated)))

    errors = []

    for source, formula in candidates:
        try:
            expr = cached_parse(formula)
            engine.validate_dimension(expr, is_3d)
            return expr, formula, source
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    if not key and not candidates:
        raise ValueError("这看起来像自然语言描述，但没有配置 DEEPSEEK_API_KEY，无法调用模型翻译。")

    if errors:
        raise ValueError("；".join(errors))

    raise ValueError("没有得到可解析的公式。")


engine = init_engine()

if "zoom_val" not in st.session_state:
    st.session_state.zoom_val = 1.0

if "drag_mode" not in st.session_state:
    st.session_state.drag_mode = "turntable"

if "needs_camera_sync" not in st.session_state:
    st.session_state.needs_camera_sync = False


with st.sidebar:
    st.header("⚙️ 工具配置")

    if not get_api_key():
        st.warning("未检测到 DEEPSEEK_API_KEY：标准公式仍可直接计算，自然语言翻译不可用。")

    if st.button("🔄 清除缓存并刷新"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.zoom_val = 1.0
        st.rerun()

    mode = st.radio("选择模式", ["一元函数 (2D)", "二元函数 (3D)"])
    is_3d = mode == "二元函数 (3D)"

    default_val = "x**2 + y**2" if is_3d else "sin(x) + x**2/5"

    user_input = st.text_input(
        "描述或输入函数",
        value=default_val,
        help="例：x的平方加sin x；或直接输入 sin(x)+x**2",
    )

    st.markdown("### 绘图区间")

    if is_3d:
        xy_range = st.slider("x/y 范围", 2, 30, 8)
        grid_size = st.slider("网格精度", 41, 161, 101, step=20)
    else:
        x_range = st.slider("x 范围", 2, 30, 10)

        st.markdown("### 图层显示")
        show_f = st.checkbox("函数 f(x)", value=True)
        show_deriv = st.checkbox("导函数 f'(x)", value=True)
        show_integral = st.checkbox("原函数 F(x)", value=False)


st.title("🚀 基于 DeepSeek V3 的微积分绘图 Agent")
st.caption("支持自然语言翻译、符号求导/积分、梯度、曲率与交互式绘图。")

with st.expander("💡 使用指南", expanded=True):
    st.markdown(
        """
- 可以直接输入公式：`sin(x)+x**2`、`Abs(x)`、`x**2+y**2`。
- 也可以输入描述：`x 的平方加 sin x`、`x 和 y 的平方和`。
- 幂运算推荐写 `**`；绝对值可写 `Abs(x)` 或 `|x|`。
- 2D 悬停可看曲率 κ；3D 悬停可看数值梯度。
"""
    )


def render_controls(is_3d: bool):
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])

    with c1:
        if st.button("➕ 放大", use_container_width=True):
            st.session_state.zoom_val *= 0.7
            st.session_state.needs_camera_sync = True
            st.rerun()

    with c2:
        if st.button("➖ 缩小", use_container_width=True):
            st.session_state.zoom_val *= 1.4
            st.session_state.needs_camera_sync = True
            st.rerun()

    with c3:
        if is_3d:
            target = "平移" if st.session_state.drag_mode == "turntable" else "旋转"
            if st.button(f"🔄 切换到{target}", use_container_width=True):
                st.session_state.drag_mode = "pan" if st.session_state.drag_mode == "turntable" else "turntable"
                st.rerun()
        else:
            st.button("📍 2D 默认平移", disabled=True, use_container_width=True)

    with c4:
        if st.button("🏠 重置", use_container_width=True):
            st.session_state.zoom_val = 1.0
            st.session_state.drag_mode = "turntable"
            st.session_state.needs_camera_sync = True
            st.rerun()


if user_input:
    try:
        expr, formula, source = parse_or_translate(user_input, is_3d)

        st.success(f"解析来源：{source}；内部公式：`{formula}`")
        st.latex(rf"f({'x,y' if is_3d else 'x'}) = {sp.latex(expr)}")

        render_controls(is_3d)

        config = {
            "displayModeBar": True,
            "scrollZoom": True,
        }

        if is_3d:
            fig = engine.generate_3d_plot(expr, -xy_range, xy_range, grid_size)

            if fig is None:
                st.error("该函数暂时无法生成 3D 图像，可能存在过多奇点或数值溢出。")
            else:
                if st.session_state.needs_camera_sync:
                    z = st.session_state.zoom_val
                    fig.update_layout(
                        scene_camera=dict(
                            eye=dict(
                                x=1.5 * z,
                                y=1.5 * z,
                                z=1.5 * z,
                            )
                        )
                    )
                    st.session_state.needs_camera_sync = False

                fig.update_layout(
                    scene_dragmode=st.session_state.drag_mode,
                    height=720,
                )

                st.plotly_chart(fig, use_container_width=True, config=config)

            analysis = engine.get_analysis_3d(expr)

            st.markdown("### 📝 二元函数分析")
            st.latex(rf"f_x={sp.latex(analysis['fx'])}\quad f_y={sp.latex(analysis['fy'])}")
            st.latex(rf"|\nabla f|={sp.latex(analysis['grad_norm'])}")
            st.latex(rf"K={sp.latex(analysis['gaussian'])}\quad H={sp.latex(analysis['mean'])}")
            st.caption("K 为高斯曲率，H 为平均曲率。复杂函数的曲率表达式可能较长，属于正常现象。")

        else:
            derivative, second_derivative, integral, curvature = engine.get_analysis_2d(expr)

            items = []

            if show_f:
                items.append((expr, "f(x)", "#1f77b4"))

            if show_deriv:
                items.append((derivative, "f'(x)", "#d62728"))

            if show_integral:
                items.append((integral, "F(x)", "#ff7f0e"))

            if not items:
                st.info("请至少勾选一个图层。")
            else:
                fig = engine.generate_2d_plot(items, -x_range, x_range)

                if st.session_state.needs_camera_sync:
                    z = st.session_state.zoom_val
                    fig.update_layout(
                        xaxis=dict(
                            range=[-x_range * z, x_range * z]
                        )
                    )
                    st.session_state.needs_camera_sync = False

                fig.update_layout(height=570)
                st.plotly_chart(fig, use_container_width=True, config=config)

            st.markdown("### 📝 一元函数分析")
            st.latex(rf"f'(x)={sp.latex(derivative)}")
            st.latex(rf"f''(x)={sp.latex(second_derivative)}")
            st.latex(rf"F(x)=\int f(x)\,dx={sp.latex(integral)}")
            st.latex(rf"\kappa(x)={sp.latex(curvature)}")

    except Exception as exc:
        st.error(f"处理失败：{exc}")
        st.info("可以试试把表达式写得更标准一些，比如 `sin(x)+x**2`、`Abs(x)`、`x**2+y**2`。")
