"""Streamlit app for a DeepSeek-powered calculus and plotting agent."""
from __future__ import annotations

import os
import re

import streamlit as st
import sympy as sp

from ai_logic import MathAgent
from math_engine import MathEngine, normalize_formula

st.set_page_config(page_title="微积分绘图 Agent", layout="wide")


@st.cache_resource
def init_engine() -> MathEngine:
    return MathEngine()


def get_api_key() -> str | None:
    """Read the DeepSeek API key from Streamlit secrets or environment variables."""
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    return os.getenv("DEEPSEEK_API_KEY")


@st.cache_resource
def init_agent(api_key: str | None) -> MathAgent | None:
    if not api_key:
        return None

    return MathAgent(api_key)


@st.cache_data(show_spinner=False)
def cached_ai_translate(input_str: str, is_3d: bool, api_key_marker: str) -> str | None:
    """Translate natural language to formula. The marker avoids caching the real key."""
    try:
        agent = init_agent(get_api_key())
    except RuntimeError as exc:
        st.warning(str(exc))
        return None

    if agent is None:
        return None

    return agent.chat_to_formula(input_str, is_3d=is_3d)


@st.cache_data(show_spinner=False)
def cached_ai_translate_equation(input_str: str, is_surface: bool, api_key_marker: str) -> str | None:
    """Translate natural language to an implicit curve/surface equation."""
    try:
        agent = init_agent(get_api_key())
    except RuntimeError as exc:
        st.warning(str(exc))
        return None

    if agent is None:
        return None

    return agent.chat_to_equation(input_str, is_surface=is_surface)


@st.cache_data(show_spinner=False)
def cached_parse(formula: str):
    return init_engine().parse_expression(formula)


def looks_like_formula(text: str) -> bool:
    """Prefer direct parsing for formula-like input to avoid unnecessary model calls."""
    return bool(
        re.search(
            r"[xyzpi]|sin|cos|tan|log|ln|sqrt|exp|Abs|abs|绝对值|\d|\*|/|\||\^|=",
            text,
            flags=re.I,
        )
    )


def parse_or_translate(user_input: str, is_3d: bool) -> tuple[sp.Expr, str, str]:
    """Parse standard formulas first, then fall back to DeepSeek translation."""
    engine = init_engine()
    candidates: list[tuple[str, str]] = []

    if looks_like_formula(user_input):
        candidates.append(("直接解析", normalize_formula(user_input)))

    key = get_api_key()
    translated = None

    if key:
        translated = cached_ai_translate(user_input, is_3d, "has-key")

    if translated:
        candidates.append(("DeepSeek 翻译", normalize_formula(translated)))

    errors: list[str] = []

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

def parse_implicit_or_translate(user_input: str, is_surface: bool) -> tuple[sp.Expr, str, str]:
    """Parse implicit equations first, then fall back to DeepSeek equation translation."""
    engine = init_engine()
    candidates: list[tuple[str, str]] = []

    if looks_like_formula(user_input):
        candidates.append(("直接解析", user_input))

    key = get_api_key()
    translated = None

    if key:
        translated = cached_ai_translate_equation(user_input, is_surface, "has-key")

    if translated:
        candidates.append(("DeepSeek 方程翻译", translated))

    errors: list[str] = []

    for source, equation in candidates:
        try:
            expr = engine.parse_implicit_equation(equation)
            engine.validate_implicit_dimension(expr, is_surface)
            return expr, equation, source
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    if not key and not candidates:
        raise ValueError("这看起来像自然语言方程描述，但没有配置 DEEPSEEK_API_KEY，无法调用模型翻译。")

    if errors:
        raise ValueError("；".join(errors))

    raise ValueError("没有得到可解析的一般方程。")


def fmt(value: float | None) -> str:
    if value is None:
        return "未定义"
    return f"{value:.8g}"


def render_value_table(values: dict[str, float | None]) -> None:
    st.table([{"项目": key, "数值": fmt(value)} for key, value in values.items()])


engine = init_engine()

if "zoom_val" not in st.session_state:
    st.session_state.zoom_val = 1.0

if "drag_mode" not in st.session_state:
    st.session_state.drag_mode = "turntable"

if "needs_camera_sync" not in st.session_state:
    st.session_state.needs_camera_sync = False


with st.sidebar:
    st.header("工具配置")

    if not get_api_key():
        st.warning("未检测到 DEEPSEEK_API_KEY：标准公式仍可直接计算，自然语言翻译不可用。")

    if st.button("清除缓存并刷新"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.zoom_val = 1.0
        st.rerun()

    mode = st.radio("选择模式", ["一元函数 (2D)", "二元函数 (3D)", "一般曲线/曲面方程"])
    is_3d = mode == "二元函数 (3D)"
    is_implicit = mode == "一般曲线/曲面方程"

    if is_implicit:
        implicit_kind = st.radio("方程类型", ["平面曲线 F(x,y)=0", "空间曲面 F(x,y,z)=0"])
        is_surface = implicit_kind == "空间曲面 F(x,y,z)=0"
        default_val = "x**2 + y**2 = 4" if not is_surface else "x**2 + y**2 + z**2 = 9"
        user_input = st.text_input(
            "输入一般方程",
            value=default_val,
            help="例如：x**2+y**2=4、y**2=x**3-x、x**2+y**2+z**2=9。",
        )
        st.markdown("### 绘图区间")
        implicit_range = st.slider("坐标范围", 2, 20, 6 if is_surface else 8)
        if is_surface:
            implicit_grid = st.slider("采样精度", 21, 71, 45, step=8)
        else:
            implicit_grid = st.slider("采样精度", 101, 501, 301, step=50)
        st.markdown("### 指定点计算")
        eval_x = st.number_input("x", value=1.0, format="%.6f")
        eval_y = st.number_input("y", value=1.0, format="%.6f")
        if is_surface:
            eval_z = st.number_input("z", value=1.0, format="%.6f")
    else:
        default_val = "x**2 + y**2" if is_3d else "1/x"
        user_input = st.text_input(
            "描述或输入函数",
            value=default_val,
            help="例如：1/x、sin(x)+x**2、Abs(x)、x**2+y**2，也可以输入中文描述。",
        )

        st.markdown("### 绘图区间")

        if is_3d:
            xy_range = st.slider("x/y 范围", 2, 30, 8)
            grid_size = st.slider("网格精度", 41, 161, 101, step=20)
            st.markdown("### 指定点计算")
            eval_x = st.number_input("x", value=1.0, format="%.6f")
            eval_y = st.number_input("y", value=1.0, format="%.6f")
        else:
            x_range = st.slider("x 范围", 2, 30, 10)
            plot_style = st.radio("2D 显示方式", ["分标签显示", "同图显示"], index=0)
            st.markdown("### 指定点计算")
            eval_x = st.number_input("x", value=1.0, format="%.6f")

            st.markdown("### 图层显示")
            show_f = st.checkbox("函数 f(x)", value=True)
            show_deriv = st.checkbox("导函数 f'(x)", value=False)
            show_integral = st.checkbox("原函数 F(x)", value=False)


st.title("基于 DeepSeek V3 的微积分绘图 Agent")
st.caption("支持自然语言翻译、求导、积分、梯度、曲率、指定点计算、一般曲线/曲面方程与交互式绘图。")

with st.expander("功能介绍与输入示例", expanded=True):
    st.markdown(
        """
这个应用用于把函数描述转成可计算的数学表达式，并自动完成微积分分析与交互式绘图。没有配置 DeepSeek API 时，也可以直接输入标准公式进行计算。

- 一元函数：绘制 `f(x)`，可选显示导函数 `f'(x)` 和原函数 `F(x)`，计算一阶导、二阶导、不定积分、曲率，并在指定 `x` 处返回对应数值。
- 二元函数：绘制 `z=f(x,y)` 的 3D 曲面，计算偏导、梯度模、高斯曲率、平均曲率，并在指定 `(x,y)` 处返回函数值和曲率等数值。
- 一般曲线/曲面方程：支持 `F(x,y)=0` 平面曲线和 `F(x,y,z)=0` 空间曲面，例如 `x**2+y**2=4`、`x**2+y**2+z**2=9`；配置 DeepSeek 后也可以输入“半径为 3 的球面”“单位圆”。
- 间断函数：`1/x`、`tan(x)` 等会自动在渐近线附近断开，避免画出错误的竖线。
- 绝对值：支持 `Abs(x)`、`abs(x)`、`|x|`、`｜x｜`、`绝对值x`、`x的绝对值`。
"""
    )


def render_controls(is_3d: bool) -> None:
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])

    with c1:
        if st.button("放大", use_container_width=True):
            st.session_state.zoom_val *= 0.7
            st.session_state.needs_camera_sync = True
            st.rerun()

    with c2:
        if st.button("缩小", use_container_width=True):
            st.session_state.zoom_val *= 1.4
            st.session_state.needs_camera_sync = True
            st.rerun()

    with c3:
        if is_3d:
            target = "平移" if st.session_state.drag_mode == "turntable" else "旋转"
            if st.button(f"切换到{target}", use_container_width=True):
                st.session_state.drag_mode = "pan" if st.session_state.drag_mode == "turntable" else "turntable"
                st.rerun()
        else:
            st.button("2D 默认平移", disabled=True, use_container_width=True)

    with c4:
        if st.button("重置", use_container_width=True):
            st.session_state.zoom_val = 1.0
            st.session_state.drag_mode = "turntable"
            st.session_state.needs_camera_sync = True
            st.rerun()


def render_2d_plot(items: list[tuple[sp.Expr, str, str]], x_range: int, key: str) -> None:
    fig = engine.generate_2d_plot(items, -x_range, x_range)

    if st.session_state.needs_camera_sync:
        z = st.session_state.zoom_val
        fig.update_layout(xaxis=dict(range=[-x_range * z, x_range * z]))
        st.session_state.needs_camera_sync = False

    fig.update_layout(height=560, uirevision="constant")
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
        key=key,
    )


if user_input:
    try:
        if is_implicit:
            expr, equation_text, source = parse_implicit_or_translate(user_input, is_surface)
            st.success(f"解析来源：{source}；输入/翻译方程：`{equation_text}`；内部形式：`{sp.sstr(expr)} = 0`")
            st.latex(rf"F= {sp.latex(expr)} = 0")
            st.markdown("---")

            render_controls(is_surface)
            if is_surface:
                fig = engine.generate_implicit_surface_plot(expr, -implicit_range, implicit_range, implicit_grid)
                values = engine.evaluate_implicit_at(expr, {engine.x: eval_x, engine.y: eval_y, engine.z: eval_z})
            else:
                fig = engine.generate_implicit_curve_plot(expr, -implicit_range, implicit_range, implicit_grid)
                values = engine.evaluate_implicit_at(expr, {engine.x: eval_x, engine.y: eval_y})

            if fig is None:
                st.error("该方程暂时无法生成图像，可能存在数值溢出或变量不匹配。")
            else:
                fig.update_layout(height=680, uirevision="constant")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

            st.markdown("### 指定点方程分析")
            render_value_table(values)

        else:
            expr, formula, source = parse_or_translate(user_input, is_3d)

            st.success(f"解析来源：{source}；内部公式：`{formula}`")
            st.latex(rf"f({'x,y' if is_3d else 'x'}) = {sp.latex(expr)}")
            st.markdown("---")

            render_controls(is_3d)

            if is_3d:
                fig = engine.generate_3d_plot(expr, -xy_range, xy_range, grid_size)

                if fig is None:
                    st.error("该函数暂时无法生成 3D 图像，可能存在过多奇点或数值溢出。")
                else:
                    if st.session_state.needs_camera_sync:
                        z = st.session_state.zoom_val
                        fig.update_layout(scene_camera=dict(eye=dict(x=1.5 * z, y=1.5 * z, z=1.5 * z)))
                        st.session_state.needs_camera_sync = False

                    fig.update_layout(scene_dragmode=st.session_state.drag_mode, height=720, uirevision="constant")
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": True, "scrollZoom": True},
                    )

                analysis = engine.get_analysis_3d(expr)
                st.markdown("### 二元函数分析")
                st.latex(rf"f_x={sp.latex(analysis['fx'])}\quad f_y={sp.latex(analysis['fy'])}")
                st.latex(rf"|\nabla f|={sp.latex(analysis['grad_norm'])}")
                st.latex(rf"K={sp.latex(analysis['gaussian'])}\quad H={sp.latex(analysis['mean'])}")
                st.caption("K 为高斯曲率，H 为平均曲率。")
                st.markdown("### 指定点函数分析")
                render_value_table(engine.evaluate_3d_at(expr, eval_x, eval_y))
            else:
                derivative, second_derivative, integral, curvature = engine.get_analysis_2d(expr)
                items: list[tuple[sp.Expr, str, str]] = []

                if show_f:
                    items.append((expr, "f(x)", "#1f77b4"))
                if show_deriv:
                    items.append((derivative, "f'(x)", "#d62728"))
                if show_integral:
                    items.append((integral, "F(x)", "#ff7f0e"))

                if not items:
                    st.info("请至少勾选一个图层。")
                elif plot_style == "同图显示":
                    render_2d_plot(items, x_range, "plot2d_all")
                else:
                    tabs = st.tabs([label for _, label, _ in items])
                    for tab, item in zip(tabs, items):
                        with tab:
                            render_2d_plot([item], x_range, f"plot2d_{item[1]}")

                st.markdown("### 一元函数分析")
                st.latex(rf"f'(x)={sp.latex(derivative)}")
                st.latex(rf"f''(x)={sp.latex(second_derivative)}")
                st.latex(rf"F(x)=\int f(x)\,dx={sp.latex(integral)}")
                st.latex(rf"\kappa(x)={sp.latex(curvature)}")
                st.markdown("### 指定点函数分析")
                render_value_table(engine.evaluate_1d_at(expr, eval_x))
    except Exception as exc:
        st.error(f"处理失败：{exc}")
        st.info("可以试试标准表达式，例如 `1/x`、`sin(x)+x**2`、`Abs(x)`、`x**2+y**2`、`x**2+y**2=4`。")


