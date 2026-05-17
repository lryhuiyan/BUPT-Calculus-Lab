"""Symbolic computation and plotting engine for the calculus drawing agent."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X, Y = sp.symbols("x y", real=True)

_ALLOWED_FUNCS = {
    "x": X,
    "y": Y,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "I": sp.I,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "sign": sp.sign,
    "Piecewise": sp.Piecewise,
}

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
    implicit_application,
    function_exponentiation,
)


def _replace_abs_bars(expr: str) -> str:
    """Convert simple math bars like |x+1| into Abs(x+1)."""
    out: list[str] = []
    stack: list[int] = []

    for ch in expr:
        if ch == "|":
            if stack:
                start = stack.pop()
                inside = "".join(out[start:])
                out[start:] = [f"Abs({inside})"]
            else:
                stack.append(len(out))
        else:
            out.append(ch)

    # 如果用户输入了不成对的 |，不要强行猜，保留原式让解析报错
    if stack:
        return expr

    return "".join(out)


def normalize_formula(expr_str: str) -> str:
    """Normalize common user/model notations before parsing."""
    if not expr_str:
        raise ValueError("表达式为空。")

    s = str(expr_str).strip()

    replacements = {
        "（": "(",
        "）": ")",
        "，": ",",
        "×": "*",
        "÷": "/",
        "−": "-",
        "^": "**",
        "ln(": "log(",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"(?:math|np|numpy|sp|sympy)\.", "", s)
    s = re.sub(
        r"^(?:[zfgy]\s*=|f\s*\([^)]*\)\s*=|z\s*\([^)]*\)\s*=)",
        "",
        s,
        flags=re.I,
    )
    s = s.replace("\\cdot", "*").replace("\\", "")
    s = _replace_abs_bars(s)

    return s.strip()


@lru_cache(maxsize=128)
def cached_parse(expr_str: str) -> sp.Expr:
    clean_str = normalize_formula(expr_str)

    try:
        return parse_expr(
            clean_str,
            transformations=_TRANSFORMATIONS,
            local_dict=_ALLOWED_FUNCS,
            global_dict={"__builtins__": {}, **sp.__dict__},
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"无法解析表达式：{clean_str}") from exc


@lru_cache(maxsize=128)
def cached_analysis_2d(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    derivative = sp.diff(expr, X).doit()
    second_derivative = sp.diff(derivative, X).doit()
    integral = sp.integrate(expr, X).doit()
    curvature = sp.simplify(
        sp.Abs(second_derivative) / (1 + derivative**2) ** sp.Rational(3, 2)
    )

    return derivative, second_derivative, integral, curvature


@lru_cache(maxsize=128)
def cached_analysis_3d(expr: sp.Expr) -> dict[str, sp.Expr]:
    fx = sp.diff(expr, X).doit()
    fy = sp.diff(expr, Y).doit()

    fxx = sp.diff(fx, X).doit()
    fxy = sp.diff(fx, Y).doit()
    fyy = sp.diff(fy, Y).doit()

    grad_norm = sp.sqrt(fx**2 + fy**2)

    # 曲面 z=f(x,y) 的高斯曲率和平均曲率
    gaussian = sp.simplify(
        (fxx * fyy - fxy**2) / (1 + fx**2 + fy**2) ** 2
    )

    mean = sp.simplify(
        (
            (1 + fy**2) * fxx
            - 2 * fx * fy * fxy
            + (1 + fx**2) * fyy
        )
        / (2 * (1 + fx**2 + fy**2) ** sp.Rational(3, 2))
    )

    return {
        "fx": fx,
        "fy": fy,
        "fxx": fxx,
        "fxy": fxy,
        "fyy": fyy,
        "grad_norm": grad_norm,
        "gaussian": gaussian,
        "mean": mean,
    }


class MathEngine:
    """Core symbolic math and Plotly rendering engine."""

    def __init__(self) -> None:
        self.x, self.y = X, Y

    def parse_expression(self, expr_str: str) -> sp.Expr:
        return cached_parse(expr_str)

    def variables_in(self, expr: sp.Expr) -> set[sp.Symbol]:
        return set(expr.free_symbols)

    def validate_dimension(self, expr: sp.Expr, is_3d: bool) -> None:
        allowed = {self.x, self.y} if is_3d else {self.x}
        extra = self.variables_in(expr) - allowed

        if extra:
            names = ", ".join(sorted(str(s) for s in extra))
            raise ValueError(f"当前模式不支持变量：{names}")

        if not is_3d and self.y in expr.free_symbols:
            raise ValueError("一元函数模式只能使用变量 x。")

    def _fix_real_roots(self, expr: sp.Expr) -> sp.Expr:
        """Keep odd-denominator rational powers real on negative inputs where possible."""
        if not hasattr(expr, "atoms"):
            return expr

        fixed = expr.replace(
            sp.log,
            lambda *args: sp.log(sp.Abs(args[0]), *args[1:]),
        )

        for p in list(fixed.atoms(sp.Pow)):
            base, exp = p.as_base_exp()

            if exp.is_Rational and exp.q % 2 == 1:
                if exp.p % 2 == 0:
                    replacement = sp.Abs(base) ** exp
                else:
                    replacement = sp.sign(base) * sp.Abs(base) ** exp

                fixed = fixed.xreplace({p: replacement})

        return fixed

    @staticmethod
    def _broadcast_scalar(val, target_array: np.ndarray) -> np.ndarray:
        arr = np.asarray(val, dtype=float)

        if arr.ndim == 0:
            return np.full_like(target_array, float(arr), dtype=float)

        return arr.astype(float)

    def _eval_1d(self, expr: sp.Expr, x_vals: np.ndarray) -> np.ndarray:
        fixed = self._fix_real_roots(expr)

        if fixed.is_constant():
            return np.full_like(x_vals, float(fixed), dtype=float)

        f_np = sp.lambdify(
            self.x,
            fixed,
            modules=[{"Abs": np.abs, "sign": np.sign}, "numpy"],
        )

        with np.errstate(all="ignore"):
            y_vals = self._broadcast_scalar(f_np(x_vals), x_vals)

        y_vals[~np.isfinite(y_vals)] = np.nan

        return y_vals

    def _eval_2d(
        self,
        expr: sp.Expr,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
    ) -> np.ndarray:
        fixed = self._fix_real_roots(expr)

        if fixed.is_constant():
            return np.full_like(X_grid, float(fixed), dtype=float)

        f_np = sp.lambdify(
            (self.x, self.y),
            fixed,
            modules=[{"Abs": np.abs, "sign": np.sign}, "numpy"],
        )

        with np.errstate(all="ignore"):
            Z = self._broadcast_scalar(f_np(X_grid, Y_grid), X_grid)

        Z[~np.isfinite(Z)] = np.nan

        return Z

    @staticmethod
    def _break_large_jumps(y_vals: np.ndarray, factor: float = 8.0) -> np.ndarray:
        """Insert NaNs where neighboring samples jump too hard, avoiding fake vertical lines."""
        y = y_vals.copy()
        finite = np.isfinite(y)

        if finite.sum() < 4:
            return y

        diffs = np.abs(np.diff(y))
        finite_diffs = diffs[np.isfinite(diffs)]

        if finite_diffs.size == 0:
            return y

        threshold = max(np.nanmedian(finite_diffs) * factor, 50.0)
        jump_idx = np.where(diffs > threshold)[0] + 1
        y[jump_idx] = np.nan

        return y

    def get_analysis_2d(self, expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        return cached_analysis_2d(expr)

    def get_analysis_3d(self, expr: sp.Expr) -> dict[str, sp.Expr]:
        return cached_analysis_3d(expr)

    def generate_2d_plot(
        self,
        expr_list: Iterable[tuple[sp.Expr, str, str]],
        x_min: float = -10,
        x_max: float = 10,
    ) -> go.Figure:
        fig = go.Figure()

        x_main = np.linspace(x_min, x_max, 1601)
        x_micro = np.linspace(-1e-4, 1e-4, 301)
        x_vals = np.sort(np.unique(np.concatenate([x_main, x_micro])))

        for expr, label, color in expr_list:
            try:
                y_vals = self._break_large_jumps(self._eval_1d(expr, x_vals))
            except Exception:
                continue

            try:
                derivative, second_derivative, _, _ = self.get_analysis_2d(expr)
                yp = self._eval_1d(derivative, x_vals)
                ypp = self._eval_1d(second_derivative, x_vals)

                with np.errstate(all="ignore"):
                    curvature_vals = np.abs(ypp) / (1 + yp**2) ** 1.5

                curvature_vals[~np.isfinite(curvature_vals)] = np.nan

            except Exception:
                curvature_vals = np.full_like(x_vals, np.nan, dtype=float)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2.4),
                    customdata=curvature_vals,
                    hovertemplate=(
                        "<b>%{name}</b><br>"
                        "x=%{x:.4f}<br>"
                        "y=%{y:.4f}<br>"
                        "曲率 κ=%{customdata:.4f}"
                        "<extra></extra>"
                    ),
                    connectgaps=False,
                )
            )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                range=[x_min, x_max],
                zeroline=True,
                gridcolor="#eeeeee",
            ),
            yaxis=dict(
                zeroline=True,
                gridcolor="#eeeeee",
            ),
            dragmode="pan",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    def generate_3d_plot(
        self,
        expr: sp.Expr,
        xy_min: float = -8,
        xy_max: float = 8,
        grid_size: int = 101,
    ) -> go.Figure | None:
        t = np.linspace(xy_min, xy_max, grid_size)
        X_grid, Y_grid = np.meshgrid(t, t)

        try:
            Z = self._eval_2d(expr, X_grid, Y_grid)
            Z_plot = np.where(np.abs(Z) < 500, Z, np.nan)

            GY, GX = np.gradient(Z, t, t)
            grad_norm = np.sqrt(GX**2 + GY**2)
            custom = np.stack((GX, GY, grad_norm), axis=-1)

        except Exception:
            return None

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_plot,
                    customdata=custom,
                    colorscale="Viridis",
                    opacity=0.92,
                    contours=dict(
                        z=dict(
                            show=True,
                            usecolormap=True,
                            project_z=True,
                        )
                    ),
                    hovertemplate=(
                        "x=%{x:.3f}<br>"
                        "y=%{y:.3f}<br>"
                        "z=%{z:.3f}"
                        "<br>fx≈%{customdata[0]:.3f}"
                        "<br>fy≈%{customdata[1]:.3f}"
                        "<br>|∇f|≈%{customdata[2]:.3f}"
                        "<extra></extra>"
                    ),
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor="white",
            scene=dict(
                xaxis=dict(range=[xy_min, xy_max]),
                yaxis=dict(range=[xy_min, xy_max]),
                zaxis=dict(title="z"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        return fig
