"""Symbolic computation and Plotly rendering engine for the calculus drawing agent."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.calculus.util import continuous_domain
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
    """Convert |x+1| style input into Abs(x+1)."""
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

    if stack:
        return expr

    return "".join(out)


def _replace_chinese_abs(expr: str) -> str:
    """Convert common Chinese absolute-value descriptions into Abs(...)."""
    s = expr.strip()
    atom = r"(?:[xy]|\d+(?:\.\d+)?|\([^()]+\))"

    s = re.sub(rf"绝对值\s*({atom})", r"Abs(\1)", s)
    s = re.sub(rf"({atom})\s*的绝对值", r"Abs(\1)", s)
    s = re.sub(rf"abs\s*({atom})", r"Abs(\1)", s, flags=re.I)

    return s


def normalize_formula(expr_str: str) -> str:
    """Normalize user/model formula text before SymPy parsing."""
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
        "｜": "|",
        "∣": "|",
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
    s = _replace_chinese_abs(s)
    s = _replace_abs_bars(s)

    return s.strip()


@lru_cache(maxsize=128)
def cached_parse(expr_str: str) -> sp.Expr:
    """Parse a string into a SymPy expression."""
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
    """For y=f(x): first derivative, second derivative, integral, curvature."""
    derivative = sp.diff(expr, X).doit()
    second_derivative = sp.diff(derivative, X).doit()
    integral = sp.integrate(expr, X).doit()
    curvature = sp.simplify(
        sp.Abs(second_derivative) / (1 + derivative**2) ** sp.Rational(3, 2)
    )

    return derivative, second_derivative, integral, curvature


@lru_cache(maxsize=128)
def cached_analysis_3d(expr: sp.Expr) -> dict[str, sp.Expr]:
    """For z=f(x,y): partials, gradient norm, Gaussian and mean curvature."""
    fx = sp.diff(expr, X).doit()
    fy = sp.diff(expr, Y).doit()

    fxx = sp.diff(fx, X).doit()
    fxy = sp.diff(fx, Y).doit()
    fyy = sp.diff(fy, Y).doit()

    grad_norm = sp.sqrt(fx**2 + fy**2)
    gaussian = sp.simplify((fxx * fyy - fxy**2) / (1 + fx**2 + fy**2) ** 2)
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
    """Core symbolic calculation and Plotly rendering engine."""

    def __init__(self) -> None:
        self.x, self.y = X, Y

    def parse_expression(self, expr_str: str) -> sp.Expr:
        return cached_parse(expr_str)

    def validate_dimension(self, expr: sp.Expr, is_3d: bool) -> None:
        allowed = {self.x, self.y} if is_3d else {self.x}
        extra = set(expr.free_symbols) - allowed

        if extra:
            names = ", ".join(sorted(str(s) for s in extra))
            raise ValueError(f"当前模式不支持变量：{names}")

        if not is_3d and self.y in expr.free_symbols:
            raise ValueError("一元函数模式只能使用变量 x。")

    def _fix_real_roots(self, expr: sp.Expr) -> sp.Expr:
        """Make odd rational powers and log render nicely on real-valued plots."""
        if not hasattr(expr, "atoms"):
            return expr

        fixed = expr.replace(sp.log, lambda *args: sp.log(sp.Abs(args[0]), *args[1:]))

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

    @staticmethod
    def _break_large_jumps(y_vals: np.ndarray, factor: float = 8.0) -> np.ndarray:
        """Break lines around sudden numeric jumps to avoid fake vertical segments."""
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

    @staticmethod
    def _set_nan_near_breaks(
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        break_points: Iterable[float],
    ) -> np.ndarray:
        y = y_vals.copy()

        for point in break_points:
            spacing = np.min(np.abs(x_vals - point))
            width = max(spacing * 1.5, 1e-8)
            y[np.abs(x_vals - point) <= width] = np.nan

        return y

    def _domain_break_points(
        self,
        expr: sp.Expr,
        x_min: float,
        x_max: float,
    ) -> list[float]:
        """Find likely one-dimensional discontinuity points inside the visible range."""
        points: set[float] = set()
        interval = sp.Interval(float(x_min), float(x_max))

        try:
            domain = continuous_domain(expr, self.x, interval)
            intervals = domain.args if isinstance(domain, sp.Union) else (domain,)

            for part in intervals:
                if not isinstance(part, sp.Interval):
                    continue

                for end in (part.start, part.end):
                    if end in (-sp.oo, sp.oo):
                        continue

                    val = float(end.evalf())
                    if x_min < val < x_max:
                        points.add(val)
        except Exception:
            pass

        try:
            denominator = sp.denom(sp.together(expr))
            roots = sp.solveset(denominator, self.x, domain=interval)

            for root in roots:
                if root.is_real:
                    val = float(root.evalf())
                    if x_min < val < x_max:
                        points.add(val)
        except Exception:
            pass

        return sorted(points)

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
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        fixed = self._fix_real_roots(expr)

        if fixed.is_constant():
            return np.full_like(x_grid, float(fixed), dtype=float)

        f_np = sp.lambdify(
            (self.x, self.y),
            fixed,
            modules=[{"Abs": np.abs, "sign": np.sign}, "numpy"],
        )

        with np.errstate(all="ignore"):
            z = self._broadcast_scalar(f_np(x_grid, y_grid), x_grid)

        z[~np.isfinite(z)] = np.nan

        return z

    def get_analysis_2d(self, expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        return cached_analysis_2d(expr)

    def get_analysis_3d(self, expr: sp.Expr) -> dict[str, sp.Expr]:
        return cached_analysis_3d(expr)

    def generate_2d_plot(
        self,
        expr_list: Iterable[tuple[sp.Expr, str, str]],
        x_min: float = -10,
        x_max: float = 10,
        clip_value: float = 80,
    ) -> go.Figure:
        """Generate robust 2D plots for ordinary and discontinuous functions."""
        items = list(expr_list)
        fig = go.Figure()

        break_points: set[float] = set()
        for expr, _, _ in items:
            break_points.update(self._domain_break_points(expr, x_min, x_max))

        x_main = np.linspace(x_min, x_max, 1601)
        local_samples = [
            np.linspace(point - 1e-3, point + 1e-3, 301)
            for point in sorted(break_points)
        ]
        x_vals = np.sort(np.unique(np.concatenate([x_main, *local_samples]))) if local_samples else x_main

        plot_arrays: list[np.ndarray] = []

        for expr, label, color in items:
            try:
                y_vals = self._eval_1d(expr, x_vals)
                y_vals = self._set_nan_near_breaks(x_vals, y_vals, break_points)
                y_vals = self._break_large_jumps(y_vals)

                y_plot = y_vals.copy()
                y_plot[np.abs(y_plot) > clip_value] = np.nan
            except Exception:
                continue

            try:
                derivative, second_derivative, _, _ = self.get_analysis_2d(expr)
                yp = self._eval_1d(derivative, x_vals)
                ypp = self._eval_1d(second_derivative, x_vals)
                yp = self._set_nan_near_breaks(x_vals, yp, break_points)
                ypp = self._set_nan_near_breaks(x_vals, ypp, break_points)

                with np.errstate(all="ignore"):
                    curvature_vals = np.abs(ypp) / (1 + yp**2) ** 1.5

                curvature_vals[~np.isfinite(curvature_vals)] = np.nan
            except Exception:
                curvature_vals = np.full_like(x_vals, np.nan, dtype=float)

            finite_mask = np.isfinite(y_plot)
            segment_edges = np.flatnonzero(np.diff(finite_mask.astype(int)) != 0) + 1
            segments = np.split(np.arange(len(x_vals)), segment_edges)
            shown_legend = False

            for segment in segments:
                if segment.size < 2 or not finite_mask[segment].all():
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=x_vals[segment],
                        y=y_plot[segment],
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=2.5),
                        customdata=curvature_vals[segment],
                        hovertemplate=(
                            "<b>%{name}</b><br>"
                            "x=%{x:.4f}<br>"
                            "y=%{y:.4f}<br>"
                            "曲率 k=%{customdata:.4f}"
                            "<extra></extra>"
                        ),
                        connectgaps=False,
                        legendgroup=label,
                        showlegend=not shown_legend,
                    )
                )
                shown_legend = True

            finite_y = y_plot[np.isfinite(y_plot)]
            if finite_y.size > 0:
                plot_arrays.append(finite_y)

        if plot_arrays:
            all_y = np.concatenate(plot_arrays)
            y_low, y_high = np.percentile(all_y, [2, 98])

            if abs(y_high - y_low) < 1e-6:
                y_low -= 1
                y_high += 1

            pad = max((y_high - y_low) * 0.15, 1.0)
            y_range = [float(y_low - pad), float(y_high + pad)]

            if break_points and np.nanmax(np.abs(all_y)) > 8:
                y_range = [min(y_range[0], -10.0), max(y_range[1], 10.0)]
        else:
            y_range = [-10, 10]

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(range=[x_min, x_max], zeroline=True, gridcolor="#f0f0f0"),
            yaxis=dict(range=y_range, zeroline=True, gridcolor="#f0f0f0"),
            dragmode="pan",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    def generate_3d_plot(
        self,
        expr: sp.Expr,
        xy_min: float = -8,
        xy_max: float = 8,
        grid_size: int = 101,
    ) -> go.Figure | None:
        """Generate a transparent light-blue surface with pale-gray curve grid lines."""
        t = np.linspace(xy_min, xy_max, grid_size)
        x_grid, y_grid = np.meshgrid(t, t)

        try:
            z = self._eval_2d(expr, x_grid, y_grid)
            z_plot = np.where(np.abs(z) < 200, z, np.nan)

            z_for_grad = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            gy, gx = np.gradient(z_for_grad, t, t)
            grad_norm = np.sqrt(gx**2 + gy**2)
            custom = np.stack((gx, gy, grad_norm), axis=-1)
        except Exception:
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_plot,
                customdata=custom,
                surfacecolor=np.zeros_like(z_plot, dtype=float),
                colorscale=[[0.0, "#8fd8ff"], [1.0, "#8fd8ff"]],
                cmin=0,
                cmax=1,
                showscale=False,
                opacity=0.58,
                lighting=dict(
                    ambient=0.95,
                    diffuse=0.45,
                    specular=0.05,
                    roughness=1.0,
                    fresnel=0.02,
                ),
                hovertemplate=(
                    "<b>x</b>: %{x:.2f}<br>"
                    "<b>y</b>: %{y:.2f}<br>"
                    "<b>z</b>: %{z:.2f}<br>"
                    "<b>fx</b>: %{customdata[0]:.2f}<br>"
                    "<b>fy</b>: %{customdata[1]:.2f}<br>"
                    "<b>|grad f|</b>: %{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

        step = max(grid_size // 10, 6)
        line_style = dict(color="rgba(0,0,0,0.42)", width=2.0)

        for idx in range(0, grid_size, step):
            fig.add_trace(
                go.Scatter3d(
                    x=x_grid[idx, :],
                    y=y_grid[idx, :],
                    z=z_plot[idx, :],
                    mode="lines",
                    line=line_style,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=x_grid[:, idx],
                    y=y_grid[:, idx],
                    z=z_plot[:, idx],
                    mode="lines",
                    line=line_style,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig.update_layout(
            paper_bgcolor="white",
            scene=dict(
                xaxis=dict(
                    range=[xy_min, xy_max],
                    backgroundcolor="white",
                    gridcolor="rgba(220,220,220,0.25)",
                    zerolinecolor="rgba(180,180,180,0.35)",
                ),
                yaxis=dict(
                    range=[xy_min, xy_max],
                    backgroundcolor="white",
                    gridcolor="rgba(220,220,220,0.25)",
                    zerolinecolor="rgba(180,180,180,0.35)",
                ),
                zaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(220,220,220,0.20)",
                    zerolinecolor="rgba(180,180,180,0.35)",
                ),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        return fig





