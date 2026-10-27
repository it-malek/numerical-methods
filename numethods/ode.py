"""Initial-value ODE solvers: Euler, midpoint, RK4, adaptive RK45.

All solvers integrate ``dy/dt = f(t, y)`` with ``y(t0) = y0``. They handle
both scalar and vector-valued ``y`` uniformly: the input ``y0`` is converted
to a NumPy array, and the output ``y`` has shape ``(n_steps, len(y0))`` if
``y0`` was vector-valued, ``(n_steps,)`` if scalar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._utils import ConvergenceError


@dataclass
class ODESolution:
    """Result of an IVP integration.

    Attributes:
        t: 1-D array of time points, shape ``(n,)``.
        y: Solution values. Shape ``(n,)`` for scalar problems,
            ``(n, d)`` for d-dimensional systems.
        n_evaluations: Number of times the right-hand side was evaluated.
        method: Solver name.
        success: True iff the solver completed without error.
    """

    t: np.ndarray
    y: np.ndarray
    n_evaluations: int
    method: str = ""
    success: bool = True


def _prepare(t_span: tuple[float, float], y0, n_steps: int):
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    t0, tf = t_span
    if tf <= t0:
        raise ValueError(f"t_span must satisfy t0 < tf, got {t_span}")
    y0_arr = np.asarray(y0, dtype=float)
    scalar = y0_arr.ndim == 0
    if scalar:
        y0_arr = y0_arr.reshape(1)
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    Y = np.empty((n_steps + 1, y0_arr.size))
    Y[0] = y0_arr
    return t, Y, h, scalar


def _finalize(t: np.ndarray, Y: np.ndarray, scalar: bool, n_eval: int, method: str) -> ODESolution:
    y = Y[:, 0] if scalar else Y
    return ODESolution(t=t, y=y, n_evaluations=n_eval, method=method, success=True)


def euler(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0,
    n_steps: int,
) -> ODESolution:
    """Explicit (forward) Euler. Global truncation error: O(h)."""
    t, Y, h, scalar = _prepare(t_span, y0, n_steps)
    for i in range(n_steps):
        Y[i + 1] = Y[i] + h * np.asarray(f(t[i], Y[i] if not scalar else Y[i, 0]), dtype=float).reshape(-1)
    return _finalize(t, Y, scalar, n_eval=n_steps, method="euler")


def midpoint(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0,
    n_steps: int,
) -> ODESolution:
    """Midpoint method (RK2). Global error: O(h^2)."""
    t, Y, h, scalar = _prepare(t_span, y0, n_steps)
    for i in range(n_steps):
        yi = Y[i] if not scalar else Y[i, 0]
        k1 = np.asarray(f(t[i], yi), dtype=float).reshape(-1)
        y_mid = Y[i] + 0.5 * h * k1
        ymid_arg = y_mid if not scalar else y_mid[0]
        k2 = np.asarray(f(t[i] + 0.5 * h, ymid_arg), dtype=float).reshape(-1)
        Y[i + 1] = Y[i] + h * k2
    return _finalize(t, Y, scalar, n_eval=2 * n_steps, method="midpoint")


def rk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0,
    n_steps: int,
) -> ODESolution:
    """Classical 4-stage Runge-Kutta. Global error: O(h^4)."""
    t, Y, h, scalar = _prepare(t_span, y0, n_steps)
    for i in range(n_steps):
        yi = Y[i] if not scalar else Y[i, 0]
        k1 = np.asarray(f(t[i], yi), dtype=float).reshape(-1)
        y2 = Y[i] + 0.5 * h * k1
        k2 = np.asarray(f(t[i] + 0.5 * h, y2 if not scalar else y2[0]), dtype=float).reshape(-1)
        y3 = Y[i] + 0.5 * h * k2
        k3 = np.asarray(f(t[i] + 0.5 * h, y3 if not scalar else y3[0]), dtype=float).reshape(-1)
        y4 = Y[i] + h * k3
        k4 = np.asarray(f(t[i] + h, y4 if not scalar else y4[0]), dtype=float).reshape(-1)
        Y[i + 1] = Y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _finalize(t, Y, scalar, n_eval=4 * n_steps, method="rk4")


# Dormand-Prince (5(4)) coefficients - same Butcher tableau used by SciPy's RK45.
_DP_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DP_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_DP_E = np.array(
    [
        71 / 57600,
        0.0,
        -71 / 16695,
        71 / 1920,
        -17253 / 339200,
        22 / 525,
        -1 / 40,
    ]
)


def rk45_adaptive(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float | None = None,
    max_steps: int = 100_000,
) -> ODESolution:
    """Adaptive Dormand-Prince 5(4) with proportional step-size control.

    Forward integration only: finite ``t0 < tf`` and finite positive ``h0``
    (or None). ``y0`` is a finite scalar or nonempty vector. The RHS returns
    the same number of components. Scalar ``rtol`` and ``atol`` must be finite
    and positive. A step passes when the RMS of the embedded error divided
    by ``atol + rtol*max(abs(y), abs(y_new))`` is at most one. This local
    estimate is not a guaranteed global error bound.

    ``max_steps`` is a positive integer counting accepted *and* rejected
    attempts. Endpoint arrival on the last attempt succeeds. Nonfinite RHS,
    stages or error arithmetic, budget exhaustion and inability to advance
    at representable time spacing raise ``ConvergenceError`` with the accepted
    trajectory (``success=False``). Invalid inputs raise ``ValueError``.
    """
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, (bool, np.bool_)) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    t0, tf = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(tf) or tf <= t0:
        raise ValueError(f"t_span must have finite t0 < tf, got {t_span}")
    y_arr = np.asarray(y0, dtype=float)
    scalar = y_arr.ndim == 0
    if y_arr.ndim > 1 or y_arr.size == 0 or not np.all(np.isfinite(y_arr)):
        raise ValueError("y0 must be a finite scalar or nonempty vector")
    y = np.atleast_1d(y_arr).copy()
    if h0 is None:
        # Scaling endpoints first avoids overflow in tf-t0 on wide intervals.
        h0 = max(tf * 0.01 - t0 * 0.01, float(np.nextafter(t0, tf) - t0))
    if not np.isscalar(h0) or not np.isfinite(h0) or h0 <= 0:
        raise ValueError("h0 must be finite and positive")
    h, t = float(h0), t0
    ts, ys = [t], [y.copy()]
    n_eval = 0

    def solution(success):
        Y = np.asarray(ys)
        return ODESolution(np.asarray(ts), Y[:, 0] if scalar else Y,
                           n_eval, "rk45_adaptive", success)

    def failure(reason):
        return ConvergenceError(f"rk45_adaptive: {reason}", result=solution(False))

    def evaluate(time, state):
        nonlocal n_eval
        if not np.all(np.isfinite(state)) or not np.isfinite(time):
            raise failure("nonfinite stage")
        n_eval += 1
        value = np.asarray(f(time, state if not scalar else state[0]), dtype=float).reshape(-1)
        if value.size != y.size:
            raise ValueError("RHS output size must match y0")
        if not np.all(np.isfinite(value)):
            raise failure("nonfinite RHS output")
        return value

    for _ in range(max_steps):
        # Time resolution depends on t. Never force-accept a rejected step.
        spacing = float(np.nextafter(t, tf) - t)
        h = min(h, tf - t)
        if h < spacing or t + h <= t:
            raise failure("step-size underflow at representable time spacing")
        t_new = min(t + h, tf)
        h = t_new - t  # Tableau and error estimate use the actual time advance.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                k = np.zeros((7, y.size))
                k[0] = evaluate(t, y)
                for stage in range(1, 7):
                    yi = y + h * sum(_DP_A[stage][j] * k[j] for j in range(stage))
                    k[stage] = evaluate(t + _DP_C[stage] * h, yi)
                y_new = y + h * (_DP_B5 @ k)
                err_vec = h * (_DP_E @ k)
                sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
                # Stable RMS avoids squaring very large or tiny ratios.
                err = float(np.hypot.reduce(err_vec / sc) / np.sqrt(y.size))
                if not np.all(np.isfinite(y_new)) or not np.isfinite(err):
                    raise failure("nonfinite state or error estimate")
        except (FloatingPointError, OverflowError) as exc:
            raise failure("nonfinite RHS, stage or error arithmetic") from exc
        if err <= 1.0:
            t, y = t_new, y_new
            ts.append(t)
            ys.append(y.copy())
            if t == tf:
                return solution(True)
        factor = 5.0 if err == 0 else min(5.0, max(0.1, 0.9 * err ** (-0.2)))
        h *= factor
        if err > 1.0:
            # Multiplication may shrink h without changing its rounded
            # endpoint. A rejected attempt must make representable progress
            # toward a smaller step, or report that none is available.
            if t + h >= t_new:
                h = float(np.nextafter(t_new, t) - t)
            if h < spacing or t + h <= t:
                raise failure("step-size underflow: error control needs a smaller step")
    raise failure(f"did not reach tf in {max_steps} steps")
