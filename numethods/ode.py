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
    """Adaptive Runge-Kutta 4(5) (Dormand-Prince) with PI step-size control.

    Args:
        f: Right-hand side ``f(t, y)``.
        t_span: ``(t0, tf)`` with ``t0 < tf``.
        y0: Initial state (scalar or vector).
        rtol: Relative tolerance.
        atol: Absolute tolerance (component-wise).
        h0: Initial step (auto-chosen if None).
        max_steps: Cap on accepted+rejected steps to prevent infinite loops.

    Returns:
        ODESolution with non-uniform ``t`` (the points actually accepted).

    Raises:
        ConvergenceError: If ``max_steps`` is reached.
    """
    if rtol <= 0 or atol <= 0:
        raise ValueError("rtol and atol must be positive")
    t0, tf = t_span
    if tf <= t0:
        raise ValueError(f"t_span must satisfy t0 < tf, got {t_span}")
    y_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    scalar = np.asarray(y0).ndim == 0
    if h0 is None:
        h0 = (tf - t0) * 0.01
    h = float(h0)
    t = float(t0)
    y = y_arr.copy()
    ts = [t]
    ys = [y.copy()]
    n_eval = 0
    for _ in range(max_steps):
        if t >= tf:
            break
        if t + h > tf:
            h = tf - t
        # Compute the 7 RK stages.
        k = np.zeros((7, y.size))
        k[0] = np.asarray(f(t, y if not scalar else y[0]), dtype=float).reshape(-1)
        for s in range(1, 7):
            yi = y + h * sum(_DP_A[s][j] * k[j] for j in range(s))
            k[s] = np.asarray(
                f(t + _DP_C[s] * h, yi if not scalar else yi[0]), dtype=float
            ).reshape(-1)
        n_eval += 7
        y_new = y + h * (_DP_B5 @ k)
        # Local error estimate: |y5 - y4| approximated by E.K coefficients.
        err_vec = h * (_DP_E @ k)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = float(np.sqrt(np.mean((err_vec / sc) ** 2)))
        if err <= 1.0 or h <= 1e-14:
            t += h
            y = y_new
            ts.append(t)
            ys.append(y.copy())
        # Step-size update with safety factor; clamp to [0.1, 5] x h.
        if err == 0:
            factor = 5.0
        else:
            factor = 0.9 * err ** (-0.2)
            factor = min(5.0, max(0.1, factor))
        h *= factor
    else:
        raise ConvergenceError(
            f"rk45_adaptive: did not reach tf in {max_steps} steps",
            result=ODESolution(
                t=np.asarray(ts),
                y=(np.asarray(ys)[:, 0] if scalar else np.asarray(ys)),
                n_evaluations=n_eval,
                method="rk45_adaptive",
                success=False,
            ),
        )
    Y = np.asarray(ys)
    return ODESolution(
        t=np.asarray(ts),
        y=(Y[:, 0] if scalar else Y),
        n_evaluations=n_eval,
        method="rk45_adaptive",
        success=True,
    )
