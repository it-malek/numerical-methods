"""Scalar root-finding: bisection, Newton, secant, fixed-point iteration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ._utils import (
    ConvergenceError,
    finite_difference_derivative,
    validate_max_iter,
    validate_tol,
)


@dataclass
class RootResult:
    """Outcome of a root-finding call.

    Attributes:
        root: Best estimate of the root.
        iterations: Number of iterations performed.
        converged: Whether the convergence criterion was met.
        residual: Absolute value of ``f(root)`` (or step size for fixed-point).
        history: Sequence of iterates ``[x0, x1, ...]`` for convergence plotting.
        method: Name of the algorithm.
    """

    root: float
    iterations: int
    converged: bool
    residual: float
    history: list[float] = field(default_factory=list)
    method: str = ""


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> RootResult:
    """Bisection method on a sign-changing bracket ``[a, b]``.

    Linear convergence (one bit per iteration). Always succeeds when the
    initial bracket contains a root and ``f`` is continuous.

    Args:
        f: Continuous scalar function.
        a: Left bracket endpoint.
        b: Right bracket endpoint.
        tol: Absolute tolerance on the bracket width.
        max_iter: Maximum iterations.

    Raises:
        ValueError: If ``f(a)`` and ``f(b)`` have the same sign.
        ConvergenceError: If ``max_iter`` exceeded.

    Examples:
        >>> r = bisection(lambda x: x**2 - 2, 0.0, 2.0)
        >>> abs(r.root - 2**0.5) < 1e-9
        True
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    fa, fb = f(a), f(b)
    if fa == 0:
        return RootResult(a, 0, True, 0.0, [a], method="bisection")
    if fb == 0:
        return RootResult(b, 0, True, 0.0, [b], method="bisection")
    if np.sign(fa) == np.sign(fb):
        raise ValueError(
            f"f(a) and f(b) must have opposite signs (got f(a)={fa}, f(b)={fb})"
        )
    history = []
    for k in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        history.append(m)
        if abs(b - a) < 2 * tol or fm == 0:
            return RootResult(
                root=float(m),
                iterations=k,
                converged=True,
                residual=float(abs(fm)),
                history=history,
                method="bisection",
            )
        if np.sign(fa) == np.sign(fm):
            a, fa = m, fm
        else:
            b, fb = m, fm
    raise ConvergenceError(
        f"bisection did not converge in {max_iter} iterations",
        result=RootResult(
            root=float(0.5 * (a + b)),
            iterations=max_iter,
            converged=False,
            residual=float(abs(f(0.5 * (a + b)))),
            history=history,
            method="bisection",
        ),
    )


def newton(
    f: Callable[[float], float],
    x0: float,
    fprime: Callable[[float], float] | None = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> RootResult:
    """Newton's method.

    If ``fprime`` is None, uses a central finite-difference approximation.

    Args:
        f: Scalar function whose root is sought.
        x0: Initial guess.
        fprime: Optional analytic derivative.
        tol: Absolute tolerance on |f(x)|.
        max_iter: Maximum iterations.

    Raises:
        ConvergenceError: If the derivative vanishes or ``max_iter`` exceeded.
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    deriv = fprime if fprime is not None else (lambda x: finite_difference_derivative(f, x))
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < tol:
            return RootResult(
                root=x,
                iterations=k - 1,
                converged=True,
                residual=float(abs(fx)),
                history=history,
                method="newton",
            )
        dfx = deriv(x)
        if dfx == 0 or not np.isfinite(dfx):
            raise ConvergenceError(
                f"Newton: derivative vanished or non-finite at iteration {k} (x={x})",
                result=RootResult(x, k, False, float(abs(fx)), history, "newton"),
            )
        x = x - fx / dfx
        history.append(x)
    fx = f(x)
    if abs(fx) < tol:
        return RootResult(x, max_iter, True, float(abs(fx)), history, "newton")
    raise ConvergenceError(
        f"Newton did not converge in {max_iter} iterations",
        result=RootResult(x, max_iter, False, float(abs(fx)), history, "newton"),
    )


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> RootResult:
    """Secant method: Newton with the derivative replaced by a finite difference.

    Order of convergence ≈ 1.618 (golden ratio).
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    if x0 == x1:
        raise ValueError("x0 and x1 must differ")
    a, b = float(x0), float(x1)
    fa, fb = f(a), f(b)
    history = [a, b]
    for k in range(1, max_iter + 1):
        if abs(fb) < tol:
            return RootResult(b, k - 1, True, float(abs(fb)), history, "secant")
        if fb == fa:
            raise ConvergenceError(
                f"secant: f(a) == f(b) at iteration {k}, slope undefined",
                result=RootResult(b, k, False, float(abs(fb)), history, "secant"),
            )
        c = b - fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b, fb = c, f(c)
        history.append(c)
    if abs(fb) < tol:
        return RootResult(b, max_iter, True, float(abs(fb)), history, "secant")
    raise ConvergenceError(
        f"secant did not converge in {max_iter} iterations",
        result=RootResult(b, max_iter, False, float(abs(fb)), history, "secant"),
    )


def fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> RootResult:
    """Fixed-point iteration ``x_{k+1} = g(x_k)``.

    Detects divergence by tracking whether successive step sizes are growing
    over a window - when they consistently grow, raise ``ConvergenceError``.

    Args:
        g: Iteration function. A fixed point of ``g`` is a zero of ``f(x) = g(x) - x``.
        x0: Starting value.
        tol: Stop when ``|x_{k+1} - x_k| < tol``.
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    x = float(x0)
    history = [x]
    prev_step = float("inf")
    growth_streak = 0
    for k in range(1, max_iter + 1):
        x_new = float(g(x))
        if not np.isfinite(x_new):
            raise ConvergenceError(
                f"fixed-point: iterate became non-finite at step {k}",
                result=RootResult(x, k, False, float(abs(x_new - x)), history, "fixed_point"),
            )
        step = abs(x_new - x)
        history.append(x_new)
        if step < tol:
            return RootResult(x_new, k, True, step, history, "fixed_point")
        if step > prev_step:
            growth_streak += 1
        else:
            growth_streak = 0
        if growth_streak >= 5:
            raise ConvergenceError(
                f"fixed-point: iterates appear to diverge (growth streak at step {k})",
                result=RootResult(x_new, k, False, step, history, "fixed_point"),
            )
        prev_step = step
        x = x_new
    raise ConvergenceError(
        f"fixed-point did not converge in {max_iter} iterations",
        result=RootResult(x, max_iter, False, prev_step, history, "fixed_point"),
    )
