"""Internal shared utilities for numethods.

These helpers are not part of the public API; they are used by the
public modules to keep their implementations focused.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ConvergenceError(RuntimeError):
    """Raised when an iterative method fails to converge in ``max_iter`` steps.

    The partial result is exposed via the :attr:`result` attribute so users
    can inspect what went wrong (e.g. the last iterate, the residual history).
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


def validate_tol(tol: float) -> None:
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be positive and finite, got {tol!r}")


def validate_max_iter(max_iter: int) -> None:
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter!r}")


def finite_difference_derivative(
    f: Callable[[float], float], x: float, h: float | None = None
) -> float:
    """Central-difference scalar derivative ``(f(x+h) - f(x-h)) / (2h)``.

    The default step ``h`` is chosen via the standard cube-root-of-epsilon
    heuristic, which balances truncation and roundoff error.
    """
    if h is None:
        h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2.0 * h)


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, h: float | None = None
) -> np.ndarray:
    """Central-difference gradient of a scalar-valued function of a vector."""
    x = np.asarray(x, dtype=float)
    if h is None:
        h = np.cbrt(np.finfo(float).eps)
    grad = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        step = h * max(1.0, abs(x[i]))
        e[i] = step
        grad[i] = (f(x + e) - f(x - e)) / (2.0 * step)
    return grad


def numerical_hessian(
    f: Callable[[np.ndarray], float], x: np.ndarray, h: float | None = None
) -> np.ndarray:
    """Central-difference Hessian of a scalar-valued function of a vector."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if h is None:
        h = np.power(np.finfo(float).eps, 0.25)
    H = np.empty((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros_like(x)
            ej = np.zeros_like(x)
            ei[i] = h
            ej[j] = h
            if i == j:
                val = (f(x + ei) - 2 * fx + f(x - ei)) / (h * h)
            else:
                val = (
                    f(x + ei + ej)
                    - f(x + ei - ej)
                    - f(x - ei + ej)
                    + f(x - ei - ej)
                ) / (4 * h * h)
            H[i, j] = val
            H[j, i] = val
    return H
