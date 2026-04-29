"""Numerical differentiation: finite differences and Richardson extrapolation.

All routines accept either scalar ``x`` or a NumPy array of points and
broadcast naturally. Forward and backward differences are O(h); central
difference is O(h^2); Richardson extrapolation lifts central to O(h^4).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayLike = float | np.ndarray


def _eval(f: Callable, x):
    """Call ``f(x)`` and return a NumPy array (preserving scalar shape)."""
    return np.asarray(f(x), dtype=float)


def forward_diff(
    f: Callable[[ArrayLike], ArrayLike], x: ArrayLike, h: float = 1e-5
) -> ArrayLike:
    """First-order forward-difference derivative ``(f(x+h) - f(x)) / h``.

    Truncation error: O(h).

    Args:
        f: Scalar-valued callable.
        x: Point or array of points at which to evaluate the derivative.
        h: Step size. Must be positive.

    Returns:
        Approximate derivative at each ``x``.

    Raises:
        ValueError: If ``h <= 0``.
    """
    if h <= 0:
        raise ValueError(f"step size h must be positive, got {h}")
    return (_eval(f, x + h) - _eval(f, x)) / h


def backward_diff(
    f: Callable[[ArrayLike], ArrayLike], x: ArrayLike, h: float = 1e-5
) -> ArrayLike:
    """First-order backward-difference derivative ``(f(x) - f(x-h)) / h``.

    Truncation error: O(h).
    """
    if h <= 0:
        raise ValueError(f"step size h must be positive, got {h}")
    return (_eval(f, x) - _eval(f, x - h)) / h


def central_diff(
    f: Callable[[ArrayLike], ArrayLike], x: ArrayLike, h: float = 1e-5
) -> ArrayLike:
    """Second-order central-difference derivative ``(f(x+h) - f(x-h)) / (2h)``.

    Truncation error: O(h^2).
    """
    if h <= 0:
        raise ValueError(f"step size h must be positive, got {h}")
    return (_eval(f, x + h) - _eval(f, x - h)) / (2.0 * h)


def richardson_extrapolation(
    f: Callable[[ArrayLike], ArrayLike], x: ArrayLike, h: float = 1e-2
) -> ArrayLike:
    """Richardson-extrapolated derivative, O(h^4) accurate.

    Computes the central-difference approximations at step ``h`` and ``h/2``
    and combines them to cancel the leading O(h^2) error term::

        D_R = (4 D(h/2) - D(h)) / 3
    """
    if h <= 0:
        raise ValueError(f"step size h must be positive, got {h}")
    d_h = central_diff(f, x, h)
    d_h2 = central_diff(f, x, h / 2.0)
    return (4.0 * d_h2 - d_h) / 3.0


def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    """Central-difference Jacobian for a vector-valued ``f: R^n -> R^m``.

    Args:
        f: Callable mapping length-``n`` vector to length-``m`` vector.
        x: Point of evaluation, shape ``(n,)``.
        h: Step size.

    Returns:
        Array of shape ``(m, n)`` where entry ``[i, j] = ∂f_i / ∂x_j``.

    Examples:
        >>> import numpy as np
        >>> J = numerical_jacobian(lambda v: np.array([v[0]**2, v[0]*v[1]]),
        ...                        np.array([1.0, 2.0]))
        >>> np.allclose(J, [[2.0, 0.0], [2.0, 1.0]], atol=1e-6)
        True
    """
    if h <= 0:
        raise ValueError(f"step size h must be positive, got {h}")
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(_eval(f, x))
    n = x.size
    m = f0.size
    J = np.empty((m, n))
    for j in range(n):
        e = np.zeros_like(x)
        step = h * max(1.0, abs(x[j]))
        e[j] = step
        plus = np.atleast_1d(_eval(f, x + e))
        minus = np.atleast_1d(_eval(f, x - e))
        J[:, j] = (plus - minus) / (2.0 * step)
    return J
