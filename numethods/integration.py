"""Numerical integration: composite Newton-Cotes, Romberg, Monte Carlo, adaptive Simpson."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class IntegrationResult:
    """Outcome of a numerical integration call.

    Attributes:
        value: Approximation to the integral.
        n_evaluations: Number of times the integrand was evaluated.
        error_estimate: Approximate error if available, else None.
        method: Name of the rule used.
        history: For iterative/adaptive rules, the per-step value sequence.
    """

    value: float
    n_evaluations: int
    error_estimate: float | None = None
    method: str = ""
    history: list[float] = field(default_factory=list)


def trapezoid(
    f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int
) -> IntegrationResult:
    """Composite trapezoidal rule with ``n`` equal subintervals.

    Truncation error: O((b - a)^3 / n^2 * max|f''|).

    Args:
        f: Integrand. Must accept and return NumPy arrays for vectorization.
        a: Lower limit.
        b: Upper limit.
        n: Number of subintervals (must be >= 1).

    Returns:
        IntegrationResult with the approximate integral.

    Raises:
        ValueError: If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    val = h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])
    return IntegrationResult(value=float(val), n_evaluations=n + 1, method="trapezoid")


def simpson(
    f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int
) -> IntegrationResult:
    """Composite Simpson's rule. ``n`` must be even.

    Truncation error: O((b - a)^5 / n^4 * max|f^(4)|), so exact on cubics.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError(f"n must be a positive even integer, got {n}")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    val = (h / 3.0) * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())
    return IntegrationResult(value=float(val), n_evaluations=n + 1, method="simpson")


def romberg(
    f: Callable[[np.ndarray], np.ndarray], a: float, b: float, max_levels: int = 8
) -> IntegrationResult:
    """Romberg integration: Richardson extrapolation on the trapezoidal rule.

    Builds a triangular table ``R[k, j]`` where ``R[k, 0]`` is the trapezoidal
    rule with 2^k panels and each subsequent column cancels successive even
    powers of h. The ``(max_levels-1, max_levels-1)`` entry is returned.

    Args:
        f: Integrand (vectorized).
        a: Lower limit.
        b: Upper limit.
        max_levels: Number of refinement levels (>= 1).
    """
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")
    R = np.zeros((max_levels, max_levels))
    n_eval = 0
    history: list[float] = []

    # Level 0: trapezoid with 1 interval.
    R[0, 0] = 0.5 * (b - a) * (f(np.array([a]))[0] + f(np.array([b]))[0])
    n_eval += 2
    history.append(R[0, 0])

    for k in range(1, max_levels):
        n_panels = 2**k
        h = (b - a) / n_panels
        # Sum the new midpoint evaluations only (the even ones came in earlier rows).
        x_new = a + h * np.arange(1, n_panels, 2)
        y_new = np.asarray(f(x_new), dtype=float)
        R[k, 0] = 0.5 * R[k - 1, 0] + h * y_new.sum()
        n_eval += x_new.size
        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
        history.append(R[k, k])

    err = abs(R[max_levels - 1, max_levels - 1] - R[max_levels - 1, max_levels - 2]) if max_levels > 1 else None
    return IntegrationResult(
        value=float(R[max_levels - 1, max_levels - 1]),
        n_evaluations=n_eval,
        error_estimate=err,
        method="romberg",
        history=history,
    )


def monte_carlo(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> IntegrationResult:
    """Plain Monte Carlo integration: ``(b - a) * mean(f(U[a, b]))``.

    Convergence is O(n^{-1/2}) - slow but dimension-independent.

    Args:
        f: Integrand (vectorized).
        a: Lower limit.
        b: Upper limit.
        n_samples: Number of uniform samples.
        rng: Optional NumPy generator for reproducibility.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.uniform(a, b, size=n_samples)
    y = np.asarray(f(samples), dtype=float)
    val = (b - a) * y.mean()
    # Standard error of the mean estimate.
    err = (b - a) * y.std(ddof=1) / np.sqrt(n_samples) if n_samples > 1 else None
    return IntegrationResult(
        value=float(val),
        n_evaluations=n_samples,
        error_estimate=float(err) if err is not None else None,
        method="monte_carlo",
    )


def _simpson_one(f: Callable, a: float, b: float, fa: float, fb: float, fm: float) -> float:
    """Simpson's rule on a single interval with cached endpoint/midpoint values."""
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb)


def _adaptive_simpson_recurse(
    f: Callable,
    a: float,
    b: float,
    fa: float,
    fb: float,
    fm: float,
    whole: float,
    tol: float,
    depth: int,
    counter: list[int],
    max_depth: int,
) -> float:
    """Recursive worker for adaptive Simpson - splits if refined estimate disagrees."""
    m = 0.5 * (a + b)
    lm = 0.5 * (a + m)
    rm = 0.5 * (m + b)
    flm = float(f(np.array([lm]))[0])
    frm = float(f(np.array([rm]))[0])
    counter[0] += 2
    left = _simpson_one(f, a, m, fa, fm, flm)
    right = _simpson_one(f, m, b, fm, fb, frm)
    diff = left + right - whole
    if depth >= max_depth or abs(diff) <= 15.0 * tol:
        return left + right + diff / 15.0
    return _adaptive_simpson_recurse(
        f, a, m, fa, fm, flm, left, tol / 2.0, depth + 1, counter, max_depth
    ) + _adaptive_simpson_recurse(
        f, m, b, fm, fb, frm, right, tol / 2.0, depth + 1, counter, max_depth
    )


def adaptive_simpson(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 20,
) -> IntegrationResult:
    """Adaptive Simpson's rule with recursive bisection.

    Subdivides until the local Richardson error estimate falls below ``tol``,
    or ``max_depth`` recursion levels are reached.

    Args:
        f: Integrand (vectorized).
        a: Lower limit.
        b: Upper limit.
        tol: Local tolerance.
        max_depth: Maximum recursion depth per branch.
    """
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    fa = float(f(np.array([a]))[0])
    fb = float(f(np.array([b]))[0])
    fm = float(f(np.array([0.5 * (a + b)]))[0])
    counter = [3]  # endpoints + midpoint
    whole = _simpson_one(f, a, b, fa, fb, fm)
    val = _adaptive_simpson_recurse(
        f, a, b, fa, fb, fm, whole, tol, 0, counter, max_depth
    )
    return IntegrationResult(
        value=float(val),
        n_evaluations=counter[0],
        error_estimate=tol,
        method="adaptive_simpson",
    )
