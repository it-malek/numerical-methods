"""Numerical integration: composite Newton-Cotes, Romberg, Monte Carlo, adaptive Simpson."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ._utils import ConvergenceError, validate_tol


@dataclass
class IntegrationResult:
    """Outcome of a numerical integration call.

    Attributes:
        value: Approximation to the integral.
        n_evaluations: Number of times the integrand was evaluated.
        error_estimate: Approximate error if available, else None.
        method: Name of the rule used.
        history: For iterative/adaptive rules, the per-step value sequence.
        converged: Adaptive Simpson's local tests all passed; None for other rules.
        requested_tolerance: Adaptive Simpson's requested absolute tolerance,
            distinct from its heuristic error_estimate; None for other rules.
    """

    value: float
    n_evaluations: int
    error_estimate: float | None = None
    method: str = ""
    history: list[float] = field(default_factory=list)
    converged: bool | None = None
    requested_tolerance: float | None = None


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
) -> tuple[float, float, bool]:
    """Recursive worker for adaptive Simpson - splits if refined estimate disagrees."""
    m = 0.5 * a + 0.5 * b
    lm = 0.5 * a + 0.5 * m
    rm = 0.5 * m + 0.5 * b
    if lm in (a, m) or rm in (m, b):
        return whole, float("inf"), False  # No representable refinement remains.
    flm = f(lm)
    frm = f(rm)
    counter[0] += 2
    left = _simpson_one(f, a, m, fa, fm, flm)
    right = _simpson_one(f, m, b, fm, fb, frm)
    diff = left + right - whole
    if not all(np.isfinite(v) for v in (left, right, diff)):
        raise ConvergenceError("adaptive_simpson: nonfinite quadrature arithmetic")
    # For smooth f in the fourth-order regime, refinement divides the
    # leading Simpson error by 16. The difference estimates 15 times that error.
    error = abs(diff) / 15.0
    if error <= tol:
        return left + right + diff / 15.0, error, True
    if depth >= max_depth:
        return left + right + diff / 15.0, error, False
    lv, le, lc = _adaptive_simpson_recurse(
        f, a, m, fa, fm, flm, left, tol / 2.0, depth + 1, counter, max_depth
    )
    rv, re, rc = _adaptive_simpson_recurse(
        f, m, b, fm, fb, frm, right, tol / 2.0, depth + 1, counter, max_depth
    )
    return lv + rv, le + re, lc and rc


def adaptive_simpson(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 20,
) -> IntegrationResult:
    """Adaptive Simpson quadrature with an absolute error *target* ``tol``.

    Accept leaves with ``abs(S_two - S_one)/15 <= local_tol``; halve the
    tolerance at each split and sum leaf estimates. The corrected integral
    uses Richardson extrapolation. The estimate describes the uncorrected
    refined Simpson rule; it is neither an achieved error nor a rigorous bound
    on the returned value. Smoothness and resolved variation are assumed;
    sampling may miss narrow peaks or oscillations entirely.

    Supports finite reversed and zero-length intervals. ``f`` takes a length-1
    NumPy array and returns one finite value. ``max_depth`` is a positive
    integer; root depth is zero. Depth/spacing exhaustion raises
    ``ConvergenceError`` with a nonconverged ``IntegrationResult`` for the
    entire interval. Nonfinite evaluations/arithmetic raise ``ConvergenceError``
    with no result (no valid interval-wide estimate is available).
    """
    validate_tol(tol)
    if not isinstance(max_depth, (int, np.integer)) or isinstance(max_depth, (bool, np.bool_)) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("interval endpoints must be finite")
    if a == b:
        return IntegrationResult(0.0, 0, 0.0, "adaptive_simpson",
                                 converged=True, requested_tolerance=tol)

    def evaluate(t):
        values = np.asarray(f(np.array([t])), dtype=float)
        if values.shape != (1,):
            raise ValueError("f must return a length-1 array for a length-1 input")
        if not np.isfinite(values[0]):
            raise ConvergenceError("adaptive_simpson: nonfinite integrand evaluation")
        return float(values[0])

    counter = [3]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            fa, fb, fm = evaluate(a), evaluate(b), evaluate(0.5 * a + 0.5 * b)
            whole = _simpson_one(f, a, b, fa, fb, fm)
            if not np.isfinite(whole):
                raise ConvergenceError("adaptive_simpson: nonfinite quadrature arithmetic")
            val, error, converged = _adaptive_simpson_recurse(
                evaluate, a, b, fa, fb, fm, whole, tol, 0, counter, max_depth
            )
            if not np.isfinite(val) or (not np.isfinite(error) and converged):
                raise ConvergenceError("adaptive_simpson: nonfinite quadrature arithmetic")
    except (FloatingPointError, OverflowError) as exc:
        raise ConvergenceError("adaptive_simpson: nonfinite evaluation or arithmetic") from exc
    result = IntegrationResult(float(val), counter[0], float(error), "adaptive_simpson",
                               converged=converged, requested_tolerance=tol)
    if not converged:
        raise ConvergenceError("adaptive_simpson: depth limit or representable spacing exhausted", result=result)
    return result
