"""Taylor polynomial expansion and remainder bounds.

Two flavors of Taylor expansion are provided:

1. **Symbolic** - when ``f`` is a SymPy expression, exact derivatives are
   computed and the polynomial is returned as a callable evaluating the
   exact rational/real coefficients.
2. **Numerical** - when ``f`` is a plain Python callable, derivatives are
   approximated via central differences. This is intentionally limited to
   moderate orders (n <= ~8) because the noise in repeated finite
   differences grows quickly.
"""

from __future__ import annotations

from collections.abc import Callable
from math import factorial

import numpy as np
import sympy as sp

ScalarFunc = Callable[[float], float]


def _numerical_nth_derivative(f: ScalarFunc, a: float, n: int, h: float) -> float:
    """Approximate ``f^{(n)}(a)`` via the central finite-difference stencil.

    Uses the symmetric coefficients ``sum_k (-1)^k C(n,k) f(a + (n/2 - k) h)``.
    """
    if n == 0:
        return float(f(a))
    total = 0.0
    for k in range(n + 1):
        sign = (-1) ** k
        coef = 1
        # binomial coefficient C(n, k)
        for j in range(k):
            coef = coef * (n - j) // (j + 1)
        total += sign * coef * f(a + (n / 2 - k) * h)
    return total / (h**n)


def taylor_polynomial(
    f: ScalarFunc | sp.Expr,
    a: float,
    n: int,
    x: float | np.ndarray | None = None,
    *,
    var: sp.Symbol | None = None,
    h: float | None = None,
) -> float | np.ndarray | ScalarFunc:
    """Evaluate (or build) the degree-``n`` Taylor polynomial of ``f`` at ``a``.

    Args:
        f: Either a SymPy expression in a single variable, or a numeric
            callable ``f(x) -> float``.
        a: Expansion point.
        n: Degree of the polynomial (non-negative integer).
        x: If given, the polynomial is evaluated at ``x`` and the value
            returned. If ``None``, a callable approximating ``f`` is returned.
        var: For SymPy input, the symbol to expand in. If ``None`` and
            ``f.free_symbols`` is a singleton, that symbol is used.
        h: Finite-difference step for numerical derivatives. Defaults to
            a heuristic based on machine epsilon.

    Returns:
        A scalar / array of evaluations, or a callable if ``x is None``.

    Raises:
        ValueError: If ``n`` is negative or the SymPy variable is ambiguous.

    Examples:
        >>> import math
        >>> abs(taylor_polynomial(math.sin, 0.0, 5, 0.5) - math.sin(0.5)) < 1e-3
        True
    """
    if n < 0:
        raise ValueError(f"degree n must be >= 0, got {n}")

    if isinstance(f, sp.Expr):
        if var is None:
            free = list(f.free_symbols)
            if len(free) != 1:
                raise ValueError(
                    "SymPy expression has multiple free symbols; pass var= explicitly"
                )
            var = free[0]
        # series(...).removeO() truncates the big-O term cleanly.
        poly_expr = sp.series(f, var, a, n + 1).removeO()
        poly = sp.lambdify(var, poly_expr, modules="numpy")
        return poly(x) if x is not None else poly

    # Numerical branch: precompute derivatives via finite differences.
    if h is None:
        # A larger h than usual: high-order central differences are noisy.
        h = max(1e-3, np.cbrt(np.finfo(float).eps))

    coeffs = np.array(
        [_numerical_nth_derivative(f, a, k, h) / factorial(k) for k in range(n + 1)]
    )

    def poly(xv):
        xv = np.asarray(xv, dtype=float)
        # Horner-style evaluation from highest degree down.
        result = np.full_like(xv, coeffs[-1], dtype=float)
        for c in coeffs[-2::-1]:
            result = result * (xv - a) + c
        return result if result.shape else float(result)

    return poly(x) if x is not None else poly


def _validate_remainder_inputs(a: float, n: int, x: float) -> None:
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 0:
        raise ValueError("degree n must be a non-negative integer")
    if not np.isfinite(a) or not np.isfinite(x):
        raise ValueError("a and x must be finite")


def taylor_remainder_bound(
    f: ScalarFunc | sp.Expr,
    a: float,
    n: int,
    x: float,
    *,
    var: sp.Symbol | None = None,
    M: float | None = None,
) -> float:
    """Conditional Lagrange bound ``M*abs(x-a)**(n+1)/(n+1)!``.

    The caller must supply finite nonnegative ``M`` bounding ``abs(f^(n+1))``
    on the *entire* closed interval between ``a`` and ``x``, and f must have
    the derivatives required by Taylor's theorem there. Neither assumption
    is certified by this program; ``f`` and ``var`` are retained for call
    compatibility, but are not evaluated.

    Compatibility: calls omitting ``M`` now raise ``ValueError``. Use
    :func:`taylor_remainder_estimate` for the former sampled heuristic.
    The return type remains float; no numerical guarantee is implied when
    a caller's M is invalid or the polynomial uses approximate derivatives.
    """
    _validate_remainder_inputs(a, n, x)
    if M is None:
        raise ValueError("supply a derivative bound M, or use taylor_remainder_estimate")
    if not np.isfinite(M) or M < 0:
        raise ValueError("M must be finite and nonnegative")
    return float(M * abs(x - a) ** (n + 1) / factorial(n + 1))


def taylor_remainder_estimate(
    f: ScalarFunc | sp.Expr,
    a: float,
    n: int,
    x: float,
    *,
    var: sp.Symbol | None = None,
) -> float:
    """Sampled Lagrange-form remainder estimate, **not a bound**.

    Symbolic derivatives are sampled at 257 equally spaced nodes. Callable
    derivatives use central finite differences (h=1e-3) at 33 nodes, adding
    differentiation error and sampling outside the interval near endpoints.
    Finite sampling can miss arbitrarily narrow peaks. Even zero is not a
    certificate of zero error. High numerical derivative orders are noisy.
    Reversed intervals are supported. Nonfinite samples raise ValueError.
    """
    _validate_remainder_inputs(a, n, x)
    lo, hi = min(a, x), max(a, x)
    if isinstance(f, sp.Expr):
        if var is None:
            free = list(f.free_symbols)
            if len(free) != 1:
                raise ValueError("SymPy expression has multiple free symbols; pass var= explicitly")
            var = free[0]
        deriv_fn = sp.lambdify(var, sp.diff(f, var, n + 1), modules="numpy")
        vals = np.abs(deriv_fn(np.linspace(lo, hi, 257)))
    else:
        grid = np.linspace(lo, hi, 33)
        h = max(1e-3, np.cbrt(np.finfo(float).eps))
        vals = np.abs([_numerical_nth_derivative(f, g, n + 1, h) for g in grid])
    if not np.all(np.isfinite(vals)):
        raise ValueError("derivative samples must be finite")
    M = float(np.max(vals))
    return float(M * abs(x - a) ** (n + 1) / factorial(n + 1))
