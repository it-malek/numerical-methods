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


def taylor_remainder_bound(
    f: ScalarFunc | sp.Expr,
    a: float,
    n: int,
    x: float,
    *,
    var: sp.Symbol | None = None,
    M: float | None = None,
) -> float:
    """Lagrange remainder bound ``|R_n(x)| <= M |x-a|^{n+1} / (n+1)!``.

    Args:
        f: Function (callable or SymPy expression).
        a: Expansion point.
        n: Polynomial degree.
        x: Evaluation point.
        var: SymPy variable (for symbolic ``f``).
        M: A user-supplied bound on ``|f^{(n+1)}(xi)|`` over the interval
            between ``a`` and ``x``. If ``None`` and ``f`` is symbolic,
            this is computed by sampling the symbolic ``(n+1)``-th
            derivative on a grid; otherwise a finite-difference estimate
            is used.

    Returns:
        A non-negative upper bound on the truncation error.
    """
    if n < 0:
        raise ValueError(f"degree n must be >= 0, got {n}")

    if M is None:
        if isinstance(f, sp.Expr):
            if var is None:
                free = list(f.free_symbols)
                if len(free) != 1:
                    raise ValueError(
                        "SymPy expression has multiple free symbols; pass var= explicitly"
                    )
                var = free[0]
            deriv = sp.diff(f, var, n + 1)
            deriv_fn = sp.lambdify(var, deriv, modules="numpy")
            # Probe densely over the interval to estimate the sup.
            lo, hi = (a, x) if a <= x else (x, a)
            grid = np.linspace(lo, hi, 257)
            vals = np.abs(deriv_fn(grid))
            M = float(np.nanmax(vals))
        else:
            # Numerical fallback: sample the (n+1)-th derivative.
            lo, hi = (a, x) if a <= x else (x, a)
            grid = np.linspace(lo, hi, 33)
            h = max(1e-3, np.cbrt(np.finfo(float).eps))
            vals = [
                abs(_numerical_nth_derivative(f, g, n + 1, h)) for g in grid
            ]
            M = float(np.max(vals))

    return M * abs(x - a) ** (n + 1) / factorial(n + 1)
