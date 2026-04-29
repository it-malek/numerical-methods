"""numethods - classical numerical methods, implemented from scratch.

Public API re-exported here for convenience::

    from numethods import newton, rk4, simpson, trapezoid

Submodule organization:

* :mod:`numethods.taylor`           - Taylor polynomial expansion
* :mod:`numethods.rootfinding`      - bisection, Newton, secant, fixed-point
* :mod:`numethods.differentiation`  - finite differences, Richardson, Jacobian
* :mod:`numethods.integration`      - trapezoid, Simpson, Romberg, Monte Carlo
* :mod:`numethods.ode`              - Euler, midpoint, RK4, adaptive RK45
* :mod:`numethods.linsys`           - LU, Jacobi, Gauss-Seidel, SOR, CG
* :mod:`numethods.optimization`     - gradient descent, Newton minimization
* :mod:`numethods.regression`       - OLS, ridge, polynomial features
"""

from __future__ import annotations

from ._utils import ConvergenceError

__version__ = "0.1.0"

_LAZY_IMPORTS = {
    # taylor
    "taylor_polynomial": "numethods.taylor",
    "taylor_remainder_bound": "numethods.taylor",
    # rootfinding
    "RootResult": "numethods.rootfinding",
    "bisection": "numethods.rootfinding",
    "newton": "numethods.rootfinding",
    "secant": "numethods.rootfinding",
    "fixed_point": "numethods.rootfinding",
    # differentiation
    "forward_diff": "numethods.differentiation",
    "backward_diff": "numethods.differentiation",
    "central_diff": "numethods.differentiation",
    "richardson_extrapolation": "numethods.differentiation",
    "numerical_jacobian": "numethods.differentiation",
    # integration
    "IntegrationResult": "numethods.integration",
    "trapezoid": "numethods.integration",
    "simpson": "numethods.integration",
    "romberg": "numethods.integration",
    "monte_carlo": "numethods.integration",
    "adaptive_simpson": "numethods.integration",
    # ode
    "ODESolution": "numethods.ode",
    "euler": "numethods.ode",
    "midpoint": "numethods.ode",
    "rk4": "numethods.ode",
    "rk45_adaptive": "numethods.ode",
    # linsys
    "IterativeSolveResult": "numethods.linsys",
    "lu_decomposition": "numethods.linsys",
    "lu_solve": "numethods.linsys",
    "gauss_elimination": "numethods.linsys",
    "jacobi": "numethods.linsys",
    "gauss_seidel": "numethods.linsys",
    "sor": "numethods.linsys",
    "conjugate_gradient": "numethods.linsys",
    # optimization
    "OptimizationResult": "numethods.optimization",
    "gradient_descent": "numethods.optimization",
    "gradient_descent_backtracking": "numethods.optimization",
    "newton_minimize": "numethods.optimization",
    # regression
    "OLSRegression": "numethods.regression",
    "RidgeRegression": "numethods.regression",
    "polynomial_features": "numethods.regression",
}

__all__ = ["ConvergenceError", *sorted(_LAZY_IMPORTS)]


def __getattr__(name: str):  # pragma: no cover - lazy re-export glue
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'numethods' has no attribute {name!r}") from exc

    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
