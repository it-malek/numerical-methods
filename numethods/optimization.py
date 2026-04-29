"""Unconstrained minimization: gradient descent (with backtracking) and Newton."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ._utils import (
    ConvergenceError,
    numerical_gradient,
    numerical_hessian,
    validate_max_iter,
    validate_tol,
)


@dataclass
class OptimizationResult:
    """Outcome of a minimization call.

    Attributes:
        x: Best estimate of the minimizer.
        fun: Objective value at ``x``.
        iterations: Number of iterations performed.
        converged: Whether the gradient-norm criterion was met.
        grad_norm: ``||∇f(x)||_2`` at the final iterate.
        history: List of iterates for convergence plotting.
        method: Algorithm name.
    """

    x: np.ndarray
    fun: float
    iterations: int
    converged: bool
    grad_norm: float
    history: list[np.ndarray] = field(default_factory=list)
    method: str = ""


def _resolve_grad(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray] | None,
) -> Callable[[np.ndarray], np.ndarray]:
    if grad is not None:
        return lambda x: np.asarray(grad(x), dtype=float)
    return lambda x: numerical_gradient(f, x)


def _resolve_hess(
    f: Callable[[np.ndarray], float],
    hess: Callable[[np.ndarray], np.ndarray] | None,
) -> Callable[[np.ndarray], np.ndarray]:
    if hess is not None:
        return lambda x: np.asarray(hess(x), dtype=float)
    return lambda x: numerical_hessian(f, x)


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    lr: float = 1e-2,
    momentum: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> OptimizationResult:
    """Steepest descent with optional Polyak momentum.

    Update::

        v_{k+1} = momentum * v_k - lr * ∇f(x_k)
        x_{k+1} = x_k + v_{k+1}

    Args:
        f: Scalar objective.
        grad: Gradient callable; if None, uses central finite differences.
        x0: Starting point (1-D array-like).
        lr: Fixed step size.
        momentum: Momentum coefficient in ``[0, 1)``.
        tol: Stop when ``||∇f|| < tol``.
        max_iter: Iteration cap.

    Raises:
        ConvergenceError: If not converged within ``max_iter``.
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not 0 <= momentum < 1:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}")
    grad_fn = _resolve_grad(f, grad)
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        g = grad_fn(x)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            return OptimizationResult(
                x=x, fun=float(f(x)), iterations=k - 1, converged=True,
                grad_norm=gn, history=history, method="gradient_descent",
            )
        v = momentum * v - lr * g
        x = x + v
        history.append(x.copy())
    g = grad_fn(x)
    gn = float(np.linalg.norm(g))
    raise ConvergenceError(
        f"gradient_descent did not converge in {max_iter} iterations (||g||={gn:.3e})",
        result=OptimizationResult(
            x=x, fun=float(f(x)), iterations=max_iter, converged=False,
            grad_norm=gn, history=history, method="gradient_descent",
        ),
    )


def gradient_descent_backtracking(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    lr_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> OptimizationResult:
    """Steepest descent with Armijo backtracking line search.

    At each step, find the largest ``alpha = lr_init * rho^j`` (j = 0, 1, ...)
    such that the Armijo sufficient-decrease condition holds::

        f(x - alpha g) <= f(x) - c1 * alpha * ||g||^2

    Args:
        f: Scalar objective.
        grad: Gradient callable; if None, finite differences.
        x0: Starting point.
        lr_init: Initial trial step.
        c1: Armijo constant in (0, 1).
        rho: Backtracking factor in (0, 1).
        tol: Convergence tolerance on gradient norm.
        max_iter: Outer iteration cap.
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    if not 0 < c1 < 1:
        raise ValueError(f"c1 must be in (0, 1), got {c1}")
    if not 0 < rho < 1:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    grad_fn = _resolve_grad(f, grad)
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        g = grad_fn(x)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            return OptimizationResult(
                x=x, fun=float(f(x)), iterations=k - 1, converged=True,
                grad_norm=gn, history=history,
                method="gradient_descent_backtracking",
            )
        fx = f(x)
        alpha = lr_init
        # Limit line search to ~50 halvings.
        for _ in range(50):
            x_new = x - alpha * g
            if f(x_new) <= fx - c1 * alpha * gn * gn:
                break
            alpha *= rho
        x = x_new
        history.append(x.copy())
    g = grad_fn(x)
    gn = float(np.linalg.norm(g))
    raise ConvergenceError(
        f"gradient_descent_backtracking did not converge in {max_iter} iters (||g||={gn:.3e})",
        result=OptimizationResult(
            x=x, fun=float(f(x)), iterations=max_iter, converged=False,
            grad_norm=gn, history=history,
            method="gradient_descent_backtracking",
        ),
    )


def newton_minimize(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray] | None,
    hess: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
    damping: float = 0.0,
) -> OptimizationResult:
    """Newton's method for minimization with an optional Levenberg-style ridge.

    Update: ``x_{k+1} = x_k - (H + damping I)^{-1} g``. Set ``damping > 0``
    when the Hessian may become indefinite or near-singular.

    Args:
        f: Objective.
        grad: Gradient (None => central-difference).
        hess: Hessian (None => central-difference).
        x0: Starting point.
        tol: Stop on ``||g|| < tol``.
        max_iter: Iteration cap.
        damping: Small ridge added to the Hessian for numerical stability.
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    if damping < 0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    grad_fn = _resolve_grad(f, grad)
    hess_fn = _resolve_hess(f, hess)
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        g = grad_fn(x)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            return OptimizationResult(
                x=x, fun=float(f(x)), iterations=k - 1, converged=True,
                grad_norm=gn, history=history, method="newton_minimize",
            )
        H = hess_fn(x)
        try:
            step = np.linalg.solve(H + damping * np.eye(x.size), g)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceError(
                f"newton_minimize: Hessian is singular at iteration {k}",
                result=OptimizationResult(
                    x=x, fun=float(f(x)), iterations=k, converged=False,
                    grad_norm=gn, history=history, method="newton_minimize",
                ),
            ) from exc
        x = x - step
        history.append(x.copy())
    g = grad_fn(x)
    gn = float(np.linalg.norm(g))
    if gn < tol:
        return OptimizationResult(
            x=x, fun=float(f(x)), iterations=max_iter, converged=True,
            grad_norm=gn, history=history, method="newton_minimize",
        )
    raise ConvergenceError(
        f"newton_minimize did not converge in {max_iter} iterations (||g||={gn:.3e})",
        result=OptimizationResult(
            x=x, fun=float(f(x)), iterations=max_iter, converged=False,
            grad_norm=gn, history=history, method="newton_minimize",
        ),
    )
