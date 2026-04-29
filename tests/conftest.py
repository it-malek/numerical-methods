"""Shared fixtures for the numethods test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def rosenbrock():
    """Classic Rosenbrock 'banana' function with analytic gradient and Hessian."""
    a, b = 1.0, 100.0

    def f(x):
        x = np.asarray(x, dtype=float)
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def grad(x):
        x = np.asarray(x, dtype=float)
        return np.array(
            [
                -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
                2 * b * (x[1] - x[0] ** 2),
            ]
        )

    def hess(x):
        x = np.asarray(x, dtype=float)
        return np.array(
            [
                [2 - 4 * b * (x[1] - 3 * x[0] ** 2), -4 * b * x[0]],
                [-4 * b * x[0], 2 * b],
            ]
        )

    return f, grad, hess, np.array([1.0, 1.0])  # f, grad, hess, minimizer


@pytest.fixture
def quadratic_bowl():
    """f(x) = x^T A x / 2 - b^T x, with minimizer x* = A^{-1} b."""
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 1.0])
    x_star = np.linalg.solve(A, b)

    def f(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * x @ A @ x - b @ x

    def grad(x):
        x = np.asarray(x, dtype=float)
        return A @ x - b

    def hess(_x):
        return A

    return f, grad, hess, x_star


@pytest.fixture
def exp_decay_ode():
    """dy/dt = -y, y(0)=1, exact: y(t) = exp(-t)."""

    def f(_t, y):
        return -y

    return f, 1.0, lambda t: np.exp(-t)


@pytest.fixture
def harmonic_ode():
    """dy/dt = [y[1], -y[0]], y(0)=[1,0], exact: [cos(t), -sin(t)]."""

    def f(_t, y):
        return np.array([y[1], -y[0]])

    def exact(t):
        return np.array([np.cos(t), -np.sin(t)])

    return f, np.array([1.0, 0.0]), exact


@pytest.fixture
def spd_matrix(rng):
    """A random 5x5 symmetric positive-definite matrix."""
    n = 5
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)
