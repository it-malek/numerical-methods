"""Tests for numethods.optimization."""

from __future__ import annotations

import numpy as np
import pytest

from numethods import ConvergenceError
from numethods.optimization import (
    gradient_descent,
    gradient_descent_backtracking,
    newton_minimize,
)


class TestGradientDescent:
    def test_quadratic_bowl(self, quadratic_bowl):
        f, grad, _hess, x_star = quadratic_bowl
        res = gradient_descent(f, grad, x0=np.array([5.0, -3.0]), lr=0.1, tol=1e-8)
        np.testing.assert_allclose(res.x, x_star, atol=1e-6)
        assert res.converged

    def test_finite_difference_grad(self, quadratic_bowl):
        f, _grad, _hess, x_star = quadratic_bowl
        res = gradient_descent(f, None, x0=np.array([2.0, -1.0]), lr=0.1, tol=1e-6)
        np.testing.assert_allclose(res.x, x_star, atol=1e-4)

    def test_momentum_accelerates(self, quadratic_bowl):
        f, grad, _hess, _x = quadratic_bowl
        x0 = np.array([5.0, -3.0])
        plain = gradient_descent(f, grad, x0=x0, lr=0.05, tol=1e-8, max_iter=10_000)
        with_mom = gradient_descent(f, grad, x0=x0, lr=0.05, momentum=0.85, tol=1e-8, max_iter=10_000)
        assert with_mom.iterations < plain.iterations

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            gradient_descent(lambda x: 0.0, lambda x: x, x0=np.zeros(2), lr=0)

    def test_invalid_momentum(self):
        with pytest.raises(ValueError):
            gradient_descent(lambda x: 0.0, lambda x: x, x0=np.zeros(2), momentum=1.5)

    def test_max_iter_exceeded(self, quadratic_bowl):
        f, grad, _h, _x = quadratic_bowl
        with pytest.raises(ConvergenceError):
            gradient_descent(f, grad, x0=np.array([5.0, -3.0]), lr=0.1, tol=1e-30, max_iter=2)


class TestBacktracking:
    def test_quadratic_bowl(self, quadratic_bowl):
        f, grad, _h, x_star = quadratic_bowl
        res = gradient_descent_backtracking(f, grad, x0=np.array([5.0, -3.0]), tol=1e-8)
        np.testing.assert_allclose(res.x, x_star, atol=1e-6)

    def test_rosenbrock(self, rosenbrock):
        # Plain gradient descent on Rosenbrock is slow; verify substantial
        # progress to the optimum within a reasonable budget.
        f, grad, _h, x_star = rosenbrock
        res = gradient_descent_backtracking(
            f, grad, x0=np.array([-1.2, 1.0]), tol=1e-3, max_iter=20_000
        )
        np.testing.assert_allclose(res.x, x_star, atol=1e-2)

    def test_invalid_c1(self, quadratic_bowl):
        f, grad, _h, _x = quadratic_bowl
        with pytest.raises(ValueError):
            gradient_descent_backtracking(f, grad, x0=np.zeros(2), c1=2.0)

    def test_invalid_rho(self, quadratic_bowl):
        f, grad, _h, _x = quadratic_bowl
        with pytest.raises(ValueError):
            gradient_descent_backtracking(f, grad, x0=np.zeros(2), rho=0.0)


class TestNewtonMinimize:
    def test_quadratic_one_step(self, quadratic_bowl):
        f, grad, hess, x_star = quadratic_bowl
        res = newton_minimize(f, grad, hess, x0=np.array([5.0, -3.0]), tol=1e-12)
        np.testing.assert_allclose(res.x, x_star, atol=1e-9)
        # Newton must converge in 1 iteration on a strict quadratic.
        assert res.iterations <= 1

    def test_rosenbrock_quadratic_convergence(self, rosenbrock):
        f, grad, hess, x_star = rosenbrock
        res = newton_minimize(f, grad, hess, x0=np.array([1.2, 1.2]), tol=1e-10)
        np.testing.assert_allclose(res.x, x_star, atol=1e-6)
        # Verify error sequence shrinks superlinearly near the optimum.
        errs = np.array([np.linalg.norm(h - x_star) for h in res.history])
        # At least the last few should be decreasing rapidly
        late = errs[errs > 1e-12]
        if len(late) > 3:
            assert late[-1] < late[-2] ** 1.3 * 10

    def test_finite_diff_fallbacks(self, quadratic_bowl):
        f, _g, _h, x_star = quadratic_bowl
        res = newton_minimize(f, None, None, x0=np.array([3.0, -2.0]), tol=1e-6)
        np.testing.assert_allclose(res.x, x_star, atol=1e-4)

    def test_invalid_damping(self, quadratic_bowl):
        f, grad, hess, _x = quadratic_bowl
        with pytest.raises(ValueError):
            newton_minimize(f, grad, hess, x0=np.zeros(2), damping=-1.0)
