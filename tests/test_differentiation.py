"""Tests for numethods.differentiation."""

from __future__ import annotations

import numpy as np
import pytest

from numethods.differentiation import (
    backward_diff,
    central_diff,
    forward_diff,
    numerical_jacobian,
    richardson_extrapolation,
)


class TestForwardDiff:
    def test_sin_at_zero(self):
        # d/dx sin(x) at x=0 is 1
        assert forward_diff(np.sin, 0.0, 1e-6) == pytest.approx(1.0, abs=1e-5)

    def test_array_input(self):
        x = np.array([0.0, np.pi / 4, np.pi / 2])
        out = forward_diff(np.sin, x, 1e-6)
        np.testing.assert_allclose(out, np.cos(x), atol=1e-5)

    def test_invalid_step(self):
        with pytest.raises(ValueError, match="positive"):
            forward_diff(np.sin, 0.0, h=0.0)
        with pytest.raises(ValueError):
            forward_diff(np.sin, 0.0, h=-1e-5)

    def test_first_order_convergence(self):
        # Error should scale linearly with h.
        f = np.sin
        x0 = 1.0
        truth = np.cos(x0)
        hs = np.array([1e-2, 1e-3, 1e-4])
        errs = np.array([abs(forward_diff(f, x0, h) - truth) for h in hs])
        rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
        assert np.all(rates > 0.9)


class TestBackwardDiff:
    def test_sin_at_zero(self):
        assert backward_diff(np.sin, 0.0, 1e-6) == pytest.approx(1.0, abs=1e-5)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            backward_diff(np.sin, 0.0, h=-1e-5)


class TestCentralDiff:
    def test_polynomial_exact_to_quadratic(self):
        # Central diff is exact for quadratics modulo roundoff.
        f = lambda x: 3 * x**2 + 2 * x + 1
        for x0 in [-1.0, 0.0, 2.5]:
            truth = 6 * x0 + 2
            assert central_diff(f, x0, 1e-3) == pytest.approx(truth, abs=1e-9)

    def test_second_order_convergence(self):
        f = np.sin
        x0 = 1.0
        truth = np.cos(x0)
        hs = np.array([1e-1, 1e-2, 1e-3])
        errs = np.array([abs(central_diff(f, x0, h) - truth) for h in hs])
        rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
        assert np.all(rates > 1.9)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            central_diff(np.sin, 0.0, h=0)


class TestRichardson:
    def test_more_accurate_than_central(self):
        f = np.sin
        x0 = 0.7
        truth = np.cos(x0)
        h = 0.1
        c_err = abs(central_diff(f, x0, h) - truth)
        r_err = abs(richardson_extrapolation(f, x0, h) - truth)
        assert r_err < c_err

    def test_fourth_order_convergence(self):
        f = np.sin
        x0 = 1.0
        truth = np.cos(x0)
        hs = np.array([0.4, 0.2, 0.1])
        errs = np.array([abs(richardson_extrapolation(f, x0, h) - truth) for h in hs])
        rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
        assert np.all(rates > 3.5)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            richardson_extrapolation(np.sin, 0.0, h=-0.1)


class TestNumericalJacobian:
    def test_identity_map(self):
        f = lambda v: v
        J = numerical_jacobian(f, np.array([0.5, -1.2, 3.0]))
        np.testing.assert_allclose(J, np.eye(3), atol=1e-6)

    def test_known_jacobian(self):
        # f(x, y) = (x^2 + y, x*y); J = [[2x, 1], [y, x]]
        def f(v):
            return np.array([v[0] ** 2 + v[1], v[0] * v[1]])

        x0 = np.array([1.5, -0.5])
        J = numerical_jacobian(f, x0)
        truth = np.array([[2 * x0[0], 1.0], [x0[1], x0[0]]])
        np.testing.assert_allclose(J, truth, atol=1e-6)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            numerical_jacobian(lambda v: v, np.array([1.0]), h=0)
