"""Tests for numethods.rootfinding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from numethods import ConvergenceError
from numethods.rootfinding import bisection, fixed_point, newton, secant


class TestBisection:
    def test_sqrt_two(self):
        r = bisection(lambda x: x**2 - 2, 0.0, 2.0, tol=1e-12)
        assert r.root == pytest.approx(math.sqrt(2), abs=1e-10)
        assert r.converged
        assert r.method == "bisection"

    def test_endpoint_is_root(self):
        r = bisection(lambda x: x - 1.0, 1.0, 2.0)
        assert r.root == 1.0
        assert r.iterations == 0

    def test_no_sign_change_raises(self):
        with pytest.raises(ValueError, match="opposite signs"):
            bisection(lambda x: x**2 + 1, -1.0, 1.0)

    def test_max_iter_exceeded(self):
        with pytest.raises(ConvergenceError) as info:
            bisection(lambda x: x**2 - 2, 0.0, 2.0, tol=1e-30, max_iter=5)
        assert info.value.result is not None
        assert not info.value.result.converged

    def test_invalid_tol(self):
        with pytest.raises(ValueError):
            bisection(lambda x: x, -1, 1, tol=0)


class TestNewton:
    def test_sqrt_two_with_derivative(self):
        r = newton(lambda x: x**2 - 2, x0=1.0, fprime=lambda x: 2 * x)
        assert r.root == pytest.approx(math.sqrt(2), abs=1e-12)
        assert r.converged

    def test_finite_difference_fallback(self):
        r = newton(lambda x: x**2 - 2, x0=1.0)
        assert r.root == pytest.approx(math.sqrt(2), abs=1e-7)

    def test_cos_minus_x(self):
        r = newton(
            lambda x: math.cos(x) - x,
            x0=0.5,
            fprime=lambda x: -math.sin(x) - 1,
        )
        assert r.root == pytest.approx(0.7390851332151607, abs=1e-12)

    def test_quadratic_convergence(self):
        # |e_{k+1}| ~ C * e_k^2 for Newton near a simple root.
        r = newton(
            lambda x: x**2 - 2,
            x0=2.0,
            fprime=lambda x: 2 * x,
        )
        truth = math.sqrt(2)
        errs = np.abs(np.array(r.history) - truth)
        # Look at consecutive error ratios - they should drop superlinearly.
        ratios = errs[1:] / errs[:-1] ** 2
        # Ratios should be roughly bounded - not blow up
        finite = ratios[np.isfinite(ratios) & (errs[:-1] > 1e-12)]
        assert np.all(finite < 5)

    def test_zero_derivative_raises(self):
        # f'(0) = 0 but f(0) = -1, so we hit the zero-derivative branch.
        with pytest.raises(ConvergenceError, match="derivative"):
            newton(lambda x: x**3 - 1.0, x0=0.0, fprime=lambda x: 3 * x**2)


class TestSecant:
    def test_sqrt_two(self):
        r = secant(lambda x: x**2 - 2, 0.0, 2.0)
        assert r.root == pytest.approx(math.sqrt(2), abs=1e-10)

    def test_equal_initial_raises(self):
        with pytest.raises(ValueError):
            secant(lambda x: x, 1.0, 1.0)

    def test_max_iter_exceeded(self):
        with pytest.raises(ConvergenceError):
            secant(lambda x: x**2 - 2, 0.0, 2.0, tol=1e-30, max_iter=2)


class TestFixedPoint:
    def test_cos_iteration(self):
        # Famous fixed point: cos(x) = x at ~0.739.
        r = fixed_point(math.cos, x0=0.5)
        assert r.root == pytest.approx(0.7390851332151607, abs=1e-9)
        assert r.converged

    def test_diverges_when_g_expands(self):
        # g(x) = 2x has fixed point at 0 but iteration diverges from any nonzero x0.
        with pytest.raises(ConvergenceError, match="diverge"):
            fixed_point(lambda x: 2 * x, x0=0.5)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError):
            fixed_point(math.cos, x0=0.5, max_iter=0)
