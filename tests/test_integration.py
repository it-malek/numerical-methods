"""Tests for numethods.integration."""

from __future__ import annotations

import numpy as np
import pytest

from numethods.integration import (
    adaptive_simpson,
    monte_carlo,
    romberg,
    simpson,
    trapezoid,
)


class TestTrapezoid:
    def test_linear_exact(self):
        # Trapezoid is exact for linear functions.
        res = trapezoid(lambda x: 2 * x + 3, 0.0, 1.0, n=4)
        assert res.value == pytest.approx(4.0, abs=1e-12)

    def test_sin_over_pi(self):
        res = trapezoid(np.sin, 0.0, np.pi, n=200)
        assert res.value == pytest.approx(2.0, abs=1e-3)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            trapezoid(np.sin, 0.0, 1.0, n=0)

    def test_second_order_convergence(self):
        f = lambda x: np.exp(x)
        truth = np.e - 1.0
        ns = np.array([4, 16, 64, 256])
        errs = np.array([abs(trapezoid(f, 0.0, 1.0, n).value - truth) for n in ns])
        # error should drop ~16x when n quadruples.
        ratios = errs[:-1] / errs[1:]
        assert np.all(ratios > 12)


class TestSimpson:
    def test_cubic_exact(self):
        # Simpson is exact for cubics.
        f = lambda x: x**3 + 2 * x**2 - x + 1
        truth = 1.0 / 4 + 2.0 / 3 - 0.5 + 1.0
        res = simpson(f, 0.0, 1.0, n=2)
        assert res.value == pytest.approx(truth, abs=1e-12)

    def test_sin_over_pi(self):
        res = simpson(np.sin, 0.0, np.pi, n=20)
        assert res.value == pytest.approx(2.0, abs=1e-4)

    def test_must_be_even(self):
        with pytest.raises(ValueError):
            simpson(np.sin, 0.0, 1.0, n=3)
        with pytest.raises(ValueError):
            simpson(np.sin, 0.0, 1.0, n=0)

    def test_fourth_order_convergence(self):
        f = lambda x: np.exp(x)
        truth = np.e - 1.0
        ns = np.array([4, 8, 16, 32])
        errs = np.array([abs(simpson(f, 0.0, 1.0, n).value - truth) for n in ns])
        ratios = errs[:-1] / errs[1:]
        # halving h reduces error by ~16 for O(h^4)
        assert np.all(ratios > 14)


class TestRomberg:
    def test_smooth_integrand(self):
        res = romberg(np.sin, 0.0, np.pi, max_levels=6)
        assert res.value == pytest.approx(2.0, abs=1e-10)

    def test_history_returned(self):
        res = romberg(lambda x: x**2, 0.0, 1.0, max_levels=4)
        assert len(res.history) == 4
        assert res.value == pytest.approx(1.0 / 3, abs=1e-12)

    def test_invalid_levels(self):
        with pytest.raises(ValueError):
            romberg(np.sin, 0.0, 1.0, max_levels=0)


class TestMonteCarlo:
    def test_seeded_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = monte_carlo(np.sin, 0.0, np.pi, 1000, rng=rng1).value
        b = monte_carlo(np.sin, 0.0, np.pi, 1000, rng=rng2).value
        assert a == b

    def test_converges_to_truth(self):
        rng = np.random.default_rng(0)
        res = monte_carlo(np.sin, 0.0, np.pi, 100_000, rng=rng)
        assert res.value == pytest.approx(2.0, abs=5e-2)
        assert res.error_estimate is not None
        assert res.error_estimate > 0

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            monte_carlo(np.sin, 0.0, 1.0, n_samples=0)


class TestAdaptiveSimpson:
    def test_smooth_function(self):
        res = adaptive_simpson(np.sin, 0.0, np.pi, tol=1e-10)
        assert res.value == pytest.approx(2.0, abs=1e-9)

    def test_tight_tolerance(self):
        res = adaptive_simpson(lambda x: np.exp(x), 0.0, 1.0, tol=1e-12)
        assert res.value == pytest.approx(np.e - 1.0, abs=1e-10)

    def test_invalid_tol(self):
        with pytest.raises(ValueError):
            adaptive_simpson(np.sin, 0.0, 1.0, tol=0.0)

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            adaptive_simpson(np.sin, 0.0, 1.0, max_depth=0)
