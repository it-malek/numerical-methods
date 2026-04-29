"""Tests for numethods.ode."""

from __future__ import annotations

import numpy as np
import pytest

from numethods.ode import euler, midpoint, rk4, rk45_adaptive


def _conv_rate(errs, hs):
    """Estimate empirical order from successive halvings of h."""
    log_e = np.log(errs)
    log_h = np.log(hs)
    return np.polyfit(log_h, log_e, 1)[0]


class TestEuler:
    def test_exp_decay(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        sol = euler(f, (0.0, 1.0), y0, n_steps=1000)
        assert abs(sol.y[-1] - exact(1.0)) < 0.01
        assert sol.t.shape == sol.y.shape == (1001,)

    def test_first_order_convergence(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        ns = [50, 100, 200, 400]
        hs = np.array([1.0 / n for n in ns])
        errs = np.array([abs(euler(f, (0.0, 1.0), y0, n).y[-1] - exact(1.0)) for n in ns])
        order = _conv_rate(errs, hs)
        assert order == pytest.approx(1.0, abs=0.15)

    def test_invalid_steps(self, exp_decay_ode):
        f, y0, _ = exp_decay_ode
        with pytest.raises(ValueError):
            euler(f, (0.0, 1.0), y0, n_steps=0)

    def test_invalid_span(self, exp_decay_ode):
        f, y0, _ = exp_decay_ode
        with pytest.raises(ValueError):
            euler(f, (1.0, 0.0), y0, n_steps=10)


class TestMidpoint:
    def test_exp_decay(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        sol = midpoint(f, (0.0, 1.0), y0, n_steps=200)
        assert abs(sol.y[-1] - exact(1.0)) < 1e-4

    def test_second_order_convergence(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        ns = [50, 100, 200, 400]
        hs = np.array([1.0 / n for n in ns])
        errs = np.array([abs(midpoint(f, (0.0, 1.0), y0, n).y[-1] - exact(1.0)) for n in ns])
        order = _conv_rate(errs, hs)
        assert order == pytest.approx(2.0, abs=0.2)


class TestRK4:
    def test_exp_decay(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        sol = rk4(f, (0.0, 1.0), y0, n_steps=50)
        assert abs(sol.y[-1] - exact(1.0)) < 1e-7

    def test_fourth_order_convergence(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        ns = [10, 20, 40, 80]
        hs = np.array([1.0 / n for n in ns])
        errs = np.array([abs(rk4(f, (0.0, 1.0), y0, n).y[-1] - exact(1.0)) for n in ns])
        order = _conv_rate(errs, hs)
        assert order == pytest.approx(4.0, abs=0.3)

    def test_vector_system(self, harmonic_ode):
        f, y0, exact = harmonic_ode
        sol = rk4(f, (0.0, 2.0), y0, n_steps=200)
        np.testing.assert_allclose(sol.y[-1], exact(2.0), atol=1e-6)
        assert sol.y.shape == (201, 2)


class TestRK45Adaptive:
    def test_exp_decay_tight(self, exp_decay_ode):
        f, y0, exact = exp_decay_ode
        sol = rk45_adaptive(f, (0.0, 1.0), y0, rtol=1e-10, atol=1e-12)
        assert abs(sol.y[-1] - exact(1.0)) < 1e-8

    def test_harmonic_oscillator(self, harmonic_ode):
        f, y0, exact = harmonic_ode
        sol = rk45_adaptive(f, (0.0, np.pi), y0, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sol.y[-1], exact(np.pi), atol=1e-6)

    def test_lotka_volterra_runs(self):
        # alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
        def f(_t, y):
            return np.array([1.0 * y[0] - 0.1 * y[0] * y[1],
                             0.075 * y[0] * y[1] - 1.5 * y[1]])

        sol = rk45_adaptive(f, (0.0, 15.0), np.array([10.0, 5.0]),
                            rtol=1e-7, atol=1e-9)
        assert sol.success
        # Populations remain positive and bounded.
        assert np.all(sol.y > 0)
        assert np.all(sol.y < 1000)

    def test_invalid_tol(self, exp_decay_ode):
        f, y0, _ = exp_decay_ode
        with pytest.raises(ValueError):
            rk45_adaptive(f, (0.0, 1.0), y0, rtol=0.0)

    def test_invalid_span(self, exp_decay_ode):
        f, y0, _ = exp_decay_ode
        with pytest.raises(ValueError):
            rk45_adaptive(f, (1.0, 0.0), y0)
