"""Tests for numethods.taylor."""

from __future__ import annotations

import math

import pytest
import sympy as sp

from numethods.taylor import taylor_polynomial, taylor_remainder_bound


class TestTaylorSymbolic:
    def test_exp_at_zero(self):
        x = sp.Symbol("x")
        # T_4 of exp at 0 = 1 + x + x^2/2 + x^3/6 + x^4/24
        val = taylor_polynomial(sp.exp(x), 0.0, 4, 0.5)
        expected = 1 + 0.5 + 0.5**2 / 2 + 0.5**3 / 6 + 0.5**4 / 24
        assert val == pytest.approx(expected, rel=1e-12)

    def test_sin_matches_truth_to_high_order(self):
        x = sp.Symbol("x")
        for xv in [-0.4, -0.1, 0.2, 0.6]:
            approx = taylor_polynomial(sp.sin(x), 0.0, 11, xv)
            assert approx == pytest.approx(math.sin(xv), abs=1e-10)

    def test_cos_even_orders_only(self):
        x = sp.Symbol("x")
        # cos has only even-order derivatives nonzero at 0; T_5 == T_4
        v4 = taylor_polynomial(sp.cos(x), 0.0, 4, 0.3)
        v5 = taylor_polynomial(sp.cos(x), 0.0, 5, 0.3)
        assert v4 == pytest.approx(v5, rel=1e-15)

    def test_returns_callable_when_x_none(self):
        x = sp.Symbol("x")
        p = taylor_polynomial(sp.cos(x), 0.0, 6)
        assert callable(p)
        assert p(0.0) == pytest.approx(1.0)

    def test_ambiguous_symbol_raises(self):
        x, y = sp.symbols("x y")
        with pytest.raises(ValueError, match="multiple free symbols"):
            taylor_polynomial(x + y, 0.0, 2, 0.1)


class TestTaylorNumerical:
    def test_polynomial_recovered_exactly(self):
        # T_3 of a cubic must match the cubic itself.
        f = lambda x: 1 + 2 * x - 3 * x**2 + 0.5 * x**3
        for xv in [-0.4, -0.1, 0.0, 0.3, 0.7]:
            approx = taylor_polynomial(f, 0.0, 3, xv, h=1e-2)
            assert approx == pytest.approx(f(xv), abs=1e-3)

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            taylor_polynomial(math.sin, 0.0, -1, 0.1)


class TestTaylorRemainder:
    def test_bound_is_actual_upper_bound(self):
        x = sp.Symbol("x")
        # Compare actual error to bound for sin near 0
        for n in [3, 5, 7]:
            for xv in [0.2, 0.4, 0.6]:
                approx = taylor_polynomial(sp.sin(x), 0.0, n, xv)
                actual_err = abs(math.sin(xv) - approx)
                bound = taylor_remainder_bound(sp.sin(x), 0.0, n, xv)
                assert actual_err <= bound + 1e-15

    def test_zero_at_expansion_point(self):
        x = sp.Symbol("x")
        assert taylor_remainder_bound(sp.cos(x), 1.0, 4, 1.0) == pytest.approx(0.0)

    def test_user_supplied_M(self):
        # M=1 for sin; |R_3(x)| <= |x|^4 / 24
        bound = taylor_remainder_bound(math.sin, 0.0, 3, 0.5, M=1.0)
        assert bound == pytest.approx(0.5**4 / 24)

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError):
            taylor_remainder_bound(math.sin, 0.0, -1, 0.1, M=1.0)
