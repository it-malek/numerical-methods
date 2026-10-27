"""Small analytic regressions for convergence and truthful diagnostics."""

import math

import numpy as np
import pytest
import sympy as sp

from numethods import ConvergenceError, adaptive_simpson, conjugate_gradient, rk45_adaptive
from numethods.taylor import taylor_remainder_bound


class TestCGContract:
    @pytest.mark.parametrize("b,x0", [([0., 0.], None), ([1., 2.], [1., 2.]),
                                     ([1., 2.], [1. + 1e-12, 2.])])
    def test_initial_convergence(self, b, x0):
        result = conjugate_gradient(np.eye(2), b, x0=x0)
        assert result.converged and result.iterations == 0
        assert result.history == []
        scale = np.linalg.norm(b) or 1.
        assert result.residual_norm == pytest.approx(np.linalg.norm(result.x - b) / scale)

    def test_true_residual_and_iteration_accounting(self):
        A = np.array([[4., 1.], [1., 3.]])
        b = np.array([1., 2.])
        result = conjugate_gradient(A, b, tol=1e-12)
        np.testing.assert_allclose(result.x, [1/11, 7/11], atol=1e-14)
        assert result.residual_norm == pytest.approx(np.linalg.norm(A @ result.x - b) / np.linalg.norm(b), abs=1e-16)
        assert result.iterations == len(result.history) == 2

    def test_zero_rhs_uses_absolute_residual(self):
        result = conjugate_gradient(np.eye(2), [0., 0.], x0=[1e-12, 0.], tol=1e-10)
        assert result.iterations == 0
        assert result.residual_norm == pytest.approx(1e-12)

    @pytest.mark.parametrize("A", [np.zeros((2, 2)), -np.eye(2)])
    def test_curvature_breakdown(self, A):
        with pytest.raises(ConvergenceError, match="curvature") as caught:
            conjugate_gradient(A, [1., 0.])
        result = caught.value.result
        assert not result.converged and result.iterations == 0
        assert result.residual_norm == 1.

    @pytest.mark.parametrize("which", ["A", "b", "x0"])
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_nonfinite_input(self, which, value):
        args = dict(A=np.eye(2), b=np.ones(2), x0=np.zeros(2))
        args[which].flat[0] = value
        with pytest.raises(ValueError, match="finite"):
            conjugate_gradient(**args)

    def test_nonfinite_arithmetic(self):
        with pytest.raises(ConvergenceError, match="nonfinite") as caught:
            conjugate_gradient(np.eye(2) * 1e308, [2., 2.])
        assert not caught.value.result.converged
        assert np.all(np.isfinite(caught.value.result.x))

    def test_residual_product_underflow_is_not_zero_residual(self):
        # Norms remain representable even though squared residuals do not.
        with pytest.raises(ConvergenceError):
            conjugate_gradient(np.eye(1), [1e-200])

    def test_malformed_dimensions(self):
        for A, b, x0 in [(np.ones((2, 3)), [1., 2.], None),
                         (np.eye(2), [1.], None), (np.eye(2), [1., 2.], [1.])]:
            with pytest.raises(ValueError):
                conjugate_gradient(A, b, x0)

    def test_budget_failure(self):
        with pytest.raises(ConvergenceError) as caught:
            conjugate_gradient(np.diag([1., 2.]), [1., 1.], max_iter=1)
        assert caught.value.result.iterations == 1
        assert len(caught.value.result.history) == 1


class TestSimpsonContract:
    @pytest.mark.parametrize("f,truth", [(lambda x: 0*x + 3, 3.), (lambda x: x**3, .25)])
    def test_exact_polynomials_have_computed_zero_error(self, f, truth):
        result = adaptive_simpson(f, 0., 1., tol=1e-7)
        assert result.value == pytest.approx(truth, abs=1e-15)
        assert result.error_estimate == 0.
        assert result.converged and result.requested_tolerance == 1e-7
        assert result.n_evaluations == 5

    def test_local_estimate_formula(self):
        # S(one panel)=5/24, S(two panels)=77/384 for x^4 on [0,1].
        result = adaptive_simpson(lambda x: x**4, 0., 1., tol=1.)
        assert result.error_estimate == pytest.approx(1/1920)
        assert result.value == pytest.approx(1/5)

    def test_accumulates_leaf_errors(self):
        # At depth 1 each half's Richardson estimate is 1/(1920*32).
        with pytest.raises(ConvergenceError) as caught:
            adaptive_simpson(lambda x: x**4, 0., 1., tol=1e-16, max_depth=1)
        result = caught.value.result
        assert result.error_estimate == pytest.approx(1/30720)
        assert result.n_evaluations == 9

    @pytest.mark.parametrize("a,b", [(0., 1.), (1., 0.), (1., 1.)])
    def test_interval_conventions(self, a, b):
        result = adaptive_simpson(np.exp, a, b, tol=1e-10)
        assert result.value == pytest.approx(math.exp(b) - math.exp(a), abs=1e-10)
        assert result.converged and 0 <= result.error_estimate <= 1e-10
        assert result.error_estimate != result.requested_tolerance

    def test_depth_exhaustion_is_failure(self):
        with pytest.raises(ConvergenceError, match="depth") as caught:
            adaptive_simpson(np.exp, 0., 1., tol=1e-16, max_depth=1)
        result = caught.value.result
        assert not result.converged
        assert np.isfinite(result.value)
        assert result.error_estimate > result.requested_tolerance

    @pytest.mark.parametrize("kwargs", [{"tol": np.nan}, {"tol": np.inf}, {"tol": 0.},
                                        {"max_depth": True}, {"max_depth": 1.5},
                                        {"max_depth": 0}, {"a": np.inf}, {"b": np.nan}])
    def test_invalid_parameters(self, kwargs):
        args = dict(a=0., b=1.) | kwargs
        with pytest.raises(ValueError):
            adaptive_simpson(np.exp, **args)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_evaluation(self, bad):
        with pytest.raises(ConvergenceError, match="nonfinite"):
            adaptive_simpson(lambda x: np.full_like(x, bad), 0., 1.)

    def test_nonfinite_refinement_evaluation(self):
        with pytest.raises(ConvergenceError, match="nonfinite") as caught:
            adaptive_simpson(lambda x: np.where(x == .25, np.nan, np.exp(x)), 0., 1.)
        assert caught.value.result is None

    def test_nonfinite_arithmetic_from_finite_values(self):
        with pytest.raises(ConvergenceError, match="nonfinite"):
            adaptive_simpson(lambda x: np.full_like(x, 1e308), 0., 1.)

    def test_malformed_output(self):
        with pytest.raises(ValueError, match="length-1"):
            adaptive_simpson(lambda x: np.array([1., 2.]), 0., 1.)

    def test_spacing_exhaustion(self):
        with pytest.raises(ConvergenceError, match="spacing") as caught:
            adaptive_simpson(np.exp, 1., np.nextafter(1., np.inf))
        assert not caught.value.result.converged
        assert caught.value.result.error_estimate == np.inf


class TestRK45Contract:
    @pytest.mark.parametrize("kwargs", [{"h0": 0.}, {"h0": -1.}, {"h0": np.nan},
                                        {"h0": np.inf}, {"rtol": np.nan}, {"atol": np.inf},
                                        {"atol": -1.}, {"max_steps": 0}, {"max_steps": 1.5},
                                        {"max_steps": True}, {"rtol": 0.}])
    def test_invalid_controls(self, kwargs):
        with pytest.raises(ValueError):
            rk45_adaptive(lambda t, y: -y, (0., 1.), 1., **kwargs)

    @pytest.mark.parametrize("span,y0", [((0., np.inf), 1.), ((np.nan, 1.), 1.),
                                        ((0., 1.), np.nan), ((1., 0.), 1.)])
    def test_invalid_problem(self, span, y0):
        with pytest.raises(ValueError):
            rk45_adaptive(lambda t, y: -y, span, y0)

    def test_endpoint_on_last_attempt(self):
        result = rk45_adaptive(lambda t, y: 1., (0., 1.), 0., h0=1., max_steps=1)
        assert result.success and result.t[-1] == 1.
        assert result.y[-1] == pytest.approx(1.)
        assert result.n_evaluations == 7

    def test_real_budget_exhaustion(self):
        with pytest.raises(ConvergenceError, match="steps") as caught:
            rk45_adaptive(lambda t, y: 1., (0., 1.), 0., h0=.1, max_steps=1)
        assert not caught.value.result.success
        assert caught.value.result.t[-1] == .1
        assert caught.value.result.n_evaluations == 7

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_rhs(self, bad):
        with pytest.raises(ConvergenceError, match="nonfinite") as caught:
            rk45_adaptive(lambda t, y: bad, (0., 1e-15), 1., h0=1e-15)
        assert not caught.value.result.success
        np.testing.assert_array_equal(caught.value.result.y, [1.])
        assert caught.value.result.n_evaluations == 1

    def test_unrepresentable_step(self):
        with pytest.raises(ConvergenceError, match="underflow") as caught:
            rk45_adaptive(lambda t, y: 0., (1., 2.), 1., h0=1e-30)
        assert caught.value.result.t.tolist() == [1.]
        assert caught.value.result.n_evaluations == 0

    def test_later_stage_failure_counts_evaluations(self):
        with pytest.raises(ConvergenceError, match="nonfinite") as caught:
            rk45_adaptive(lambda t, y: 0. if t == 0. else np.nan,
                          (0., 1.), 1., h0=.1)
        assert caught.value.result.n_evaluations == 2
        assert caught.value.result.t.tolist() == [0.]

    def test_intermediate_overflow(self):
        with pytest.raises(ConvergenceError, match="nonfinite") as caught:
            rk45_adaptive(lambda t, y: np.full_like(y, 1e308),
                          (0., 10.), 1e308, h0=10.)
        assert not caught.value.result.success
        assert np.all(np.isfinite(caught.value.result.y))

    def test_malformed_rhs(self):
        with pytest.raises(ValueError, match="size"):
            rk45_adaptive(lambda t, y: [1., 2., 3.], (0., 1.), [1., 2.])

    def test_rejected_small_step_is_not_force_accepted(self):
        # One ULP at t=1e-3 is far smaller than the old 1e-14 cutoff.
        t0 = 1e-3
        tf = np.nextafter(t0, np.inf)
        h = tf - t0
        with pytest.raises(ConvergenceError, match="underflow") as caught:
            rk45_adaptive(lambda t, y: y/h, (t0, tf), 1., h0=h,
                          rtol=1e-12, atol=1e-14, max_steps=2)
        assert caught.value.result.t.tolist() == [t0]

    def test_rejected_step_shrinks_its_representable_endpoint(self):
        # The first error ratio is just over 1; multiplying a 2-ULP step by
        # the controller factor rounds back to the same 2-ULP endpoint.
        h = 2*np.spacing(1.)
        result = rk45_adaptive(lambda t, y: y/h, (1., 1. + h), 1., h0=h,
                               rtol=1e-14, atol=5e-4, max_steps=3)
        assert result.success and result.t[-1] == 1. + h
        assert result.n_evaluations == 21  # reject, then two 1-ULP steps
        assert result.y[-1] == pytest.approx(math.e, rel=1e-4)

    @pytest.mark.parametrize("scale", [1e-20, 1., 1e20])
    def test_scale_aware_progress(self, scale):
        result = rk45_adaptive(lambda t, y: 1/scale, (0., scale), 0., h0=scale, max_steps=1)
        assert result.success and result.t[-1] == scale
        assert result.y[-1] == pytest.approx(1.)

    def test_scalar_and_vector_analytic(self):
        for y0 in [1., [1., 2.]]:
            result = rk45_adaptive(lambda t, y: -y, (0., 2.), y0, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(result.y[-1], np.asarray(y0)*math.exp(-2), rtol=1e-7, atol=1e-9)
            assert result.success and result.t[-1] == 2.
            assert np.all(np.diff(result.t) > 0)


class TestTaylorContract:
    def test_bound_requires_M(self):
        with pytest.raises(ValueError, match="M"):
            taylor_remainder_bound(math.sin, 0., 3, .5)

    def test_missed_peak_is_only_an_estimate(self):
        from numethods import taylor_remainder_estimate

        x = sp.Symbol("x")
        center = sp.Rational(1, 2) + sp.Rational(1, 512)
        f = sp.erf(100000*(x-center))
        estimate = taylor_remainder_estimate(f, 0., 0, 1.)
        actual = math.erf(100000*(1-float(center))) - math.erf(-100000*float(center))
        assert actual == 2. and estimate < 1e-20
        with pytest.raises(ValueError, match="M"):
            taylor_remainder_bound(f, 0., 0, 1.)
        # Analytic sup |f'| = 2*100000/sqrt(pi), independently of the grid.
        assert taylor_remainder_bound(f, 0., 0, 1., M=200000/math.sqrt(math.pi)) > actual

    @pytest.mark.parametrize("a,x", [(0., .5), (.5, 0.), (.5, .5)])
    def test_analytic_bound_formula(self, a, x):
        assert taylor_remainder_bound(math.sin, a, 3, x, M=1.) == pytest.approx(abs(x-a)**4/24)

    def test_smooth_estimates_both_derivative_paths(self):
        from numethods.taylor import taylor_remainder_estimate

        x = sp.Symbol("x")
        for a, target in [(0., 1.), (1., 0.)]:
            symbolic = taylor_remainder_estimate(sp.exp(x), a, 0, target)
            numeric = taylor_remainder_estimate(math.exp, a, 0, target)
            assert symbolic == pytest.approx(math.e)
            assert numeric == pytest.approx(math.e, rel=1e-6)

    @pytest.mark.parametrize("M", [-1., np.nan, np.inf])
    def test_invalid_M(self, M):
        with pytest.raises(ValueError):
            taylor_remainder_bound(math.sin, 0., 3, .5, M=M)

    @pytest.mark.parametrize("n", [-1, 1.5, True])
    def test_invalid_degree(self, n):
        from numethods.taylor import taylor_remainder_estimate

        for helper in [taylor_remainder_estimate, taylor_remainder_bound]:
            with pytest.raises(ValueError):
                helper(math.sin, 0., n, 1.)

    def test_constant_symbolic_derivative_and_zero_interval(self):
        from numethods.taylor import taylor_remainder_estimate

        x = sp.Symbol("x")
        assert taylor_remainder_estimate(x**2, 0., 1, 2.) == pytest.approx(4.)
        assert taylor_remainder_estimate(sp.exp(x), 1., 4, 1.) == 0.

    def test_nonfinite_derivative_samples(self):
        from numethods.taylor import taylor_remainder_estimate

        with pytest.raises(ValueError, match="finite"):
            taylor_remainder_estimate(lambda x: np.nan, 0., 0, 1.)
