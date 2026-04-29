"""Tests for numethods.regression."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLR
from sklearn.linear_model import Ridge as SkRidge

from numethods.regression import OLSRegression, RidgeRegression, polynomial_features


@pytest.fixture
def synth_linear(rng):
    n, d = 100, 4
    X = rng.standard_normal((n, d))
    true_beta = np.array([1.5, -2.0, 0.5, 0.0])
    y = X @ true_beta + 3.0 + 0.05 * rng.standard_normal(n)
    return X, y, true_beta, 3.0


class TestPolynomialFeatures:
    def test_degree_three(self):
        x = np.array([1.0, 2.0, 3.0])
        out = polynomial_features(x, degree=3)
        np.testing.assert_allclose(
            out,
            np.array([[1, 1, 1], [2, 4, 8], [3, 9, 27]], dtype=float),
        )

    def test_with_bias(self):
        out = polynomial_features(np.array([1.0, 2.0]), degree=2, include_bias=True)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[:, 0], 1.0)

    def test_invalid_degree(self):
        with pytest.raises(ValueError):
            polynomial_features(np.array([1.0]), degree=0)


class TestOLSRegression:
    def test_qr_matches_sklearn(self, synth_linear):
        X, y, _, _ = synth_linear
        model = OLSRegression(method="qr").fit(X, y)
        ref = SkLR().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, atol=1e-9)
        assert model.intercept_ == pytest.approx(ref.intercept_, abs=1e-9)

    def test_normal_matches_qr(self, synth_linear):
        X, y, _, _ = synth_linear
        m_qr = OLSRegression(method="qr").fit(X, y)
        m_n = OLSRegression(method="normal").fit(X, y)
        np.testing.assert_allclose(m_qr.coef_, m_n.coef_, atol=1e-9)

    def test_no_intercept(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        model = OLSRegression(method="qr", fit_intercept=False).fit(X, y)
        assert model.coef_[0] == pytest.approx(2.0)
        assert model.intercept_ == 0.0

    def test_score_perfect_fit(self, synth_linear):
        X, _, true_beta, intercept = synth_linear
        # Construct y exactly from true_beta to get R^2 = 1.
        y_true = X @ true_beta + intercept
        model = OLSRegression().fit(X, y_true)
        assert model.score(X, y_true) == pytest.approx(1.0, abs=1e-10)

    def test_predict_before_fit(self):
        with pytest.raises(RuntimeError):
            OLSRegression().predict(np.array([[1.0]]))

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            OLSRegression(method="lol")

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            OLSRegression().fit(np.ones((5, 2)), np.ones(7))


class TestRidgeRegression:
    def test_alpha_zero_matches_ols(self, synth_linear):
        X, y, _, _ = synth_linear
        m_ridge = RidgeRegression(alpha=0.0).fit(X, y)
        m_ols = OLSRegression().fit(X, y)
        np.testing.assert_allclose(m_ridge.coef_, m_ols.coef_, atol=1e-9)
        assert m_ridge.intercept_ == pytest.approx(m_ols.intercept_, abs=1e-9)

    def test_matches_sklearn_ridge(self, synth_linear):
        X, y, _, _ = synth_linear
        alpha = 2.5
        ours = RidgeRegression(alpha=alpha).fit(X, y)
        ref = SkRidge(alpha=alpha, solver="cholesky").fit(X, y)
        np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-8)
        assert ours.intercept_ == pytest.approx(ref.intercept_, abs=1e-8)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            RidgeRegression(alpha=-1.0)

    def test_predict_before_fit(self):
        with pytest.raises(RuntimeError):
            RidgeRegression().predict(np.array([[1.0]]))

    def test_score(self, synth_linear):
        X, y, _, _ = synth_linear
        model = RidgeRegression(alpha=0.1).fit(X, y)
        assert model.score(X, y) > 0.95
