"""Tests for numethods.linsys."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as spl

from numethods import ConvergenceError
from numethods.linsys import (
    conjugate_gradient,
    gauss_elimination,
    gauss_seidel,
    jacobi,
    lu_decomposition,
    lu_solve,
    sor,
)


def diagonally_dominant(rng, n=6):
    """Construct an n x n strictly diagonally dominant matrix."""
    A = rng.standard_normal((n, n))
    A = A + n * np.eye(n) * np.sign(np.diag(A) + 1e-12)
    # Force diagonal dominance
    A = A + np.diag(np.abs(A).sum(axis=1))
    return A


class TestLU:
    def test_pa_equals_lu(self, rng):
        A = rng.standard_normal((5, 5))
        L, U, P = lu_decomposition(A)
        np.testing.assert_allclose(P @ A, L @ U, atol=1e-10)
        # L is unit lower-triangular
        assert np.allclose(np.tril(L), L)
        np.testing.assert_allclose(np.diag(L), np.ones(5))
        # U is upper-triangular
        assert np.allclose(np.triu(U), U)

    def test_lu_solve_matches_scipy(self, rng):
        A = rng.standard_normal((5, 5))
        b = rng.standard_normal(5)
        L, U, P = lu_decomposition(A)
        x = lu_solve(L, U, P, b)
        np.testing.assert_allclose(x, spl.solve(A, b), atol=1e-9)

    def test_singular_raises(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(ConvergenceError, match="singular"):
            lu_decomposition(A)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            lu_decomposition(np.ones((3, 4)))


class TestGaussElimination:
    def test_matches_scipy(self, rng):
        A = rng.standard_normal((6, 6))
        b = rng.standard_normal(6)
        x = gauss_elimination(A, b)
        np.testing.assert_allclose(x, spl.solve(A, b), atol=1e-9)

    def test_b_dimension_mismatch(self):
        with pytest.raises(ValueError):
            gauss_elimination(np.eye(3), np.ones(4))


class TestJacobi:
    def test_diagonally_dominant_converges(self, rng):
        A = diagonally_dominant(rng, n=5)
        b = rng.standard_normal(5)
        res = jacobi(A, b, tol=1e-10)
        assert res.converged
        np.testing.assert_allclose(res.x, np.linalg.solve(A, b), atol=1e-8)
        assert len(res.history) == res.iterations

    def test_non_dominant_diverges(self):
        # This matrix is symmetric but NOT diagonally dominant; Jacobi diverges.
        A = np.array([[1.0, 2.0], [2.0, 1.0]])
        b = np.array([3.0, 3.0])
        with pytest.raises(ConvergenceError):
            jacobi(A, b, tol=1e-10, max_iter=20)

    def test_zero_diagonal_raises(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            jacobi(A, np.array([1.0, 1.0]))


class TestGaussSeidel:
    def test_diagonally_dominant_converges(self, rng):
        A = diagonally_dominant(rng, n=5)
        b = rng.standard_normal(5)
        res = gauss_seidel(A, b, tol=1e-10)
        assert res.converged
        np.testing.assert_allclose(res.x, np.linalg.solve(A, b), atol=1e-8)

    def test_faster_than_jacobi(self, rng):
        A = diagonally_dominant(rng, n=5)
        b = rng.standard_normal(5)
        gs = gauss_seidel(A, b, tol=1e-8)
        jc = jacobi(A, b, tol=1e-8)
        # Gauss-Seidel should converge in at most as many iterations.
        assert gs.iterations <= jc.iterations


class TestSOR:
    def test_omega_one_matches_gauss_seidel(self, rng):
        A = diagonally_dominant(rng, n=5)
        b = rng.standard_normal(5)
        res = sor(A, b, omega=1.0, tol=1e-10)
        np.testing.assert_allclose(res.x, np.linalg.solve(A, b), atol=1e-8)

    def test_invalid_omega(self, rng):
        A = diagonally_dominant(rng)
        b = np.ones(A.shape[0])
        with pytest.raises(ValueError):
            sor(A, b, omega=0.0)
        with pytest.raises(ValueError):
            sor(A, b, omega=2.5)


class TestConjugateGradient:
    def test_spd_solve(self, spd_matrix, rng):
        A = spd_matrix
        b = rng.standard_normal(A.shape[0])
        res = conjugate_gradient(A, b, tol=1e-12)
        np.testing.assert_allclose(res.x, np.linalg.solve(A, b), atol=1e-8)
        assert res.converged
        # CG converges in <= n iterations in exact arithmetic.
        assert res.iterations <= A.shape[0] + 1

    def test_history_decreases(self, spd_matrix):
        b = np.ones(spd_matrix.shape[0])
        res = conjugate_gradient(spd_matrix, b, tol=1e-12)
        # Residuals should be (mostly) monotonically decreasing.
        h = np.array(res.history)
        # allow small numerical noise
        decreases = (h[1:] / h[:-1] < 1.5).mean()
        assert decreases > 0.6
