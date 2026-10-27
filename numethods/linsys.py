"""Linear systems: LU with partial pivoting, Gauss elimination, Jacobi/GS/SOR, CG.

All implementations are pure NumPy. They are not as fast as ``scipy.linalg``
(which calls into LAPACK), but expose the same algorithmic structure for
study and testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._utils import ConvergenceError, validate_max_iter, validate_tol


@dataclass
class IterativeSolveResult:
    """Outcome of an iterative linear solve.

    Attributes:
        x: Best estimate of the solution.
        iterations: Number of sweeps performed.
        converged: Whether the solver's residual test was achieved.
        residual_norm: Final relative residual norm; absolute when b is zero.
        history: Per-iteration residual norms (for log-log convergence plots).
        method: Algorithm name.
    """

    x: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float
    history: list[float] = field(default_factory=list)
    method: str = ""


def _validate_square(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    return A


def _validate_b(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).ravel()
    if b.size != A.shape[0]:
        raise ValueError(f"b length {b.size} does not match A's {A.shape[0]} rows")
    return b


def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU factorization with partial pivoting: returns ``(L, U, P)``.

    Such that ``P @ A = L @ U``, where ``L`` is unit lower-triangular,
    ``U`` is upper-triangular, and ``P`` is a permutation matrix.

    Raises:
        ValueError: If ``A`` is not square.
        ConvergenceError: If ``A`` is singular (a pivot is zero after pivoting).
    """
    A = _validate_square(A)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n - 1):
        # Partial pivoting: pick the largest |U[i, k]| in rows k..n-1.
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot, k]) < 1e-15:
            raise ConvergenceError(
                f"lu_decomposition: matrix is singular at column {k}",
                result=None,
            )
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        factors = U[k + 1 :, k] / U[k, k]
        L[k + 1 :, k] = factors
        U[k + 1 :, k:] -= np.outer(factors, U[k, k:])
    if abs(U[-1, -1]) < 1e-15:
        raise ConvergenceError(
            "lu_decomposition: matrix is singular (zero final pivot)",
            result=None,
        )
    return L, U, P


def lu_solve(L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` given an LU factorization ``P A = L U``."""
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)
    P = np.asarray(P, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = L.shape[0]
    Pb = P @ b
    # Forward substitution: L y = P b
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]
    # Back substitution: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def gauss_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` by Gauss elimination with partial pivoting.

    A pedagogical convenience wrapper over :func:`lu_decomposition` /
    :func:`lu_solve`.
    """
    A = _validate_square(A)
    b = _validate_b(A, b)
    L, U, P = lu_decomposition(A)
    return lu_solve(L, U, P, b)


def _iterative_setup(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    A = _validate_square(A)
    b = _validate_b(A, b)
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.size != n:
        raise ValueError(f"x0 length {x.size} does not match dimension {n}")
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0:
        bnorm = 1.0  # Avoid division by zero on b == 0 systems.
    return A, b, x, bnorm


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> IterativeSolveResult:
    """Jacobi iteration. Converges when ``A`` is strictly diagonally dominant.

    Update: ``x_{k+1} = D^{-1} (b - (L+U) x_k)`` where ``A = D + L + U``.
    """
    validate_tol(tol)
    validate_max_iter(max_iter)
    A, b, x, bnorm = _iterative_setup(A, b, x0)
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Jacobi requires non-zero diagonal entries in A")
    R = A - np.diag(D)
    history: list[float] = []
    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        res = float(np.linalg.norm(A @ x_new - b) / bnorm)
        history.append(res)
        if res < tol:
            return IterativeSolveResult(x_new, k, True, res, history, "jacobi")
        x = x_new
    raise ConvergenceError(
        f"jacobi did not converge in {max_iter} iterations (rel. residual {res:.3e})",
        result=IterativeSolveResult(x, max_iter, False, res, history, "jacobi"),
    )


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> IterativeSolveResult:
    """Gauss-Seidel iteration: like Jacobi but using updated entries immediately."""
    validate_tol(tol)
    validate_max_iter(max_iter)
    A, b, x, bnorm = _iterative_setup(A, b, x0)
    n = A.shape[0]
    if np.any(np.diag(A) == 0):
        raise ValueError("Gauss-Seidel requires non-zero diagonal entries in A")
    history: list[float] = []
    for k in range(1, max_iter + 1):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1 :] @ x[i + 1 :]
            x[i] = (b[i] - s) / A[i, i]
        res = float(np.linalg.norm(A @ x - b) / bnorm)
        history.append(res)
        if res < tol:
            return IterativeSolveResult(x.copy(), k, True, res, history, "gauss_seidel")
    raise ConvergenceError(
        f"gauss_seidel did not converge in {max_iter} iterations (rel. residual {res:.3e})",
        result=IterativeSolveResult(x.copy(), max_iter, False, res, history, "gauss_seidel"),
    )


def sor(
    A: np.ndarray,
    b: np.ndarray,
    omega: float,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> IterativeSolveResult:
    """Successive over-relaxation. ``omega == 1`` reduces to Gauss-Seidel.

    Convergence requires ``omega in (0, 2)``; the optimal ``omega`` for SPD
    systems is ``2 / (1 + sqrt(1 - rho^2))`` with ``rho`` the spectral radius
    of the Jacobi iteration matrix.
    """
    if not 0 < omega < 2:
        raise ValueError(f"omega must be in (0, 2), got {omega}")
    validate_tol(tol)
    validate_max_iter(max_iter)
    A, b, x, bnorm = _iterative_setup(A, b, x0)
    n = A.shape[0]
    if np.any(np.diag(A) == 0):
        raise ValueError("SOR requires non-zero diagonal entries in A")
    history: list[float] = []
    for k in range(1, max_iter + 1):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1 :] @ x[i + 1 :]
            x_gs = (b[i] - s) / A[i, i]
            x[i] = (1 - omega) * x[i] + omega * x_gs
        res = float(np.linalg.norm(A @ x - b) / bnorm)
        history.append(res)
        if res < tol:
            return IterativeSolveResult(x.copy(), k, True, res, history, "sor")
    raise ConvergenceError(
        f"sor did not converge in {max_iter} iterations (rel. residual {res:.3e})",
        result=IterativeSolveResult(x.copy(), max_iter, False, res, history, "sor"),
    )


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
) -> IterativeSolveResult:
    """Conjugate gradient for a real symmetric positive-definite (SPD) system.

    The caller supplies SPD ``A``; symmetry/definiteness are not certified.
    Stop when the *true* residual ``||b-Ax|| / ||b|| < tol``. For zero ``b``,
    use the absolute residual instead. An adequate initial guess returns zero
    iterations and an empty history; otherwise history has one entry per update.

    Defaults ``max_iter`` to ``n`` (the exact-arithmetic termination bound,
    not a floating-point guarantee). Invalid inputs raise ``ValueError``;
    nonfinite arithmetic, nonpositive curvature or exhausted iterations raise
    ``ConvergenceError`` carrying the last completed iterate.
    """
    validate_tol(tol)
    A = _validate_square(A)
    b = _validate_b(A, b)
    x = np.zeros(b.size) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    if x.size != b.size:
        raise ValueError("x0 length must match the system dimension")
    if not all(np.all(np.isfinite(v)) for v in (A, b, x)):
        raise ValueError("A, b and x0 must be finite")
    if max_iter is None:
        max_iter = b.size
    validate_max_iter(max_iter)
    history: list[float] = []
    res = float("inf")

    def failure(reason):
        return ConvergenceError(
            f"conjugate_gradient: {reason}",
            result=IterativeSolveResult(x.copy(), len(history), False, res,
                                        history.copy(), "conjugate_gradient"),
        )

    # hypot avoids squaring tiny/large entries just to measure a norm.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            bnorm = float(np.hypot.reduce(b)) or 1.0
            r = b - A @ x
            res = float(np.hypot.reduce(r) / bnorm)
            if not np.isfinite(res):
                raise failure("nonfinite initial residual")
            if res < tol:
                return IterativeSolveResult(x, 0, True, res, [], "conjugate_gradient")
            p = r.copy()
            rs_old = float(r @ r)
            for _ in range(max_iter):
                Ap = A @ p
                curvature = float(p @ Ap)
                if not np.isfinite(curvature) or not np.isfinite(rs_old):
                    raise failure("nonfinite curvature or residual product")
                if curvature <= 0:
                    raise failure("nonpositive curvature; SPD iteration broke down")
                if rs_old <= 0:
                    raise failure("residual product underflow; iteration broke down")
                alpha = rs_old / curvature
                x_new = x + alpha * p
                r_new = r - alpha * Ap
                # Recurrence residuals can drift: report and test b-Ax itself.
                res_new = float(np.hypot.reduce(b - A @ x_new) / bnorm)
                if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(r_new))
                        and np.isfinite(res_new)):
                    raise failure("nonfinite iterate or residual")
                x, r, res = x_new, r_new, res_new
                history.append(res)
                if res < tol:
                    return IterativeSolveResult(x, len(history), True, res, history,
                                                "conjugate_gradient")
                rs_new = float(r @ r)
                if not np.isfinite(rs_new):
                    raise failure("nonfinite residual product")
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
    except (FloatingPointError, OverflowError) as exc:
        raise failure("nonfinite arithmetic during iteration") from exc
    raise failure(f"did not converge in {max_iter} iterations (residual {res:.3e})")
