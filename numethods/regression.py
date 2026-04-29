"""Linear regression: ordinary least squares (normal-eq + QR), ridge, polynomial features."""

from __future__ import annotations

from typing import Literal

import numpy as np


def polynomial_features(X: np.ndarray, degree: int, include_bias: bool = False) -> np.ndarray:
    """Expand a 1-D feature vector into ``[x, x^2, ..., x^degree]`` columns.

    For multivariate ``X`` (shape ``(n, d)``) only individual-feature powers
    are emitted (no cross-terms), which is the most common pedagogical use.

    Args:
        X: Input array, shape ``(n,)`` or ``(n, d)``.
        degree: Highest polynomial degree to include (>= 1).
        include_bias: If True, prepend an all-ones column.

    Returns:
        Expanded design matrix, shape ``(n, k)``.

    Raises:
        ValueError: If ``degree < 1``.
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    blocks = [X**p for p in range(1, degree + 1)]
    out = np.hstack(blocks)
    if include_bias:
        out = np.hstack([np.ones((out.shape[0], 1)), out])
    return out


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


class OLSRegression:
    """Ordinary least squares regression.

    Two solution methods are available:

    * ``"normal"`` - solve ``(X^T X) β = X^T y``. Fast, but sensitive to
      ill-conditioning since ``cond(X^T X) = cond(X)^2``.
    * ``"qr"`` - solve ``R β = Q^T y`` from the thin QR. Numerically stable.

    Attributes:
        coef_: Coefficient vector, shape ``(n_features,)``.
        intercept_: Intercept (if ``fit_intercept=True``).
        method: The fit method actually used.
    """

    def __init__(
        self,
        method: Literal["normal", "qr"] = "qr",
        fit_intercept: bool = True,
    ) -> None:
        if method not in {"normal", "qr"}:
            raise ValueError(f"method must be 'normal' or 'qr', got {method!r}")
        self.method = method
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> OLSRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.size:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.size} entries"
            )
        Xd = _add_bias(X) if self.fit_intercept else X
        if self.method == "normal":
            beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)
        else:  # qr
            Q, R = np.linalg.qr(Xd, mode="reduced")
            beta = np.linalg.solve(R, Q.T @ y)
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("OLSRegression: call fit(...) before predict(...)")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination R^2."""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 0.0 if ss_res > 0 else 1.0
        return 1.0 - ss_res / ss_tot


class RidgeRegression:
    """Tikhonov-regularized least squares: ``min ||y - Xβ||^2 + α ||β||^2``.

    The intercept is excluded from regularization (a standard convention).

    Attributes:
        alpha: Ridge penalty (>= 0).
        coef_: Coefficient vector.
        intercept_: Intercept term.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xc = X - x_mean
            yc = y - y_mean
            n_features = X.shape[1]
            beta = np.linalg.solve(
                Xc.T @ Xc + self.alpha * np.eye(n_features),
                Xc.T @ yc,
            )
            self.coef_ = beta
            self.intercept_ = float(y_mean - x_mean @ beta)
        else:
            n_features = X.shape[1]
            self.coef_ = np.linalg.solve(
                X.T @ X + self.alpha * np.eye(n_features),
                X.T @ y,
            )
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegression: call fit(...) before predict(...)")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 0.0 if ss_res > 0 else 1.0
        return 1.0 - ss_res / ss_tot
