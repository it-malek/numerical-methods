# Method reference: which routine should I use?

A one-page cheat sheet. Each row maps a problem you might have to the
recommended `numethods` routine, with a short note on when to prefer it.

## Taylor expansion

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Build a Taylor polynomial around a point           | `taylor_polynomial(f, a, n, x=None, var=..., h=...)`  | Works with symbolic or numerical `f`. Returns a callable if `x` is omitted. |
| Bound the truncation error                         | `taylor_remainder_bound(f, a, n, x, var=..., M=...)` | Uses the Lagrange remainder form. Supply a derivative bound `M` when you have one. |

## Root-finding (scalar)

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Find a root in a known sign-changing bracket       | `bisection(f, a, b)`             | Always succeeds. Slow (linear). Use as a baseline. |
| Find a root, I have $f'$                            | `newton(f, x0, fprime=...)`     | Quadratic near simple roots. Diverges if $f'(x_0)$ is small. |
| Find a root, no $f'$ available                      | `newton(f, x0)` or `secant(f, x0, x1)` | Newton uses central FD; secant is derivative-free, superlinear. |
| Solve a fixed-point equation $g(x) = x$             | `fixed_point(g, x0)`             | Detects divergence. Requires $|g'| < 1$ near the fixed point. |

## Differentiation

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Approximate $f'(x)$ to first order                  | `forward_diff(f, x, h)`         | Cheapest. $\mathcal{O}(h)$. |
| Approximate $f'(x)$ to second order                 | `central_diff(f, x, h)`         | Default for smooth $f$. $\mathcal{O}(h^2)$. |
| Get $\mathcal{O}(h^4)$ accuracy without coding higher stencils| `richardson_extrapolation(f, x, h)` | Extrapolates two central differences. |
| Approximate the Jacobian of a vector-valued $f$     | `numerical_jacobian(f, x, h)`   | Central FD per column. $\mathcal{O}(n)$ evaluations. |

## Integration

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Quick second-order quadrature                       | `trapezoid(f, a, b, n)`         | $\mathcal{O}(h^2)$. Robust. |
| Higher-order on smooth integrands                   | `simpson(f, a, b, n)`           | $\mathcal{O}(h^4)$. `n` must be even. |
| Best accuracy on smooth $f$ with few evaluations    | `romberg(f, a, b, max_levels)`  | Richardson on trapezoid. Often machine precision in 6–8 levels. |
| Adaptive quadrature with a target tolerance         | `adaptive_simpson(f, a, b, tol)`| Splits where curvature is high. |
| Random / high-dimensional / discontinuous integrands| `monte_carlo(f, a, b, n_samples, rng)`  | $\mathcal{O}(n^{-1/2})$. Pass a seeded RNG for reproducibility. |

## ODEs (initial-value problems)

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Quick first look / debugging                        | `euler(f, t_span, y0, n)`       | $\mathcal{O}(h)$. Fixed step. |
| Fixed-step second-order                             | `midpoint(...)`                  | $\mathcal{O}(h^2)$. |
| Fixed-step accurate solution                        | `rk4(...)`                       | $\mathcal{O}(h^4)$. The default workhorse. |
| Adaptive step with a target tolerance               | `rk45_adaptive(f, t_span, y0, rtol, atol)` | Dormand-Prince. Good for varying time scales. Handles vector-valued `y`. |

## Linear systems $A x = b$

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Solve a small dense system                          | `gauss_elimination(A, b)`        | LU with partial pivoting. |
| Reuse a factorization for many right-hand sides     | `lu_decomposition(A)` + `lu_solve(...)` | Factor once, solve many times. |
| Iterative on a diagonally-dominant matrix           | `jacobi(A, b)` / `gauss_seidel(A, b)` | Simple iterative; GS usually halves the iteration count. |
| Iterative with relaxation                           | `sor(A, b, omega)`               | Tune `omega in (0, 2)` for fastest convergence. |
| SPD system, large or sparse                         | `conjugate_gradient(A, b)`       | Converges in $\le n$ exact steps for $n \times n$ SPD $A$. |

## Optimization (unconstrained)

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Quick descent with hand-tuned step                  | `gradient_descent(f, grad, x0, lr=...)` | Add `momentum=0.9` for ill-conditioned problems. |
| Robust descent with no learning-rate tuning         | `gradient_descent_backtracking(f, grad, x0)` | Armijo line search. Recommended default. |
| Quadratic convergence near the optimum              | `newton_minimize(f, grad, hess, x0)` | Needs $\nabla^2 f$ (or finite-difference Hessian). |
| No analytic gradient                                | Pass `grad=None` (and `hess=None`) | Falls back to central FD. Slower but works. |

## Regression

| I want to...                                       | Use                              | Notes |
|----------------------------------------------------|----------------------------------|-------|
| Plain linear regression                             | `OLSRegression()`                | Default `method="qr"` is numerically stable. |
| Compare to closed-form normal equations             | `OLSRegression(method="normal")` | Squares the condition number; use only for small problems. |
| Regularize against multicollinearity / overfit      | `RidgeRegression(alpha=...)`     | Closed-form ridge. |
| Polynomial fit                                      | `polynomial_features(X, degree)` then any regressor | Degree-only expansion; no cross terms. |

## Tolerances and tuning notes

- **Most iterative methods** take `tol` and `max_iter`, but not all solvers
  share the same controls. For example, `romberg` uses `max_levels`,
  `adaptive_simpson` uses `tol` and `max_depth`, `rk45_adaptive` uses
  `rtol`, `atol`, and `max_steps`, and the fixed-step ODE solvers use
  `n_steps`.
- Defaults are conservative; tighten `tol` for higher precision and raise
  the iteration or step cap if you see `ConvergenceError` on a problem you
  believe should converge.
- **Custom `ConvergenceError`** carries the partial result on its
  `.result` attribute – catch it and inspect the iterates to debug.
- **Vectorize integrands** when possible: every quadrature routine accepts
  array input via NumPy broadcasting, so writing `f(x) = np.sin(x)` is faster
  than a Python loop calling `math.sin`.
