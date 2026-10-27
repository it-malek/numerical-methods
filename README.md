# numethods

A small numerical methods library implementing the classical algorithms from
**MATH 240 (Computational Mathematics)** at Lake Forest College, with automated
tests, convergence benchmarks, and worked examples.

The package covers Taylor expansion, root-finding, finite differences,
quadrature, ODE solvers, direct and iterative linear solvers, unconstrained
optimization, and linear regression. Each routine is implemented from
scratch on top of NumPy, validated against SciPy / scikit-learn, and
benchmarked to confirm the textbook convergence rates.

## Quick start

```bash
git clone https://github.com/it-malek/numerical-methods.git
cd numerical-methods
pip install -e ".[dev]"
```

```python
from numethods import newton, rk4, simpson

# Solve x^2 - 2 = 0 with Newton's method
root = newton(lambda x: x**2 - 2, x0=1.0, fprime=lambda x: 2*x)
print(root.root)        # 1.4142135623730951
print(root.iterations)  # 5
print(root.converged)   # True

# Integrate sin(x) from 0 to pi via composite Simpson
import numpy as np
val = simpson(np.sin, 0.0, np.pi, n=20).value
print(val)              # ≈ 2.0000

# Solve dy/dt = -y, y(0)=1 with RK4
sol = rk4(lambda t, y: -y, t_span=(0.0, 1.0), y0=1.0, n_steps=50)
print(sol.y[-1])        # ≈ exp(-1) = 0.3679
```

## What's included

| Module                       | Methods |
|------------------------------|---------|
| `numethods.taylor`           | `taylor_polynomial`, `taylor_remainder_estimate`, `taylor_remainder_bound` |
| `numethods.rootfinding`      | `bisection`, `newton`, `secant`, `fixed_point` |
| `numethods.differentiation`  | `forward_diff`, `backward_diff`, `central_diff`, `richardson_extrapolation`, `numerical_jacobian` |
| `numethods.integration`      | `trapezoid`, `simpson`, `romberg`, `monte_carlo`, `adaptive_simpson` |
| `numethods.ode`              | `euler`, `midpoint`, `rk4`, `rk45_adaptive` |
| `numethods.linsys`           | `lu_decomposition`, `lu_solve`, `gauss_elimination`, `jacobi`, `gauss_seidel`, `sor`, `conjugate_gradient` |
| `numethods.optimization`     | `gradient_descent`, `gradient_descent_backtracking`, `newton_minimize` |
| `numethods.regression`       | `OLSRegression`, `RidgeRegression`, `polynomial_features` |

Every iterative method returns a structured result (`RootResult`,
`OptimizationResult`, `IterativeSolveResult`, ...) that includes the full
convergence history, so you can plot it, inspect failed runs, or use it
in tests.

## Numerical reliability and failure results

- `conjugate_gradient` assumes real symmetric positive-definite input; it does
  not certify that assumption. It checks the initial **true** residual before
  dividing, so `conjugate_gradient(np.eye(2), np.zeros(2))` returns zero
  iterations, residual zero and an empty history. The tolerance is relative to
  `||b||` for nonzero `b`, and absolute for zero `b`. Reported residuals and
  history use `b - A @ x`. Invalid curvature and nonfinite arithmetic raise
  `ConvergenceError` with the last completed iterate; finite invalid inputs
  such as an indefinite matrix are not necessarily detected before iteration.
- `adaptive_simpson` treats `tol` as an absolute target. Its result exposes
  `requested_tolerance`, `error_estimate` (the sum of leaf Richardson
  estimates), and `converged`. A constant integrand has estimated error zero,
  regardless of the requested tolerance. Exhausting depth or representable
  refinement raises `ConvergenceError` with the full-interval partial result.
  Nonfinite evaluations/arithmetic raise the same exception with no result.
  Estimates assume smooth, sufficiently resolved variation and can miss
  peaks; they are not guaranteed bounds on the corrected integral's error.
- `rk45_adaptive` integrates forward with positive finite step sizes and
  scalar tolerances. Its budget counts accepted and rejected attempts.
  Reaching the endpoint on the final attempt succeeds. If a step fails error
  control and cannot shrink at the current floating-point time spacing, it
  raises `ConvergenceError`; small steps do not bypass the error test.
  Failure results contain only accepted states and `success=False`.
- `taylor_remainder_estimate` uses sampled derivatives and returns a heuristic
  float, which can underestimate the error even for smooth functions.
  **Compatibility change:** `taylor_remainder_bound` now requires explicit
  `M`. It returns `M*abs(x-a)**(n+1)/(n+1)!`, conditional on Taylor's theorem
  applying and the caller's `M` bounding the required derivative throughout
  the entire interval. The program does not certify `M`. For example,
  `taylor_remainder_bound(np.sin, 0., 3, .5, M=1.)` is `.5**4/24`.
  Omitting `M` raises `ValueError`; use the estimate helper for the former
  automatic behavior. This bounds Taylor truncation, not rounding or errors
  in numerically approximated polynomial coefficients.

Catch `ConvergenceError` and inspect `.result` to diagnose a failed solve;
never interpret a partial result as successful convergence. Other integration
rules leave the new adaptive-only metadata fields as `None`.

## Examples

The `examples/` folder contains the original lab notebooks, cleaned up
and rewritten on top of the library. They are teaching examples; the unit tests below verify library behavior.
The Taylor example uses an analytic derivative bound. Stored notebook outputs
are historical and are not regenerated by the test suite.

| Notebook | Topic |
|----------|-------|
| `examples/00_recap_numpy_sympy.ipynb`              | NumPy / SymPy / Matplotlib refresher |
| `examples/01_taylor_polynomials.ipynb`             | Taylor expansion and Lagrange remainder |
| `examples/02_rootfinding.ipynb`                    | Bisection, Newton, secant on $\sqrt{2}$ and $\cos x = x$ |
| `examples/03_integration_and_differentiation.ipynb`| Quadrature, finite differences, $\pi$ via half-disc area |
| `examples/04_odes_and_dynamical_systems.ipynb`     | Damped oscillator, Lotka–Volterra, SIR sweep on $R_0$ |
| `examples/05_linear_systems.ipynb`                 | LU vs Jacobi vs Gauss–Seidel vs CG |
| `examples/06_optimization.ipynb`                   | Gradient descent and Newton on Rosenbrock |
| `examples/07_regression.ipynb`                     | OLS and ridge with polynomial features |

## Benchmarks

Two notebooks under `benchmarks/`:

- **`convergence_rates.ipynb`** – log-log plots showing each method's
  empirical convergence order matches theory ($\mathcal{O}(h)$, $\mathcal{O}(h^2)$, $\mathcal{O}(h^4)$,
  quadratic for Newton).
- **`runtime_vs_scipy.ipynb`** – head-to-head timing against SciPy on the
  same inputs. SciPy wins on speed (it's compiled), but the gap on small
  problems is modest.

Sample plot from the convergence notebook (ODE solvers, exponential decay):

![ODE convergence](docs/teaser_ode_convergence.png)

The slopes of 1, 2, and 4 visible above match the theoretical orders of
the Euler, midpoint, and RK4 schemes exactly.

## Testing

```bash
pytest                           # library tests and deterministic regressions
pytest --cov=numethods           # measure coverage for this checkout
ruff check numethods tests       # style / lints
```

## Method reference

See [`docs/method_reference.md`](docs/method_reference.md) for a one-page
"I want to do X, which method should I use?" cheat sheet.

## Course attribution

The original lab assignments come from MATH 240 (Introduction to
Computational Mathematics) at Lake Forest College. The library and
infrastructure here are an expanded version of that coursework.

## License

MIT – see [LICENSE](LICENSE).
