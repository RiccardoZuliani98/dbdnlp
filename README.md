# dbdnlp

`dbdnlp` solves parametric nonlinear programs (NLPs) with [CasADi](https://web.casadi.org/) and computes how the solution changes with the parameters. It also handles degenerate solutions, where the standard implicit-function-theorem derivative does not exist.

It solves problems of the form

```
min_x  f(x, θ)
s.t.   g(x, θ) <= 0
       h(x, θ)  = 0
```

and returns the sensitivities `dx/dθ`, `dλ/dθ` and `dν/dθ` at a primal-dual solution `(x, λ, ν)`. The Lagrangian is `L = f + λᵀg + νᵀh` with `λ >= 0`.

## Installation

The package needs Python >= 3.14 and depends on `casadi` and `scipy`. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```python
import casadi as ca
from dbdnlp import DifferentiableNLP

x = ca.SX.sym("x", 2)
theta = ca.SX.sym("theta", 1)

nlp = DifferentiableNLP(
    x=x,
    theta=theta,
    f=ca.sumsqr(x - theta),
    g=x[0] + x[1] - 1,        # g(x, θ) <= 0
    solver_opts={"ipopt.print_level": 0, "print_time": False},
)

sol = nlp.solve(theta=[2.0])
d = nlp.differentiate(sol.theta, sol.x, sol.lam, sol.nu,
                      mode="projected", options={"rho_1": 1e-6, "rho_2": 1e-6})
print(d["x"])  # dx/dθ, shape (n_x, n_theta)
```

## Main components

- **`DifferentiableNLP`** ([src/dbdnlp/_impl/nlp.py](src/dbdnlp/_impl/nlp.py)) builds the CasADi solver (IPOPT by default; set `solver_opts["plugin"]` to use another) and compiles the KKT and derivative functions once, at construction.
  - `solve(theta, ...)` solves the NLP, with optional warm starts, and returns an `NLPSolution` that includes the KKT residuals.
  - `evaluate(theta, x, lam, nu)` wraps any primal-dual point as an `NLPSolution` without solving.
  - `differentiate(theta, x, lam, nu, mode, options)` computes the sensitivities. It has four modes:
    - `"full"` differentiates the full KKT system.
    - `"active_set"` solves the reduced symmetric KKT system on the active constraints. Options: `lstsq` (least-squares solution) and `prox` (adds a proximal term).
    - `"regularized"` uses a uniform regularization `ρ₁ I`, `ρ₂ I`.
    - `"projected"` applies the regularization only on the degenerate subspaces, `ρ₁ Π_P`, `ρ₂ Π_U`. At a nondegenerate solution it gives exactly the `"active_set"` result. The linear system is solved with a sparse LU.
  - `check_properties(...)` checks LICQ, strict complementarity, SOSC and strong SOSC, and returns the dimension of the kernel of the KKT matrix.
  - `degenerate_subspaces(...)` returns the projectors and bases of the degenerate subspaces: `U`, the left null space of the active constraint Jacobian, and `P`, the directions where the reduced Hessian is flat. For large problems it computes the null spaces with a sparse method and falls back to a dense SVD.
- **`build_regularized_nlp`** ([src/dbdnlp/_impl/nlp_reg.py](src/dbdnlp/_impl/nlp_reg.py)) builds an augmented primal-dual regularized NLP from an existing problem's symbolic data (`nlp.sym`). The result is another `DifferentiableNLP`, which is strongly convex locally and differentiable even when LICQ or strict complementarity fail.
- **`NLPSolution`**, `NLPSymbolic` and `NLPFunctions` ([src/dbdnlp/_impl/types.py](src/dbdnlp/_impl/types.py)) are the data containers for solutions, symbolic expressions and compiled functions.

## Tests

```bash
uv run pytest
```

## License

MIT, © 2026 Automatic Control Laboratory, ETH Zurich, 8092 Zurich, Switzerland. See [LICENSE](LICENSE).
