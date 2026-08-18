# degnlp — degenerate parametric NLPs + a differentiator-agnostic harness

Tiers 1 and 2 of the test plan: scalar catastrophe normal forms and
constraint-side degeneracies, each with an **analytic solution map**, plus a
harness whose only contract with your code is a numeric Jacobian.

```
degnlp/
  core.py       Layout, Branch, Functions, Example; Jacobian coercion; branch selection
  examples.py   the 9 examples + declared exponents + registry
  metrics.py    every check; all of them consume a plain array
  reference.py  baseline differentiators (regularised KKT, pinv, FD oracle)
  report.py     run_suite / validate_examples
  symbolic.py   optional CasADi mirrors + a transcription check
tests/          56 tests, `pytest -q`
demo.py         end-to-end walk-through
```

No hard dependency beyond numpy. CasADi is optional and only used by
`symbolic.py`.

## The contract

```python
Differentiator = Callable[[Example], Array]   # -> shape (n + m_g + m_h, p)
```

That is all. `coerce_jac` also accepts

* an `(n, p)` array — read as the `x` block, duals treated as unreported,
* a dict with any of `{"x", "lam", "nu"}`,
* any object with `.jac` / `.jacobian` / `.J`.

Unreported rows become `NaN` and are skipped, so a method that only produces
`dx/dtheta` is still fully testable on the primal blocks. Nothing else about
your differentiator is observable to the tests.

### Plugging in `DifferentiableNLP`

```python
from degnlp import examples, metrics, report

def my_diff(ex, varrho=1e-3, rho=1e-3):
    x, th, f, g, h = ex.build()                 # CasADi mirror of the example
    nlp = DifferentiableNLP(x, th, f, g, h)     # your class
    eta = nlp.solve(ex.theta_bar, phi0=ex.phi_bar.phi)
    return nlp.differentiate(eta, varrho=varrho, rho=rho)   # -> (n+m_g+m_h, p)

print(report.run_suite(my_diff))
ex = examples.get("fold")
print(metrics.eps_sweep(ex, lambda e: my_diff(ex, e, e), side="+"))
```

Two things worth doing before trusting any number:

1. `symbolic.validate(ex)` — confirms the CasADi and numpy descriptions of the
   same example agree at the base point, so a transcription slip cannot be
   mistaken for regularisation error.
2. `metrics.reg_jac_exactness(ex, J, varrho, rho)` — on `licq_two_eq` and
   `mixed_kernel_quadratic` the regularised KKT Jacobian is available in closed
   form. Any discrepancy there is a bug, full stop. Note that your P2 carries a
   slack `z` and a *theta-dependent* proximal centre `g_theta(xbar)`; the closed
   forms here assume the plain `K + diag(varrho I, -rho I)` system, so a
   mismatch localised to `d_theta g` is the first thing to check.

## Examples

| name | tier | degeneracy | solution map along the default direction |
|---|---|---|---|
| `fold` | 1 | SOSC, `beta != 0` | `x = +-sqrt(theta)`; **none** for `theta < 0` |
| `cusp` | 1 | SOSC, `beta = 0` | unique `theta^(1/3)`; persists, Hölder-1/3 |
| `pitchfork` | 1 | SOSC, `alpha = 0` | `0` for `theta <= 0`, `+-sqrt(theta)` after; non-unique |
| `swallowtail` | 1 | A3 germ, `p = 2` | up to 3 roots; discriminant `4 t1^3 = 27 t2^2` |
| `mixed_kernel_fold` | 2 | mixed kernel + fold | `x2 = +-sqrt(-2 b theta / c)`, one side only |
| `mixed_kernel_quadratic` | 2 | mixed kernel, ratio trap | **unbounded** for `theta != 0`; line of KKT points at 0 |
| `licq_two_eq` | 2 | pure LICQ | compatible dir: analytic; incompatible dir: **infeasible** |
| `tangency` | 2 | LICQ + SOSC, `d = 2` | `const` on one side, Hölder-1/2 on the other |
| `wachter_biegler` | 2 | none (control) | smooth; isolates bias from degeneracy |

Every example declares its Hölder exponent per side, and
`report.validate_examples()` checks the library against its own claims with no
differentiator involved. All 15 example/direction pairs currently measure what
they declare.

The base point matters and is part of the example: `tangency` has a whole
segment of valid multipliers at `theta = 0`, and `tangency(multiplier=s)` lets
you pick. `lam1 = kappa/2` is the limit of the nondegenerate branch and is the
choice that makes `ker K` two-dimensional.

## Metrics

All take a Jacobian:

* `taylor` — the workhorse. Predicted vs true step over a log grid of `t`, both
  sides, reporting absolute error, distance to the solution **set** (the right
  metric when minimisers are non-unique), relative error, the fitted order of
  the error in `t`, and `t_valid` = largest `t` where the prediction is still
  within tolerance. Verdicts distinguish *second order*, *first order (biased)*,
  *sublinear: true map not differentiable*, *no solution to predict*, and
  *constant solution, derivative reports spurious motion*.
* `hoelder` — properties of the true map only; validates the examples.
* `asymmetry` — forward vs backward difference quotients vs the reported
  directional derivative. Separates folds (one side missing) from kinks (both
  finite, different).
* `step_residual` — KKT residual of `phi_bar + t J d`, against the frozen point.
  Meaningful even where no nearby solution exists: does the predicted step make
  a better warm start?
* `loss_gradient` — cosine similarity and relative error of `J^T grad loss`
  against a finite difference of `loss(phi*(theta))`. The downstream criterion.
* `eps_sweep` — `taylor` across regularisation weights; fits `|J d| ~ eps^-1`
  and `t_valid ~ eps^2`.
* `solution_bias` — the `O(eps)` row of the theory table. Deliberately separate
  because it needs a *solution*, not a Jacobian.
* `reg_jac_exactness` — closed-form comparison where available.

## What the baseline already shows

Running `demo.py` with the reference regularised-KKT differentiator:

* **`fold`**: `|J| ~ eps^-1.00`, `t_valid ~ eps^2.00`, solution bias `~ eps^0.95`
  — the three predicted rates, to two digits.
* **ratio trap** (`mixed_kernel_quadratic`, `b = 1`): `cond(K+R)` rises from
  4e4 to 4e8 as `varrho/rho` crosses `b^2 = 1`, with both weights strictly
  positive throughout.
* **`tangency`**: the reported `dx2/dtheta` is `1/||v||^2 = 1/5`, while the truth
  is `0` on one side and `1` on the other — and that factor is **independent of
  rho**. With parallel active constraints the dual regularisation splits the
  perturbation between primal motion and constraint violation, and the resulting
  primal error does not vanish as `rho -> 0`. Worth a figure.
* **`fold` downstream**: cosine 1.0 with 68% relative error. The gradient is
  useless as a magnitude and fine as a direction.
* **`step_residual`** on `fold`: the predicted step *increases* the residual by
  ~1000x at `t = 1e-1` and reduces it by 1e-7 at `t = 1e-4 ~ eps^2`. The `eps^2`
  window is visible without any reference to the analytic solution.

## Next

Tier 3 (strict-complementarity failure, active-set switching, MPCC) needs one
addition to `core`: examples whose active set changes with `theta`, and a metric
for the `O(rho)` displacement of the switching surface implied by eliminating
`z` from P2. The `Branch.kind` field and `distance_to_set` are already in place
for it.
