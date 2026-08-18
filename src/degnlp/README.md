# degnlp — degenerate parametric NLPs + a differentiator-agnostic harness

Tiers 1 and 2 of the test plan: scalar catastrophe normal forms and
constraint-side degeneracies, each with an **analytic solution map**, built on
your `NLPSolution` / `NLPSymbolic` / `NLPFunctions` types.

```
degnlp/
  types.py      imports your types (falls back to a vendored copy); DM<->numpy helpers
  adapter.py    NLPSymbolic/NLPFunctions construction, the NLP_FACTORY hook,
                KKT matrix, kernel basis, LICQ/SOSC/strict-complementarity analysis
  core.py       Dims, make_solution, Example, branch selection, coerce_jac
  examples.py   the 9 examples, declared exponents and properties, registry
  metrics.py    every check; all of them consume a plain Jacobian
  reference.py  baseline differentiators (regularised KKT, least-squares, FD oracle)
  report.py     run_suite / validate_examples / property_table
tests/          75 tests, `pytest -q`
demo.py         end-to-end walk-through
```

## Wiring it to the project

Two hooks, both one-liners.

**1. Types.** `types.py` tries `dbdnlp._impl.types` first and falls back to the
vendored `_types_vendored.py` (your `types_new.py`, verbatim). Fix `_CANDIDATES`
if the module path differs and the vendored copy stops being used. `TYPES_SOURCE`
reports which one is live.

**2. The NLP object.** Examples describe themselves as CasADi builders
`() -> (primal, param, cost, ineq, eq)` and construct `Example.nlp` through
`adapter.NLP_FACTORY`:

```python
from dbdnlp._impl.nlp import DifferentiableNLP
from degnlp import adapter

adapter.set_factory(lambda primal, param, cost, ineq, eq: DifferentiableNLP(...))
```

The default factory returns `ReferenceNLP`, a minimal stand-in exposing exactly
the surface the harness uses — `sym`, `funcs`, `n_x`, `n_ineq`, `n_eq`,
`n_theta` — so everything runs today and swaps over without touching the
examples. `test_factory_hook_is_used` pins that contract.

Everything downstream calls `nlp.funcs.kkt` and `nlp.funcs.ift` **positionally**,
in the documented output order, so any object with those two `ca.Function`s and
the dimension attributes works.

## The differentiator contract

```python
Differentiator = Callable[[nlp, NLPSolution], ArrayLike]   # (n_phi, n_theta)
```

`coerce_jac` also accepts an `(n_x, n_theta)` block, a dict keyed by
`x`/`lam`/`nu`, a `ca.DM`, or an object with `.jac`/`.jacobian`/`.J`. Unreported
rows stay NaN and are skipped, so a method that only produces `dx/dtheta` is
still fully testable on the primal block.

```python
from degnlp import examples, metrics, report

def my_diff(nlp, solution, varrho=1e-3, rho=1e-3):
    return nlp.differentiate(solution, varrho=varrho, rho=rho)

my_diff.name = "P2"
print(report.run_suite(my_diff))

ex = examples.get("fold")
print(metrics.eps_sweep(ex, lambda e: my_diff(ex.nlp, ex.phi_bar, e, e), side="+"))
```

Run `metrics.reg_jac_exactness(ex, J, varrho, rho)` on `licq_two_eq` and
`mixed_kernel_quadratic` before interpreting anything else — there the
regularised KKT Jacobian is closed form, so a discrepancy is a bug. Those closed
forms assume the plain `K + diag(varrho I, -rho I)` system; your P2 carries a
slack `z` and a *theta-dependent* proximal centre `g_theta(xbar)`, so a mismatch
confined to the `d_theta` columns points straight at that term.

## Solutions and branch labels

Every ground-truth branch is an `NLPSolution` with `kkt_residuals` filled in by
`funcs.kkt`. The second-order character lives in `properties['kind']`
(`'min'`, `'saddle'`, `'max'`, `'degenerate'`, `'reference'`) — read it with
`core.kind_of`, since `properties` is now `Dict[str, Any]` and this is not a
boolean. `Example.phi_bar` additionally carries the measured hypotheses.

## Examples

| name | tier | degeneracy | solution map along the default direction |
|---|---|---|---|
| `fold` | 1 | SOSC, `beta != 0` | `x = +-sqrt(theta)`; **none** for `theta < 0` |
| `cusp` | 1 | SOSC, `beta = 0` | unique `theta^(1/3)`; persists, Hölder-1/3 |
| `pitchfork` | 1 | SOSC, `alpha = 0` | `0` then `+-sqrt(theta)`; non-unique |
| `swallowtail` | 1 | A3 germ, `n_theta = 2` | up to 3 roots; discriminant `4 t1^3 = 27 t2^2` |
| `mixed_kernel_fold` | 2 | mixed kernel + fold | `x2 = +-sqrt(-2 b theta / c)`, one side only |
| `mixed_kernel_quadratic` | 2 | mixed kernel, ratio trap | **unbounded** for `theta != 0`; line of KKT points at 0 |
| `licq_two_eq` | 2 | pure LICQ | compatible dir: analytic; incompatible dir: **infeasible** |
| `tangency` | 2 | LICQ + SOSC, `d = 2` | `const` on one side, Hölder-1/2 on the other |
| `wachter_biegler` | 2 | none (control) | smooth; isolates bias from degeneracy |

Both claims are checked against measurement, with no differentiator involved:
`report.validate_examples()` compares declared Hölder exponents to observed ones
(15/15 agree), and `report.property_table()` reads LICQ / SOSC / strict
complementarity / `dim ker K` off `funcs.ift`:

```
example                           LICQ   strict comp          SOSC     K nonsing  dim ker K
fold                               yes           yes            NO            NO          1
licq_two_eq                         NO           yes           yes            NO          1
tangency                            NO           yes            NO            NO          2
wachter_biegler                    yes           yes           yes           yes          0
```

`tangency` is the one to watch: at `theta = 0` the multiplier set is a segment,
and `tangency(multiplier=s)` picks a point in it. Only `lam1 = kappa/2` — the
limit of the `theta < 0` branch — sees the full two-dimensional kernel; the
other choices see dimension 1. The base multiplier is part of the problem
specification, not a detail.

## Metrics

All take a Jacobian: `taylor` (predicted vs true step, both sides, with fitted
error order, distance-to-solution-set, and `t_valid`), `hoelder` (true map only),
`asymmetry` (folds vs kinks), `step_residual` (warm-start value),
`loss_gradient` (cosine + relative error downstream), `eps_sweep`
(`|J d| ~ eps^-1`, `t_valid ~ eps^2`), `reg_jac_exactness`. `solution_bias` is
separate because the `O(eps)` row of the theory table needs a *solution*, not a
Jacobian.

## What the baseline already shows

`demo.py` with the reference regularised-KKT differentiator:

* **`fold`**: `|J| ~ eps^-1.00`, `t_valid ~ eps^2.00`, solution bias `~ eps^0.95`
  — the three predicted rates to two digits.
* **ratio trap** (`mixed_kernel_quadratic`, `b = 1`): `cond(K+R)` rises 4e4 → 4e8
  as `varrho/rho` crosses `b^2 = 1`, both weights strictly positive throughout.
* **`tangency`**: reported `dx2/dtheta` is `1/||v||^2 = 1/5` while the truth is
  `0` on one side and `1` on the other — and that factor is **independent of
  rho**. With parallel active constraints the dual regularisation splits the
  perturbation between primal motion and constraint violation, and the resulting
  primal error does not vanish as `rho -> 0`. Worth a figure.
* **`fold` downstream**: cosine 1.0 with 68% relative error — useless as a
  magnitude, fine as a direction.
* **`step_residual`**: the predicted step *increases* the KKT residual ~1000x at
  `t = 1e-1` and drives it to zero at `t = 1e-4 ~ eps^2`. The window is visible
  without reference to the analytic solution.

## Next

Tier 3 (strict-complementarity failure, active-set switching, MPCC) needs
examples whose active set changes with `theta`, plus a metric for the `O(rho)`
displacement of the switching surface implied by eliminating `z` from P2.
`properties['kind']`, `adapter.active_set` and `distance_to_set` are already in
place for it.
