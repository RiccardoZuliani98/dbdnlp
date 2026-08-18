# degnlp — degenerate parametric NLPs + a differentiator-agnostic harness

Tiers 1 and 2 of the test plan: scalar catastrophe normal forms and
constraint-side degeneracies, each with an **analytic solution map**, running
directly on `DifferentiableNLP`.

```
degnlp/
  types.py      re-exports the project types; Sensitivities; DM -> numpy helpers
  core.py       Dims, Example, select_branch, loglog_slope, conversion helpers
  examples.py   the 9 examples (each owning its DifferentiableNLP), registry
  metrics.py    every check; all of them consume one sensitivity dict
  reference.py  three baselines, all thin wrappers over differentiate()
  report.py     run_suite / validate_examples / property_table
tests/          94 tests, `pytest -q`
demo.py         end-to-end walk-through
nlp_main.py     DifferentiableNLP + evaluate() + dim_kernel;  see nlp_main.patch
```

## The contract

```python
Differentiator = Callable[[nlp, NLPSolution], Sensitivities]
Sensitivities  = Dict[str, ArrayLike]   # 'x', 'lam', 'nu' — all three required
```

Deliberately the exact return type of `DifferentiableNLP.differentiate`, modulo
currying mode and options, so no adapter is ever needed. `stack_sensitivities`
is the whole interface: it concatenates the three blocks in `phi` order and
raises on a missing key or a wrong shape. Anything a differentiator cannot
compute it reports as **NaN**, which propagates into the results — never a
silent zero, which would score as a confident (and often plausible-looking)
wrong answer.

That strictness is what removed `coerce_jac`: 40 lines of guessing what the
caller meant, replaced by 12 that state what is required.

```python
from degnlp import examples, metrics, report
from degnlp.reference import IFT, Regularized

print(report.run_suite(Regularized(rho_1=1e-3, rho_2=1e-3)))
print(report.run_suite(IFT(mode="active_set")))          # the baseline to beat

ex = examples.get("fold")
print(metrics.eps_sweep(ex, lambda e: Regularized(e, e)(ex.nlp, ex.phi_bar), side="+"))
```

A new mode in `differentiate` means one more four-line wrapper in
`reference.py`, and nothing else.

## What lives where

`adapter.py` is gone; so are `coerce_jac`, `kkt_matrix` and `kernel_dim`.

| was | now |
|---|---|
| `build_symbolic` / `build_functions` | `DifferentiableNLP.__init__` |
| `ReferenceNLP`, `NLP_FACTORY` | `DifferentiableNLP(*build())` — the builder tuple **is** its positional signature |
| `eval_kkt`, `eval_ift`, `active_set` | `funcs.kkt`, `funcs.ift`, `differentiate` |
| `solution_properties` | `check_properties` / `annotate_solution` |
| `RegularizedKKT` (a reimplementation of P2) | `differentiate(mode='regularized')` |
| `kkt_matrix`, `kernel_dim` | `check_properties['dim_kernel']` |
| `coerce_jac` | `stack_sensitivities` + a strict contract |
| `distance_to_set` | `select_branch` — the nearest branch *is* the distance to the set |
| `residual_norm` | folded into `Example.residual` |

`kkt_matrix` was the worse of the two offences: it hard-coded one assembly of
the KKT system inside a harness whose entire purpose is not to know how anyone
assembles anything. What it was actually needed for is `dim ker K`, which is a
property of the *KKT point* and belongs beside LICQ and SOSC. `check_properties`
already has `H_L` and `J_act` in hand, so it costs two lines there.

`Example` now takes a constructed `DifferentiableNLP` rather than a builder, so
`core` no longer knows how one is made. What remains is `Dims` (block slicing of
`phi`), `Example`, `select_branch`, `loglog_slope`, and three conversion
helpers.

`Example.residual` computes the Euclidean norm over all five KKT residual
blocks, which overlaps with the residual `check_properties` reports — that one
is a max-norm over stationarity / equality / dual feasibility, the right choice
there since those are the conditions its analysis presumes, and the wrong one
for comparing a reference point against zero.

## Additions to `DifferentiableNLP`

Two, totalling ~70 added lines of which about half is docstring:

**`evaluate(theta, x, lam=None, nu=None, properties=None) -> NLPSolution`** plus
a five-line `_col` helper. Packages an arbitrary primal-dual point with its KKT
residuals; `solve` now calls it rather than duplicating that block. The harness
needs it because ground truth is analytic — nine examples' worth of closed-form
KKT points must become `NLPSolution`s with no solver running. Independently
useful for scoring a warmstart, wrapping a point from elsewhere, or re-checking
a stored solution at a new `theta`. `_col` exists because `ca.DM(np.zeros(0))`
works but `ca.DM([])` does not, and six of the nine examples have an empty
multiplier block.

**`check_properties` now also reports `dim_kernel`**, the dimension of the null
space of `[[H_L, J_act^T], [J_act, 0]]`. Zero exactly when the IFT applies; its
value counts the independent degeneracy directions, which `licq` and `sosc`
separately do not (`tangency` has one of each).

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

Everything is checked against measurement rather than asserted.
`report.validate_examples()` compares declared Hölder exponents to observed ones
(15/15 agree); `report.property_table()` reports what `check_properties` finds;
and the declared `ker K` generators are verified against `funcs.ift` — both that
each satisfies `H u + J^T v = 0`, `J u = 0`, and that there are `dim_kernel` of
them, so a mis-specified base point cannot pass silently.

```
example                       LICQ       SCS      SOSC     SSOSC  dim ker K
fold                           yes       yes        NO        NO          1
licq_two_eq                     NO       yes       yes       yes          1
tangency                        NO       yes        NO        NO          2
wachter_biegler                yes       yes       yes       yes          0
```

Branch labels (`'min'`, `'saddle'`, `'max'`, `'degenerate'`) live in
`properties['kind']` — read with `core.kind_of`. That slot works because
`properties` is `Dict[str, Any]`; the label is not boolean and does not deserve
a subclass of `NLPSolution`.

## What the baselines show

`demo.py`:

* **`IFT(mode='active_set')` raises `LinAlgError: singular matrix` on all eight
  degenerate examples and is second-order exact on the control.** `run_suite`
  records the exception as a note rather than crashing.
* **`fold` under `regularized`**: `|J| ~ eps^-1.00`, `t_valid ~ eps^2.00`,
  solution bias `~ eps^0.95` — the three predicted rates to two digits.
* **ratio trap** (`mixed_kernel_quadratic`, `b = 1`): sweeping `rho_1/rho_2`
  from 0.01 to 100, `|dx1/dtheta|` runs 1.0e-2 → **1.0e4 at ratio 1** → 1.0e0,
  matching the closed form to five digits at every point. Both weights are
  strictly positive throughout.
* **`tangency`**: reported `dx2/dtheta` is `1/||v||^2 = 1/5` while the truth is
  `0` on one side and `1` on the other — **independent of `rho_2`**. With
  parallel active constraints the dual regularisation splits the perturbation
  between primal motion and constraint violation, and that error does not vanish
  as `rho_2 -> 0`.
* **weak activity silently discards a perturbation.** Choosing `lam1 = 0.5` on
  `tangency` makes `lam2` weakly active, so the `regularized` mode's right-hand
  multiplier `lam / (lam - rho_1 g)` is `0/0`, clamps to zero, and the entire
  `theta`-dependence entering through `g2` disappears: the derivative becomes
  exactly `0`. Here that is the correct *right*-derivative, but it arrives via
  the sign of the `eps` clamp rather than a decision — worth making explicit
  before Tier 3, which is all about weak activity.

## Next

Tier 3 (strict-complementarity failure, active-set switching, MPCC) needs
examples whose active set changes with `theta`, plus a metric for the `O(rho_2)`
displacement of the switching surface implied by eliminating `z` from P2.
`properties['kind']`, `n_weakly_active`, and the `err_set` metric are in place,
and the clamp behaviour above is the first thing to pin down.
