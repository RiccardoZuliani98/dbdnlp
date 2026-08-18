"""
Walk-through of the harness with the baseline differentiators.

Replace ``RegularizedKKT`` with your own callable  Example -> Jacobian  and
nothing else changes.
"""

import numpy as np

from degnlp import examples as ex_mod
from degnlp import metrics as M
from degnlp import report
from degnlp.reference import FiniteDifferenceOracle, RegularizedKKT
from degnlp.report import jacobian_for


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
rule("0. the library checks itself: measured vs declared Hoelder exponents")
for h in report.validate_examples():
    flag = "ok " if all(h.matches_expected.values()) else "BAD"
    got = {s: ("none" if not h.exists[s] else f"{h.exponent[s]:.3f}") for s in "+-"}
    print(f" {flag}  {h.example:<24}{h.direction:<14}{got}")

# --------------------------------------------------------------------------
rule("0b. hypotheses measured at each reference point (from funcs.ift)")
print(report.property_table())

rule("1. full suite, regularised KKT at varrho = rho = 1e-3 (primal block)")
print(report.run_suite(RegularizedKKT(varrho=1e-3, rho=1e-3)))

rule("2. same suite, finite-difference oracle (positive control)")
print(report.run_suite(FiniteDifferenceOracle(t=1e-6)))

# --------------------------------------------------------------------------
rule("3. fold: the three predicted rates")
fold = ex_mod.get("fold")
sweep = M.eps_sweep(
    fold,
    lambda e: jacobian_for(RegularizedKKT(varrho=e, rho=e), fold),
    eps=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
    ts=np.geomspace(1e-2, 1e-10, 17),
    side="+",
)
print(sweep)
print("  theory: |J| ~ eps^-1, t_valid ~ eps^2, linear err @ fixed t ~ eps^-1")
print()
sol = fold.ls["reg_solution"]
print(M.solution_bias(fold, lambda th, e: np.array([sol(th, e, e)]), t=1e-2))
print("  theory: bias = varrho / 2  ->  order 1")

# --------------------------------------------------------------------------
rule("4. mixed kernel: the varrho/rho ratio trap (critical ratio = b^2 = 1)")
mk = ex_mod.get("mixed_kernel_quadratic")
rho = 1e-4
print(f"{'varrho/rho':>12}{'cond(K+R)':>14}{'|dx1/dtheta|':>16}{'|dx2/dtheta|':>16}")
for ratio in (1e-2, 1e-1, 0.5, 0.9, 1.0, 1.1, 2.0, 1e1, 1e2):
    diff = RegularizedKKT(varrho=ratio * rho, rho=rho)
    J = jacobian_for(diff, mk)
    print(f"{ratio:>12.3g}{diff.cond:>14.3e}{abs(J[0, 0]):>16.3e}{abs(J[1, 0]):>16.3e}")
print("  closed form: dx1/dtheta = 1/(1 + rho(1+varrho) - rho b^2/varrho)")

# --------------------------------------------------------------------------
rule("5. licq_two_eq: compatible vs incompatible perturbation")
lq = ex_mod.get("licq_two_eq")
J = jacobian_for(RegularizedKKT(varrho=1e-3, rho=1e-3), lq)
print("J =\n", J, "\n")
for dname in ("compatible", "incompatible"):
    rep = M.taylor(lq, J, direction=dname, block="x")
    print(f" {dname:<14} +: {rep.verdict['+']:<40} -: {rep.verdict['-']}")
print("\n multiplier rows blow up like 1/rho:")
for rho in (1e-2, 1e-3, 1e-4):
    Jr = jacobian_for(RegularizedKKT(varrho=1e-3, rho=rho), lq)
    print(f"   rho={rho:.0e}   |dnu/dtheta| = {np.linalg.norm(Jr[1:]):.3e}")

# --------------------------------------------------------------------------
rule("6. tangency: strongly one-sided, d = 2 kernel")
tg = ex_mod.get("tangency")
J = jacobian_for(RegularizedKKT(varrho=1e-3, rho=1e-3), tg)
asym = M.asymmetry(tg, J, direction="loosen", t=1e-5, block="x")
print(f" forward  dx/dtheta = {asym.forward}")
print(f" backward dx/dtheta = {asym.backward}")
print(f" reported dx/dtheta = {asym.jac_dir}")
print(f" -> {asym.note}")
print(
    "    note: the reported dx2/dtheta is 1/||v||^2 = 1/5 for *every* rho -- with\n"
    "    two parallel active constraints the dual regularisation splits the\n"
    "    perturbation between motion and violation, and that factor does not\n"
    "    vanish as rho -> 0."
)
print("\n multiplier choice at theta_bar changes K (segment 2 lam1 + lam2 = 1):")
for s in (0.0, 0.5, 1.0):
    ex_s = ex_mod.tangency(multiplier=s * 0.5)
    d = RegularizedKKT(varrho=1e-3, rho=1e-3)
    Js = jacobian_for(d, ex_s)
    print(
        f"   lam1 = {s*0.5:.3f}   cond = {d.cond:.3e}   "
        f"dim ker K = {ex_s.kernel_numeric().shape[1]}   dx/dtheta = {Js[:2, 0]}"
    )
print("    only lam1 = kappa/2 sees the full d = 2 kernel: the base multiplier")
print("    you pick out of a degenerate multiplier set changes the problem.")

# --------------------------------------------------------------------------
rule("7. downstream gradient and warm-start residual (fold)")
lg = M.loss_gradient(fold, jacobian_for(RegularizedKKT(varrho=1e-2, rho=1e-2), fold),
                     loss=lambda phi: float(phi[0]), t=1e-5)
print(" ", lg)
print("  cosine 1 with large relative error = still a descent direction")
print()
print(M.step_residual(fold, jacobian_for(RegularizedKKT(varrho=1e-2, rho=1e-2), fold)))
