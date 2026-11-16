# Copyright (c) 2026 Automatic Control Laboratory, ETH Zurich, 8092 Zurich, Switzerland, Riccardo Zuliani
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the project root.

"""
Tests for the regularized problem P_2 and its sensitivities.

Covers four claims:
  1. P_2 anchored at a KKT point of P_1 reproduces that point exactly (fixed point).
  2. P_2 restores LICQ unconditionally, and SOSC/SSOSC subject to a curvature
     condition on varrho (always satisfied for a convex QP).
  3. The solution map of P_2 is differentiable: its sensitivities match central
     finite differences.
  4. differentiate(mode="regularized") on P_1 equals differentiate(mode="full")
     applied to P_2 itself, restricted to the (x, lam, nu) rows and theta columns.
"""
import numpy as np
import casadi as ca
import pytest
from scipy.linalg import null_space

from dbdnlp.qp import generate_qp
from dbdnlp import DifferentiableNLP, build_regularized_nlp

# =====================================================================
# CONFIGURATION
# =====================================================================

DIMENSIONS = [(15, 3, 5), (30, 8, 12)]
BOOLEANS = [True, False]

N_THETA = 3
VARRHO = 1e-2          # rho_1, primal penalty
RHO = 1e-3             # rho_2, dual penalty
FD_EPS = 1e-6

TOL_SOLUTION = 1e-6
TOL_DEGENERATE = 1e-4  # see note below
TOL_DERIV = 1e-5       # relative, vs. central differences
TOL_CROSS = 1e-8       # relative, analytic vs. analytic

# When SCS fails for P_1 it also fails for P_2.
# An interior-point method cannot drive a component where both z_i and its
# multiplier vanish below ~sqrt(mu_min); measured floor is ~2e-6 here, and no
# combination of mu_strategy / bound_relax_factor / mu_target improves it. The
# looser tolerance is a property of the problem class, not slack in the code.

SOLVER_OPTS = {
    "ipopt.print_level": 0, "print_time": 0,
    "ipopt.tol": 1e-12, "ipopt.constr_viol_tol": 1e-12,
    "ipopt.dual_inf_tol": 1e-12, "ipopt.acceptable_tol": 1e-14,
}


# =====================================================================
# HELPERS
# =====================================================================

def build_parametric_nlp(qp, nx, neq, nineq, n_theta, seed):
    """
    Wrap a QPData as a *parametric* NLP. theta enters the linear cost term and
    both right-hand sides, so that theta = 0 reproduces the generated QP exactly
    and dsol/dtheta is non-trivial.

        f(x, th) = 1/2 x'Qx + (q + P_q th)'x
        g(x, th) = Gx - (h + P_h th) <= 0
        h(x, th) = Ax - (b + P_b th)  = 0
    """
    rng = np.random.default_rng(seed + 9973)
    P_q = rng.normal(size=(nx, n_theta)) * 0.1
    P_h = rng.normal(size=(nineq, n_theta)) * 0.1 if nineq else np.zeros((0, n_theta))
    P_b = rng.normal(size=(neq, n_theta)) * 0.1 if neq else np.zeros((0, n_theta))

    x  = ca.MX.sym("x", nx)           # type: ignore
    th = ca.MX.sym("theta", n_theta)  # type: ignore

    f = 0.5 * ca.mtimes([x.T, ca.DM(qp.Q), x]) + ca.mtimes((ca.DM(qp.q) + ca.mtimes(ca.DM(P_q), th)).T, x)
    g = ca.mtimes(ca.DM(qp.G), x) - (ca.DM(qp.h) + ca.mtimes(ca.DM(P_h), th)) if nineq else None
    h = ca.mtimes(ca.DM(qp.A), x) - (ca.DM(qp.b) + ca.mtimes(ca.DM(P_b), th)) if neq else None

    return DifferentiableNLP(x=x, theta=th, f=f, g=g, h=h, solver_opts=dict(SOLVER_OPTS))

def anchored_params(theta, x_bar, lam_bar, nu_bar, varrho=VARRHO, rho=RHO):
    """Assemble p_reg = [theta; rho_1; rho_2; x_bar; lam_bar; nu_bar]."""
    return np.concatenate([
        np.asarray(theta, float).flatten(),
        [varrho], 
        [rho],
        np.asarray(x_bar, float).flatten(),
        np.asarray(lam_bar, float).flatten(),
        np.asarray(nu_bar, float).flatten()
    ])

def solve_P2(base, reg, theta, phi_bar, varrho=VARRHO, rho=RHO):
    """Solve P_2(theta, phi_bar) and split xi = [x; lam; nu; z]."""
    x_bar, lam_bar, nu_bar = phi_bar
    nx, ng, nh = base.n_x, base.n_ineq, base.n_eq
    z0 = np.array(base.funcs.ineq(x_bar, theta)).flatten() if ng else np.zeros(0)
    xi0 = np.concatenate([x_bar, lam_bar, nu_bar, z0])
    sol = reg.solve(anchored_params(theta, *phi_bar, varrho=varrho, rho=rho), x_warmstart=xi0)
    xi = np.array(sol.x).flatten()
    return xi[:nx], xi[nx:nx + ng], xi[nx + ng:nx + ng + nh], xi[nx + ng + nh:], sol

def as_np(dm):
    return np.array(ca.DM(dm).full())

def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))

@pytest.fixture
def setup(request):
    """(base nlp, regularized nlp, theta0, phi_bar) for one parameter combination."""
    nx, neq, nineq, licq, scs, sosc = request.param
    seed = 42 + nx + neq + nineq + int(licq) + 2 * int(scs) + 4 * int(sosc)
    qp, true_sol = generate_qp(
        nx=nx, neq=neq, nineq=nineq, seed=seed,
        licq=licq, scs=scs, sosc=sosc
    )
    base = build_parametric_nlp(qp, nx, neq, nineq, N_THETA, seed)
    reg = build_regularized_nlp(base.sym, solver_opts=dict(SOLVER_OPTS))
    theta0 = np.zeros(N_THETA)
    phi_bar = (
        np.asarray(true_sol.x, float).flatten(),
        np.asarray(true_sol.lam, float).flatten(),
        np.asarray(true_sol.nu, float).flatten()
    )
    return base, reg, theta0, phi_bar, dict(licq=licq, scs=scs, sosc=sosc)


def _params(scs_values=BOOLEANS):
    return [
        pytest.param((
                nx, neq, nineq, licq, scs, sosc
            ), id=f"nx{nx}-neq{neq}-nineq{nineq}-licq{int(licq)}-scs{int(scs)}-sosc{int(sosc)}"
        )
            for (nx, neq, nineq) in DIMENSIONS
            for licq in BOOLEANS 
            for scs in scs_values # this way we can enforce scs = True always
            for sosc in BOOLEANS
        ]


# =====================================================================
# 1. FIXED POINT: P_2 anchored at a KKT point of P_1 reproduces it
# =====================================================================

@pytest.mark.parametrize("setup", _params(), indirect=True)
def test_P2_reproduces_the_anchor_solution(setup):
    base, reg, theta0, phi_bar, flags = setup
    x_bar, lam_bar, nu_bar = phi_bar
    tol = TOL_SOLUTION if flags["scs"] else TOL_DEGENERATE

    # sanity: the anchor really is a KKT point of P_1
    stat, ineq, eq, slack, dfeas = base.funcs.kkt(x_bar, lam_bar, nu_bar, theta0)
    for name, v in [
        ("stationarity", stat), ("inequality", ineq), ("equality", eq),
        ("slackness", slack), ("dual_feasibility", dfeas)
    ]:
        assert float(ca.norm_inf(v)) < TOL_SOLUTION, f"anchor violates {name}"

    x, lam, nu, z, _ = solve_P2(base, reg, theta0, phi_bar)

    assert np.max(np.abs(x - x_bar)) < tol
    assert np.max(np.abs(lam - lam_bar)) < tol
    assert np.max(np.abs(nu - nu_bar)) < tol

    # the slack must land on g(x_bar, theta)
    if base.n_ineq:
        g_bar = np.array(base.funcs.ineq(x_bar, theta0)).flatten()
        assert np.max(np.abs(z - g_bar)) < tol

    # and the recovered point must still zero the ORIGINAL KKT system
    stat, ineq, eq, slack, dfeas = base.funcs.kkt(x, lam, nu, theta0)
    assert max(float(ca.norm_inf(v)) for v in (stat, ineq, eq, slack, dfeas)) < tol


@pytest.mark.parametrize("setup", _params(), indirect=True)
def test_P2_pins_down_duals_even_without_LICQ(setup):
    """Without LICQ the duals of P_1 are non-unique; the rho-term makes P_2's unique."""
    base, reg, theta0, phi_bar, flags = setup
    if flags["licq"]:
        pytest.skip("only meaningful when LICQ fails")
    tol = TOL_SOLUTION if flags["scs"] else TOL_DEGENERATE
    _, lam, nu, _, _ = solve_P2(base, reg, theta0, phi_bar)
    assert np.max(np.abs(lam - phi_bar[1])) < tol
    assert np.max(np.abs(nu - phi_bar[2])) < tol


# =====================================================================
# 2. PROPERTIES: what P_2 does and does not repair
# =====================================================================

def _reduced_hessian_min_eig(base, theta, x, lam, nu, active_tol=1e-6):
    H, _, J_g, _, J_h, _ = base.funcs.ift(x, lam, nu, theta)
    H = as_np(H).reshape(base.n_x, base.n_x)
    J_g = as_np(J_g).reshape(base.n_ineq, base.n_x)
    J_h = as_np(J_h).reshape(base.n_eq, base.n_x)
    g = np.array(base.funcs.ineq(x, theta)).flatten() if base.n_ineq else np.zeros(0)
    act = np.where(g >= -active_tol)[0]
    J = np.vstack([J_h, J_g[act, :]])
    Z = np.eye(base.n_x) if J.shape[0] == 0 else null_space(J)
    if Z.shape[1] == 0:
        return np.inf
    Hr = Z.T @ H @ Z
    return float(np.min(np.linalg.eigvalsh(0.5 * (Hr + Hr.T))))


@pytest.mark.parametrize("setup", _params(), indirect=True)
def test_P2_restores_LICQ_unconditionally(setup):
    """For rho > 0 the -rho*I blocks make the equality rows independent of e_z."""
    base, reg, theta0, phi_bar, flags = setup
    x, lam, nu, z, sol = solve_P2(base, reg, theta0, phi_bar)
    props = reg.check_properties(anchored_params(theta0, *phi_bar), sol.x, sol.lam, sol.nu)
    assert props["licq"] is True, f"P_2 lost LICQ (P_1 licq={flags['licq']})"


@pytest.mark.parametrize("setup", _params(), indirect=True)
def test_P2_restores_second_order_conditions(setup):
    """Verify that SOSC is restored."""
    base, reg, theta0, phi_bar, flags = setup
    mu_min = _reduced_hessian_min_eig(base, theta0, *phi_bar)
    # varrho = max(VARRHO, -mu_min + 1e-3) if np.isfinite(mu_min) else VARRHO
    varrho = VARRHO

    x, lam, nu, z, sol = solve_P2(base, reg, theta0, phi_bar, varrho=varrho)
    props = reg.check_properties(
        anchored_params(theta0, *phi_bar, varrho=varrho), 
        sol.x, sol.lam, sol.nu
    )

    assert props["sosc"] is True, f"P_2 failed SOSC (P_1 sosc={flags['sosc']}, mu_min={mu_min:.2e})"
    if flags["scs"]:
        assert props["ssosc"] is True, "P_2 failed SSOSC despite SCS holding for P_1"
        assert props["sosc_exact"] is True


@pytest.mark.parametrize("setup", _params(), indirect=True)
def test_P2_does_not_restore_SCS(setup):
    """
    At the fixed point the multiplier of z_i <= 0 equals lam_bar_i, and z_i = 0
    iff g_i(x_bar) = 0. Weak activity therefore carries over from P_1 verbatim:
    the regularization repairs LICQ and curvature, NOT strict complementarity.
    """
    base, reg, theta0, phi_bar, flags = setup
    if base.n_ineq == 0:
        pytest.skip("no inequalities")
    x, lam, nu, z, sol = solve_P2(base, reg, theta0, phi_bar)
    # active_tol must exceed the achievable solve accuracy at a degenerate
    # component, or the weakly active z_i is misread as strictly inactive.
    props = reg.check_properties(
        anchored_params(theta0, *phi_bar), 
        sol.x, sol.lam, sol.nu,
        active_tol=TOL_DEGENERATE
    )
    assert props["scs"] is flags["scs"], "SCS of P_2 should mirror SCS of P_1"


# =====================================================================
# 3. DIFFERENTIABILITY: sensitivities of P_2 vs. finite differences
# =====================================================================

@pytest.mark.parametrize("setup", _params(scs_values=[True]), indirect=True)
def test_regularized_sensitivity_matches_finite_differences(setup):
    """
    Central differences of the P_2 solution map theta -> (x, lam, nu), with the
    anchor phi_bar held FIXED (it is a parameter of P_2, not a function of theta).
    """
    base, reg, theta0, phi_bar, flags = setup
    nx, ng, nh = base.n_x, base.n_ineq, base.n_eq

    d = base.differentiate(
        theta0, *phi_bar, mode="regularized",
        options={"rho_1": VARRHO, "rho_2": RHO}
    )
    analytic = np.vstack([as_np(d["x"]), as_np(d["lam"]), as_np(d["nu"])])

    fd = np.zeros((nx + ng + nh, N_THETA))
    for j in range(N_THETA):
        tp, tm = theta0.copy(), theta0.copy()
        tp[j] += FD_EPS
        tm[j] -= FD_EPS
        xp, lp, np_, _, _ = solve_P2(base, reg, tp, phi_bar)
        xm, lm, nm, _, _ = solve_P2(base, reg, tm, phi_bar)
        fd[:, j] = (np.concatenate([xp, lp, np_]) - np.concatenate([xm, lm, nm])) / (2 * FD_EPS)

    assert rel_err(analytic, fd) < TOL_DERIV, f"rel err {rel_err(analytic, fd):.3e}"


@pytest.mark.parametrize("setup", _params(scs_values=[True]), indirect=True)
@pytest.mark.parametrize("varrho,rho", [(1e-3, 1e-3), (1e-2, 1e-4), (1e-1, 1e-1), (1.0, 1.0)])
def test_regularized_sensitivity_exact_for_all_penalties(setup, varrho, rho):
    """The IFT formula is exact for every varrho, rho > 0, not just asymptotically."""
    base, reg, theta0, phi_bar, flags = setup
    nx, ng, nh = base.n_x, base.n_ineq, base.n_eq

    d = base.differentiate(theta0, *phi_bar, mode="regularized",
                           options={"rho_1": varrho, "rho_2": rho})
    analytic = np.vstack([as_np(d["x"]), as_np(d["lam"]), as_np(d["nu"])])

    fd = np.zeros((nx + ng + nh, N_THETA))
    for j in range(N_THETA):
        tp, tm = theta0.copy(), theta0.copy()
        tp[j] += FD_EPS
        tm[j] -= FD_EPS
        xp, lp, np_, _, _ = solve_P2(base, reg, tp, phi_bar, varrho, rho)
        xm, lm, nm, _, _ = solve_P2(base, reg, tm, phi_bar, varrho, rho)
        fd[:, j] = (np.concatenate([xp, lp, np_]) - np.concatenate([xm, lm, nm])) / (2 * FD_EPS)

    assert rel_err(analytic, fd) < TOL_DERIV


# =====================================================================
# 4. CROSS-CHECK: regularized mode on P_1 == full mode on P_2
# =====================================================================

@pytest.mark.parametrize("setup", _params(scs_values=[True]), indirect=True)
def test_regularized_mode_equals_full_KKT_differentiation_of_P2(setup):
    """
    differentiate(mode="regularized") is a hand-derived reduction of the KKT
    system of P_2 (z eliminated via z = min(0, g(x_bar) + lam/varrho)). Applying
    the generic mode="full" differentiation directly to P_2 must give the same
    answer on the (x, lam, nu) rows and the theta columns of p_reg.

    P_2 satisfies LICQ by construction; mode="full" additionally needs strict
    complementarity for its diag(g)/diag(lam) block to be nonsingular, which is
    why this is restricted to scs=True.
    """
    base, reg, theta0, phi_bar, flags = setup
    nx, ng, nh = base.n_x, base.n_ineq, base.n_eq

    d_red = base.differentiate(
        theta0, *phi_bar, mode="regularized",
        options={"rho_1": VARRHO, "rho_2": RHO}
    )
    reduced = np.vstack([as_np(d_red["x"]), as_np(d_red["lam"]), as_np(d_red["nu"])])

    p = anchored_params(theta0, *phi_bar)
    _, _, _, _, sol = solve_P2(base, reg, theta0, phi_bar)
    d_full = reg.differentiate(p, sol.x, sol.lam, sol.nu, mode="full")

    # rows: xi = [x; lam; nu; z]  ->  keep the first nx+ng+nh
    # cols: p_reg = [theta; ...]  ->  keep the first N_THETA
    full = as_np(d_full["x"])[:nx + ng + nh, :N_THETA]

    assert rel_err(reduced, full) < TOL_CROSS, f"rel err {rel_err(reduced, full):.3e}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])