"""Self-consistency tests for the example library and the harness.

Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from degnlp import examples as ex_mod
from degnlp import metrics as M
from degnlp import report
from degnlp.reference import FiniteDifferenceOracle, RegularizedKKT

ALL = list(ex_mod.REGISTRY)


# ---------------------------------------------------------------------------
# the examples describe themselves correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_reference_point_is_kkt(name):
    ex = ex_mod.get(name)
    assert ex.check_reference_point() < 1e-6


@pytest.mark.parametrize("name", ALL)
def test_solve_reproduces_reference_point(name):
    ex = ex_mod.get(name)
    branches = ex.solve_at(ex.theta_bar)
    assert branches, "solve() must return the base point at theta_bar"
    d = min(np.linalg.norm(b.x - ex.phi_bar.x) for b in branches)
    assert d < 1e-9


@pytest.mark.parametrize("name", ALL)
def test_declared_hoelder_exponents(name):
    ex = ex_mod.get(name)
    for dname in ex.directions:
        h = M.hoelder(ex, direction=dname)
        for side, ok in h.matches_expected.items():
            assert ok, f"{name}/{dname} side {side}: got {h.exponent[side]}, want {h.expected.get(side)}"


@pytest.mark.parametrize("name", ALL)
def test_declared_kernel_is_a_kernel(name):
    """K @ phi_k == 0 for the declared generators, K built from FD derivatives."""
    ex = ex_mod.get(name)
    Nmat = ex.kernel_matrix()
    if Nmat.shape[1] == 0:
        return
    lay, fns = ex.layout, ex.functions
    th, x = ex.theta_bar, ex.phi_bar.x
    lam, nu = ex.phi_bar.lam, ex.phi_bar.nu
    H = fns.hess_xx_L(x, lam, nu, th)
    G = fns.jac_g(x, th)
    A = fns.jac_h(x, th)
    K = np.zeros((lay.size, lay.size))
    K[: lay.n, : lay.n] = H
    if lay.m_g:
        K[: lay.n, lay.blocks["lam"]] = G.T
        K[lay.blocks["lam"], : lay.n] = G
    if lay.m_h:
        K[: lay.n, lay.blocks["nu"]] = A.T
        K[lay.blocks["nu"], : lay.n] = A
    assert np.max(np.abs(K @ Nmat)) < 1e-5, f"{name}: declared kernel is not in ker K"


# ---------------------------------------------------------------------------
# the harness certifies a correct derivative and flags a wrong one
# ---------------------------------------------------------------------------
def test_oracle_is_second_order_on_regular_example():
    ex = ex_mod.get("wachter_biegler")
    J = FiniteDifferenceOracle(t=1e-6)(ex)
    rep = M.taylor(ex, J, direction="eq1", block="x")
    assert rep.slope["+"] > 1.5
    assert "second order" in rep.verdict["+"]


def test_zero_jacobian_is_first_order_on_regular_example():
    ex = ex_mod.get("wachter_biegler")
    J = np.zeros((ex.layout.size, ex.p))
    rep = M.taylor(ex, J, direction="eq1", block="x")
    assert 0.9 < rep.slope["+"] < 1.2


def test_fold_defeats_every_jacobian():
    """No matter what matrix is supplied, the error is only O(sqrt(t)).

    A large entry can beat sqrt(t) over a window of t (that is exactly the
    O(1/eps) regularised branch), so the probe has to go deep enough in t.
    """
    ex = ex_mod.get("fold")
    ts = np.geomspace(1e-4, 1e-12, 9)
    for J in (np.zeros((1, 1)), np.array([[1.0]]), np.array([[1e3]])):
        rep = M.taylor(ex, J, block="x", ts=ts)
        assert rep.slope["+"] < 0.6, (J, rep.slope)
        assert not np.isfinite(rep.true_slope["-"])   # no solution on that side


def test_partial_jacobian_is_accepted():
    """A differentiator that only produces dx/dtheta is still testable."""
    ex = ex_mod.get("licq_two_eq")
    exact = M.taylor(ex, {"x": np.array([[0.5, 0.5]])}, direction="compatible", block="x")
    assert "exact" in exact.verdict["+"]
    wrong = M.taylor(ex, {"x": np.array([[0.5, 0.4]])}, direction="compatible", block="x")
    assert 0.9 < wrong.slope["+"] < 1.2


# ---------------------------------------------------------------------------
# closed-form regularised Jacobians
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["mixed_kernel_quadratic", "licq_two_eq"])
def test_reference_kkt_matches_closed_form(name):
    ex = ex_mod.get(name)
    varrho, rho = 1e-2, 1e-3
    J = RegularizedKKT(varrho=varrho, rho=rho)(ex)
    out = M.reg_jac_exactness(ex, J, varrho, rho)
    assert out is not None and out["rel_err"] < 1e-5, out


def test_ratio_trap_is_singular_at_the_predicted_ratio():
    ex = ex_mod.get("mixed_kernel_quadratic")
    b2 = ex.ls["critical_ratio"]
    rho = 1e-4
    norms = {}
    for ratio in (0.25 * b2, b2, 4.0 * b2):
        J = RegularizedKKT(varrho=ratio * rho, rho=rho)(ex)
        norms[ratio] = np.linalg.norm(J)
    assert norms[b2] > 10 * norms[0.25 * b2]
    assert norms[b2] > 10 * norms[4.0 * b2]


# ---------------------------------------------------------------------------
# rates predicted by the Lyapunov-Schmidt analysis
# ---------------------------------------------------------------------------
def test_fold_saturation_and_knee_orders():
    ex = ex_mod.get("fold")
    sweep = M.eps_sweep(
        ex,
        lambda e: RegularizedKKT(varrho=e, rho=e)(ex),
        eps=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
        ts=np.geomspace(1e-2, 1e-10, 17),
        side="+",
    )
    assert -1.2 < sweep.saturation_order < -0.8      # |J| ~ 1/eps
    assert 1.6 < sweep.knee_order < 2.4              # good only for |mu| = O(eps^2)


def test_fold_solution_bias_is_first_order():
    ex = ex_mod.get("fold")
    sol = ex.ls["reg_solution"]
    sb = M.solution_bias(ex, lambda th, e: np.array([sol(th, e, e)]), t=1e-2)
    assert 0.85 < sb.order < 1.15, sb


def test_relaxed_solution_bias_licq():
    """On the incompatible direction P1 has no solution, so the honest reference
    is the relaxation itself: violation is O(|mu|) and independent of eps."""
    ex = ex_mod.get("licq_two_eq")
    relaxed = ex.ls["relaxed_solution"]
    d = ex.direction("incompatible")
    for e in (1e-2, 1e-3):
        theta = ex.theta_bar + 1e-3 * d
        x = relaxed(theta, e, e)
        viol = np.abs(ex.functions.hval(np.array([x]), theta))
        assert np.max(viol) > 1e-5


def test_suite_runs_end_to_end():
    s = report.run_suite(RegularizedKKT(varrho=1e-3, rho=1e-3))
    assert s.rows
    assert max(s.self_check.values()) < 1e-6


# ---------------------------------------------------------------------------
# optional symbolic mirror
# ---------------------------------------------------------------------------
casadi = pytest.importorskip("casadi", reason="casadi not installed")


@pytest.mark.parametrize("name", ALL)
def test_symbolic_matches_numeric(name):
    from degnlp import symbolic

    ex = ex_mod.get(name)
    assert ex.build is not None
    out = symbolic.validate(ex)
    assert max(out.values()) < 1e-5, out
