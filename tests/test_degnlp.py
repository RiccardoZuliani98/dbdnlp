"""Self-consistency tests for the example library and the harness.

Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from dbdnlp._impl.nlp import DifferentiableNLP

from degnlp import examples as ex_mod
from degnlp import metrics as M
from degnlp import report
from degnlp.core import kind_of, phi_np, stack_sensitivities
from degnlp.reference import FiniteDifferenceOracle, Regularized
from degnlp.report import jacobian_for
from degnlp.types import NLPSolution, to_np

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
    branches = ex.solve(ex.theta_bar)
    assert branches, "solve() must return the base point at theta_bar"
    xbar = np.asarray(ex.reference[0], dtype=float)
    assert min(np.linalg.norm(np.asarray(b.x).reshape(-1) - xbar) for b in branches) < 1e-9


@pytest.mark.parametrize("name", ALL)
def test_solutions_are_nlpsolutions_with_kind(name):
    """Branches must be usable anywhere an NLPSolution is expected, and carry
    their second-order label in ``properties``."""
    ex = ex_mod.get(name)
    for sol in ex.solve(ex.theta_bar):
        assert isinstance(sol, NLPSolution)
        assert kind_of(sol) != "unknown"
        assert set(sol.kkt_residuals) == {
            "stationarity", "inequality", "equality", "slackness", "dual_feasibility",
        }
        assert phi_np(sol).size == ex.dims.n_phi


@pytest.mark.parametrize("name", ALL)
def test_reference_properties_are_measured(name):
    """LICQ/SOSC/nonsingularity come from funcs.ift, not from assertion.

    This is the check that catches a mis-specified base point: pick the wrong
    multiplier out of a degenerate multiplier set and the measured kernel
    dimension stops matching the declared one.
    """
    ex = ex_mod.get(name)
    props = ex.phi_bar.properties
    want = ex.ls["expected_properties"]
    for key in ("licq", "scs", "sosc"):
        assert props[key] is want[key], f"{name}: {key}"
    assert props["dim_kernel"] == want["dim_kernel"]
    assert (props["dim_kernel"] == 0) is (props["licq"] and props["ssosc"])
    assert props["kkt_residual"] < 1e-9


@pytest.mark.parametrize("name", ALL)
def test_declared_hoelder_exponents(name):
    ex = ex_mod.get(name)
    for dname in ex.directions:
        h = M.hoelder(ex, direction=dname)
        for side, ok in h.matches_expected.items():
            assert ok, f"{name}/{dname} side {side}: got {h.exponent[side]}, want {h.expected.get(side)}"


def _amax(v) -> float:
    """max |v|, zero for the empty vector (unconstrained examples)."""
    v = np.asarray(v, dtype=float)
    return float(np.max(np.abs(v))) if v.size else 0.0


@pytest.mark.parametrize("name", ALL)
def test_declared_kernel_is_a_kernel(name):
    """The declared generators really span ker K, and there are the right number
    of them -- checked against ``check_properties['dim_kernel']``."""
    ex = ex_mod.get(name)
    declared = ex.kernel_matrix()
    measured = ex.phi_bar.properties["dim_kernel"]
    assert measured == declared.shape[1], f"{name}: kernel dimension mismatch"
    if declared.shape[1] == 0:
        return

    # A generator (u, v) of ker K satisfies  H_xx u + J^T v = 0  and  J u = 0,
    # written out from funcs.ift rather than from any assembled matrix.
    s = ex.phi_bar
    H, _, J_g, _, J_h, _ = ex.nlp.funcs.ift(s.x, s.lam, s.nu, s.theta)
    H = to_np(H).reshape(ex.dims.n_x, ex.dims.n_x)
    J = np.vstack([
        to_np(J_g).reshape(ex.dims.n_ineq, ex.dims.n_x),
        to_np(J_h).reshape(ex.dims.n_eq, ex.dims.n_x),
    ])
    for k in range(declared.shape[1]):
        u = ex.dims.rows(declared[:, k], "x")
        v = ex.dims.rows(declared[:, k], "dual")
        assert _amax(H @ u + J.T @ v) < 1e-8, f"{name}: generator {k} stationarity"
        assert _amax(J @ u) < 1e-8, f"{name}: generator {k} feasibility"


# ---------------------------------------------------------------------------
# the harness certifies a correct derivative and flags a wrong one
# ---------------------------------------------------------------------------
def test_oracle_is_second_order_on_regular_example():
    ex = ex_mod.get("wachter_biegler")
    J = jacobian_for(FiniteDifferenceOracle(t=1e-6), ex)
    rep = M.taylor(ex, J, direction="eq1", block="x")
    assert rep.slope["+"] > 1.5
    assert "second order" in rep.verdict["+"]


def _zeros(ex):
    d = ex.dims
    return {
        "x": np.zeros((d.n_x, d.n_theta)), 
        "lam": np.zeros((d.n_ineq, d.n_theta)),
        "nu": np.zeros((d.n_eq, d.n_theta))
    }


def test_zero_jacobian_is_first_order_on_regular_example():
    ex = ex_mod.get("wachter_biegler")
    J = _zeros(ex)
    rep = M.taylor(ex, J, direction="eq1", block="x")
    assert 0.9 < rep.slope["+"] < 1.2


def test_fold_defeats_every_jacobian():
    """No matter what matrix is supplied, the error is only O(sqrt(t)).

    A large entry can beat sqrt(t) over a window of t (that is exactly the
    O(1/eps) regularised branch), so the probe has to go deep enough in t.
    """
    ex = ex_mod.get("fold")
    ts = np.geomspace(1e-4, 1e-12, 9)
    empty = {"lam": np.zeros((0, 1)), "nu": np.zeros((0, 1))}
    for dx in (0.0, 1.0, 1e3):
        rep = M.taylor(ex, {"x": np.array([[dx]]), **empty}, block="x", ts=ts)
        assert rep.slope["+"] < 0.6, (dx, rep.slope)
        assert not np.isfinite(rep.true_slope["-"])   # no solution on that side


def test_contract_is_enforced():
    """All three blocks are required; a partial answer is an error, not a
    silently-zeroed row."""
    ex = ex_mod.get("licq_two_eq")
    full = {"x": np.array([[0.5, 0.5]]), "lam": np.zeros((0, 2)), "nu": np.zeros((2, 2))}
    assert "exact" in M.taylor(ex, full, direction="compatible", block="x").verdict["+"]
    with pytest.raises(KeyError):
        M.taylor(ex, {"x": np.array([[0.5, 0.5]])}, direction="compatible")
    with pytest.raises(ValueError):
        M.taylor(ex, {**full, "nu": np.zeros((5, 2))}, direction="compatible")


def test_nan_blocks_propagate_rather_than_read_as_zero():
    """A differentiator reporting NaN must not be scored as if it said zero."""
    ex = ex_mod.get("licq_two_eq")
    nan = {"x": np.full((1, 2), np.nan), "lam": np.zeros((0, 2)), "nu": np.zeros((2, 2))}
    rep = M.taylor(ex, nan, direction="compatible", block="x")
    assert not np.isfinite(rep.slope["+"])
    assert all(not np.isfinite(r["err"]) for r in rep.table.rows if r["exists"])


# ---------------------------------------------------------------------------
# closed-form regularised Jacobians
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["mixed_kernel_quadratic", "licq_two_eq"])
def test_reference_kkt_matches_closed_form(name):
    ex = ex_mod.get(name)
    varrho, rho = 1e-2, 1e-3
    J = jacobian_for(Regularized(rho_1=varrho, rho_2=rho), ex)
    out = M.reg_jac_exactness(ex, J, varrho, rho)
    assert out is not None and out["rel_err"] < 1e-5, out


def test_ratio_trap_is_singular_at_the_predicted_ratio():
    ex = ex_mod.get("mixed_kernel_quadratic")
    b2 = ex.ls["critical_ratio"]
    rho = 1e-4
    norms = {}
    for ratio in (0.25 * b2, b2, 4.0 * b2):
        J = stack_sensitivities(jacobian_for(Regularized(rho_1=ratio * rho, rho_2=rho), ex), ex.dims)
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
        lambda e: jacobian_for(Regularized(rho_1=e, rho_2=e), ex),
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
        viol = np.abs(np.asarray(ex.nlp.funcs.eq(x, theta)).reshape(-1))
        assert np.max(viol) > 1e-5


def test_suite_runs_end_to_end():
    s = report.run_suite(Regularized(rho_1=1e-3, rho_2=1e-3))
    assert s.rows
    assert max(s.self_check.values()) < 1e-6


# ---------------------------------------------------------------------------
# the NLP factory is swappable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_example_carries_a_differentiable_nlp(name):
    """Each example owns a constructed DifferentiableNLP, and Dims agrees with it."""
    ex = ex_mod.get(name)
    assert isinstance(ex.nlp, DifferentiableNLP)
    assert (ex.nlp.n_x, ex.nlp.n_ineq, ex.nlp.n_eq, ex.nlp.n_theta) == (
        ex.dims.n_x, ex.dims.n_ineq, ex.dims.n_eq, ex.dims.n_theta
    )


@pytest.mark.parametrize("name", ALL)
def test_differentiate_dict_is_consumed_directly(name):
    """differentiate() returns {'x','lam','nu'}; the harness must take it as is."""
    ex = ex_mod.get(name)
    out = ex.nlp.differentiate(
        ex.phi_bar.theta, ex.phi_bar.x, ex.phi_bar.lam, ex.phi_bar.nu,
        mode="regularized", options={"rho_1": 1e-2, "rho_2": 1e-2},
    )
    assert set(out) == {"x", "lam", "nu"}
    assert stack_sensitivities(out, ex.dims).shape == (ex.dims.n_phi, ex.dims.n_theta)


def test_evaluate_round_trips_a_point():
    """DifferentiableNLP.evaluate must package an arbitrary point, empty
    multiplier blocks included."""
    ex = ex_mod.get("fold")
    sol = ex.nlp.evaluate([0.25], [0.5])
    assert isinstance(sol, NLPSolution)
    assert sol.lam.shape == (0, 1) and sol.nu.shape == (0, 1)
    assert np.max(np.abs(to_np(sol.kkt_residuals["stationarity"]))) < 1e-12
    with pytest.raises(ValueError):
        ex.nlp.evaluate([0.0], [0.0, 0.0])


@pytest.mark.parametrize("name", ALL)
def test_sensitivity_blocks_stack_in_phi_order(name):
    ex = ex_mod.get(name)
    sens = jacobian_for(Regularized(rho_1=1e-2, rho_2=1e-2), ex)
    J = stack_sensitivities(sens, ex.dims)
    d = ex.dims
    assert J.shape == (d.n_phi, d.n_theta)
    for key in ("x", "lam", "nu"):
        block = to_np(sens[key]).reshape(-1)
        assert np.allclose(d.rows(J, key).reshape(-1), block)
