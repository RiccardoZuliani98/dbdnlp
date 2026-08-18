"""
Tier 1 -- scalar catastrophe normal forms (SOSC-type degeneracy).
Tier 2 -- constraint-side degeneracy (LICQ-type, mixed kernel, existence).

Each example supplies a CasADi builder in the codebase convention
``() -> (primal, param, cost, ineq, eq)`` and an analytic ``branches`` function
returning every nearby KKT point as an ``NLPSolution``, so
ground truth needs no solver.  ``EXPECTED`` records what the theory predicts,
which lets the library check itself (``metrics.hoelder``).

Conventions:  L = f + lam^T g + nu^T h,  g <= 0,  lam >= 0.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import casadi as ca
import numpy as np

from .core import Example
from .types import NLPSolution

#: Parameter values with |theta| <= TOL are treated as exactly degenerate.
#: Keep probe steps ``t`` comfortably above this (the default grid stops at 1e-6).
TOL = 1e-15

Factory = Callable[..., NLPSolution]


def _a(*vals) -> np.ndarray:
    return np.array(vals, dtype=float)


# ===========================================================================
# Tier 1: scalar normal forms, unconstrained
# ===========================================================================
def fold() -> Example:
    """f = x^3/3 - theta x.   Fold (A2).

    KKT points x = +-sqrt(theta): two for theta > 0, none for theta < 0.  The
    minimiser is x = +sqrt(theta), so the solution map is Hoelder-1/2 on one side
    and *does not exist* on the other -- the canonical failure of every IFT-based
    sensitivity scheme.

    LS data: alpha = -1, beta = f''' = 2, so a = -mu and w = +-sqrt(mu).
    Regularised: x = (-varrho + sqrt(varrho^2 + 4 theta))/2, hence dx/dtheta =
    1/varrho at theta = 0, bias -varrho/2, fold displaced to theta = -varrho^2/4.
    """

    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 1)
        return x, th, x[0] ** 3 / 3 - th[0] * x[0], None, None

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t = float(theta[0])
        if t < -TOL:
            return []
        if abs(t) <= TOL:
            return [at([0.0], kind="degenerate")]
        r = np.sqrt(t)
        return [at([r], kind="min"), at([-r], kind="max")]

    def reg_solution(theta, varrho, rho):
        t = float(np.atleast_1d(theta)[0])
        disc = varrho**2 + 4 * t
        return (-varrho + np.sqrt(disc)) / 2.0 if disc >= 0 else np.nan

    return Example(
        name="fold",
        tags=("tier1", "sosc", "fold", "existence"),
        build=build,
        theta_bar=_a(0.0),
        branches=branches,
        directions={"unfold": _a(1.0)},
        reference=([0.0], [], []),
        kernel=((_a(1.0), np.zeros(0), np.zeros(0)),),
        ls={
            "alpha": _a(-1.0),
            "beta": 2.0,
            "kernel_type": ("sosc",),
            "reg_solution": reg_solution,
            "reg_window": lambda varrho, rho: varrho**2 / 4.0,
        },
        notes="Canonical A2 fold; regularisation translates it, does not remove it.",
    )


def cusp() -> Example:
    """f = x^4/4 - theta x.   Cusp (A3), beta = 0.

    Unique KKT point x = cbrt(theta) for every theta: the solution *persists* but
    is only Hoelder-1/3.  Because x^3 + varrho x is strictly monotone the
    regularised map is smooth for all theta -- regularisation as a genuine cure
    rather than a translation.
    """

    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 1)
        return x, th, x[0] ** 4 / 4 - th[0] * x[0], None, None

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        x = float(np.cbrt(float(theta[0])))
        return [at([x], kind="min" if abs(x) > TOL else "degenerate")]

    def reg_solution(theta, varrho, rho):
        t = float(np.atleast_1d(theta)[0])
        roots = np.roots([1.0, 0.0, varrho, -t])
        real = roots[np.abs(roots.imag) < 1e-9].real
        return float(real[np.argmin(np.abs(real))]) if real.size else np.nan

    return Example(
        name="cusp",
        tags=("tier1", "sosc", "cusp"),
        build=build,
        theta_bar=_a(0.0),
        branches=branches,
        directions={"unfold": _a(1.0)},
        reference=([0.0], [], []),
        kernel=((_a(1.0), np.zeros(0), np.zeros(0)),),
        ls={
            "alpha": _a(-1.0),
            "beta": 0.0,
            "quartic": 6.0,
            "kernel_type": ("sosc",),
            "reg_solution": reg_solution,
        },
        notes="beta = 0: solution persists, regularised map smooth for all theta.",
    )


def pitchfork() -> Example:
    """f = x^4/4 - theta x^2/2.   Pitchfork; alpha = 0 (perturbation tangential).

    theta <= 0: unique minimiser x = 0.  theta > 0: minimisers +-sqrt(theta) and
    a saddle at 0.  The minimiser set is non-unique after the bifurcation, so the
    honest error metric is distance-to-set.  Regularisation displaces the
    bifurcation to theta = varrho.
    """

    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 1)
        return x, th, x[0] ** 4 / 4 - th[0] * x[0] ** 2 / 2, None, None

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t = float(theta[0])
        if t <= TOL:
            return [at([0.0], kind="degenerate" if abs(t) <= TOL else "min")]
        r = np.sqrt(t)
        return [at([r], kind="min"), at([-r], kind="min"), at([0.0], kind="saddle")]

    return Example(
        name="pitchfork",
        tags=("tier1", "sosc", "pitchfork", "nonunique"),
        build=build,
        theta_bar=_a(0.0),
        branches=branches,
        directions={"unfold": _a(1.0)},
        reference=([0.0], [], []),
        kernel=((_a(1.0), np.zeros(0), np.zeros(0)),),
        ls={
            "alpha": _a(0.0),
            "beta": 0.0,
            "kernel_type": ("sosc",),
            "bifurcation_point": lambda varrho, rho: varrho,
        },
        notes="alpha = 0: w = 0 persists; C(mu) governs. Symmetry -> non-unique minimisers.",
    )


def swallowtail() -> Example:
    """f = x^4/4 - theta1 x^2/2 - theta2 x.   Two-parameter unfolding of x^4.

    Up to three KKT points; two annihilate across the discriminant
    4 theta1^3 = 27 theta2^2.  The example where (varrho, rho) cannot suffice:
    they occupy the removable quadratic slot of the unfolding, and there are two
    genuinely transverse directions to cover.
    """

    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 2)
        return x, th, x[0] ** 4 / 4 - th[0] * x[0] ** 2 / 2 - th[1] * x[0], None, None

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t1, t2 = float(theta[0]), float(theta[1])
        out = []
        for r in np.roots([1.0, 0.0, -t1, -t2]):
            if abs(r.imag) > 1e-9:
                continue
            x = float(r.real)
            curv = 3 * x**2 - t1
            kind = "min" if curv > 1e-10 else ("max" if curv < -1e-10 else "degenerate")
            out.append(at([x], kind=kind, curvature=curv))
        return out

    return Example(
        name="swallowtail",
        tags=("tier1", "sosc", "cusp", "twoparam"),
        build=build,
        theta_bar=_a(0.0, 0.0),
        branches=branches,
        directions={"tilt": _a(0.0, 1.0), "split": _a(1.0, 0.0), "diag": _a(1.0, 1.0)},
        reference=([0.0], [], []),
        kernel=((_a(1.0), np.zeros(0), np.zeros(0)),),
        ls={
            "alpha": _a(0.0, -1.0),
            "beta": 0.0,
            "kernel_type": ("sosc",),
            "discriminant": lambda th: 4 * float(th[0]) ** 3 - 27 * float(th[1]) ** 2,
        },
        notes="A3 germ with full unfolding; shows (varrho, rho) are not enough.",
    )


# ===========================================================================
# Tier 2: constraint-side degeneracy
# ===========================================================================
def mixed_kernel(b: float = 1.0, c: float = 1.0, name: Optional[str] = None) -> Example:
    """f = x1^2/2 + b x1 x2 + (c/6) x2^3,   h = x1 - theta.

    SOSC fails on ker J_hx = span{(0,1)} with a *mixed* kernel direction
    phi = ((0,1), -b): both primal and dual components nonzero, so
    M = varrho |u|^2 - rho |v|^2 = varrho - rho b^2 vanishes at varrho/rho = b^2.
    The "both weights positive is not enough" example.

    c != 0:  KKT points x2 = +-sqrt(-2 b theta / c), a fold on one side only.
    c == 0:  unbounded below for theta != 0 (f is linear in x2) and a whole line
             of KKT points at theta = 0.  The regularised derivative is finite
             while nothing it approximates exists, and everything is closed form:

                 dx1/dtheta = 1 / (1 + rho(1 + varrho) - rho b^2 / varrho).
    """

    def build():
        x = ca.SX.sym("x", 2)
        th = ca.SX.sym("theta", 1)
        f = 0.5 * x[0] ** 2 + b * x[0] * x[1] + (c / 6.0) * x[1] ** 3
        return x, th, f, None, ca.vertcat(x[0] - th[0])

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t = float(theta[0])
        if abs(c) <= TOL:
            if abs(b * t) <= TOL:
                return [at([0.0, 0.0], nu=[0.0], kind="degenerate", continuum=_a(0.0, 1.0))]
            return []  # unbounded below: no KKT point at all
        disc = -2.0 * b * t / c
        if disc < -TOL:
            return []
        if abs(disc) <= TOL:
            return [at([t, 0.0], nu=[-t], kind="degenerate")]
        r = np.sqrt(disc)
        out = []
        for x2 in (r, -r):
            curv = c * x2  # Z^T H_xx Z with Z = (0,1)
            kind = "min" if curv > 1e-10 else ("max" if curv < -1e-10 else "degenerate")
            out.append(at([t, x2], nu=[-(t + b * x2)], kind=kind, curvature=curv))
        return out

    def reg_jac_exact(varrho: float, rho: float) -> np.ndarray:
        """Exact dphi/dtheta of K + diag(varrho I, -rho I) (rows x1, x2, nu)."""
        H = np.array([[1.0, b], [b, 0.0]])
        A = np.array([[1.0, 0.0]])
        K = np.block([[H + varrho * np.eye(2), A.T], [A, -rho * np.eye(1)]])
        rhs = np.zeros((3, 1))
        rhs[2, 0] = 1.0
        return np.linalg.solve(K, rhs)

    return Example(
        name=name or ("mixed_kernel_fold" if abs(c) > TOL else "mixed_kernel_quadratic"),
        tags=("tier2", "sosc", "mixed", "ratio_trap")
        + (("fold", "existence") if abs(c) > TOL else ("existence", "unbounded")),
        build=build,
        theta_bar=_a(0.0),
        branches=branches,
        directions={"unfold": _a(1.0)},
        reference=([0.0, 0.0], [], [0.0]),
        kernel=((_a(0.0, 1.0), np.zeros(0), _a(-b)),),
        ls={
            "alpha": _a(b),
            "beta": c,
            "kernel_type": ("mixed",),
            "critical_ratio": b**2,
            "M": lambda varrho, rho: (varrho - rho * b**2) / (1.0 + b**2),
            "reg_jac_exact": reg_jac_exact,
            "b": b,
            "c": c,
        },
        notes=(
            "Sweep varrho/rho across b^2: cond(K+R) spikes and the derivative "
            "diverges although both regularisation weights are positive."
        ),
    )


def licq_two_eq() -> Example:
    """f = x^2/2,  h1 = x - theta1,  h2 = x - theta2.   Pure LICQ failure.

    ker K = span{((0), (1,-1))} is a pure dual direction, so beta = 0 and the
    reduced equation degenerates to the compatibility condition
    v^T J_htheta mu = 0, i.e. mu1 = mu2:

      * ``compatible`` (1,1):   x = theta1 is analytic; only the multiplier is
        non-unique (the line nu1 + nu2 = -x).
      * ``incompatible`` (1,-1): P1 is *infeasible*.  The regularised problem
        still solves, so its derivative is that of a relaxation, with violation
        O(|mu|) and multipliers O(|mu|/rho):

            x_reg = (theta1 + theta2) / (2 + rho(1 + varrho)).
    """

    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 2)
        return x, th, 0.5 * x[0] ** 2, None, ca.vertcat(x[0] - th[0], x[0] - th[1])

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t1, t2 = float(theta[0]), float(theta[1])
        if abs(t1 - t2) > TOL:
            return []  # infeasible
        x = t1
        return [
            at([x], nu=[-x / 2.0, -x / 2.0], kind="min",
               multiplier_line=(_a(1.0, -1.0), -x))
        ]

    def reg_jac_exact(varrho: float, rho: float) -> np.ndarray:
        K = np.block(
            [[np.array([[1.0 + varrho]]), np.ones((1, 2))],
             [np.ones((2, 1)), -rho * np.eye(2)]]
        )
        rhs = np.zeros((3, 2))
        rhs[1, 0] = 1.0
        rhs[2, 1] = 1.0
        return np.linalg.solve(K, rhs)

    def relaxed_solution(theta, varrho: float, rho: float) -> float:
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        return float((th[0] + th[1]) / (2.0 + rho * (1.0 + varrho)))

    return Example(
        name="licq_two_eq",
        tags=("tier2", "licq", "infeasible", "nonunique_dual"),
        build=build,
        theta_bar=_a(0.0, 0.0),
        branches=branches,
        directions={"compatible": _a(1.0, 1.0), "incompatible": _a(1.0, -1.0)},
        reference=([0.0], [], [0.0, 0.0]),
        kernel=((np.zeros(1), np.zeros(0), _a(1.0, -1.0)),),
        ls={
            "alpha": _a(-1.0, 1.0),
            "beta": 0.0,
            "kernel_type": ("licq",),
            "reg_jac_exact": reg_jac_exact,
            "relaxed_solution": relaxed_solution,
        },
        notes="One example, two directions: benign vs infeasible LICQ failure.",
    )


_TANGENCY_KERNEL = (
    (_a(1.0, 0.0), np.zeros(2), np.zeros(0)),    # pure primal (SOSC)
    (np.zeros(2), _a(1.0, -2.0), np.zeros(0)),   # pure dual (LICQ)
)


def tangency(kappa: float = 0.5, multiplier: Optional[float] = None) -> Example:
    """min -x2 - (kappa/2) x1^2  s.t.  x1^2 + (x2-1)^2 <= 1,  x2 <= 2 + theta.

    At theta = 0 the active gradients (0,2) and (0,1) are parallel: LICQ fails and
    the multiplier set is the segment {2 lam1 + lam2 = 1, lam >= 0}.  Taking
    lam1 = kappa/2 (the limit of the theta < 0 branch) also kills SOSC in the x1
    direction, so ker K is two dimensional -- one pure-primal and one pure-dual
    generator, coupled through B[phi_A, phi_A, phi_B] = 2 v1 != 0.

    Solution map (kappa < 1):
      theta > 0:  unique min x = (0, 2), *constant*  -> derivative 0
      theta = 0:  unique min x = (0, 2), multiplier set a segment
      theta < 0:  two minimisers (+-sqrt(-2 theta - theta^2), 2 + theta) plus a
                  saddle at (0, 2 + theta)  -> Hoelder-1/2
    """
    lam1_bar = kappa / 2.0 if multiplier is None else multiplier

    def build():
        x = ca.SX.sym("x", 2)
        th = ca.SX.sym("theta", 1)
        f = -x[1] - 0.5 * kappa * x[0] ** 2
        g = ca.vertcat(x[0] ** 2 + (x[1] - 1.0) ** 2 - 1.0, x[1] - 2.0 - th[0])
        return x, th, f, g, None

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t = float(theta[0])
        if t > TOL:
            return [at([0.0, 2.0], lam=[0.5, 0.0], kind="min")]
        if abs(t) <= TOL:
            return [
                at([0.0, 2.0], lam=[lam1_bar, 1.0 - 2.0 * lam1_bar], kind="min",
                   multiplier_segment=(0.0, 0.5))
            ]
        if t <= -1.0 + TOL:
            return []
        r = np.sqrt(max(-2.0 * t - t**2, 0.0))
        lam = [kappa / 2.0, 1.0 - kappa * (1.0 + t)]
        return [
            at([r, 2.0 + t], lam=lam, kind="min"),
            at([-r, 2.0 + t], lam=lam, kind="min"),
            at([0.0, 2.0 + t], lam=[0.0, 1.0], kind="saddle"),
        ]

    return Example(
        name="tangency",
        tags=("tier2", "licq", "sosc", "d2", "nonunique", "onesided"),
        build=build,
        theta_bar=_a(0.0),
        branches=branches,
        directions={"tighten": _a(-1.0), "loosen": _a(1.0)},
        reference=([0.0, 2.0], [lam1_bar, 1.0 - 2.0 * lam1_bar], []),
        kernel=_TANGENCY_KERNEL,
        ls={
            "beta": 0.0,
            "kernel_type": ("sosc", "licq"),
            "coupling_B_AAB": 2.0,
            "kappa": kappa,
            "lam1_bar": lam1_bar,
            "multiplier_family": lambda s: np.array([0.5 * s, 1.0 - s]),
        },
        notes=(
            "d = 2 kernel with primal-dual coupling; outside the d = 1 fold "
            "classification, and the multiplier chosen at theta = 0 changes K."
        ),
    )


def wachter_biegler() -> Example:
    """min x1 s.t. x1^2 - x2 - 1 - theta1 = 0, x1 - x3 - 1/2 - theta2 = 0, x2,x3 >= 0.

    Control example: notoriously hard for line-search interior point, but LICQ,
    SOSC and strict complementarity all *hold*, and
    x1 = max(sqrt(1 + theta1), 1/2 + theta2) is smooth near theta = 0.  Any error
    reported here is regularisation bias or a bug, not degeneracy.
    """

    def build():
        x = ca.SX.sym("x", 3)
        th = ca.SX.sym("theta", 2)
        g = ca.vertcat(-x[1], -x[2])
        h = ca.vertcat(x[0] ** 2 - x[1] - 1.0 - th[0], x[0] - x[2] - 0.5 - th[1])
        return x, th, x[0], g, h

    def branches(theta: np.ndarray, at: Factory) -> List[NLPSolution]:
        t1, t2 = float(theta[0]), float(theta[1])
        if 1.0 + t1 < 0:
            return []
        cand_a, cand_b = np.sqrt(1.0 + t1), 0.5 + t2
        x1 = max(cand_a, cand_b)
        x2, x3 = x1**2 - 1.0 - t1, x1 - 0.5 - t2
        if cand_a >= cand_b:      # x2 = 0 active, x3 > 0
            lam, nu = _a(1.0 / (2 * x1), 0.0), _a(-1.0 / (2 * x1), 0.0)
        else:                     # x3 = 0 active, x2 > 0
            lam, nu = _a(0.0, 1.0), _a(0.0, -1.0)
        return [at([x1, x2, x3], lam=lam, nu=nu, kind="min")]

    return Example(
        name="wachter_biegler",
        tags=("tier2", "control", "regular"),
        build=build,
        theta_bar=_a(0.0, 0.0),
        branches=branches,
        directions={"eq1": _a(1.0, 0.0), "eq2": _a(0.0, 1.0), "diag": _a(1.0, 1.0)},
        reference=([1.0, 0.0, 0.5], [0.5, 0.0], [-0.5, 0.0]),
        kernel=(),
        ls={"kernel_type": (), "beta": None},
        notes="Nondegenerate but numerically nasty; isolates bias from degeneracy.",
    )


# ===========================================================================
# expected behaviour of the *true* solution map, per example and direction
# ===========================================================================
#   value  = Hoelder exponent of ||phi(theta) - phi_bar|| along the direction
#   None   = no KKT point exists on that side
#   inf    = the tracked solution does not move at all
EXPECTED: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "fold": {"unfold": {"+": 0.5, "-": None}},
    "cusp": {"unfold": {"+": 1.0 / 3.0, "-": 1.0 / 3.0}},
    "pitchfork": {"unfold": {"+": 0.5, "-": np.inf}},
    "swallowtail": {
        "tilt": {"+": 1.0 / 3.0, "-": 1.0 / 3.0},
        "split": {"+": 0.5, "-": np.inf},
    },
    "mixed_kernel_fold": {"unfold": {"+": None, "-": 0.5}},
    "mixed_kernel_quadratic": {"unfold": {"+": None, "-": None}},
    "licq_two_eq": {
        "compatible": {"+": 1.0, "-": 1.0},
        "incompatible": {"+": None, "-": None},
    },
    "tangency": {
        "loosen": {"+": np.inf, "-": 0.5},
        "tighten": {"+": 0.5, "-": np.inf},
    },
    "wachter_biegler": {
        "eq1": {"+": 1.0, "-": 1.0},
        "eq2": {"+": 1.0, "-": 1.0},
        "diag": {"+": 1.0, "-": 1.0},
    },
}


#: which hypotheses are meant to fail at each reference point.  Checked against
#: what ``adapter.solution_properties`` actually measures, so a mis-specified
#: example (wrong multiplier, wrong base point) cannot pass silently.
EXPECTED_PROPERTIES: Dict[str, Dict[str, object]] = {
    "fold":                   {"licq": True,  "sosc": False, "dim_kernel": 1},
    "cusp":                   {"licq": True,  "sosc": False, "dim_kernel": 1},
    "pitchfork":              {"licq": True,  "sosc": False, "dim_kernel": 1},
    "swallowtail":            {"licq": True,  "sosc": False, "dim_kernel": 1},
    "mixed_kernel_fold":      {"licq": True,  "sosc": False, "dim_kernel": 1},
    "mixed_kernel_quadratic": {"licq": True,  "sosc": False, "dim_kernel": 1},
    "licq_two_eq":            {"licq": False, "sosc": True,  "dim_kernel": 1},
    "tangency":               {"licq": False, "sosc": False, "dim_kernel": 2},
    "wachter_biegler":        {"licq": True,  "sosc": True,  "dim_kernel": 0},
}


# ===========================================================================
# registry
# ===========================================================================
REGISTRY: Dict[str, Callable[[], Example]] = {
    # tier 1
    "fold": fold,
    "cusp": cusp,
    "pitchfork": pitchfork,
    "swallowtail": swallowtail,
    # tier 2
    "mixed_kernel_fold": lambda: mixed_kernel(b=1.0, c=1.0),
    "mixed_kernel_quadratic": lambda: mixed_kernel(b=1.0, c=0.0),
    "licq_two_eq": licq_two_eq,
    "tangency": tangency,
    "wachter_biegler": wachter_biegler,
}


def get(name: str) -> Example:
    if name not in REGISTRY:
        raise KeyError(f"unknown example {name!r}; available: {sorted(REGISTRY)}")
    ex = REGISTRY[name]()
    ex.ls.setdefault("expected", EXPECTED.get(ex.name, {}))
    ex.ls.setdefault("expected_properties", EXPECTED_PROPERTIES.get(ex.name, {}))
    return ex


def all_examples(tags=None) -> List[Example]:
    out = [get(k) for k in REGISTRY]
    if tags:
        want = set(tags)
        out = [e for e in out if want.issubset(set(e.tags))]
    return out
