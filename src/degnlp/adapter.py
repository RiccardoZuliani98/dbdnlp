"""
Construction of the NLP object the examples hand to a differentiator, and the
numerical analysis that reads degeneracy off it.

``Example.nlp`` is whatever ``NLP_FACTORY`` returns.  By default that is
:class:`ReferenceNLP`, a minimal object exposing exactly the surface the harness
uses -- ``sym`` (``NLPSymbolic``), ``funcs`` (``NLPFunctions``), and the four
dimensions.  Point it at the real class once and every example follows::

    from dbdnlp._impl.nlp import DifferentiableNLP
    from degnlp import adapter

    adapter.set_factory(
        lambda primal, param, cost, ineq, eq: DifferentiableNLP(...)
    )

Everything downstream reads ``nlp.funcs.kkt`` and ``nlp.funcs.ift`` positionally,
in the documented output order, so any object with those two CasADi functions
and the dimension attributes works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import casadi as ca
import numpy as np

from .types import (
    NLPFunctions,
    NLPSolution,
    NLPSymbolic,
    Symbolic,
    mat,
    to_dm,
    to_np,
)

INF = ca.inf

Builder = Callable[
    [], Tuple[Symbolic, Symbolic, Symbolic, Optional[Symbolic], Optional[Symbolic]]
]


# ---------------------------------------------------------------------------
# symbolic + compiled function construction
# ---------------------------------------------------------------------------
# TODO: redundant, can be pulled directly from DifferentiableNLP
def build_symbolic(
    primal: Symbolic,
    param: Symbolic,
    cost: Symbolic,
    ineq: Optional[Symbolic] = None,
    eq: Optional[Symbolic] = None,
) -> NLPSymbolic:
    """Assemble an :class:`NLPSymbolic` from ``f``, ``g <= 0``, ``h == 0``."""
    n_ineq = 0 if ineq is None else int(ineq.numel())
    n_eq = 0 if eq is None else int(eq.numel())
    ineq = ca.SX.zeros(0, 1) if ineq is None else ca.reshape(ineq, n_ineq, 1)
    eq = ca.SX.zeros(0, 1) if eq is None else ca.reshape(eq, n_eq, 1)
    dual_ineq = ca.SX.sym("lam", n_ineq)
    dual_eq = ca.SX.sym("nu", n_eq)
    return NLPSymbolic(
        cost=cost,
        ineq=ineq,
        eq=eq,
        cst=ca.vertcat(ineq, eq),
        primal=primal,
        param=param,
        dual_ineq=dual_ineq,
        dual_eq=dual_eq,
        dual_cst=ca.vertcat(dual_ineq, dual_eq),
    )

#TODO: this is redundant and can be pulled from DifferentiableNLP directly
def build_functions(sym: NLPSymbolic) -> NLPFunctions:
    """Compile the :class:`NLPFunctions` bundle for a symbolic NLP."""
    x, th = sym.primal, sym.param
    lam, nu = sym.dual_ineq, sym.dual_eq
    n_ineq, n_eq = int(sym.ineq.numel()), int(sym.eq.numel())

    L = sym.cost
    if n_ineq:
        L = L + ca.dot(lam, sym.ineq)
    if n_eq:
        L = L + ca.dot(nu, sym.eq)
    grad_x_L = ca.jacobian(L, x).T

    args = [x, lam, nu, th]
    names = ["x", "lam", "nu", "theta"]
    return NLPFunctions(
        cost=ca.Function("cost", [x, th], [sym.cost], ["x", "theta"], ["f"]),
        ineq=ca.Function("ineq", [x, th], [sym.ineq], ["x", "theta"], ["g"]),
        eq=ca.Function("eq", [x, th], [sym.eq], ["x", "theta"], ["h"]),
        cst=ca.Function("cst", [x, th], [sym.cst], ["x", "theta"], ["c"]),
        lagr=ca.Function("lagr", args, [L], names, ["L"]),
        kkt=ca.Function(
            "kkt",
            args,
            [
                grad_x_L,
                ca.fmax(sym.ineq, 0),
                sym.eq,
                sym.ineq * lam if n_ineq else ca.SX.zeros(0, 1),
                ca.fmin(lam, 0),
            ],
            names,
            ["stationarity", "inequality", "equality", "slackness", "dual_feasibility"],
        ),
        ift=ca.Function(
            "ift",
            args,
            [
                ca.hessian(L, x)[0],
                ca.jacobian(grad_x_L, th),
                ca.jacobian(sym.ineq, x),
                ca.jacobian(sym.ineq, th),
                ca.jacobian(sym.eq, x),
                ca.jacobian(sym.eq, th),
            ],
            names,
            ["H_xx", "H_xtheta", "J_gx", "J_gtheta", "J_hx", "J_htheta"],
        ),
        lbg=ca.vertcat(-INF * ca.DM.ones(n_ineq, 1), ca.DM.zeros(n_eq, 1)),
        ubg=ca.DM.zeros(n_ineq + n_eq, 1),
    )


# ---------------------------------------------------------------------------
# the NLP object
# ---------------------------------------------------------------------------
# TODO: I can add these to NLP and get rid of this
@dataclass(frozen=True)
class ReferenceNLP:
    """Minimal stand-in for ``DifferentiableNLP``: symbolics, compiled
    functions, dimensions.  Replaced wholesale via :func:`set_factory`."""

    sym: NLPSymbolic
    funcs: NLPFunctions
    n_x: int
    n_ineq: int
    n_eq: int
    n_theta: int

    @property
    def n_c(self) -> int:
        return self.n_ineq + self.n_eq

    @property
    def n_phi(self) -> int:
        return self.n_x + self.n_ineq + self.n_eq


def _default_factory(primal, param, cost, ineq, eq):
    sym = build_symbolic(primal, param, cost, ineq, eq)
    return ReferenceNLP(
        sym=sym,
        funcs=build_functions(sym),
        n_x=int(sym.primal.numel()),
        n_ineq=int(sym.ineq.numel()),
        n_eq=int(sym.eq.numel()),
        n_theta=int(sym.param.numel()),
    )


NLP_FACTORY: Callable[..., object] = _default_factory


def set_factory(factory: Callable[..., object]) -> None:
    """Install the project's NLP constructor.

    Called as ``factory(primal, param, cost, ineq, eq)`` where ``ineq``/``eq``
    may be ``None``.  The returned object must expose ``funcs.kkt``,
    ``funcs.ift`` and ``n_x`` / ``n_ineq`` / ``n_eq`` (``n_theta`` is inferred
    from ``theta_bar`` if absent).
    """
    global NLP_FACTORY
    NLP_FACTORY = factory


def reset_factory() -> None:
    set_factory(_default_factory)


def make_nlp(build: Builder):
    """Run an example's builder through the installed factory."""
    primal, param, cost, ineq, eq = build()
    return NLP_FACTORY(primal, param, cost, ineq, eq)


# ---------------------------------------------------------------------------
# evaluation helpers (positional calls: works for any ca.Function)
# ---------------------------------------------------------------------------
# TODO: this can be merged to NLP
def _dims(nlp, n_theta: Optional[int] = None) -> Tuple[int, int, int, int]:
    n_th = getattr(nlp, "n_theta", None) if n_theta is None else n_theta
    if n_th is None:
        n_th = int(nlp.sym.param.numel())
    return int(nlp.n_x), int(nlp.n_ineq), int(nlp.n_eq), int(n_th)

# TODO: this function and the next are also redundant
def eval_kkt(nlp, x, lam, nu, theta) -> Dict[str, ca.DM]:
    """``funcs.kkt`` as a dict, in the documented output order."""
    n_x, n_ineq, n_eq, n_theta = _dims(nlp)
    out = nlp.funcs.kkt(
        to_dm(x, n_x), to_dm(lam, n_ineq), to_dm(nu, n_eq), to_dm(theta, n_theta)
    )
    keys = ("stationarity", "inequality", "equality", "slackness", "dual_feasibility")
    if isinstance(out, dict):
        return {k: ca.DM(out[k]) for k in keys}
    return {k: ca.DM(v) for k, v in zip(keys, out)}


def residual_norm(residuals: Dict[str, ca.DM]) -> float:
    parts = [to_np(v).reshape(-1) for v in residuals.values()]
    parts = [p for p in parts if p.size] or [np.zeros(1)]
    return float(np.linalg.norm(np.concatenate(parts)))

#TODO: I think this is redundant
def eval_ift(nlp, solution: NLPSolution, n_theta: Optional[int] = None) -> Dict[str, np.ndarray]:
    """``funcs.ift`` at a solution, returned with exact shapes."""
    n_x, n_ineq, n_eq, n_th = _dims(nlp, n_theta)
    out = nlp.funcs.ift(
        to_dm(solution.x, n_x),
        to_dm(solution.lam, n_ineq),
        to_dm(solution.nu, n_eq),
        to_dm(solution.theta, n_th),
    )
    keys = ("H_xx", "H_xtheta", "J_gx", "J_gtheta", "J_hx", "J_htheta")
    vals = [out[k] for k in keys] if isinstance(out, dict) else list(out)
    shapes = [
        (n_x, n_x), (n_x, n_th), (n_ineq, n_x), (n_ineq, n_th), (n_eq, n_x), (n_eq, n_th)
    ]
    return {k: mat(v, *s) for k, v, s in zip(keys, vals, shapes)}

# TODO: this can also be returned elsewhere
def active_set(nlp, solution: NLPSolution, tol: float = 1e-8) -> np.ndarray:
    n_x, n_ineq, n_eq, n_th = _dims(nlp)
    if n_ineq == 0:
        return np.zeros(0, dtype=bool)
    g = to_np(nlp.funcs.ineq(to_dm(solution.x, n_x), to_dm(solution.theta, n_th)))
    return np.abs(g.reshape(n_ineq)) <= tol


# ---------------------------------------------------------------------------
# KKT matrix, kernel, degeneracy flags
# ---------------------------------------------------------------------------
# TODO: also redundant since it can be generated using the standard NLP
# differentiator on the regularized problem.
# TODO: also, everything should be differentiator-agnostic.
def kkt_matrix(
    nlp,
    solution: NLPSolution,
    varrho: float = 0.0,
    rho: float = 0.0,
    active: Optional[np.ndarray] = None,
    n_theta: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The symmetric primal-dual matrix and the rhs ``-D_theta F``.

        [ H_xx + varrho I   J_A^T   J_h^T ]        [ H_xtheta ]
        [ J_A              -rho I     0   ]   -    [ J_Atheta ]
        [ J_h                0     -rho I ]        [ J_htheta ]

    Rows/columns cover ``(x, lam_A, nu)`` for the active set ``A``.
    """
    n_x, n_ineq, n_eq, n_th = _dims(nlp, n_theta)
    d = eval_ift(nlp, solution, n_theta=n_th)
    act = active_set(nlp, solution) if active is None else np.asarray(active, dtype=bool)
    Ja, Jat = d["J_gx"][act], d["J_gtheta"][act]
    na = Ja.shape[0]
    size = n_x + na + n_eq

    K = np.zeros((size, size))
    K[:n_x, :n_x] = d["H_xx"] + varrho * np.eye(n_x)
    if na:
        K[:n_x, n_x : n_x + na] = Ja.T
        K[n_x : n_x + na, :n_x] = Ja
        K[n_x : n_x + na, n_x : n_x + na] = -rho * np.eye(na)
    if n_eq:
        K[:n_x, n_x + na :] = d["J_hx"].T
        K[n_x + na :, :n_x] = d["J_hx"]
        K[n_x + na :, n_x + na :] = -rho * np.eye(n_eq)

    rhs = np.zeros((size, n_th))
    rhs[:n_x] = -d["H_xtheta"]
    if na:
        rhs[n_x : n_x + na] = -Jat
    if n_eq:
        rhs[n_x + na :] = -d["J_htheta"]
    return K, rhs

#TODO: this belongs elsewhere
def kernel_basis(
    nlp, solution: NLPSolution, tol: float = 1e-8, active: Optional[np.ndarray] = None
) -> np.ndarray:
    """Orthonormal basis of ``ker K`` on the active branch, in ``(x, lam_A, nu)``."""
    K, _ = kkt_matrix(nlp, solution, active=active)
    if K.shape[0] == 0:
        return np.zeros((0, 0))
    _, s, Vt = np.linalg.svd(K)
    cols = [Vt[i] for i in range(s.size) if s[i] <= tol * max(1.0, s[0])]
    return np.stack(cols, axis=1) if cols else np.zeros((K.shape[0], 0))

# TODO: redundant, this already exists inside the NLP
# TODO: I'll make this an external helper that I can pull into NLP or wherever.
def solution_properties(nlp, solution: NLPSolution, tol: float = 1e-8) -> Dict[str, object]:
    """LICQ, strict complementarity, SOSC, and nonsingularity of the primal-dual
    matrix -- i.e. exactly the hypotheses the IFT needs.  Suitable for
    ``NLPSolution.properties``."""
    n_x, n_ineq, n_eq, n_th = _dims(nlp)
    d = eval_ift(nlp, solution)
    act = active_set(nlp, solution, tol=tol)
    lam = to_np(solution.lam).reshape(n_ineq) if n_ineq else np.zeros(0)
    Ja = np.vstack([d["J_gx"][act], d["J_hx"]]) if (act.any() or n_eq) else np.zeros((0, n_x))

    licq = True if Ja.shape[0] == 0 else (
        int(np.linalg.matrix_rank(Ja, tol=1e-9)) == Ja.shape[0]
    )
    strict = bool(np.all(lam[act] > tol)) if act.any() else True

    # the critical cone is the null space of the active Jacobian; its dimension
    # is governed by the *rank*, not the row count -- which differ precisely in
    # the LICQ-degenerate cases this library is about
    sosc, sosc_semi = True, True
    rank_Ja = int(np.linalg.matrix_rank(Ja, tol=1e-9)) if Ja.shape[0] else 0
    if rank_Ja < n_x:
        if Ja.shape[0]:
            _, s, Vt = np.linalg.svd(Ja)
            Z = Vt[rank_Ja:].T
        else:
            Z = np.eye(n_x)
        if Z.shape[1]:
            ev = np.linalg.eigvalsh(Z.T @ d["H_xx"] @ Z)
            sosc = bool(np.min(ev) > tol)
            sosc_semi = bool(np.min(ev) > -tol)
    K, _ = kkt_matrix(nlp, solution)
    nonsingular = bool(K.shape[0] and np.linalg.matrix_rank(K, tol=1e-9) == K.shape[0])
    return {
        "licq": licq,
        "strict_complementarity": strict,
        "sosc": sosc,
        "sosc_semidefinite": sosc_semi,
        "kkt_nonsingular": nonsingular,
        "n_active": int(act.sum()),
    }
