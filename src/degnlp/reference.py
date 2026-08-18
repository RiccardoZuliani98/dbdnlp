"""
Baseline differentiators.

Signature matches :class:`degnlp.types.Differentiator`::

    (nlp, solution: NLPSolution) -> d phi / d theta      (n_phi, n_theta)

They exist so the harness can be validated and so you have something to plot
against.  They are deliberately *not* a model of P2: the system below is the
plain ``K + diag(varrho I, -rho I)``, with no slack variable ``z`` and no
theta-dependent proximal centre ``g_theta(xbar)``.  Everything is read off
``nlp.funcs.ift``, so they work against a ``DifferentiableNLP`` or the
``ReferenceNLP`` fallback without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .adapter import active_set, kkt_matrix
from .core import Dims, phi_np, select_branch
from .types import ArrayLike, NLPSolution, to_np


@dataclass
class RegularizedKKT:
    """dphi/dtheta from the regularised, active-set-frozen KKT system.

        [ H_xx + varrho I   J_A^T   J_h^T ] [ dx     ]     [ H_xtheta ]
        [ J_A              -rho I     0   ] [ dlam_A ] = - [ J_Atheta ]
        [ J_h                0     -rho I ] [ dnu    ]     [ J_htheta ]

    ``cond`` is recorded after each call so the varrho/rho ratio trap is visible.
    """

    varrho: float = 1e-3
    rho: float = 1e-3
    active_tol: float = 1e-8
    name: str = "reg-kkt"
    cond: Optional[float] = None
    active: Optional[np.ndarray] = None

    def __call__(self, nlp, solution: NLPSolution) -> ArrayLike:
        n_theta = int(to_np(solution.theta).size)
        dims = Dims.from_nlp(nlp, n_theta=n_theta)
        act = active_set(nlp, solution, tol=self.active_tol)
        self.active = act
        K, rhs = kkt_matrix(
            nlp, solution, varrho=self.varrho, rho=self.rho, active=act, n_theta=n_theta
        )
        self.cond = float(np.linalg.cond(K)) if K.shape[0] else float("nan")
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]

        n_x, na, n_eq = dims.n_x, int(act.sum()), dims.n_eq
        J = np.zeros((dims.n_phi, n_theta))
        J[dims.blocks["x"]] = sol[:n_x]
        if na:
            J[np.where(act)[0] + n_x] = sol[n_x : n_x + na]
        if n_eq:
            J[dims.blocks["nu"]] = sol[n_x + na :]
        return J


@dataclass
class PseudoInverseKKT(RegularizedKKT):
    """The unregularised system with a least-squares fallback.

    What you get if you ignore the degeneracy and let the linear solver cope: a
    plausible-looking matrix that is wrong in a specific, diagnosable way.
    """

    varrho: float = 0.0
    rho: float = 0.0
    name: str = "pinv-kkt"

    def __call__(self, nlp, solution: NLPSolution) -> ArrayLike:
        self.varrho, self.rho = 0.0, 0.0
        n_theta = int(to_np(solution.theta).size)
        dims = Dims.from_nlp(nlp, n_theta=n_theta)
        act = active_set(nlp, solution, tol=self.active_tol)
        self.active = act
        K, rhs = kkt_matrix(nlp, solution, active=act, n_theta=n_theta)
        self.cond = float(np.linalg.cond(K)) if K.shape[0] else float("nan")
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        n_x, na, n_eq = dims.n_x, int(act.sum()), dims.n_eq
        J = np.zeros((dims.n_phi, n_theta))
        J[dims.blocks["x"]] = sol[:n_x]
        if na:
            J[np.where(act)[0] + n_x] = sol[n_x : n_x + na]
        if n_eq:
            J[dims.blocks["nu"]] = sol[n_x + na :]
        return J


@dataclass
class FiniteDifferenceOracle:
    """Central (or one-sided) difference of the *true* solution map.

    A positive control: on nondegenerate examples the harness must certify this
    as second order.  Needs the example itself, not just ``(nlp, solution)``, so
    it is constructed per example via :meth:`bind`.
    """

    example: object = None
    t: float = 1e-5
    kind: Optional[str] = "min"
    name: str = "fd-oracle"

    def bind(self, example) -> "FiniteDifferenceOracle":
        return FiniteDifferenceOracle(example=example, t=self.t, kind=self.kind)

    def __call__(self, nlp, solution: NLPSolution) -> ArrayLike:
        ex = self.example
        if ex is None:
            raise ValueError("FiniteDifferenceOracle must be bound to an example")
        dims = ex.dims
        phibar = ex.phi_bar_np
        J = np.full((dims.n_phi, dims.n_theta), np.nan)
        for j in range(dims.n_theta):
            e = np.zeros(dims.n_theta)
            e[j] = self.t
            sides = {}
            for s, key in ((+1.0, "+"), (-1.0, "-")):
                points = ex.solve(ex.theta_bar + s * e)
                if points:
                    p = select_branch(points, phibar, dims, kind=self.kind)
                    sides[key] = (phi_np(p) - phibar) / (s * self.t)
            if "+" in sides and "-" in sides:
                J[:, j] = 0.5 * (sides["+"] + sides["-"])
            elif sides:
                J[:, j] = next(iter(sides.values()))
        return J
