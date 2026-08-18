"""
Baseline differentiators.

These exist so the harness can be validated and so you have something to plot
against.  They are deliberately *not* a model of your P2: the primal-dual
regularised KKT system below is the plain
``K + diag(varrho I, -rho I)`` system, with no slack variable and no
theta-dependent proximal centre ``g_theta(xbar)``.  Any differentiator with the
same signature can replace them.

    Differentiator:  Example -> array of shape (n + m_g + m_h, p)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import Array, Example, select_branch


# ---------------------------------------------------------------------------
@dataclass
class RegularizedKKT:
    """dphi/dtheta from the regularised, active-set-frozen KKT system.

    Assembles

        [ H + varrho I   G_A^T   A_h^T ] [ dx     ]     [ d_theta grad_x L ]
        [ G_A           -rho I    0    ] [ dlam_A ] = - [ d_theta g_A      ]
        [ A_h             0     -rho I ] [ dnu    ]     [ d_theta h        ]

    and reports ``cond`` so the ratio trap is visible.
    """

    varrho: float = 1e-3
    rho: float = 1e-3
    active_tol: float = 1e-8
    name: str = "reg-kkt"
    cond: Optional[float] = None
    active: Optional[Array] = None

    def __call__(self, example: Example) -> Array:
        lay, fns = example.layout, example.functions
        th = example.theta_bar
        x = example.phi_bar.x
        lam = example.phi_bar.lam
        nu = example.phi_bar.nu

        H = fns.hess_xx_L(x, lam, nu, th)
        Ah = fns.jac_h(x, th)
        Gall = fns.jac_g(x, th)
        gval = fns.gval(x, th)
        act = np.abs(gval) <= self.active_tol if gval.size else np.zeros(0, dtype=bool)
        self.active = act
        G = Gall[act] if gval.size else np.zeros((0, lay.n))

        na, nh = G.shape[0], Ah.shape[0]
        K = np.zeros((lay.n + na + nh, lay.n + na + nh))
        K[: lay.n, : lay.n] = H + self.varrho * np.eye(lay.n)
        if na:
            K[: lay.n, lay.n : lay.n + na] = G.T
            K[lay.n : lay.n + na, : lay.n] = G
            K[lay.n : lay.n + na, lay.n : lay.n + na] = -self.rho * np.eye(na)
        if nh:
            K[: lay.n, lay.n + na :] = Ah.T
            K[lay.n + na :, : lay.n] = Ah
            K[lay.n + na :, lay.n + na :] = -self.rho * np.eye(nh)

        rhs = np.zeros((K.shape[0], example.p))
        rhs[: lay.n] = -fns.dtheta_grad_x_L(x, lam, nu, th)
        if na:
            rhs[lay.n : lay.n + na] = -fns.dtheta_g(x, th)[act]
        if nh:
            rhs[lay.n + na :] = -fns.dtheta_h(x, th)

        self.cond = float(np.linalg.cond(K))
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]

        J = np.zeros((lay.size, example.p))
        J[lay.blocks["x"]] = sol[: lay.n]
        if na:
            idx = np.where(act)[0] + lay.n
            J[idx] = sol[lay.n : lay.n + na]
        if nh:
            J[lay.blocks["nu"]] = sol[lay.n + na :]
        return J


@dataclass
class PseudoInverseKKT(RegularizedKKT):
    """The unregularised system with a Moore-Penrose fallback.

    What you get if you ignore the degeneracy and let the linear solver cope --
    a useful straw man, since it returns a plausible-looking matrix that is
    wrong in a specific, diagnosable way.
    """

    varrho: float = 0.0
    rho: float = 0.0
    name: str = "pinv-kkt"

    def __call__(self, example: Example) -> Array:
        self.varrho, self.rho = 0.0, 0.0
        lay = example.layout
        # reuse the assembly, but solve with pinv
        base = RegularizedKKT(varrho=0.0, rho=0.0, active_tol=self.active_tol)
        try:
            J = base(example)
        except Exception:
            J = np.zeros((lay.size, example.p))
        self.cond = base.cond
        self.active = base.active
        return J


@dataclass
class FiniteDifferenceOracle:
    """Central (or one-sided) difference of the *true* solution map.

    A positive control: on nondegenerate examples the harness must certify this
    as second order.  Where the solution map is one-sided this falls back to the
    available side, and where it does not exist it returns NaNs.
    """

    t: float = 1e-5
    kind: Optional[str] = "min"
    name: str = "fd-oracle"

    def __call__(self, example: Example) -> Array:
        lay = example.layout
        phibar = example.phi_bar.phi
        J = np.full((lay.size, example.p), np.nan)
        for j in range(example.p):
            e = np.zeros(example.p)
            e[j] = self.t
            sides = {}
            for s, key in ((+1.0, "+"), (-1.0, "-")):
                br = example.solve_at(example.theta_bar + s * e)
                if br:
                    b = select_branch(br, phibar, lay, kind=self.kind)
                    sides[key] = (b.phi - phibar) / (s * self.t)
            if "+" in sides and "-" in sides:
                J[:, j] = 0.5 * (sides["+"] + sides["-"])
            elif sides:
                J[:, j] = next(iter(sides.values()))
        return J
