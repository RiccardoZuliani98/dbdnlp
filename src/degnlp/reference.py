"""
Baseline differentiators.

All three satisfy :class:`degnlp.types.Differentiator`::

    (nlp, solution: NLPSolution) -> d phi / d theta

The first two are thin adapters over ``DifferentiableNLP.differentiate``: they
add no mathematics, they only turn the ``(theta, x, lam, nu, mode, options)``
signature into the ``(nlp, solution)`` one the harness iterates over, so a suite
run compares *your* modes against each other rather than against a
reimplementation.  The returned ``{'x', 'lam', 'nu'}`` dict is passed through
untouched -- it is already the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .core import phi_np, select_branch
from .types import NLPSolution, Sensitivities


@dataclass
class IFT:
    """``differentiate(mode='full')`` or ``mode='active_set'``.

    The unregularised implicit-function derivative.  On a degenerate example the
    linear system is singular, so this either raises or returns nonsense -- which
    is precisely the baseline the regularisation has to beat.  ``run_suite``
    records a raised exception as a note rather than crashing.
    """

    mode: str = "active_set"
    options: Optional[Dict[str, Any]] = None
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"ift-{self.mode}"

    def __call__(self, nlp, solution: NLPSolution) -> Sensitivities:
        return nlp.differentiate(
            solution.theta, solution.x, solution.lam, solution.nu,
            mode=self.mode, options=self.options,
        )


@dataclass
class Regularized:
    """``differentiate(mode='regularized')`` at fixed ``(rho_1, rho_2)``.

    ``rho_1`` is the primal weight (``varrho`` in the write-up), ``rho_2`` the
    dual one.  Sweeping ``rho_1/rho_2`` along a mixed kernel direction is what
    exposes the ratio trap.
    """

    rho_1: float = 1e-3
    rho_2: float = 1e-3
    name: str = "regularized"

    def __call__(self, nlp, solution: NLPSolution) -> Sensitivities:
        return nlp.differentiate(
            solution.theta, solution.x, solution.lam, solution.nu,
            mode="regularized", options={"rho_1": self.rho_1, "rho_2": self.rho_2},
        )


@dataclass
class FiniteDifferenceOracle:
    """Central (or one-sided) difference of the *true* solution map.

    A positive control: on nondegenerate examples the harness must certify this
    as second order.  Needs the example's analytic branches rather than just
    ``(nlp, solution)``, so it is bound per example via :meth:`bind`;
    ``report.jacobian_for`` does that automatically.
    """

    # TODO: describe this better
    example: object = None
    t: float = 1e-5
    kind: Optional[str] = "min"
    name: str = "fd-oracle"

    def bind(self, example) -> "FiniteDifferenceOracle":
        return FiniteDifferenceOracle(example=example, t=self.t, kind=self.kind)

    def __call__(self, nlp, solution: NLPSolution) -> Sensitivities:
        ex = self.example
        if ex is None:
            raise ValueError("FiniteDifferenceOracle must be bound to an example")
        dims, phibar = ex.dims, ex.phi_bar_np
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
        # NaN columns mean "no solution on either side", which is a real answer
        return {k: J[dims.blocks[k]] for k in ("x", "lam", "nu")}
