"""
Core datatypes for the degenerate-NLP test library.

Design contract
---------------
The *examples* know everything about the mathematics of P1: the CasADi
description handed to ``DifferentiableNLP``, a reference (possibly degenerate)
primal-dual point, and -- where it exists in closed form -- the exact set of
nearby KKT points as a function of theta.  Every solution is an ``NLPSolution``
built by ``DifferentiableNLP.evaluate``; the second-order character of a branch
lives in ``properties['kind']``.

The *harness* (``metrics.py``) knows nothing about how a derivative was
obtained.  A differentiator is any callable

    (nlp, solution: NLPSolution) -> {'x': (n_x, n_theta),
                                     'lam': (n_ineq, n_theta),
                                     'nu': (n_eq, n_theta)}

which is exactly what ``DifferentiableNLP.differentiate`` returns.  All three
keys are required; ``stack_sensitivities`` is the only thing the harness does
with them.  Anything a differentiator cannot compute it must report as NaN --
never as a silent zero, and never by omitting the block.

Sign conventions:  L = f + lam^T g + nu^T h,  g(x, theta) <= 0,  lam >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbdnlp._impl.nlp import DifferentiableNLP

from .types import NLPSolution, Sensitivities, mat, to_np


Array = np.ndarray


# ---------------------------------------------------------------------------
# dimensions and blocks of phi = (x, lam, nu)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dims:
    """Block structure of ``NLPSolution.phi``, read off the NLP."""

    n_x: int
    n_ineq: int = 0
    n_eq: int = 0
    n_theta: int = 1

    @classmethod
    def of(cls, nlp) -> "Dims":
        return cls(int(nlp.n_x), int(nlp.n_ineq), int(nlp.n_eq), int(nlp.n_theta))

    @property
    def n_phi(self) -> int:
        return self.n_x + self.n_ineq + self.n_eq

    @property
    def blocks(self) -> Dict[str, slice]:
        return {
            "x": slice(0, self.n_x),
            "lam": slice(self.n_x, self.n_x + self.n_ineq),
            "nu": slice(self.n_x + self.n_ineq, self.n_phi),
            "dual": slice(self.n_x, self.n_phi),
            "all": slice(0, self.n_phi),
        }

    def rows(self, arr, block: str = "all") -> Array:
        if block not in self.blocks:
            raise KeyError(f"unknown block {block!r}; use one of {sorted(self.blocks)}")
        return np.asarray(arr, dtype=float)[self.blocks[block]]

# ---------------------------------------------------------------------------
# the example container
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """A parametric NLP with a *known* degeneracy at ``theta_bar``."""

    name: str
    tags: Tuple[str, ...]
    nlp: DifferentiableNLP
    theta_bar: Array
    #: reference primal-dual point at ``theta_bar`` as ``(x, lam, nu)``
    reference: Tuple[Sequence[float], Sequence[float], Sequence[float]]
    #: analytic set of nearby KKT points; called as ``branches(theta, at)`` where
    #: ``at(x, lam, nu, kind, **meta) -> NLPSolution``
    branches: Callable[[Array, Callable[..., NLPSolution]], List[NLPSolution]]
    directions: Dict[str, Array]
    #: unnormalised generators of ker K in phi-space, as ``(u, v_ineq, v_eq)``
    kernel: Tuple[Tuple[Array, Array, Array], ...] = ()
    #: analytic Lyapunov-Schmidt data / closed forms, free-form
    ls: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    _phi_bar: Optional[NLPSolution] = None

    @property
    def dims(self) -> Dims:
        return Dims.of(self.nlp)

    @property
    def p(self) -> int:
        return int(np.size(self.theta_bar))

    # -- solutions ---------------------------------------------------------
    def at(self, theta, x, lam=None, nu=None, kind: str = "unknown", **meta) -> NLPSolution:
        """One KKT point as an ``NLPSolution``, residuals evaluated by CasADi."""
        return self.nlp.evaluate(theta, x, lam, nu, properties={"kind": kind, **meta})

    def solve(self, theta: Array, with_properties: bool = False) -> List[NLPSolution]:
        """Every nearby KKT point at ``theta``, analytically -- no solver call."""

        # ensure proper type in theta
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        # fix theta value in "at" method
        def at(x, lam=None, nu=None, kind: str = "unknown", **meta) -> NLPSolution:
            return self.at(theta, x, lam, nu, kind=kind, **meta)

        # compute explicit solution using "branches"
        out = list(self.branches(theta, at))

        # annotate if requested
        if with_properties:
            for s in out:
                self.nlp.annotate_solution(s)
        return out

    @property
    def phi_bar(self) -> NLPSolution:
        """Reference solution at ``theta_bar``, annotated by ``check_properties``."""
        if self._phi_bar is None:
            x, lam, nu = self.reference
            sol = self.at(self.theta_bar, x, lam, nu, kind="reference")
            self.nlp.annotate_solution(sol)
            self._phi_bar = sol
        return self._phi_bar

    @property
    def phi_bar_np(self) -> Array:
        return phi_np(self.phi_bar)

    # -- helpers -----------------------------------------------------------
    def direction(self, name: Optional[str] = None) -> Array:
        if name is None:
            name = next(iter(self.directions))
        d = np.atleast_1d(np.asarray(self.directions[name], dtype=float))
        nrm = float(np.linalg.norm(d))
        return d / nrm if nrm > 0 else d

    def kernel_matrix(self, normalise: bool = True) -> Array:
        """The hand-derived generators of ker K, as columns in phi-space."""
        if not self.kernel:
            return np.zeros((self.dims.n_phi, 0))
        cols = []
        for u, vg, vh in self.kernel:
            col = np.concatenate([np.ravel(u), np.ravel(vg), np.ravel(vh)])
            cols.append(col / np.linalg.norm(col) if normalise else col)
        return np.stack(cols, axis=1)

    def residual(self, x, lam, nu, theta) -> float:
        """Scalar KKT residual norm, evaluated exactly by CasADi."""

        # evaluate solution
        solution = self.nlp.evaluate(theta, x, lam, nu)

        # convert to list of np arrays
        parts = [to_np(v).reshape(-1) for v in solution.kkt_residuals.values()]

        # normalize zero vectors
        parts = [p for p in parts if p.size] or [np.zeros(1)]

        # compute combined norm
        return float(np.linalg.norm(np.concatenate(parts)))

    def residual_phi(self, phi: Array, theta) -> float:
        d = self.dims
        return self.residual(d.rows(phi, "x"), d.rows(phi, "lam"), d.rows(phi, "nu"), theta)

    def check_reference_point(self) -> float:
        x, lam, nu = self.reference
        return self.residual(x, lam, nu, self.theta_bar)


# ---------------------------------------------------------------------------
# branch selection
# ---------------------------------------------------------------------------
def select_branch(
    branches: Sequence[NLPSolution],
    reference: Array,
    dims: Dims,
    kind: Optional[str] = "min",
    block: str = "all",
) -> Optional[NLPSolution]:
    """Pick the branch closest to ``reference`` in ``block``, preferring ``kind``.

    This doubles as the distance-to-the-solution-*set* query, which is the honest
    error metric when the minimiser is non-unique (pitchfork, LICQ-degenerate
    multipliers).  The nearest branch realises that minimum by construction, so
    the distance is just

        np.linalg.norm(dims.rows(phi_np(result), block)
                       - dims.rows(reference, block))

    with no second search needed.
    """
    if not branches:
        return None
    pool = [b for b in branches if kind is None or kind_of(b) == kind] or list(branches)
    ref = dims.rows(np.asarray(reference, dtype=float), block)
    dists = [np.linalg.norm(dims.rows(phi_np(b), block) - ref) for b in pool]
    return pool[int(np.argmin(dists))]


def loglog_slope(xs: Sequence[float], ys: Sequence[float], last: Optional[int] = None) -> float:
    """Fitted exponent ``p`` in ``y ~ C * x**p`` -- i.e. an observed rate.

    Taking logs turns a power law into a straight line, ``log y = log C +
    p log x``, so the least-squares slope through the ``(log x, log y)`` points
    estimates ``p``.  Every rate the harness reports is read off this way.  Feed
    it (step size, prediction error) and ``p = 2`` says the derivative is exact,
    ``p = 1`` that it is biased, ``p = 0.5`` that the true solution map is
    Hoelder and no derivative can help.  Feed it (regularisation weight,
    ``|J d|``) instead and ``p = -1`` is the ``O(1/eps)`` Lipschitz blow-up.

    Two details matter.  ``last`` keeps only the ``last`` *smallest* ``xs``,
    because a rate is an asymptotic statement and pre-asymptotic points at large
    ``x`` would tilt the fit.  Non-finite and non-positive entries are dropped
    rather than fitted, since their logs do not exist -- that is how "no solution
    on this side" (NaN) and "error exactly zero" leave the fit instead of
    poisoning it.  Fewer than two usable points returns NaN, which the callers
    read as "no rate to report".
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if last is not None:
        xs, ys = xs[:last], ys[:last]
    ok = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if int(ok.sum()) < 2:
        return float("nan")
    A = np.stack([np.log(xs[ok]), np.ones(int(ok.sum()))], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.log(ys[ok]), rcond=None)
    return float(sol[0])

# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------
def phi_np(solution: NLPSolution) -> Array:
    """``solution.phi`` (a ``ca.DM``) as a flat numpy array."""
    return to_np(solution.phi).reshape(-1)


def kind_of(solution: NLPSolution) -> str:
    """Second-order character of a branch: ``'min'``, ``'saddle'``, ``'max'``,
    ``'degenerate'``, ``'reference'`` or ``'unknown'``.

    Kept in ``properties`` because ``NLPSolution`` has one slot for
    solution-level metadata and this label is not boolean.
    """
    return str((solution.properties or {}).get("kind", "unknown"))

def stack_sensitivities(sens: Sensitivities, dims: Dims) -> Array:
    """``{'x', 'lam', 'nu'} -> (n_phi, n_theta)``, in ``phi`` order.

    The entire interface between the harness and a differentiator.  A missing
    key or a wrongly shaped block raises here rather than being papered over,
    which is the point: a differentiator that cannot fill a block should say so
    with NaN.
    """
    if not isinstance(sens, dict):
        raise TypeError(
            "a differentiator must return {'x', 'lam', 'nu'} sensitivity blocks, "
            f"as DifferentiableNLP.differentiate does; got {type(sens).__name__}"
        )
    missing = {"x", "lam", "nu"} - set(sens)
    if missing:
        raise KeyError(
            f"differentiator must return 'x', 'lam' and 'nu'; missing {sorted(missing)}"
        )
    return np.vstack(
        [
            mat(sens["x"], dims.n_x, dims.n_theta),
            mat(sens["lam"], dims.n_ineq, dims.n_theta),
            mat(sens["nu"], dims.n_eq, dims.n_theta),
        ]
    )
