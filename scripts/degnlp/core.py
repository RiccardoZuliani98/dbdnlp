"""
Core datatypes for the degenerate-NLP test library.

Design contract
---------------
The *examples* know everything about the mathematics of P1: the functions
``f, g, h``, a reference (possibly degenerate) primal-dual point, and -- where
it exists in closed form -- the exact set of nearby KKT points as a function of
theta.

The *harness* (``metrics.py``) knows nothing about how a derivative was
obtained.  Every check consumes a plain numeric Jacobian

    J  ~  d phi / d theta      of shape (n + m_g + m_h, p)

(or a subset of its rows, see ``coerce_jac``).  Swapping differentiators means
handing the harness a different array.

Sign conventions:  L = f + lam^T g + nu^T h,  g(x, theta) <= 0,  lam >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dbdnlp._impl.nlp import DifferentiableNLP
from dbdnlp._impl.types import NLPSolution

Array = np.ndarray
ThetaFn = Callable[[Array], Array]


def make_branch(
    layout: Layout,
    x,
    lam=None,
    nu=None,
    kind: str = "unknown",
    **meta,
) -> Branch:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lam = np.zeros(layout.m_g) if lam is None else np.atleast_1d(np.asarray(lam, dtype=float))
    nu = np.zeros(layout.m_h) if nu is None else np.atleast_1d(np.asarray(nu, dtype=float))
    return Branch(x=x, lam=lam, nu=nu, kind=kind, meta=dict(meta))


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
    phi_bar: NLPSolution
    directions: Dict[str, Array]
    solve: Callable[[Array], List[NLPSolution]]
    #: unnormalised generators of ker K in phi-space, as (u, v_g, v_h) tuples
    kernel: Tuple[Tuple[Array, Array, Array], ...] = ()
    #: analytic Lyapunov-Schmidt data / predictions, free-form
    ls: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    @property
    def p(self) -> int:
        return int(np.size(self.theta_bar))

    def direction(self, name: Optional[str] = None) -> Array:
        if name is None:
            name = next(iter(self.directions))
        d = np.atleast_1d(np.asarray(self.directions[name], dtype=float))
        nrm = np.linalg.norm(d)
        return d / nrm if nrm > 0 else d

    def solve_at(self, theta: Array) -> List[NLPSolution]:
        return list(self.solve(np.atleast_1d(np.asarray(theta, dtype=float))))

    def kernel_matrix(self, normalise: bool = True) -> Array:
        total_size = self.nlp.n_x + self.nlp.n_ineq + self.nlp.n_eq
        if not self.kernel:
            return np.zeros((total_size, 0))
        cols = []
        for u, vg, vh in self.kernel:
            col = np.concatenate([np.ravel(u), np.ravel(vg), np.ravel(vh)])
            if normalise:
                col = col / np.linalg.norm(col)
            cols.append(col)
        return np.stack(cols, axis=1)

    def residual(self, x: Array, lam: Array, nu: Array, theta: Array) -> float:
        # Rely entirely on CasADi exact evaluations
        stat, ineq, eq, slack, dual = self.nlp.funcs.kkt(x, lam, nu, theta)
        
        # Combine them into a single scalar residual norm
        parts = [np.array(ca.DM(p).full()).flatten() for p in (stat, ineq, eq, slack, dual)]
        return float(np.linalg.norm(np.concatenate([p for p in parts if p.size > 0])))

    def check_reference_point(self) -> float:
        return self.residual(self.phi_bar.x, self.phi_bar.lam, self.phi_bar.nu, self.theta_bar)


# ---------------------------------------------------------------------------
# Jacobian coercion + branch selection
# ---------------------------------------------------------------------------
JacLike = Union[Array, Dict[str, Array], "object"]


def coerce_jac(jac: JacLike, layout: Layout, p: int) -> Array:
    """Normalise whatever a differentiator returned into a (size, p) array.

    Accepts
      * a full ``(n + m_g + m_h, p)`` array,
      * an ``(n, p)`` array  -> interpreted as the x-block, duals set to NaN,
      * a dict with any of the keys ``x``, ``lam``, ``nu`` (missing -> NaN),
      * any object exposing ``.jac`` / ``.jacobian`` / ``.J`` with the above.

    NaN rows are *skipped* by the metrics, so a differentiator that only
    produces dx/dtheta is still fully testable on the primal blocks.
    """
    for attr in ("jac", "jacobian", "J"):
        if not isinstance(jac, (np.ndarray, dict)) and hasattr(jac, attr):
            jac = getattr(jac, attr)
            break

    out = np.full((layout.size, p), np.nan)
    if isinstance(jac, dict):
        for key, blk in jac.items():
            sl = layout.blocks[key]
            arr = np.atleast_2d(np.asarray(blk, dtype=float))
            if arr.shape != (sl.stop - sl.start, p):
                arr = arr.reshape(sl.stop - sl.start, p)
            out[sl] = arr
        return out

    arr = np.asarray(jac, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape == (layout.size, p):
        return arr
    if arr.shape == (layout.n, p):
        out[layout.blocks["x"]] = arr
        return out
    raise ValueError(
        f"cannot interpret Jacobian of shape {arr.shape}; expected "
        f"({layout.size}, {p}) or ({layout.n}, {p})"
    )


def select_branch(
    branches: Sequence[Branch],
    reference: Array,
    layout: Layout,
    kind: Optional[str] = "min",
    block: str = "all",
) -> Optional[Branch]:
    """Pick the branch closest to ``reference`` in ``block``, preferring ``kind``."""
    if not branches:
        return None
    pool = [b for b in branches if kind is None or b.kind == kind]
    if not pool:
        pool = list(branches)
    ref = layout.rows(np.asarray(reference, dtype=float), block)
    dists = [np.linalg.norm(layout.rows(b.phi, block) - ref) for b in pool]
    return pool[int(np.argmin(dists))]


def distance_to_set(
    branches: Sequence[Branch],
    point: Array,
    layout: Layout,
    kind: Optional[str] = "min",
    block: str = "all",
) -> float:
    """Distance from ``point`` to the solution *set* (right metric when the
    minimiser is non-unique, e.g. pitchfork or LICQ-degenerate multipliers)."""
    if not branches:
        return float("nan")
    pool = [b for b in branches if kind is None or b.kind == kind] or list(branches)
    pt = layout.rows(np.asarray(point, dtype=float), block)
    return float(min(np.linalg.norm(layout.rows(b.phi, block) - pt) for b in pool))


# ---------------------------------------------------------------------------
# small numerics
# ---------------------------------------------------------------------------
def loglog_slope(xs: Sequence[float], ys: Sequence[float], last: Optional[int] = None) -> float:
    """Least-squares slope of log y vs log x over the finite, positive entries.

    ``last`` restricts the fit to the ``last`` smallest ``xs`` (the asymptotic
    end), which is what you want for observed convergence orders.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if last is not None:
        xs, ys = xs[:last], ys[:last]
    ok = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    A = np.stack([np.log(xs[ok]), np.ones(ok.sum())], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.log(ys[ok]), rcond=None)
    return float(sol[0])


def geomspace(lo: float, hi: float, num: int) -> Array:
    return np.geomspace(lo, hi, num)
