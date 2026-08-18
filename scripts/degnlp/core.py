"""
Core datatypes for the degenerate-NLP test library.

Design contract
---------------
The *examples* know everything about the mathematics of P1: the functions
``f, g, h``, a reference (possibly degenerate) primal-dual point, and -- where
it exists in closed form -- the exact set of nearby KKT points as a function of
theta.  Every solution is an ``NLPSolution``; the second-order character of a
branch lives in ``properties['kind']``.

The *harness* (``metrics.py``) knows nothing about how a derivative was
obtained.  Every check consumes a plain numeric Jacobian

    J  ~  d phi / d theta      of shape (n_x + n_ineq + n_eq, n_theta)

(or a subset of its rows, see ``coerce_jac``).  Swapping differentiators means
handing the harness a different array.

Sign conventions:  L = f + lam^T g + nu^T h,  g(x, theta) <= 0,  lam >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import casadi as ca
import numpy as np

from .adapter import (
    Builder,
    eval_kkt,
    kernel_basis,
    make_nlp,
    residual_norm,
    solution_properties,
)
from .types import JacLike, NLPSolution, to_dm, to_np

Array = np.ndarray


# ---------------------------------------------------------------------------
# dimensions and blocks of phi = (x, lam, nu)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dims:
    """Sizes of one parametric NLP, named as in ``NLPSolution``."""

    n_x: int
    n_ineq: int = 0
    n_eq: int = 0
    n_theta: int = 1

    @classmethod
    def from_nlp(cls, nlp, n_theta: Optional[int] = None) -> "Dims":
        n_th = getattr(nlp, "n_theta", None) if n_theta is None else n_theta
        if n_th is None:
            n_th = int(nlp.sym.param.numel())
        return cls(int(nlp.n_x), int(nlp.n_ineq), int(nlp.n_eq), int(n_th))

    @property
    def n_c(self) -> int:
        return self.n_ineq + self.n_eq

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

    def stack(self, x, lam=None, nu=None) -> Array:
        return np.concatenate(
            [
                to_np(x).reshape(self.n_x),
                np.zeros(self.n_ineq) if lam is None else to_np(lam).reshape(self.n_ineq),
                np.zeros(self.n_eq) if nu is None else to_np(nu).reshape(self.n_eq),
            ]
        )


# ---------------------------------------------------------------------------
# solutions
# ---------------------------------------------------------------------------
def phi_np(solution: NLPSolution) -> Array:
    """``solution.phi`` (a ``ca.DM``) as a flat numpy array."""
    return to_np(solution.phi).reshape(-1)


def kind_of(solution: NLPSolution) -> str:
    """Second-order character of a branch: ``'min'``, ``'saddle'``, ``'max'``,
    ``'degenerate'``, ``'reference'`` or ``'unknown'``.

    Stored in ``properties`` because ``NLPSolution`` has one slot for
    solution-level metadata and the label is not boolean.
    """
    props = solution.properties or {}
    return str(props.get("kind", "unknown"))


def make_solution(
    nlp,
    theta,
    x,
    lam=None,
    nu=None,
    kind: str = "unknown",
    with_properties: bool = False,
    **meta: Any,
) -> NLPSolution:
    """Build an :class:`NLPSolution` with its KKT residuals filled in.

    ``kind`` and any extra keywords land in ``properties``; passing
    ``with_properties=True`` additionally runs the LICQ / SOSC / strict
    complementarity analysis, which costs an extra ``ift`` evaluation.
    """
    n_x, n_ineq, n_eq = int(nlp.n_x), int(nlp.n_ineq), int(nlp.n_eq)
    xd, lamd, nud = to_dm(x, n_x), to_dm(lam if lam is not None else [], n_ineq), to_dm(
        nu if nu is not None else [], n_eq
    )
    thd = to_dm(theta)
    sol = NLPSolution(
        x=xd,
        theta=thd,
        lam=lamd,
        nu=nud,
        kkt_residuals=eval_kkt(nlp, xd, lamd, nud, thd),
        properties={"kind": kind, **meta},
    )
    if with_properties:
        sol.properties.update(solution_properties(nlp, sol))
    return sol


# ---------------------------------------------------------------------------
# the example container
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """A parametric NLP with a *known* degeneracy at ``theta_bar``."""

    name: str
    tags: Tuple[str, ...]
    #: CasADi builder, ``() -> (primal, param, cost, ineq, eq)``
    build: Builder
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

    _nlp: Optional[object] = None

    # -- the NLP object ----------------------------------------------------
    @property
    def nlp(self):
        """The ``DifferentiableNLP`` (or ``ReferenceNLP``) for this example,
        built once through ``adapter.NLP_FACTORY``."""
        if self._nlp is None:
            self._nlp = make_nlp(self.build)
        return self._nlp

    @property
    def dims(self) -> Dims:
        return Dims.from_nlp(self.nlp, n_theta=self.p)

    @property
    def p(self) -> int:
        return int(np.size(self.theta_bar))

    # -- solutions ---------------------------------------------------------
    def at(self, theta, x, lam=None, nu=None, kind: str = "unknown", **meta) -> NLPSolution:
        return make_solution(self.nlp, theta, x, lam, nu, kind=kind, **meta)

    def solve(self, theta: Array, with_properties: bool = False) -> List[NLPSolution]:
        """Every nearby KKT point at ``theta``, analytically."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        def at(x, lam=None, nu=None, kind: str = "unknown", **meta) -> NLPSolution:
            return self.at(theta, x, lam, nu, kind=kind, **meta)

        out = list(self.branches(theta, at))
        if with_properties:
            for s in out:
                s.properties.update(solution_properties(self.nlp, s))
        return out

    def solve_at(self, theta: Array) -> List[NLPSolution]:
        return self.solve(theta)

    @property
    def phi_bar(self) -> NLPSolution:
        """Reference solution at ``theta_bar``, with its properties analysed."""
        x, lam, nu = self.reference
        return make_solution(
            self.nlp, self.theta_bar, x, lam, nu, kind="reference", with_properties=True
        )

    @property
    def phi_bar_np(self) -> Array:
        x, lam, nu = self.reference
        return self.dims.stack(x, lam, nu)

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

    def kernel_numeric(self, tol: float = 1e-8) -> Array:
        """``ker K`` measured from ``funcs.ift`` at the reference point."""
        return kernel_basis(self.nlp, self.phi_bar, tol=tol)

    def residual(self, x, lam, nu, theta) -> float:
        """Scalar KKT residual norm, evaluated exactly by CasADi."""
        return residual_norm(eval_kkt(self.nlp, x, lam, nu, theta))

    def residual_phi(self, phi: Array, theta) -> float:
        d = self.dims
        return self.residual(d.rows(phi, "x"), d.rows(phi, "lam"), d.rows(phi, "nu"), theta)

    def check_reference_point(self) -> float:
        x, lam, nu = self.reference
        return self.residual(x, lam, nu, self.theta_bar)


# ---------------------------------------------------------------------------
# Jacobian coercion + branch selection
# ---------------------------------------------------------------------------
def coerce_jac(jac: JacLike, dims: Dims) -> Array:
    """Normalise whatever a differentiator returned into an ``(n_phi, n_theta)`` array.

    Accepts
      * a full ``(n_phi, n_theta)`` array or ``ca.DM``,
      * an ``(n_x, n_theta)`` array -> the x-block, duals left unreported,
      * a dict with any of the keys ``x``, ``lam``, ``nu``,
      * any object exposing ``.jac`` / ``.jacobian`` / ``.J`` with the above.

    Unreported rows are NaN and are *skipped* by the metrics, so a
    differentiator that only produces dx/dtheta stays fully testable on the
    primal block.
    """
    for attr in ("jac", "jacobian", "J"):
        if not isinstance(jac, (np.ndarray, ca.DM, dict)) and hasattr(jac, attr):
            jac = getattr(jac, attr)
            break

    out = np.full((dims.n_phi, dims.n_theta), np.nan)
    if isinstance(jac, dict):
        for key, blk in jac.items():
            sl = dims.blocks[key]
            out[sl] = to_np(blk).reshape(sl.stop - sl.start, dims.n_theta)
        return out

    # note: np.array(ca.DM(...)) rather than to_np, which flattens (1, n) matrices
    arr = np.array(ca.DM(jac), dtype=float) if isinstance(jac, ca.DM) else np.asarray(jac, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape == (dims.n_phi, dims.n_theta):
        return arr
    if arr.shape == (dims.n_x, dims.n_theta):
        out[dims.blocks["x"]] = arr
        return out
    raise ValueError(
        f"cannot interpret Jacobian of shape {arr.shape}; expected "
        f"({dims.n_phi}, {dims.n_theta}) or ({dims.n_x}, {dims.n_theta})"
    )


def select_branch(
    branches: Sequence[NLPSolution],
    reference: Array,
    dims: Dims,
    kind: Optional[str] = "min",
    block: str = "all",
) -> Optional[NLPSolution]:
    """Pick the branch closest to ``reference`` in ``block``, preferring ``kind``."""
    if not branches:
        return None
    pool = [b for b in branches if kind is None or kind_of(b) == kind] or list(branches)
    ref = dims.rows(np.asarray(reference, dtype=float), block)
    dists = [np.linalg.norm(dims.rows(phi_np(b), block) - ref) for b in pool]
    return pool[int(np.argmin(dists))]


def distance_to_set(
    branches: Sequence[NLPSolution],
    point: Array,
    dims: Dims,
    kind: Optional[str] = "min",
    block: str = "all",
) -> float:
    """Distance from ``point`` to the solution *set* (the right metric when the
    minimiser is non-unique, e.g. pitchfork or LICQ-degenerate multipliers)."""
    if not branches:
        return float("nan")
    pool = [b for b in branches if kind is None or kind_of(b) == kind] or list(branches)
    pt = dims.rows(np.asarray(point, dtype=float), block)
    return float(min(np.linalg.norm(dims.rows(phi_np(b), block) - pt) for b in pool))


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
    if int(ok.sum()) < 2:
        return float("nan")
    A = np.stack([np.log(xs[ok]), np.ones(int(ok.sum()))], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.log(ys[ok]), rcond=None)
    return float(sol[0])


def geomspace(lo: float, hi: float, num: int) -> Array:
    return np.geomspace(lo, hi, num)
