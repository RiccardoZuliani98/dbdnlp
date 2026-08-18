"""
Numeric checks.

Everything here consumes a *plain Jacobian* -- ``np.ndarray``, ``ca.DM``, a dict
of blocks, or an object exposing ``.jac`` -- and nothing else, so any
differentiator is tested by the same code.  The two exceptions
(``eps_sweep``, ``solution_bias``) take a callable rather than an array, but
still nothing structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    Example,
    kind_of,
    loglog_slope,
    phi_np,
    select_branch,
    stack_sensitivities,
)
from .types import NLPSolution, Sensitivities, to_np

DEFAULT_TS = np.geomspace(1e-1, 1e-6, 13)
TINY = 1e-300


def _dir(example: Example, direction) -> Tuple[str, np.ndarray]:
    # if no direction is passed, pass next direction from example
    if direction is None:
        name = next(iter(example.directions))
        return name, example.direction(name)
    # if string is passed, look for a direction with that name
    if isinstance(direction, str):
        return direction, example.direction(direction)
    # otherwise, we are dealing with a custom direction (given numerically)
    # and we normalize it first.
    d = np.atleast_1d(np.asarray(direction, dtype=float))
    return "<custom>", d / max(np.linalg.norm(d), TINY)


@dataclass
class Table:
    """A tiny table with named columns, printable and easy to post-process."""

    columns: Tuple[str, ...]
    rows: List[Dict[str, object]] = field(default_factory=list)

    def col(self, name: str) -> np.ndarray:
        return np.array([r.get(name, np.nan) for r in self.rows], dtype=float)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        widths = {c: max(len(c), 11) for c in self.columns}
        head = "  ".join(f"{c:>{widths[c]}}" for c in self.columns)
        lines = [head, "-" * len(head)]
        for r in self.rows:
            cells = []
            for c in self.columns:
                v = r.get(c, "")
                if isinstance(v, bool):
                    cells.append(f"{('yes' if v else 'no'):>{widths[c]}}")
                elif isinstance(v, float):
                    cells.append(
                        f"{v:>{widths[c]}.3e}" if np.isfinite(v) else f"{'--':>{widths[c]}}"
                    )
                else:
                    cells.append(f"{str(v):>{widths[c]}}")
            lines.append("  ".join(cells))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. properties of the *true* solution map (no differentiator involved)
# ---------------------------------------------------------------------------
@dataclass
class Hoelder:
    example: str
    direction: str
    block: str
    exponent: Dict[str, float]
    exists: Dict[str, bool]
    expected: Dict[str, Optional[float]]
    table: Table

    @property
    def matches_expected(self) -> Dict[str, bool]:
        out = {}
        for s in ("+", "-"):
            exp = self.expected.get(s, "unset")
            if exp == "unset":
                out[s] = True
            elif exp is None:
                out[s] = not self.exists[s]
            elif not np.isfinite(exp):
                out[s] = self.exists[s] and not np.isfinite(self.exponent[s])
            else:
                assert not isinstance(exp,str)
                out[s] = self.exists[s] and abs(self.exponent[s] - exp) < 0.08
        return out

    def __str__(self) -> str:  # pragma: no cover
        head = f"[hoelder] {self.example} / {self.direction} / block={self.block}"
        det = "  ".join(
            f"{s}: {'none' if not self.exists[s] else f'{self.exponent[s]:.3f}'}"
            f" (exp {self.expected.get(s)}) {'OK' if self.matches_expected[s] else 'MISMATCH'}"
            for s in ("+", "-")
        )
        return f"{head}\n  {det}\n{self.table}"


def hoelder(
    example: Example,
    direction=None,
    ts: Sequence[float] | np.ndarray = DEFAULT_TS,
    block: str = "x",
    kind: Optional[str] = "min",
    fit_last: int = 6,
) -> Hoelder:
    """Observed exponent of ||phi(theta_bar + t d) - phi_bar|| ~ t^p.

    Validates the *examples* independently of any differentiator: a problem
    claimed to be Hoelder-1/2 must measure 0.5.
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    tab = Table(("t", "sign", "exists", "|d phi|", "kind"))
    dists: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    exists = {"+": False, "-": False}

    for t in ts:
        for s in (+1.0, -1.0):
            key = "+" if s > 0 else "-"
            points = example.solve(example.theta_bar + s * t * d)
            if not points:
                tab.rows.append({"t": float(t), "sign": key, "exists": False,
                                 "|d phi|": np.nan, "kind": "-"})
                continue
            exists[key] = True
            p = select_branch(points, phibar, dim, kind=kind)
            assert p is not None
            nrm = float(np.linalg.norm(dim.rows(phi_np(p) - phibar, block)))
            dists[key].append((float(t), nrm))
            tab.rows.append({"t": float(t), "sign": key, "exists": True,
                             "|d phi|": nrm, "kind": kind_of(p)})

    expo = {}
    for key in ("+", "-"):
        pts = dists[key]
        if not pts:
            expo[key] = np.nan
        elif max(v for _, v in pts) <= 1e-14:
            # TODO: why infinity?
            expo[key] = np.inf
        else:
            expo[key] = loglog_slope([a for a, _ in pts], [b for _, b in pts], last=fit_last)

    return Hoelder(
        example=example.name, direction=name, block=block, exponent=expo, exists=exists,
        expected=dict(example.ls.get("expected", {}).get(name, {})), table=tab,
    )


# ---------------------------------------------------------------------------
# 2. does a given Jacobian predict the true solution map?
# ---------------------------------------------------------------------------
@dataclass
class Taylor:
    example: str
    direction: str
    block: str
    table: Table
    slope: Dict[str, float]        # order of the prediction error in t
    true_slope: Dict[str, float]   # order of the true step;  inf = does not move
    exists: Dict[str, bool]
    t_valid: Dict[str, float]      # largest t with relative error <= tol
    verdict: Dict[str, str]
    tol: float

    def __str__(self) -> str:  # pragma: no cover
        head = f"[taylor] {self.example} / {self.direction} / block={self.block}"
        det = "  |  ".join(
            f"{s}: err ~ t^{self.slope[s]:.2f}, true ~ t^{self.true_slope[s]:.2f}, "
            f"t_valid={self.t_valid[s]:.1e} -> {self.verdict[s]}"
            for s in ("+", "-")
        )
        return f"{head}\n  {det}\n{self.table}"


def _classify(slope: float, true_slope: float, exists: bool) -> str:
    if not exists:
        return "no solution to predict"
    if not np.isfinite(true_slope):
        if not np.isfinite(slope):
            # both infinite
            return "constant solution, correctly predicted"
        return "constant solution, derivative reports spurious motion"
    if not np.isfinite(slope):
        # true_slope is finite, slope is infinite
        return "exact (no error)"
    if slope >= 1.8:
        return "second order (derivative exact)"
    if slope >= 0.9:
        return "first order (biased derivative)"
    if true_slope < 0.9:
        return "sublinear: true map is not differentiable"
    return "sublinear (unexplained)"


def taylor(
    example: Example,
    jac: Sensitivities,
    direction=None,
    ts: Sequence[float] | np.ndarray = DEFAULT_TS,
    block: str = "x",
    kind: Optional[str] = "min",
    tol: float = 0.25,
    fit_last: int = 6,
    noise_floor: float = 1e-13,
) -> Taylor:
    """Compare the first-order prediction ``phi_bar + t J d`` against truth.

    Reports, per side: absolute error, distance from the predicted point to the
    solution *set* (the right notion when the minimiser is non-unique), relative
    error (``rel ~ 1`` means the derivative carries no information), the fitted
    order of the error in t, and the largest t at which the prediction is still
    within ``tol`` relatively.
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    Jb = dim.rows(stack_sensitivities(jac, dim), block)

    tab = Table(("t", "sign", "exists", "|d phi|", "|pred|", "err", "err_set", "rel"))
    errs: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    trues: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    rels: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    exists = {"+": False, "-": False}

    for t in ts:
        for s in (+1.0, -1.0):
            key = "+" if s > 0 else "-"
            dtheta = s * t * d
            pred = Jb @ dtheta
            points = example.solve(example.theta_bar + dtheta)
            if not points:
                tab.rows.append({"t": float(t), "sign": key, "exists": False,
                                 "|d phi|": np.nan, "|pred|": float(np.linalg.norm(pred)),
                                 "err": np.nan, "err_set": np.nan, "rel": np.nan})
                continue
            exists[key] = True
            p = select_branch(points, phibar, dim, kind=kind)
            assert p is not None
            true_step = dim.rows(phi_np(p) - phibar, block)
            err = float(np.linalg.norm(true_step - pred))
            pred_phi = phibar.copy()
            pred_phi[dim.blocks[block]] = dim.rows(phibar, block) + pred
            nearest = select_branch(points, pred_phi, dim, kind=kind, block=block)
            assert nearest is not None
            err_set = float(
                np.linalg.norm(dim.rows(phi_np(nearest), block) - dim.rows(pred_phi, block))
            )
            nt = float(np.linalg.norm(true_step))
            rel = err / max(nt, TINY)
            if err > noise_floor * max(1.0, nt):
                errs[key].append((float(t), err))
            trues[key].append((float(t), nt))
            rels[key].append((float(t), rel))
            tab.rows.append({"t": float(t), "sign": key, "exists": True,
                             "|d phi|": nt, "|pred|": float(np.linalg.norm(pred)),
                             "err": err, "err_set": err_set, "rel": rel})

    slope, true_slope, tvalid, verdict = {}, {}, {}, {}
    for key in ("+", "-"):
        e, tr, rl = errs[key], trues[key], rels[key]
        slope[key] = loglog_slope([a for a, _ in e], [b for _, b in e], last=fit_last) if e else np.nan
        if not tr:
            true_slope[key] = np.nan
        elif max(v for _, v in tr) <= 1e-14:
            true_slope[key] = np.inf
        else:
            true_slope[key] = loglog_slope([a for a, _ in tr], [b for _, b in tr], last=fit_last)
        ok = [a for a, r in rl if np.isfinite(r) and r <= tol]
        tvalid[key] = max(ok) if ok else np.nan
        verdict[key] = _classify(slope[key], true_slope[key], exists[key])

    return Taylor(
        example=example.name, direction=name, block=block, table=tab, slope=slope,
        true_slope=true_slope, exists=dict(exists), t_valid=tvalid, verdict=verdict, tol=tol,
    )


# ---------------------------------------------------------------------------
# 3. one-sided asymmetry
# ---------------------------------------------------------------------------
@dataclass
class Asymmetry:
    example: str
    direction: str
    t: float
    forward: Optional[np.ndarray]
    backward: Optional[np.ndarray]
    central: Optional[np.ndarray]
    jac_dir: np.ndarray
    err_forward: float
    err_backward: float
    err_central: float
    note: str


def asymmetry(
    example: Example,
    jac: Sensitivities,
    direction=None,
    t: float = 1e-4,
    block: str = "x",
    kind: Optional[str] = "min",
) -> Asymmetry:
    """One-sided difference quotients vs the reported directional derivative.

    A fold shows up as one side being ``None``; an active-set kink as both being
    finite and different, in which case the reported Jacobian can match at most
    one of them.
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    Jd = dim.rows(stack_sensitivities(jac, dim) @ d, block)

    def side(s: float) -> Optional[np.ndarray]:
        points = example.solve(example.theta_bar + s * t * d)
        if not points:
            return None
        p = select_branch(points, phibar, dim, kind=kind)
        return dim.rows(phi_np(p) - phibar, block) / (s * t)

    fwd, bwd = side(+1.0), side(-1.0)
    cen = 0.5 * (fwd + bwd) if (fwd is not None and bwd is not None) else None

    def e(v):
        return float(np.linalg.norm(v - Jd)) if v is not None else np.nan

    if fwd is None or bwd is None:
        note = "one-sided: solution exists on a single side (fold / infeasibility)"
    elif np.linalg.norm(fwd - bwd) > 1e-3 * max(1.0, float(np.linalg.norm(fwd))):
        note = "kink: forward and backward derivatives differ"
    else:
        note = "two-sided derivative appears to exist"

    return Asymmetry(
        example=example.name, direction=name, t=float(t), forward=fwd, backward=bwd,
        central=cen, jac_dir=Jd, err_forward=e(fwd), err_backward=e(bwd),
        err_central=e(cen), note=note,
    )


# ---------------------------------------------------------------------------
# 4. KKT residual of the predicted point (practical warm-start metric)
# ---------------------------------------------------------------------------
def step_residual(
    example: Example,
    jac: Sensitivities,
    direction=None,
    ts: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
    kind: Optional[str] = "min",
) -> Table:
    """``||r_P1(phi_bar + t J d, theta_bar + t d)||`` versus the frozen point.

    Meaningful even where no nearby solution exists: is the predicted point a
    better place to start a solve than the base point?
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    J = np.nan_to_num(stack_sensitivities(jac, dim), nan=0.0)
    tab = Table(("t", "sign", "res_frozen", "res_pred", "ratio", "res_true"))
    for t in ts:
        for s in (+1.0, -1.0):
            theta = example.theta_bar + s * t * d
            r0 = example.residual_phi(phibar, theta)
            r1 = example.residual_phi(phibar + J @ (s * t * d), theta)
            points = example.solve(theta)
            rt = (
                example.residual_phi(phi_np(select_branch(points, phibar, dim, kind=kind)), theta) # type: ignore
                if points else np.nan
            )
            tab.rows.append({"t": float(t), "sign": "+" if s > 0 else "-",
                             "res_frozen": r0, "res_pred": r1,
                             "ratio": r1 / max(r0, TINY), "res_true": rt})
    return tab


# ---------------------------------------------------------------------------
# 5. downstream gradient: what a bilevel / learning user actually cares about
# ---------------------------------------------------------------------------
@dataclass
class LossGradient:
    example: str
    predicted: np.ndarray
    reference: np.ndarray
    cosine: float
    rel_error: float
    t_used: float
    note: str

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"[loss] {self.example}: cos={self.cosine:.4f}  rel={self.rel_error:.3e} "
            f"(t={self.t_used:.1e})  {self.note}"
        )


def loss_gradient(
    example: Example,
    jac: Sensitivities,
    loss: Callable[[np.ndarray], float],
    t: float = 1e-4,
    kind: Optional[str] = "min",
    fd_step: float = 1e-6,
) -> LossGradient:
    """Compare ``J^T grad_phi loss`` with a finite difference of ``loss(phi*(theta))``.

    Cosine similarity is the number that matters downstream: a derivative with
    50% relative error but cosine ~1 is still a descent direction.
    """
    dim, phibar = example.dims, example.phi_bar_np
    J = np.nan_to_num(stack_sensitivities(jac, dim), nan=0.0)

    gphi = np.zeros_like(phibar)
    for i in range(phibar.size):
        e = np.zeros_like(phibar)
        e[i] = fd_step
        gphi[i] = (loss(phibar + e) - loss(phibar - e)) / (2 * fd_step)
    predicted = J.T @ gphi

    ref = np.full(dim.n_theta, np.nan)
    missing = False
    for j in range(dim.n_theta):
        e = np.zeros(dim.n_theta)
        e[j] = t
        vals = []
        for s in (+1.0, -1.0):
            points = example.solve(example.theta_bar + s * e)
            if not points:
                missing = True
                vals.append(np.nan)
                continue
            vals.append(loss(phi_np(select_branch(points, phibar, dim, kind=kind))))
        if np.isfinite(vals[0]) and np.isfinite(vals[1]):
            ref[j] = (vals[0] - vals[1]) / (2 * t)
        elif np.isfinite(vals[0]):
            ref[j] = (vals[0] - loss(phibar)) / t
        elif np.isfinite(vals[1]):
            ref[j] = (loss(phibar) - vals[1]) / t

    ok = np.isfinite(ref)
    if ok.sum() == 0:
        cos, rel = np.nan, np.nan
    else:
        a, b = predicted[ok], ref[ok]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(a @ b / denom) if denom > 0 else np.nan
        rel = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), TINY))

    note = "one-sided reference (solution missing on a side)" if missing else "central reference"
    return LossGradient(example.name, predicted, ref, cos, rel, float(t), note)


# ---------------------------------------------------------------------------
# 6. regularisation sweep: saturation and the O(eps^2) knee
# ---------------------------------------------------------------------------
@dataclass
class EpsSweep:
    example: str
    direction: str
    block: str
    eps: np.ndarray
    table: Table
    #: order in eps of the *linearisation* error at the largest probed t.  For a
    #: fold probed at t >> eps^2 the prediction t|J d| ~ t/eps swamps the true
    #: O(sqrt t) step, so the expected value is -1, NOT +1.  The O(eps) row of the
    #: theory table is the bias of the regularised *solution*: see solution_bias.
    linear_error_order: float
    saturation_order: float    # ||J d|| ~ eps^saturation_order  (expect -1)
    knee_order: float          # t_valid ~ eps^knee_order        (expect  2)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"[eps sweep] {self.example} / {self.direction} / block={self.block}\n"
            f"  linear err @t_max ~ eps^{self.linear_error_order:.2f}   "
            f"|J d| ~ eps^{self.saturation_order:.2f}   "
            f"t_valid ~ eps^{self.knee_order:.2f}\n{self.table}"
        )


def eps_sweep(
    example: Example,
    jac_fn: Callable[[float], Sensitivities],
    eps: Sequence[float] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
    direction=None,
    ts: Sequence[float] = DEFAULT_TS,
    block: str = "x",
    side: str = "auto",
    kind: Optional[str] = "min",
    tol: float = 0.25,
) -> EpsSweep:
    """Run ``taylor`` for a family of regularisation weights and fit the rates."""
    name, d = _dir(example, direction)
    eps = np.asarray(list(eps), dtype=float)
    tab = Table(("eps", "side", "|J d|", "err@t_max", "rel@t_max", "t_valid", "slope"))
    big_t = float(max(ts))
    Jn, errs, tvals = [], [], []

    for e in eps:
        sens = jac_fn(float(e))
        rep = taylor(example, sens, direction=d, ts=ts, block=block, kind=kind, tol=tol)
        J = stack_sensitivities(sens, example.dims)
        if side == "auto":
            cand = [s for s in ("+", "-") if rep.exists[s]]
            s_use = cand[0] if cand else "+"
        else:
            s_use = side
        match = [r for r in rep.table.rows if r["sign"] == s_use and r["t"] == big_t]
        err = float(match[0]["err"]) if match else np.nan
        rel = float(match[0]["rel"]) if match else np.nan
        jd = float(np.linalg.norm(example.dims.rows(J @ d, block)))
        tab.rows.append({"eps": float(e), "side": s_use, "|J d|": jd, "err@t_max": err,
                         "rel@t_max": rel, "t_valid": rep.t_valid[s_use],
                         "slope": rep.slope[s_use]})
        Jn.append(jd)
        errs.append(err)
        tvals.append(rep.t_valid[s_use])

    return EpsSweep(
        example=example.name, direction=name, block=block, eps=eps, table=tab,
        linear_error_order=loglog_slope(eps, errs),
        saturation_order=loglog_slope(eps, Jn),
        knee_order=loglog_slope(eps, tvals),
    )


# ---------------------------------------------------------------------------
# 7. exactness check against a closed-form regularised Jacobian
# ---------------------------------------------------------------------------
def reg_jac_exactness(
    example: Example, jac: Sensitivities, varrho: float, rho: float
) -> Optional[Dict[str, float]]:
    """For the quadratic examples the regularised KKT Jacobian is closed form.

    Any discrepancy is an implementation bug, not approximation error -- run this
    before interpreting anything else.  Note the closed forms assume the plain
    ``K + diag(varrho I, -rho I)`` system: if your P2's slack block or its
    theta-dependent proximal centre changes the answer, the mismatch should show
    up in specific columns rather than everywhere.
    """
    fn = example.ls.get("reg_jac_exact")
    if fn is None:
        return None
    ref = np.asarray(fn(varrho, rho), dtype=float)
    got = stack_sensitivities(jac, example.dims)
    mask = np.isfinite(got)
    if not mask.any():
        return None
    abs_err = float(np.max(np.abs(got[mask] - ref[mask])))
    return {
        "abs_err": abs_err,
        "rel_err": abs_err / max(float(np.max(np.abs(ref[mask]))), TINY),
        "compared_entries": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# 8. bias of the regularised *solution* (needs a solve, not a Jacobian)
# ---------------------------------------------------------------------------
@dataclass
class SolutionBias:
    example: str
    direction: str
    block: str
    t: float
    eps: np.ndarray
    bias: np.ndarray
    order: float

    def __str__(self) -> str:  # pragma: no cover
        rows = "  ".join(f"{e:.1e}:{b:.2e}" for e, b in zip(self.eps, self.bias))
        return (
            f"[solution bias] {self.example} / {self.direction} at t={self.t:.1e}\n"
            f"  bias ~ eps^{self.order:.2f}   ({rows})"
        )


def solution_bias(
    example: Example,
    sol_fn: Callable[[np.ndarray, float], object],
    eps: Sequence[float] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
    direction=None,
    t: float = 1e-2,
    sign: float = +1.0,
    block: str = "x",
    kind: Optional[str] = "min",
) -> SolutionBias:
    """``||phi_reg(theta; eps) - phi_true(theta)||`` at fixed theta, as eps -> 0.

    The O(eps) entry of the theory table.  Deliberately separate from the
    Jacobian metrics: measuring it requires the regularised *solution*, so pass a
    callable ``(theta, eps) -> NLPSolution | array``.  Examples with a closed form
    expose one under ``ls['reg_solution']`` / ``ls['relaxed_solution']``.
    """
    name, d = _dir(example, direction)
    dim = example.dims
    theta = example.theta_bar + sign * t * d
    points = example.solve(theta)
    ref = (
        dim.rows(phi_np(select_branch(points, example.phi_bar_np, dim, kind=kind)), block)
        if points else None
    )
    eps = np.asarray(list(eps), dtype=float)
    bias = []
    for e in eps:
        got = sol_fn(theta, float(e))
        if got is None or ref is None:
            bias.append(np.nan)
            continue
        if isinstance(got, NLPSolution):
            vec = dim.rows(phi_np(got), block)
        else:
            arr = np.atleast_1d(to_np(got)).reshape(-1)
            vec = dim.rows(arr, block) if arr.size == dim.n_phi else arr
        bias.append(float(np.linalg.norm(vec - ref)))
    return SolutionBias(
        example=example.name, direction=name, block=block, t=float(t), eps=eps,
        bias=np.asarray(bias), order=loglog_slope(eps, bias),
    )
