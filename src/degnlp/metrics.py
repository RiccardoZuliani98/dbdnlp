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
from typing import (
    Callable, Dict, List, Optional, 
    Sequence, Tuple, cast
)

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

# Default sequence of step sizes (t) for finite difference checks, spaced logarithmically
DEFAULT_TS = np.geomspace(1e-1, 1e-6, 13)
# A tiny number to avoid division by zero
TINY = 1e-300


def _dir(example: Example, direction: Optional[str | np.ndarray]) -> Tuple[str, np.ndarray]:
    """
    Extracts and normalizes the perturbation direction.

    Args:
        example: The parametric example containing default directions.
        direction: Either a string (name of a pre-defined direction), None (picks 
            the first available direction), or a custom numeric array.

    Returns:
        A tuple containing the name of the direction and the normalized direction array.
    """
    # If no direction is passed, default to the first available direction in the example
    if direction is None:
        name = next(iter(example.directions))
        return name, example.direction(name)
    
    # If a string is passed, look up the predefined direction by name
    if isinstance(direction, str):
        return direction, example.direction(direction)
    
    # Otherwise, handle a custom numeric direction and normalize it
    d = np.atleast_1d(np.asarray(direction, dtype=float))
    return "<custom>", d / max(np.linalg.norm(d), TINY)


@dataclass
class Table:
    """A tiny table utility with named columns, used for logging and post-processing."""

    columns: Tuple[str, ...]
    rows: List[Dict[str, object]] = field(default_factory=list)

    def col(self, name: str) -> np.ndarray:
        """Extracts a specific column as a numpy float array, filling missing values with NaN."""
        return np.array([r.get(name, np.nan) for r in self.rows], dtype=float)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        """Formats the table into a readable string with aligned columns."""
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
                    # Format finite floats in scientific notation, handle NaNs
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
    """Data object storing the results of a Hölder continuity check."""
    example: str
    direction: str
    block: str
    exponent: Dict[str, float]
    exists: Dict[str, bool]
    expected: Dict[str, Optional[float]]
    table: Table

    @property
    def matches_expected(self) -> Dict[str, bool]:
        """Validates if the computed Hölder exponents match the theoretical expectations."""
        out = {}
        for s in ("+", "-"):
            exp = self.expected.get(s, "unset")
            if exp == "unset":
                out[s] = True
            elif exp is None:
                # Expects no solution to exist
                out[s] = not self.exists[s]
            elif not np.isfinite(exp):
                # Expects an infinite slope (step doesn't affect the solution)
                out[s] = self.exists[s] and not np.isfinite(self.exponent[s])
            else:
                # Expects a specific exponent (e.g., 1.0 for Lipschitz, 0.5 for square-root behavior)
                assert not isinstance(exp, str)
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
    direction: Optional[str | np.ndarray] = None,
    ts: Sequence[float] | np.ndarray = DEFAULT_TS,
    block: str = "x",
    kind: Optional[str] = "min",
    fit_last: int = 6,
) -> Hoelder:
    """
    Computes the observed Hölder exponent of the solution mapping.
    
    This function tracks the distance $\\|\\phi(\\bar{\\theta} + t d) - \\bar{\\phi}\\|$ 
    as $t \\to 0$ and fits a line in log-log space to find the exponent $p$ such 
    that the distance is proportional to $t^p$.

    Args:
        example: The parametric optimization example.
        direction: The direction of perturbation (string name or numerical array).
        ts: A sequence of step sizes (t) to evaluate.
        block: The specific variable block in the solution to monitor (default "x").
        kind: Branch selection criteria if multiple solutions exist.
        fit_last: Number of points to use for the log-log linear regression slope.

    Returns:
        A Hoelder object containing exponents for both positive and negative perturbations.
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    
    # Initialize the tracking table and data structures
    tab = Table(("t", "sign", "exists", "|d phi|", "kind"))
    dists: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    exists = {"+": False, "-": False}

    for t in ts:
        for s in (+1.0, -1.0):
            key = "+" if s > 0 else "-"
            
            # Solve the problem at the perturbed parameter value
            points = example.solve(example.theta_bar + s * t * d)
            if not points:
                tab.rows.append({"t": float(t), "sign": key, "exists": False,
                                 "|d phi|": np.nan, "kind": "-"})
                continue
            
            exists[key] = True
            p = select_branch(points, phibar, dim, kind=kind)
            assert p is not None
            
            # Compute the actual displacement in the specified variable block
            nrm = float(np.linalg.norm(dim.rows(phi_np(p) - phibar, block)))
            dists[key].append((float(t), nrm))
            tab.rows.append({"t": float(t), "sign": key, "exists": True,
                             "|d phi|": nrm, "kind": kind_of(p)})

    # Calculate log-log slope (exponent) for both perturbation directions
    expo = {}
    for key in ("+", "-"):
        pts = dists[key]
        if not pts:
            expo[key] = np.nan
        elif max(v for _, v in pts) <= 1e-14:
            # If the displacement is effectively zero, the map is constant (infinite exponent)
            expo[key] = np.inf
        else:
            expo[key] = loglog_slope([a for a, _ in pts], [b for _, b in pts], last=fit_last)

    expected = dict(cast(dict, example.ls.get("expected", {})))

    return Hoelder(
        example=example.name, direction=name, block=block, exponent=expo, exists=exists,
        expected=expected.get(name, {}), table=tab
    )


# ---------------------------------------------------------------------------
# 2. does a given Jacobian predict the true solution map?
# ---------------------------------------------------------------------------
@dataclass
class Taylor:
    """Data object storing the results of a Taylor approximation check."""
    example: str
    direction: str
    block: str
    table: Table
    slope: Dict[str, float]        # Order of the prediction error in t (ideally >= 2)
    true_slope: Dict[str, float]   # Order of the true step in t (ideally 1 for regular maps)
    exists: Dict[str, bool]
    t_valid: Dict[str, float]      # Largest step size t where rel error <= tol
    verdict: Dict[str, str]        # Human-readable interpretation of the results
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
    """
    Classifies the quality of the Jacobian based on the convergence rates 
    of the error and the true solution trajectory.
    """
    if not exists:
        return "no solution to predict"
    if not np.isfinite(true_slope):
        if not np.isfinite(slope):
            return "constant solution, correctly predicted"
        return "constant solution, derivative reports spurious motion"
    if not np.isfinite(slope):
        return "exact (no error)"
    if slope >= 1.8:
        # Error drops quadratically: indicates a perfectly accurate first derivative
        return "second order (derivative exact)"
    if slope >= 0.9:
        # Error drops linearly: derivative is fundamentally wrong (or missing terms)
        return "first order (biased derivative)"
    if true_slope < 0.9:
        # True solution movement is sublinear (e.g., Hölder-1/2); map isn't differentiable
        return "sublinear: true map is not differentiable"
    return "sublinear (unexplained)"


def taylor(
    example: Example,
    jac: Sensitivities,
    direction: Optional[str | np.ndarray] = None,
    ts: Sequence[float] | np.ndarray = DEFAULT_TS,
    block: str = "x",
    kind: Optional[str] = "min",
    tol: float = 0.25,
    fit_last: int = 6,
    noise_floor: float = 1e-13,
) -> Taylor:
    """
    Compares the first-order Taylor prediction $\\bar{\\phi} + t J d$ against the true solution.

    This acts as a robust numerical derivative test. It calculates:
      - The true displacement of the solution.
      - The expected displacement based on the provided Jacobian matrix.
      - The absolute and relative errors between prediction and truth.
      - The asymptotic order (slope in log-log space) of the error as step size shrinks.

    Args:
        example: The parametric optimization example.
        jac: The Jacobian matrix to be evaluated.
        direction: Direction of perturbation (string name or numerical array).
        ts: Sequence of perturbation step sizes.
        block: The specific variable block in the solution to check (default "x").
        kind: Branch selection criteria if multiple solutions exist.
        tol: Relative error threshold to determine `t_valid` limit.
        fit_last: Number of points to use for the log-log linear regression slope.
        noise_floor: Lower bound for absolute error threshold to ignore numerical noise.

    Returns:
        A Taylor object with detailed error evaluations, log-log slopes, and classifications.
    """
    # Normalize direction and fetch problem specifics
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np

    # Stack the given Jacobian and slice out the target block
    Jb = dim.rows(stack_sensitivities(jac, dim), block)

    # Initialize data tracking
    tab = Table(("t", "sign", "exists", "|d phi|", "|pred|", "err", "err_set", "rel"))
    errs: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    trues: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    rels: Dict[str, List[Tuple[float, float]]] = {"+": [], "-": []}
    exists = {"+": False, "-": False}

    # Loop through perturbation magnitudes
    for t in ts:
        # Loop through positive and negative perturbation directions
        for s in (+1.0, -1.0):
            key = "+" if s > 0 else "-"
            
            # Compute parameter perturbation (dtheta) and predicted step
            dtheta = s * t * d
            pred = Jb @ dtheta

            # Solve at the new parameter value to get the "true" solution
            points = example.solve(example.theta_bar + dtheta)

            if not points:
                # Log a failure to solve
                tab.rows.append({"t": float(t), "sign": key, "exists": False,
                                 "|d phi|": np.nan, "|pred|": float(np.linalg.norm(pred)),
                                 "err": np.nan, "err_set": np.nan, "rel": np.nan})
                continue
                
            exists[key] = True

            # Pick closest branch to original unperturbed solution
            p = select_branch(points, phibar, dim, kind=kind)
            assert p is not None

            # Calculate actual step and prediction residual
            true_step = dim.rows(phi_np(p) - phibar, block)
            err = float(np.linalg.norm(true_step - pred))
            
            # Determine error against the solution set (important if minimizers aren't unique)
            pred_phi = phibar.copy()
            pred_phi[dim.blocks[block]] = dim.rows(phibar, block) + pred
            nearest = select_branch(points, pred_phi, dim, kind=kind, block=block)
            assert nearest is not None
            err_set = float(
                np.linalg.norm(dim.rows(phi_np(nearest), block) - dim.rows(pred_phi, block))
            )
            
            # Compute relative error 
            nt = float(np.linalg.norm(true_step))
            rel = err / max(nt, TINY)
            
            # Filter noise floor for log-log regression tracking
            if err > noise_floor * max(1.0, nt):
                errs[key].append((float(t), err))
            
            trues[key].append((float(t), nt))
            rels[key].append((float(t), rel))
            
            # Log step metrics to table
            tab.rows.append({"t": float(t), "sign": key, "exists": True,
                             "|d phi|": nt, "|pred|": float(np.linalg.norm(pred)),
                             "err": err, "err_set": err_set, "rel": rel})

    # Post-process errors to identify convergence rates
    slope, true_slope, tvalid, verdict = {}, {}, {}, {}
    for key in ("+", "-"):
        e, tr, rl = errs[key], trues[key], rels[key]
        
        # Determine log-log slopes (exponent properties)
        slope[key] = loglog_slope([a for a, _ in e], [b for _, b in e], last=fit_last) if e else np.nan
        if not tr:
            true_slope[key] = np.nan
        elif max(v for _, v in tr) <= 1e-14:
            true_slope[key] = np.inf
        else:
            true_slope[key] = loglog_slope([a for a, _ in tr], [b for _, b in tr], last=fit_last)
            
        # Determine validation range and final classification
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
    """Data object storing the results of the one-sided asymmetry check."""
    example: str
    direction: str
    t: float
    forward: Optional[np.ndarray]
    backward: Optional[np.ndarray]
    central: Optional[np.ndarray]
    jac_dir: np.ndarray             # The reported directional derivative (J * d)
    err_forward: float
    err_backward: float
    err_central: float
    note: str                       # Human-readable interpretation of the geometry


def asymmetry(
    example: Example,
    jac: Sensitivities,
    direction: Optional[str | np.ndarray] = None,
    t: float = 1e-4,
    block: str = "x",
    kind: Optional[str] = "min",
) -> Asymmetry:
    """
    Compares one-sided difference quotients against the reported directional derivative.

    This test checks how the differentiator handles non-smoothness in the solution map:
      - A "fold" or infeasibility boundary: The solution only exists for t > 0 or t < 0.
      - An active-set "kink": The solution exists on both sides, but the forward 
        and backward derivatives are distinct.
        
    In cases of a kink, a single linear Jacobian can mathematically match at most 
    one of the branches.

    Args:
        example: The parametric optimization example.
        jac: The evaluated sensitivities (Jacobian).
        direction: The perturbation direction.
        t: A single, strictly positive step size to use for the finite differences.
        block: The variable block to check (default "x").
        kind: The requested solution character.

    Returns:
        An Asymmetry object detailing the finite differences and errors.
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    
    # Compute the directional derivative predicted by the provided Jacobian
    Jd = dim.rows(stack_sensitivities(jac, dim) @ d, block)

    def side(s: float) -> Optional[np.ndarray]:
        """Helper to calculate the one-sided finite difference quotient."""
        points = example.solve(example.theta_bar + s * t * d)
        if not points:
            return None
        p = select_branch(points, phibar, dim, kind=kind)
        assert p is not None
        # Return the difference quotient: (phi(theta + s*t*d) - phi(theta)) / (s*t)
        return dim.rows(phi_np(p) - phibar, block) / (s * t)

    # Evaluate forward (+1) and backward (-1) steps
    fwd, bwd = side(+1.0), side(-1.0)
    
    # Compute central difference if both sides exist
    cen = 0.5 * (fwd + bwd) if (fwd is not None and bwd is not None) else None

    def e(v: Optional[np.ndarray]) -> float:
        """Helper to compute absolute error against the predicted Jacobian step."""
        return float(np.linalg.norm(v - Jd)) if v is not None else np.nan

    # Classify the geometric nature of the perturbation
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
    direction: Optional[str | np.ndarray] = None,
    ts: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
    kind: Optional[str] = "min",
) -> Table:
    """
    Evaluates the predicted step as a warm-start candidate for an NLP solver.

    This function measures ``||r_P1(phi_bar + t * J * d, theta_bar + t * d)||``.
    It compares the KKT residual of the *frozen* point (doing nothing) against the 
    KKT residual of the *predicted* point (applying the Taylor step).

    This metric is deeply practical: even if the true solution map is non-differentiable 
    or a nearby solution doesn't exist, does following the derivative provide a better 
    initial guess for the optimizer than just staying put?

    Args:
        example: The parametric optimization example.
        jac: The evaluated sensitivities.
        direction: The perturbation direction.
        ts: Sequence of step sizes to evaluate.
        kind: The requested solution character.

    Returns:
        A formatted Table comparing the frozen and predicted residuals.
    """
    name, d = _dir(example, direction)
    dim, phibar = example.dims, example.phi_bar_np
    
    # Handle NaN sensitivities gracefully by replacing them with zero (no motion)
    J = np.nan_to_num(stack_sensitivities(jac, dim), nan=0.0)
    
    tab = Table(("t", "sign", "res_frozen", "res_pred", "ratio", "res_true"))
    
    for t in ts:
        for s in (+1.0, -1.0):
            # The perturbed parameter
            theta = example.theta_bar + s * t * d
            
            # r0: Residual if we just use the unperturbed solution (frozen)
            r0 = example.residual_phi(phibar, theta)
            
            # r1: Residual if we step along the Jacobian prediction (warm-start)
            r1 = example.residual_phi(phibar + J @ (s * t * d), theta)
            
            # rt: The exact residual of the true solution at the perturbed parameter
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
    """Data object storing downstream bilevel gradient accuracy."""
    example: str
    predicted: np.ndarray
    reference: np.ndarray
    cosine: float                   # Cosine similarity between predicted and true gradient
    rel_error: float                # Relative error of the gradient magnitude
    t_used: float                   # Finite difference step used for the reference
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
    """
    Compares the chain-rule gradient ``J^T grad_phi(loss)`` against the true gradient.

    In bilevel optimization or machine learning (e.g., differentiating through an argmin),
    the user fundamentally wants $\\nabla_\\theta L(\\phi^*(\\theta))$. This function checks
    if the provided Jacobian $J$ yields a useful downstream gradient.

    Crucially, it reports the **cosine similarity**. A derivative might have a 50% 
    relative error in magnitude, but if the cosine similarity is near 1.0, it is 
    still a highly effective descent direction for gradient-based learning.

    Args:
        example: The parametric optimization example.
        jac: The evaluated sensitivities (Jacobian).
        loss: A scalar-valued downstream loss function consuming phi.
        t: The step size for the finite difference approximation of the reference.
        kind: The requested solution character.
        fd_step: Step size for finite-differencing the downstream loss function itself.

    Returns:
        A LossGradient object tracking the alignment and error of the predicted gradient.
    """
    dim, phibar = example.dims, example.phi_bar_np
    
    # Replace any uncomputable derivatives with zero for the chain rule application
    J = np.nan_to_num(stack_sensitivities(jac, dim), nan=0.0)

    # 1. Compute grad_phi(loss) via central finite differences on the given loss function
    gphi = np.zeros_like(phibar)
    for i in range(phibar.size):
        e = np.zeros_like(phibar)
        e[i] = fd_step
        gphi[i] = (loss(phibar + e) - loss(phibar - e)) / (2 * fd_step)
        
    # Apply the chain rule: predicted grad_theta = J^T * grad_phi
    predicted = J.T @ gphi

    # 2. Compute the "true" reference grad_theta by applying finite differences 
    # directly to the composition L(phi*(theta))
    ref = np.full(dim.n_theta, np.nan)
    missing = False
    
    for j in range(dim.n_theta):
        e = np.zeros(dim.n_theta)
        e[j] = t
        vals = []
        
        # Evaluate the true solution map perturbed in the j-th basis direction
        for s in (+1.0, -1.0):
            points = example.solve(example.theta_bar + s * e)
            if not points:
                missing = True
                vals.append(np.nan)
                continue
            sol = select_branch(points, phibar, dim, kind=kind)
            assert sol is not None
            vals.append(loss(phi_np(sol)))
            
        # Calculate finite difference based on which sides successfully solved
        if np.isfinite(vals[0]) and np.isfinite(vals[1]):
            # Central difference
            ref[j] = (vals[0] - vals[1]) / (2 * t)
        elif np.isfinite(vals[0]):
            # Forward difference
            ref[j] = (vals[0] - loss(phibar)) / t
        elif np.isfinite(vals[1]):
            # Backward difference
            ref[j] = (loss(phibar) - vals[1]) / t

    # 3. Calculate similarity and error metrics
    ok = np.isfinite(ref)
    if ok.sum() == 0:
        # If the reference gradient is completely uncomputable
        cos, rel = np.nan, np.nan
    else:
        # Compare only the finite/computable dimensions
        a, b = predicted[ok], ref[ok]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        
        # Cosine similarity: (a dot b) / (|a| * |b|)
        cos = float(a @ b / denom) if denom > 0 else np.nan
        # Relative error
        rel = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), TINY))

    note = "one-sided reference (solution missing on a side)" if missing else "central reference"
    
    return LossGradient(example.name, predicted, ref, cos, rel, float(t), note)


# ---------------------------------------------------------------------------
# 6. regularisation sweep: saturation and the O(eps^2) knee
# ---------------------------------------------------------------------------
@dataclass
class EpsSweep:
    """Data object storing the scaling behavior of a regularized Jacobian w.r.t. epsilon."""
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
    
    #: The order at which the norm of the directional derivative scales.
    #: ||J d|| ~ eps^saturation_order (we typically expect -1 for a fold).
    saturation_order: float    
    
    #: The order at which the valid linearization radius scales.
    #: t_valid ~ eps^knee_order (we typically expect 2 for a fold).
    knee_order: float          

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
    direction: Optional[str | np.ndarray] = None,
    ts: Sequence[float] | np.ndarray = DEFAULT_TS,
    block: str = "x",
    side: str = "auto",
    kind: Optional[str] = "min",
    tol: float = 0.25,
) -> EpsSweep:
    """
    Runs the Taylor test for a family of regularization weights to fit scaling rates.

    When optimizing through a singularity (like a fold or active-set change),
    regularization (e.g., smoothing or interior point barrier eps) is often used. 
    This sweep evaluates how the local linearization degrades as epsilon shrinks.
    
    It extracts three key properties:
      1. linear_error_order: Does the prediction error explode?
      2. saturation_order: Does the step magnitude explode?
      3. knee_order: How quickly does the radius of validity shrink to zero?

    Args:
        example: The parametric optimization example.
        jac_fn: A closure that takes an epsilon value and returns the evaluated Jacobian.
        eps: Sequence of epsilon regularization values to test.
        direction: The perturbation direction.
        ts: Sequence of finite difference step sizes.
        block: The variable block to check.
        side: Target side of the perturbation ("+" or "-") or "auto" to pick one that exists.
        kind: The requested solution character.
        tol: The relative error tolerance for determining t_valid in the taylor test.

    Returns:
        An EpsSweep object tracking the structural breakdown of the derivative as eps -> 0.
    """
    name, d = _dir(example, direction)
    eps_np = np.asarray(list(eps), dtype=float)
    tab = Table(("eps", "side", "|J d|", "err@t_max", "rel@t_max", "t_valid", "slope"))
    
    # We evaluate the worst-case error at the largest tested perturbation step
    big_t = float(max(ts))
    
    Jn, errs, tvals = [], [], []

    for e in eps_np:
        # Generate the Jacobian for the specific regularization parameter
        sens = jac_fn(float(e))
        
        # Run standard taylor diagnostics on this specific smoothed system
        rep = taylor(example, sens, direction=d, ts=ts, block=block, kind=kind, tol=tol)
        J = stack_sensitivities(sens, example.dims)
        
        # Resolve which side (+ or -) of the finite difference to use
        if side == "auto":
            cand = [s for s in ("+", "-") if rep.exists[s]]
            s_use = cand[0] if cand else "+"
        else:
            s_use = side
            
        # Extract the taylor test results for the largest step size (big_t)
        match = [r for r in rep.table.rows if r["sign"] == s_use and r["t"] == big_t]
        err = float(match[0]["err"]) if match else np.nan # type: ignore
        rel = float(match[0]["rel"]) if match else np.nan # type: ignore
        
        # Magnitude of the directional derivative 
        jd = float(np.linalg.norm(example.dims.rows(J @ d, block)))
        
        tab.rows.append({"eps": float(e), "side": s_use, "|J d|": jd, "err@t_max": err,
                         "rel@t_max": rel, "t_valid": rep.t_valid[s_use],
                         "slope": rep.slope[s_use]})
        
        # Store for log-log slope extraction
        Jn.append(jd)
        errs.append(err)
        tvals.append(rep.t_valid[s_use])

    # Fit a power law (slope in log-log space) for how these metrics scale with eps
    return EpsSweep(
        example=example.name, direction=name, block=block, eps=eps_np, table=tab,
        linear_error_order=loglog_slope(eps_np.tolist(), errs),
        saturation_order=loglog_slope(eps_np.tolist(), Jn),
        knee_order=loglog_slope(eps_np.tolist(), tvals),
    )


# ---------------------------------------------------------------------------
# 7. exactness check against a closed-form regularised Jacobian
# ---------------------------------------------------------------------------
def reg_jac_exactness(
    example: Example, jac: Sensitivities, varrho: float, rho: float
) -> Optional[Dict[str, float]]:
    """
    Validates the computed Jacobian against a known, closed-form analytic matrix.

    For basic quadratic examples, the regularized KKT Jacobian can be solved exactly
    on paper. Any discrepancy found here implies a fundamental implementation bug in 
    the algorithmic differentiation or KKT system assembly—not just a mathematical 
    approximation error.
    
    Note: The closed forms assume the plain ``K + diag(varrho I, -rho I)`` system. 
    If your solver uses a different proximal center or handles slack blocks differently,
    the mismatch should show up in specific matrix columns rather than uniformly everywhere.

    Args:
        example: The parametric optimization example (which must provide 'reg_jac_exact').
        jac: The numerically/algorithmically evaluated Sensitivities.
        varrho: Primal regularization parameter.
        rho: Dual regularization parameter.

    Returns:
        A dictionary containing absolute and relative errors, or None if the 
        closed-form function doesn't exist or evaluation fails entirely.
    """
    fn = example.ls.get("reg_jac_exact")
    if fn is None:
        # No analytic Jacobian available for this specific example problem
        return None
        
    # Get the mathematically exact reference Jacobian
    ref = np.asarray(fn(varrho, rho), dtype=float) # type: ignore
    
    # Get the matrix to test
    got = stack_sensitivities(jac, example.dims)
    
    # We only compute error metrics on computable (finite) entries
    mask = np.isfinite(got)
    if not mask.any():
        return None
        
    # Worst-case absolute mismatch across all valid matrix entries
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
    """Data object storing the asymptotic error of a regularized solver as eps -> 0."""

    example: str
    direction: str
    block: str
    t: float
    eps: np.ndarray
    bias: np.ndarray  # Absolute solution error ||phi_reg(eps) - phi_true||
    order: float  # Asymptotics: bias ~ eps^order (e.g., order = 1.0 for O(eps))

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
    direction: Optional[str | np.ndarray] = None,
    t: float = 1e-2,
    sign: float = +1.0,
    block: str = "x",
    kind: Optional[str] = "min",
) -> SolutionBias:
    """Computes ||phi_reg(theta; eps) - phi_true(theta)|| at a fixed theta as eps -> 0.

    Measures the solution bias introduced by a smoothed, relaxed, or regularized
    solver formulation (e.g., interior-point barrier penalty, quadratic primal-dual
    regularization) at a fixed perturbed parameter point theta = theta_bar + sign * t * d.

    Unlike Jacobian tests, this requires solving the regularized problem directly.
    The parameter `sol_fn` must be a callable matching `(theta, eps) -> NLPSolution | Array`.

    Args:
        example: The parametric optimization example.
        sol_fn: A solver function returning the regularized solution for a given (theta, eps).
        eps: Sequence of regularization weights to sweep through.
        direction: Direction of parameter perturbation d.
        t: Fixed perturbation distance.
        sign: Direction sign factor (+1.0 or -1.0).
        block: Sliced variable block ("x", "lam", "nu", "all") to evaluate bias on.
        kind: Preferred second-order character for branch selection on the true solution.

    Returns:
        A SolutionBias object containing the observed errors and log-log slope order.
    """
    name, d = _dir(example, direction)
    dim = example.dims

    # Compute the fixed parameter point theta = theta_bar + sign * t * d
    theta = example.theta_bar + sign * t * d

    # Obtain the true analytic KKT solutions at this theta
    points = example.solve(theta)
    ref = (
        dim.rows(
            phi_np(select_branch(points, example.phi_bar_np, dim, kind=kind)), block
        )
        if points
        else None
    )

    eps_np = np.asarray(list(eps), dtype=float)
    bias = []

    for e in eps_np:
        # Solve the regularized problem at (theta, eps)
        got = sol_fn(theta, float(e))

        if got is None or ref is None:
            bias.append(np.nan)
            continue

        # Format and slice the output vector appropriately based on returned type
        if isinstance(got, NLPSolution):
            vec = dim.rows(phi_np(got), block)
        else:
            arr = np.atleast_1d(to_np(got)).reshape(-1)
            # If the solver returned full phi, slice the requested block; otherwise treat as already sliced
            vec = dim.rows(arr, block) if arr.size == dim.n_phi else arr

        # Distance between the regularized solution vector and the exact reference solution
        bias.append(float(np.linalg.norm(vec - ref)))

    # Compute log-log slope to extract the order p in: bias ~ O(eps^p)
    return SolutionBias(
        example=example.name,
        direction=name,
        block=block,
        t=float(t),
        eps=eps_np,
        bias=np.asarray(bias),
        order=loglog_slope(eps_np.tolist(), bias),
    )