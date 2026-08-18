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
    """
    Manages the dimensions and block indexing of the stacked KKT vector ``phi``.
    
    The vector ``phi`` is composed of primal variables (x), inequality multipliers (lam),
    and equality multipliers (nu). This class provides utility methods to slice arrays
    according to these structural blocks.
    """

    n_x: int
    n_ineq: int = 0
    n_eq: int = 0
    n_theta: int = 1

    @classmethod
    def of(cls, nlp) -> "Dims":
        """Instantiates the dimensions directly from a given NLP object."""
        return cls(int(nlp.n_x), int(nlp.n_ineq), int(nlp.n_eq), int(nlp.n_theta))

    @property
    def n_phi(self) -> int:
        """Total size of the KKT vector (primal + dual variables)."""
        return self.n_x + self.n_ineq + self.n_eq

    @property
    def blocks(self) -> Dict[str, slice]:
        """Dictionary of slice objects to extract specific variable blocks from phi."""
        return {
            "x": slice(0, self.n_x),
            "lam": slice(self.n_x, self.n_x + self.n_ineq),
            "nu": slice(self.n_x + self.n_ineq, self.n_phi),
            "dual": slice(self.n_x, self.n_phi),
            "all": slice(0, self.n_phi),
        }

    def rows(self, arr: Array, block: str = "all") -> Array:
        """
        Extracts a specific block of rows from an array representing the KKT system.

        Args:
            arr: The array to slice (must be compatible with phi's dimension).
            block: The name of the block to extract ("x", "lam", "nu", "dual", "all").
            
        Raises:
            KeyError: If the requested block is not recognized.
        """
        if block not in self.blocks:
            raise KeyError(f"unknown block {block!r}; use one of {sorted(self.blocks)}")
        return np.asarray(arr, dtype=float)[self.blocks[block]]


# ---------------------------------------------------------------------------
# the example container
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """
    A parametric NLP instance configured with analytical/known truth data.
    
    This class is the core test container. It encapsulates a specific mathematical
    problem, a known point of degeneracy (``theta_bar``), and the explicit closed-form
    solutions for KKT points in the neighborhood of the degeneracy. This allows the 
    validation framework to test differentiators against absolute mathematical truth 
    rather than numerical solver approximations.
    """

    name: str
    tags: Tuple[str, ...]
    nlp: DifferentiableNLP
    theta_bar: Array
    
    #: The reference (potentially degenerate) primal-dual point at ``theta_bar``
    #: structured as ``(x, lam, nu)``.
    reference: Tuple[Sequence[float], Sequence[float], Sequence[float]]
    
    #: A callable providing the analytic set of nearby KKT points.
    #: Called as ``branches(theta, at)`` where ``at`` is a factory function for NLPSolution.
    branches: Callable[[Array, Callable[..., NLPSolution]], List[NLPSolution]]
    
    #: Predetermined parameter perturbation directions of interest for testing.
    directions: Dict[str, Array]
    
    #: Unnormalized generators for the kernel of the KKT matrix in phi-space.
    #: Stored as a tuple of basis vectors, each being ``(u, v_ineq, v_eq)``.
    kernel: Tuple[Tuple[Array, Array, Array], ...] = ()
    
    #: Analytical Lyapunov-Schmidt data and closed forms (free-form dictionary).
    ls: Dict[str, object] = field(default_factory=dict)
    
    #: Optional metadata or descriptions regarding the example's mathematical nature.
    notes: str = ""

    # Internal cache for the reference solution
    _phi_bar: Optional[NLPSolution] = None

    @property
    def dims(self) -> Dims:
        """Returns the structural dimensions of this specific NLP."""
        return Dims.of(self.nlp)

    @property
    def p(self) -> int:
        """The dimension of the parameter space (theta)."""
        return int(np.size(self.theta_bar))

    # -- solutions ---------------------------------------------------------
    def at(self, theta, x, lam=None, nu=None, kind: str = "unknown", **meta) -> NLPSolution:
        """
        Constructs an evaluated KKT point for a specific parameter value.
        
        Evaluates the gradients and constraints via CasADi to generate residuals.
        """
        return self.nlp.evaluate(theta, x, lam, nu, properties={"kind": kind, **meta})

    def solve(self, theta: Array, with_properties: bool = False) -> List[NLPSolution]:
        """
        Retrieves all valid KKT branches at a given parameter value ``theta``.
        
        This relies entirely on the analytically provided ``branches`` callback, 
        ensuring results are exact and bypass numerical solver inaccuracies.
        """
        # Ensure theta is at least a 1D numpy array
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        # Partial application of the "at" method fixed to the current theta
        def at(x, lam=None, nu=None, kind: str = "unknown", **meta) -> NLPSolution:
            return self.at(theta, x, lam, nu, kind=kind, **meta)

        # Compute the explicit solution branches
        out = list(self.branches(theta, at))

        # Annotate solutions with properties if requested (e.g., active sets)
        if with_properties:
            for s in out:
                self.nlp.annotate_solution(s)
        return out

    @property
    def phi_bar(self) -> NLPSolution:
        """The cached reference solution at the central parameter ``theta_bar``."""
        if self._phi_bar is None:
            x, lam, nu = self.reference
            sol = self.at(self.theta_bar, x, lam, nu, kind="reference")
            self.nlp.annotate_solution(sol)
            self._phi_bar = sol
        return self._phi_bar

    @property
    def phi_bar_np(self) -> Array:
        """
        Returns the reference solution as a stacked flat numpy array.
        """
        return phi_np(self.phi_bar)

    # -- helpers -----------------------------------------------------------
    def direction(self, name: Optional[str] = None) -> Array:
        """
        Retrieves and normalizes a specific perturbation direction by name.
        If no name is provided, defaults to the first available direction.
        """
        if name is None:
            name = next(iter(self.directions))
        
        d = np.atleast_1d(np.asarray(self.directions[name], dtype=float))
        nrm = float(np.linalg.norm(d))
        
        # Guard against zero-division for a perfectly zero direction vector
        return d / nrm if nrm > 0 else d

    def kernel_matrix(self, normalise: bool = True) -> Array:
        """
        Constructs a matrix whose columns span the known null space (kernel)
        of the KKT matrix at the reference point.
        """
        if not self.kernel:
            return np.zeros((self.dims.n_phi, 0))
            
        cols = []
        for u, vg, vh in self.kernel:
            # Flatten and stack the primal and dual components of the kernel basis
            col = np.concatenate([np.ravel(u), np.ravel(vg), np.ravel(vh)])
            cols.append(col / np.linalg.norm(col) if normalise else col)
            
        return np.stack(cols, axis=1)

    def residual(self, x: Array, lam: Array, nu: Array, theta: Array) -> float:
        """
        Computes the absolute scalar norm of the KKT residuals for a given point.
        Uses the underlying CasADi NLP implementation for exact evaluation.
        """
        # Ask CasADi to evaluate the solution
        solution = self.nlp.evaluate(theta, x, lam, nu)

        # Extract evaluated residual vectors from the result
        parts = [to_np(v).reshape(-1) for v in solution.kkt_residuals.values()]

        # Filter out empty arrays (e.g., if there are no equality constraints)
        parts = [p for p in parts if p.size] or [np.zeros(1)]

        # Return the L2 norm of the concatenated residual vector
        return float(np.linalg.norm(np.concatenate(parts)))

    def residual_phi(self, phi: Array, theta: Array) -> float:
        """Convenience wrapper for ``residual`` that accepts a stacked phi vector."""
        d = self.dims
        return self.residual(d.rows(phi, "x"), d.rows(phi, "lam"), d.rows(phi, "nu"), theta)

    def check_reference_point(self) -> float:
        """
        Validates the provided analytical reference point by evaluating its
        actual KKT residual. For a true solution, this should be close to 0.
        """
        x, lam, nu = self.reference
        return self.residual(np.asarray(x), np.asarray(lam), np.asarray(nu), self.theta_bar)

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
    """
    Pick the branch closest to ``reference`` in ``block``, preferring a specific ``kind``.

    This function acts as a distance-to-the-solution-set query. This is the 
    honest error metric when the minimizer is non-unique (e.g., pitchfork bifurcations, 
    LICQ-degenerate multipliers). The nearest branch realizes that minimum by 
    construction, meaning the true error distance is just the Euclidean distance 
    between the closest branch and the reference.

    Args:
        branches: A sequence of available NLP solutions to choose from.
        reference: The reference point (array) to calculate the distance against.
        dims: The Dims object to slice the correct variable block.
        kind: The preferred second-order character (e.g., "min"). If provided, 
              only branches matching this kind are considered unless none match.
        block: The variable block ("x", "lam", "nu", "all") to measure distance over.

    Returns:
        The nearest NLPSolution from the provided branches, or None if empty.
    """
    if not branches:
        return None
        
    # Filter the pool to only include solutions of the requested 'kind'. 
    # If no solutions match (or kind is None), fall back to the entire set of branches.
    pool = [b for b in branches if kind is None or kind_of(b) == kind] or list(branches)
    
    # Extract the targeted block from the reference point
    ref = dims.rows(np.asarray(reference, dtype=float), block)
    
    # Compute the L2 (Euclidean) distance for each branch in the targeted block
    dists = [np.linalg.norm(dims.rows(phi_np(b), block) - ref) for b in pool]
    
    # Return the branch that minimizes this distance
    return pool[int(np.argmin(dists))]


def loglog_slope(xs: Sequence[float], ys: Sequence[float], last: Optional[int] = None) -> float:
    """
    Fits the exponent ``p`` for a power law relationship ``y ~ C * x**p``.

    Taking the logarithm turns the power law into a straight line:
        log(y) = log(C) + p * log(x)
        
    The least-squares slope through the ``(log x, log y)`` points estimates ``p``.
    This is used throughout the test suite to observe asymptotic rates:
      - p = 2.0: Derivative is exact (quadratic error shrinkage).
      - p = 1.0: Derivative is biased (linear error shrinkage).
      - p = 0.5: True solution map is Hölder-1/2 (non-differentiable).
      - p = -1.0: O(1/eps) Lipschitz blow-up (when charting against regularization weight).

    Args:
        xs: Sequence of x values (e.g., perturbation step sizes).
        ys: Sequence of y values (e.g., prediction errors).
        last: If set, fits only the ``last`` smallest ``x`` values. This is crucial 
              because rates are asymptotic; large steps pre-asymptotically tilt the fit.

    Returns:
        The fitted slope ``p``, or NaN if there are fewer than 2 valid data points.
    """
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    
    # Sort data so that the smallest x values come first
    order = np.argsort(xs_np)
    xs_np, ys_np = xs_np[order], ys_np[order]
    
    # Truncate to the smallest 'last' elements to focus on the asymptotic limit
    if last is not None:
        xs_np, ys_np = xs_np[:last], ys_np[:last]
        
    # Filter out non-finite and non-positive entries, as their log is undefined.
    # This naturally excludes "no solution" (NaN) and "exact zero error" cases.
    ok = np.isfinite(xs_np) & np.isfinite(ys_np) & (xs_np > 0) & (ys_np > 0)
    
    if int(ok.sum()) < 2:
        return float("nan")
        
    # Setup design matrix A for least squares: A = [log(x), 1]
    A = np.stack([np.log(xs_np[ok]), np.ones(int(ok.sum()))], axis=1)
    
    # Solve A * [p; log(C)] = log(y)
    sol, *_ = np.linalg.lstsq(A, np.log(ys_np[ok]), rcond=None)
    
    # sol[0] corresponds to the multiplier of log(x), which is our exponent 'p'
    return float(sol[0])


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------
def phi_np(solution: NLPSolution) -> Array:
    """
    Extracts ``solution.phi`` as a flattened 1D numpy array.
    
    Args:
        solution: The evaluated NLP solution (often backed by a CasADi DM).
        
    Returns:
        A flat numpy array containing the combined primal and dual variables.
    """
    return to_np(solution.phi).reshape(-1)


def kind_of(solution: NLPSolution) -> str:
    """
    Extracts the second-order mathematical character of a solution branch.
    
    Expected return values include: ``'min'``, ``'saddle'``, ``'max'``,
    ``'degenerate'``, ``'reference'`` or ``'unknown'``. This metadata is 
    stored in the ``properties`` dictionary because the base ``NLPSolution`` 
    class only has one generic slot for metadata.

    Args:
        solution: The evaluated NLP solution.

    Returns:
        The string label representing the branch character.
    """
    return str((solution.properties or {}).get("kind", "unknown"))


def stack_sensitivities(sens: Sensitivities, dims: Dims) -> Array:
    """
    Stacks block-based sensitivities into a single dense matrix.
    
    This function converts a dictionary of sensitivities ``{'x', 'lam', 'nu'}`` 
    into a dense matrix of shape ``(n_phi, n_theta)``, in the exact block order 
    of the KKT vector ``phi``. 
    
    This forms the strict interface boundary between the harness and a differentiator. 
    If a differentiator cannot compute a block, it MUST report it as NaN rather 
    than omitting it, ensuring strict compliance with the mathematical dimensions.

    Args:
        sens: A dictionary containing sensitivity blocks for 'x', 'lam', and 'nu'.
        dims: The Dims object providing the exact block dimensions.

    Returns:
        A vertically stacked numpy array of the sensitivities.
        
    Raises:
        TypeError: If the provided sensitivities are not in a dictionary.
        KeyError: If any of the required blocks ('x', 'lam', 'nu') are missing.
    """
    if not isinstance(sens, dict):
        raise TypeError(
            "a differentiator must return {'x', 'lam', 'nu'} sensitivity blocks, "
            f"as DifferentiableNLP.differentiate does; got {type(sens).__name__}"
        )
        
    # Strictly enforce that all three structural blocks are present
    missing = {"x", "lam", "nu"} - set(sens)
    if missing:
        raise KeyError(
            f"differentiator must return 'x', 'lam' and 'nu'; missing {sorted(missing)}"
        )
        
    # Stack the matrices vertically ensuring proper sizing via the `mat` utility
    return np.vstack(
        [
            mat(sens["x"], dims.n_x, dims.n_theta),
            mat(sens["lam"], dims.n_ineq, dims.n_theta),
            mat(sens["nu"], dims.n_eq, dims.n_theta),
        ]
    )