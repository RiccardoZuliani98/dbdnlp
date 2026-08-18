"""
Types.

Everything structural comes from the project: ``NLPSolution``, ``NLPSymbolic``,
``NLPFunctions`` and the function-signature annotations are re-exported from
``dbdnlp._impl.types`` so there is exactly one definition of each.

Defined here: ``ca.DM`` <-> ``numpy`` conversion helpers used by the harness, and
the differentiator protocol.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import casadi as ca
import numpy as np

from dbdnlp._impl.types import (  # noqa: F401
    ArrayLike,
    CostFuncType,
    CstFuncType,
    EqFuncType,
    IFTFuncType,
    IneqFuncType,
    KKTFuncType,
    LagrFuncType,
    NLPFunctions,
    NLPSolution,
    NLPSymbolic,
    Symbolic,
)


# ---------------------------------------------------------------------------
# ca.DM <-> np.ndarray
# ---------------------------------------------------------------------------
def to_np(value, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Flatten a ``ca.DM`` / ``np.ndarray`` / scalar to a numpy array."""
    if isinstance(value, (ca.DM, ca.SX, ca.MX)):
        arr = np.array(ca.DM(value).full(), dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if shape is not None:
        return arr.reshape(shape)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return arr


def mat(value, rows: int, cols: int) -> np.ndarray:
    """Coerce a possibly zero-size CasADi output to an exact ``(rows, cols)`` array."""
    arr = value if isinstance(value, np.ndarray) else np.array(ca.DM(value).full(), dtype=float)
    return np.zeros((rows, cols)) if arr.size == 0 else arr.reshape(rows, cols)


# ---------------------------------------------------------------------------
# what the harness asks of a differentiator
# ---------------------------------------------------------------------------
#: The return type of ``DifferentiableNLP.differentiate``: sensitivity blocks
#: keyed by ``'x'``, ``'lam'``, ``'nu'``, of shapes ``(n_x, n_theta)``,
#: ``(n_ineq, n_theta)``, ``(n_eq, n_theta)``.
Sensitivities = Dict[str, ArrayLike]


@runtime_checkable
class Differentiator(Protocol):
    """``(nlp, solution) -> Sensitivities``.

    Deliberately the exact signature and return type of
    ``DifferentiableNLP.differentiate``, modulo currying the mode and options,
    so writing an adapter is never necessary.  All three blocks are required:
    anything a differentiator cannot compute it reports as NaN, so that a
    missing sensitivity is visible in the results rather than indistinguishable
    from a true zero.
    """

    name: str

    def __call__(self, nlp, solution: NLPSolution) -> Sensitivities:
        ...


__all__ = [
    "ArrayLike",
    "CostFuncType",
    "CstFuncType",
    "EqFuncType",
    "IFTFuncType",
    "IneqFuncType",
    "KKTFuncType",
    "LagrFuncType",
    "NLPFunctions",
    "NLPSolution",
    "NLPSymbolic",
    "Symbolic",
    "Differentiator",
    "Sensitivities",
    "to_np",
    "mat",
]
