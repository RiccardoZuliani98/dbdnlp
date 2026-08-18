"""
Types.

The codebase types (``NLPSolution``, ``NLPSymbolic``, ``NLPFunctions``,
``Symbolic``, ``ArrayLike`` and the function-signature annotations) are imported
from your package when it is importable, and from a vendored copy of
``types.py`` otherwise.  Edit ``_CANDIDATES`` so there is a single source of
truth once this lives inside the project.

Everything defined *here* is additive: ``ca.DM``/``np.ndarray`` conversion
helpers and the differentiator protocol the harness talks to.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import casadi as ca
import numpy as np

_CANDIDATES = (
    "dbdnlp._impl.types",   # <- the real one; adjust to match your layout
    "dbdnlp.types",
)

_NAMES = (
    "Symbolic",
    "ArrayLike",
    "NLPSolution",
    "NLPSymbolic",
    "NLPFunctions",
    "CostFuncType",
    "IneqFuncType",
    "EqFuncType",
    "CstFuncType",
    "LagrFuncType",
    "KKTFuncType",
    "IFTFuncType",
)

_module = None
for _cand in _CANDIDATES:
    try:
        _module = importlib.import_module(_cand)
        break
    except ImportError:
        continue
if _module is None:
    from . import _types_vendored as _module  # vendored fallback

TYPES_SOURCE = _module.__name__

globals().update({name: getattr(_module, name) for name in _NAMES})

# re-bound explicitly so static tooling and ``from .types import X`` both work
Symbolic = _module.Symbolic
ArrayLike = _module.ArrayLike
NLPSolution = _module.NLPSolution
NLPSymbolic = _module.NLPSymbolic
NLPFunctions = _module.NLPFunctions


# ---------------------------------------------------------------------------
# ca.DM <-> np.ndarray
# ---------------------------------------------------------------------------
def to_np(value, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Flatten a ``ca.DM`` / ``np.ndarray`` / scalar to a numpy array."""
    if isinstance(value, (ca.DM, ca.SX, ca.MX)):
        arr = np.array(ca.DM(value), dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if shape is not None:
        return arr.reshape(shape)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return arr


def to_dm(value, n_rows: Optional[int] = None) -> ca.DM:
    """Build a column ``ca.DM``, handling the zero-size case cleanly.

    CasADi will not accept ``np.zeros(0)`` where a ``(0, 1)`` input is expected,
    which every unconstrained example hits on the ``lam``/``nu`` arguments.
    """
    if isinstance(value, ca.DM):
        arr = np.array(value, dtype=float).reshape(-1)
    else:
        arr = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
    if n_rows is not None:
        if arr.size == 0 or n_rows == 0:
            return ca.DM(n_rows, 1)
        arr = arr.reshape(n_rows)
    if arr.size == 0:
        return ca.DM(0, 1)
    return ca.DM(arr.reshape(-1, 1))


def mat(value, rows: int, cols: int) -> np.ndarray:
    """Coerce a possibly zero-size CasADi output to an exact ``(rows, cols)`` array."""
    arr = value if isinstance(value, np.ndarray) else np.array(ca.DM(value), dtype=float)
    if arr.size == 0:
        return np.zeros((rows, cols))
    return arr.reshape(rows, cols)


# ---------------------------------------------------------------------------
# what the harness asks of a differentiator
# ---------------------------------------------------------------------------
JacLike = Union[np.ndarray, ca.DM, Dict[str, "ArrayLike"], object]


@runtime_checkable
class Differentiator(Protocol):
    """``(nlp, solution) -> d phi / d theta`` of shape ``(n_phi, n_theta)``.

    ``coerce_jac`` also accepts an ``(n_x, n_theta)`` block, a dict keyed by
    ``x`` / ``lam`` / ``nu``, a ``ca.DM``, or an object exposing ``.jac`` /
    ``.jacobian`` / ``.J``.  Unreported rows stay unreported rather than
    silently becoming zero.
    """

    name: str

    def __call__(self, nlp, solution: "NLPSolution") -> ArrayLike:
        ...


__all__ = list(_NAMES) + [
    "TYPES_SOURCE",
    "Differentiator",
    "JacLike",
    "to_np",
    "to_dm",
    "mat",
]
