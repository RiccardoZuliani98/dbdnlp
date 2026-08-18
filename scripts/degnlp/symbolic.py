"""
Optional CasADi mirrors of the examples.

The harness needs nothing from this module -- it exists so that a differentiator
built on a symbolic NLP object (a ``DifferentiableNLP``, an ``nlpsol``, an
implicit layer) can be handed the *same* problems the analytic ground truth
describes.  ``validate`` checks the two descriptions agree at the base point, so
a transcription slip cannot masquerade as regularisation error.

Convention matches ``core.Functions``:  ``build() -> (x, theta, f, g, h)`` with
``g <= 0``, ``h == 0``, and ``g``/``h`` possibly ``None``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .core import Example

try:  # casadi is optional
    import casadi as ca

    HAVE_CASADI = True
except ImportError:  # pragma: no cover
    ca = None
    HAVE_CASADI = False

Build = Callable[[], Tuple["ca.SX", "ca.SX", "ca.SX", Optional["ca.SX"], Optional["ca.SX"]]]


def _require():
    if not HAVE_CASADI:  # pragma: no cover
        raise ImportError("casadi is required for degnlp.symbolic")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _fold() -> Build:
    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 1)
        return x, th, x[0] ** 3 / 3 - th[0] * x[0], None, None

    return build


def _cusp() -> Build:
    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 1)
        return x, th, x[0] ** 4 / 4 - th[0] * x[0], None, None

    return build


def _pitchfork() -> Build:
    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 1)
        return x, th, x[0] ** 4 / 4 - th[0] * x[0] ** 2 / 2, None, None

    return build


def _swallowtail() -> Build:
    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 2)
        f = x[0] ** 4 / 4 - th[0] * x[0] ** 2 / 2 - th[1] * x[0]
        return x, th, f, None, None

    return build


def _mixed_kernel(b: float, c: float) -> Build:
    def build():
        x = ca.SX.sym("x", 2)
        th = ca.SX.sym("theta", 1)
        f = 0.5 * x[0] ** 2 + b * x[0] * x[1] + (c / 6.0) * x[1] ** 3
        h = ca.vertcat(x[0] - th[0])
        return x, th, f, None, h

    return build


def _licq_two_eq() -> Build:
    def build():
        x = ca.SX.sym("x", 1)
        th = ca.SX.sym("theta", 2)
        h = ca.vertcat(x[0] - th[0], x[0] - th[1])
        return x, th, 0.5 * x[0] ** 2, None, h

    return build


def _tangency(kappa: float) -> Build:
    def build():
        x = ca.SX.sym("x", 2)
        th = ca.SX.sym("theta", 1)
        f = -x[1] - 0.5 * kappa * x[0] ** 2
        g = ca.vertcat(
            x[0] ** 2 + (x[1] - 1.0) ** 2 - 1.0,
            x[1] - 2.0 - th[0],
        )
        return x, th, f, g, None

    return build


def _wachter_biegler() -> Build:
    def build():
        x = ca.SX.sym("x", 3)
        th = ca.SX.sym("theta", 2)
        g = ca.vertcat(-x[1], -x[2])
        h = ca.vertcat(x[0] ** 2 - x[1] - 1.0 - th[0], x[0] - x[2] - 0.5 - th[1])
        return x, th, x[0], g, h

    return build


def builder_for(example: Example) -> Build:
    """Return the CasADi builder matching ``example`` (params read from ``ls``)."""
    _require()
    name = example.name
    if name == "fold":
        return _fold()
    if name == "cusp":
        return _cusp()
    if name == "pitchfork":
        return _pitchfork()
    if name == "swallowtail":
        return _swallowtail()
    if name.startswith("mixed_kernel"):
        return _mixed_kernel(float(example.ls["b"]), float(example.ls["c"]))
    if name == "licq_two_eq":
        return _licq_two_eq()
    if name == "tangency":
        return _tangency(float(example.ls["kappa"]))
    if name == "wachter_biegler":
        return _wachter_biegler()
    raise KeyError(f"no symbolic builder for {name!r}")


def attach(example: Example) -> Example:
    """Set ``example.build`` to the CasADi builder (no-op without casadi)."""
    if HAVE_CASADI:
        example.build = builder_for(example)
    return example


# ---------------------------------------------------------------------------
# consistency check
# ---------------------------------------------------------------------------
def validate(example: Example, tol: float = 1e-6) -> Dict[str, float]:
    """Compare the CasADi and numpy descriptions at the reference point.

    Returns max absolute discrepancies for f, g, h, grad_x L and hess_xx L.
    Anything above ``tol`` means the two descriptions of the same example have
    drifted apart.
    """
    _require()
    x_s, th_s, f_s, g_s, h_s = builder_for(example)()
    lay = example.layout
    x0, th0 = example.phi_bar.x, example.theta_bar
    lam0, nu0 = example.phi_bar.lam, example.phi_bar.nu

    lam_s = ca.SX.sym("lam", lay.m_g)
    nu_s = ca.SX.sym("nu", lay.m_h)
    L = f_s
    if g_s is not None:
        L = L + ca.dot(lam_s, g_s)
    if h_s is not None:
        L = L + ca.dot(nu_s, h_s)

    args = [x_s, th_s, lam_s, nu_s]
    F = ca.Function("F", args, [f_s])
    G = ca.Function("G", args, [g_s if g_s is not None else ca.SX.zeros(0)])
    H = ca.Function("H", args, [h_s if h_s is not None else ca.SX.zeros(0)])
    gL = ca.Function("gL", args, [ca.jacobian(L, x_s).T])
    hL = ca.Function("hL", args, [ca.hessian(L, x_s)[0]])

    a = [x0, th0, lam0, nu0]
    fns = example.functions

    def diff(sym, num):
        sym = np.asarray(sym, dtype=float).reshape(-1)
        num = np.asarray(num, dtype=float).reshape(-1)
        if sym.size == 0 and num.size == 0:
            return 0.0
        return float(np.max(np.abs(sym - num)))

    return {
        "f": diff(F(*a), fns.f(x0, th0)),
        "g": diff(G(*a), fns.gval(x0, th0)),
        "h": diff(H(*a), fns.hval(x0, th0)),
        "grad_x_L": diff(gL(*a), fns.grad_x_L(x0, lam0, nu0, th0)),
        "hess_xx_L": diff(hL(*a), fns.hess_xx_L(x0, lam0, nu0, th0)),
    }
