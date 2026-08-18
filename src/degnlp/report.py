"""
Suite-level driving and reporting.

``run_suite`` is the one-call entry point: hand it a differentiator with the
``(nlp, solution) -> jacobian`` signature and it walks every example and every
probe direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import examples as ex_mod
from . import metrics as M
from .core import Example, stack_sensitivities
from .reference import FiniteDifferenceOracle
from .types import NLPSolution, Sensitivities

# TODO: what is this? Do we need a differentiate object here, or can we pull
# from types.py?
Differentiate = Callable[[object, NLPSolution], Sensitivities]

# TODO: there's no way we need this. I'll change the signature of the differentiator methods
def jacobian_for(differentiator: Differentiate, example: Example) -> Sensitivities:
    """Call a differentiator on an example's NLP and reference solution.

    Differentiators that need the example itself (the finite-difference oracle)
    are bound first; everything else sees only ``(nlp, NLPSolution)``.
    """
    diff = differentiator
    if isinstance(diff, FiniteDifferenceOracle):
        diff = diff.bind(example)
    return diff(example.nlp, example.phi_bar)

# TODO: this needs proper explanation
@dataclass
class Row:
    example: str
    direction: str
    block: str
    side: str
    exists: bool
    true_order: float
    err_order: float
    t_valid: float
    verdict: str
    expected: object


@dataclass
class Suite:
    differentiator: str
    rows: List[Row] = field(default_factory=list)
    self_check: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, Dict[str, object]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        hdr = (
            f"{'example':<24}{'direction':<14}{'side':<6}{'true~t^':>9}"
            f"{'err~t^':>9}{'t_valid':>11}   verdict"
        )
        lines = [f"differentiator: {self.differentiator}", "", hdr, "-" * (len(hdr) + 18)]
        for r in self.rows:
            if not r.exists:
                true_s = "none"
            elif not np.isfinite(r.true_order):
                true_s = "const"
            else:
                true_s = f"{r.true_order:.2f}"
            err_s = "--" if not np.isfinite(r.err_order) else f"{r.err_order:.2f}"
            tv = "--" if not np.isfinite(r.t_valid) else f"{r.t_valid:.1e}"
            lines.append(
                f"{r.example:<24}{r.direction:<14}{r.side:<6}{true_s:>9}{err_s:>9}"
                f"{tv:>11}   {r.verdict}"
            )
        if self.self_check:
            lines += ["", "reference-point residuals (should be ~0):"]
            for k, v in self.self_check.items():
                flag = "" if v < 1e-6 else "   <-- CHECK"
                lines.append(f"  {k:<24}{v:.2e}{flag}")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


def run_suite(
    differentiator: Differentiate,
    names: Optional[Sequence[str]] = None,
    block: str = "x",
    ts: Sequence[float] = M.DEFAULT_TS,
    tol: float = 0.25,
    tags: Optional[Sequence[str]] = None,
) -> Suite:
    """Walk examples x directions, calling ``differentiator`` once per example."""
    pool = [ex_mod.get(n) for n in names] if names else ex_mod.all_examples(tags=tags)
    label = getattr(differentiator, "name", type(differentiator).__name__)
    suite = Suite(differentiator=label)

    for ex in pool:
        suite.self_check[ex.name] = ex.check_reference_point()
        suite.properties[ex.name] = dict(ex.phi_bar.properties or {})
        try:
            jac = jacobian_for(differentiator, ex)
            stack_sensitivities(jac, ex.dims)
        except Exception as exc:  # a differentiator may legitimately refuse
            suite.notes.append(f"{ex.name}: differentiator raised {exc!r}")
            continue
        for dname in ex.directions:
            rep = M.taylor(ex, jac, direction=dname, ts=ts, block=block, tol=tol)
            expected = ex.ls.get("expected", {}).get(dname, {})
            for side in ("+", "-"):
                suite.rows.append(
                    Row(
                        example=ex.name, direction=dname, block=block, side=side,
                        exists=rep.exists[side], true_order=rep.true_slope[side],
                        err_order=rep.slope[side], t_valid=rep.t_valid[side],
                        verdict=rep.verdict[side], expected=expected.get(side, "unset"),
                    )
                )
    return suite


def validate_examples(
    names: Optional[Sequence[str]] = None, block: str = "x", ts=M.DEFAULT_TS
) -> List[M.Hoelder]:
    """Check the library against its own declared exponents (no differentiator)."""
    pool = [ex_mod.get(n) for n in names] if names else ex_mod.all_examples()
    out = []
    for ex in pool:
        for dname in ex.directions:
            out.append(M.hoelder(ex, direction=dname, block=block, ts=ts))
    return out


def property_table(names: Optional[Sequence[str]] = None) -> str:
    """Which hypotheses actually fail at each reference point, as measured by
    ``DifferentiableNLP.check_properties`` rather than asserted."""
    pool = [ex_mod.get(n) for n in names] if names else ex_mod.all_examples()
    keys = ("licq", "scs", "sosc", "ssosc")
    labels = ("LICQ", "SCS", "SOSC", "SSOSC")
    hdr = f"{'example':<24}" + "".join(f"{l:>10}" for l in labels) + f"{'dim ker K':>11}"
    lines = [hdr, "-" * len(hdr)]
    for ex in pool:
        props = ex.phi_bar.properties or {}
        cells = "".join(f"{('yes' if props.get(k) else 'NO'):>10}" for k in keys)
        lines.append(f"{ex.name:<24}{cells}{props.get('dim_kernel', -1):>11}")
    return "\n".join(lines)
