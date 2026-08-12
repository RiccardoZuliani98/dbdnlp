from dataclasses import dataclass
from typing import (
    Dict, Optional, Callable, Union, 
    Annotated, Tuple
)
import casadi as ca
import numpy as np

# Type aliases
Symbolic = Union[ca.MX, ca.SX]
ArrayLike = Union[np.ndarray, ca.DM, ca.MX, ca.SX]

@dataclass
class NLPSolution:
    """
    Stores the numerical solution, parameters, dual multipliers, and KKT residual analysis.

    Dimensions
    ----------
    - n_x     : Number of primal variables
    - n_theta : Number of parameters
    - n_g     : Number of inequality constraints
    - n_h     : Number of equality constraints

    Attributes
    ----------
    x : ca.DM
        Optimal primal decision variable vector of shape (n_x, 1).
    theta : ca.DM
        Parameter vector evaluated during the solve of shape (n_theta, 1).
    lam : ca.DM
        Optimal dual multipliers for inequality constraints g(x, theta) <= 0 of shape (n_g, 1).
    nu : ca.DM
        Optimal dual multipliers for equality constraints h(x, theta) == 0 of shape (n_h, 1).
    kkt_residuals : Dict[str, ca.DM]
        Evaluated residual vectors of the KKT system containing:
        - 'stationarity' : Gradient of Lagrangian w.r.t x (n_x, 1)
        - 'inequality'   : max(g(x, theta), 0) (n_g, 1)
        - 'equality'     : h(x, theta) (n_h, 1)
        - 'slackness'    : lam * g(x, theta) (n_g, 1)
        - 'dual_feas'    : min(lam, 0) (n_g, 1)
    properties : Optional[Dict[str, bool]], default=None
        Flags indicating properties at the solution (e.g., LICQ holds, strict complementarity).
    """
    x:      Annotated[ca.DM, "(n_x, 1)"]
    theta:  Annotated[ca.DM, "(n_theta, 1)"]
    lam:    Annotated[ca.DM, "(n_g, 1)"]
    nu:     Annotated[ca.DM, "(n_h, 1)"]
    kkt_residuals: Dict[str, ca.DM]
    properties: Optional[Dict[str, bool]] = None


@dataclass
class NLPSymbolic:
    """
    Pure symbolic container for an NLP formulation and its decision/dual variables.

    Dimensions
    ----------
    - n_x     : Number of primal variables
    - n_theta : Number of parameters
    - n_g     : Number of inequality constraints
    - n_h     : Number of equality constraints
    - n_c     : Total constraints (n_g + n_h)

    Attributes
    ----------
    cost : Symbolic
        Scalar objective function expression f(x, theta) of shape (1, 1).
    ineq : Symbolic
        Inequality constraint expressions g(x, theta) <= 0 of shape (n_g, 1).
    eq : Symbolic
        Equality constraint expressions h(x, theta) == 0 of shape (n_h, 1).
    cst : Symbolic
        Stacked constraint vector c(x, theta) = [g; h] of shape (n_h + n_g, 1).
    primal : Symbolic
        Primal decision variable vector x of shape (n_x, 1).
    param : Symbolic
        Parameter vector theta of shape (n_theta, 1).
    dual_ineq : Symbolic
        Dual variable vector lambda for inequalities of shape (n_g, 1).
    dual_eq : Symbolic
        Dual variable vector nu for equalities of shape (n_h, 1).
    dual_cst : Symbolic
        Stacked dual variable vector [lambda; nu] of shape (n_c, 1).
    """
    cost:       Annotated[Symbolic, "(1, 1)"]
    ineq:       Annotated[Symbolic, "(n_g, 1)"]
    eq:         Annotated[Symbolic, "(n_h, 1)"]
    cst:        Annotated[Symbolic, "(n_h + n_g, 1)"]
    primal:     Annotated[Symbolic, "(n_x, 1)"]
    param:      Annotated[Symbolic, "(n_theta, 1)"]
    dual_ineq:  Annotated[Symbolic, "(n_g, 1)"]
    dual_eq:    Annotated[Symbolic, "(n_h, 1)"]
    dual_cst:   Annotated[Symbolic, "(n_c, 1)"]


# ---------------------------------------------------------------------------
# Function Signature Annotations
# ---------------------------------------------------------------------------

CostFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike], ca.DM],
    "Inputs: (x: n_x, theta: n_theta) -> Output: f (1x1)"
]

IneqFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike], ca.DM],
    "Inputs: (x: n_x, theta: n_theta) -> Output: g (n_g x 1)"
]

EqFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike], ca.DM],
    "Inputs: (x: n_x, theta: n_theta) -> Output: h (n_h x 1)"
]

CstFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike], ca.DM],
    "Inputs: (x: n_x, theta: n_theta) -> Output: c (n_g + n_h x 1)"
]

LagrFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], ca.DM],
    "Inputs: (x: n_x, lam: n_g, nu: n_h, theta: n_theta) -> Output: L (1x1)"
]

# CasADi functions with multiple outputs return a tuple (or list) when called positionally.
KKTFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Tuple[ca.DM, ...]],
    "Inputs: (x: n_x, lam: n_g, nu: n_h, theta: n_theta) -> "
    "Outputs: (stationarity: n_x, ineq: n_g, eq: n_h, slack: n_g, dual_feas: n_g)"
]

IFTFuncType = Annotated[
    Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Tuple[ca.DM, ...]],
    "Inputs: (x: n_x, lam: n_g, nu: n_h, theta: n_theta) -> "
    "Outputs: (H_xx: n_x x n_x, H_xtheta: n_x x n_theta, "
    "J_gx: n_g x n_x, J_gtheta: n_g x n_theta, "
    "J_hx: n_h x n_x, J_htheta: n_h x n_theta)"
]

@dataclass
class NLPFunctions:
    """
    Compiled CasADi functions evaluated during numerical solving and differentiation.

    Attributes
    ----------
    cost : ca.Function
        Evaluates objective function.
        Inputs:  [x, theta] -> Output: [f] (1, 1)
    ineq : ca.Function
        Evaluates inequality constraints.
        Inputs:  [x, theta] -> Output: [g] (n_g, 1)
    eq : ca.Function
        Evaluates equality constraints.
        Inputs:  [x, theta] -> Output: [h] (n_h, 1)
    cst : ca.Function
        Evaluates stacked constraints [g; h].
        Inputs:  [x, theta] -> Output: [c] (n_c, 1)
    lagr : ca.Function
        Evaluates the Lagrangian L(x, lam, nu, theta) = f + lam^T g + nu^T h.
        Inputs:  [x, lam, nu, theta] -> Output: [L] (1, 1)
    kkt : ca.Function
        Evaluates exact KKT residual vectors.
        Inputs:  [x, lam, nu, theta] 
        Outputs: ['stationarity', 'inequality', 'equality', 'slackness', 'dual_feasibility']
    ift : ca.Function
        Evaluates Jacobians and Hessians for Implicit Function Theorem differentiation.
        Inputs:  [x, lam, nu, theta] 
        Outputs: [H_xx (n_x, n_x), H_xtheta (n_x, n_theta), 
                  J_gx (n_g, n_x), J_gtheta (n_g, n_theta), 
                  J_hx (n_h, n_x), J_htheta (n_h, n_theta)]
    lbg : ca.DM
        Lower bound of cst function, shape (n_h + n_g, 1).
    ubg : ca.DM
        Upper bound of cst function, shape (n_h + n_g, 1).
    """
    cost:   CostFuncType
    ineq:   IneqFuncType
    eq:     EqFuncType
    cst:    CstFuncType
    lagr:   LagrFuncType
    kkt:    KKTFuncType
    ift:    IFTFuncType
    lbg:    Annotated[ca.DM, "(n_h + n_g, 1)"]
    ubg:    Annotated[ca.DM, "(n_h + n_g, 1)"]