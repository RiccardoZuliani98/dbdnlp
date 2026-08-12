import casadi as ca
import numpy as np
from typing import Dict, Optional, cast

from dbdnlp._impl.types import (
    NLPFunctions, NLPSolution, NLPSymbolic, Symbolic,
    CostFuncType, IneqFuncType, EqFuncType, CstFuncType,
    LagrFuncType, KKTFuncType, IFTFuncType
)

class DifferentiableNLP:
    """
    A class to solve and differentiate non-linear programs (NLPs) in CasADi.
    
    This solver manages a single NLP instance. Regularized versions of the problem 
    should be constructed externally via a factory function that returns a new 
    DifferentiableNLP instance.
    """

    def __init__(
        self,
        x: Symbolic,
        theta: Symbolic,
        f: Symbolic,
        g: Optional[Symbolic] = None,
        h: Optional[Symbolic] = None,
        solver_opts: Optional[Dict] = None
    ) -> None:
        """
        Constructs the NLP solver, its symbolic representation, and eagerly 
        compiles all numerical functions required for solving and differentiation.

        Parameters
        ----------
        x : Symbolic
            Decision variables vector of shape (n_x, 1).
        theta : Symbolic
            Parameters vector of shape (n_theta, 1).
        f : Symbolic
            Scalar objective function expression f(x, theta).
        g : Optional[Symbolic], optional
            Inequality constraint expressions g(x, theta) <= 0.
        h : Optional[Symbolic], optional
            Equality constraint expressions h(x, theta) == 0.
        solver_opts : Optional[Dict], optional
            Dictionary of options passed to `casadi.nlpsol`.
        """
        # 1. Normalize Constraints and Dimensions
        g = g if g is not None else ca.MX(0, 1)
        h = h if h is not None else ca.MX(0, 1)

        self.n_x = x.size1()
        self.n_theta = theta.size1()
        self.n_ineq = g.size1()
        self.n_eq = h.size1()
        self.n_c = self.n_ineq + self.n_eq

        # 2. Initialize Symbolic Multipliers
        # Assuming MX by default to match original implementation; adjust if supporting SX
        if isinstance(x, ca.SX):
            sym_type = ca.SX
        else:
            sym_type = ca.MX
        lam = cast(Symbolic, sym_type.sym('lam', self.n_ineq)) #type:ignore
        nu  = cast(Symbolic, sym_type.sym('nu', self.n_eq))  #type:ignore

        # 3. Build NLPSymbolic Dataclass
        cst = cast(Symbolic, ca.vertcat(g, h))
        dual_cst = cast(Symbolic, ca.vertcat(lam, nu))

        self.sym = NLPSymbolic(
            cost=f, ineq=g, eq=h, cst=cst, 
            primal=x, param=theta,
            dual_ineq=lam, dual_eq=nu, 
            dual_cst=dual_cst
        )

        # 4. Build Eagerly Compiled Functions
        cost_func = ca.Function('cost', [x, theta], [f])
        ineq_func = ca.Function('ineq', [x, theta], [g])
        eq_func   = ca.Function('eq',   [x, theta], [h])
        cst_func  = ca.Function('cst',  [x, theta], [cst])

        # Lagrangian: L = f + lam^T g + nu^T h
        lagr = f
        if self.n_ineq > 0: lagr += ca.dot(lam, g)
        if self.n_eq > 0: lagr += ca.dot(nu, h)
        lagr = cast(Symbolic, lagr)
        lagr_func = ca.Function('lagr', [x, theta, lam, nu], [lagr])

        # KKT Conditions
        stationarity = ca.jacobian(lagr, x).T
        inequality = ca.fmax(g, 0.0)
        equality = h
        slackness = lam * g
        dual_feasibility = ca.fmin(lam, 0.0)
        
        kkt_func = ca.Function(
            'kkt', 
            [theta, x, lam, nu], 
            [stationarity, inequality, equality, slackness, dual_feasibility]
        )

        # Implicit Function Theorem (IFT) Derivatives
        H_xx = ca.hessian(lagr, x)[0]
        dL_dx = ca.jacobian(lagr, x).T
        H_xtheta = ca.jacobian(dL_dx, theta)
        
        # Safely handle empty constraints for Jacobians
        J_gx = ca.jacobian(g, x) if self.n_ineq > 0 else ca.MX(0, self.n_x)
        J_gtheta = ca.jacobian(g, theta) if self.n_ineq > 0 else ca.MX(0, self.n_theta)
        J_hx = ca.jacobian(h, x) if self.n_eq > 0 else ca.MX(0, self.n_x)
        J_htheta = ca.jacobian(h, theta) if self.n_eq > 0 else ca.MX(0, self.n_theta)

        ift_func = ca.Function(
            'ift', 
            [x, theta, lam, nu], 
            [H_xx, H_xtheta, J_gx, J_gtheta, J_hx, J_htheta]
        )

        # Define Constraint Bounds
        # Inequalities are g(x) <= 0, Equalities are h(x) == 0
        lbg = ca.DM(np.concatenate([-np.inf * np.ones(self.n_ineq), np.zeros(self.n_eq)]))
        ubg = ca.DM(np.concatenate([np.zeros(self.n_ineq), np.zeros(self.n_eq)]))

        self.funcs = NLPFunctions(
            cost=cast(CostFuncType, cost_func),
            ineq=cast(IneqFuncType, ineq_func),
            eq=cast(EqFuncType, eq_func),
            cst=cast(CstFuncType, cst_func),
            lagr=cast(LagrFuncType, lagr_func),
            kkt=cast(KKTFuncType, kkt_func),
            ift=cast(IFTFuncType, ift_func),
            lbg=lbg, ubg=ubg
        )

        # 5. Initialize Solver Configuration
        self.opts = solver_opts if solver_opts is not None else {}
        self.plugin = self.opts.pop("plugin", "ipopt")

        nlp_dict = {
            'x': self.sym.primal,
            'p': self.sym.param,
            'f': self.sym.cost,
            'g': self.sym.cst
        }
        
        self.solver = ca.nlpsol('solver', self.plugin, nlp_dict, self.opts)