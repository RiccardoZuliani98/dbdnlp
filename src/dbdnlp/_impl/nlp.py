import casadi as ca
import numpy as np
from typing import Dict, Optional, cast, Any
from scipy.linalg import solve, null_space

from dbdnlp._impl.types import (
    NLPFunctions, NLPSolution, NLPSymbolic, Symbolic,
    CostFuncType, IneqFuncType, EqFuncType, CstFuncType,
    LagrFuncType, KKTFuncType, IFTFuncType, ArrayLike
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

        # identify symbolic type
        if isinstance(x, ca.SX):
            sym_type = ca.SX
        else:
            sym_type = ca.MX

        # 1. Normalize Constraints and Dimensions
        g = g if g is not None else sym_type(0, 1)
        h = h if h is not None else sym_type(0, 1)

        self.n_x     = cast(int, x.size1())
        self.n_theta = cast(int, theta.size1())
        self.n_ineq  = cast(int, g.size1())
        self.n_eq    = cast(int, h.size1())
        self.n_c     = self.n_ineq + self.n_eq

        # 2. Initialize Symbolic Multipliers
        lam = cast(Symbolic, sym_type.sym('lam', self.n_ineq))  #type:ignore
        nu  = cast(Symbolic, sym_type.sym('nu',  self.n_eq))    #type:ignore

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
        if self.n_eq > 0:   lagr += ca.dot(nu, h)
        lagr = cast(Symbolic, lagr)
        lagr_func = ca.Function('lagr', [x, lam, nu, theta], [lagr])

        # KKT Conditions
        stationarity     = cast(Symbolic, ca.jacobian(lagr, x).T)
        inequality       = cast(Symbolic, ca.fmax(g, 0.0))
        equality         = h
        slackness        = cast(Symbolic, lam * g)
        dual_feasibility = cast(Symbolic, ca.fmin(lam, 0.0))
        
        kkt_func = ca.Function(
            'kkt', 
            [x, lam, nu, theta],
            [stationarity, inequality, equality, slackness, dual_feasibility]
        )

        # Implicit Function Theorem (IFT) Derivatives
        H_xx     = cast(Symbolic, ca.hessian(lagr, x)[0])
        H_xtheta = cast(Symbolic, ca.jacobian(stationarity, theta))
        
        # Safely handle empty constraints for Jacobians
        J_gx     = cast(Symbolic, ca.jacobian(g, x)) if self.n_ineq > 0 else sym_type(0, self.n_x)
        J_gtheta = cast(Symbolic, ca.jacobian(g, theta)) if self.n_ineq > 0 else sym_type(0, self.n_theta)
        J_hx     = cast(Symbolic, ca.jacobian(h, x)) if self.n_eq > 0 else sym_type(0, self.n_x)
        J_htheta = cast(Symbolic, ca.jacobian(h, theta)) if self.n_eq > 0 else sym_type(0, self.n_theta)

        ift_func = ca.Function(
            'ift', 
            [x, lam, nu, theta], 
            [H_xx, H_xtheta, J_gx, J_gtheta, J_hx, J_htheta]
        )

        # Define Constraint Bounds
        # Inequalities are g(x) <= 0, Equalities are h(x) == 0
        lbg = ca.DM(np.concatenate([-np.inf * np.ones(self.n_ineq), np.zeros(self.n_eq)]))
        ubg = ca.DM(np.concatenate([np.zeros(self.n_ineq), np.zeros(self.n_eq)]))

        self.funcs = NLPFunctions(
            cost = cast(CostFuncType, cost_func),
            ineq = cast(IneqFuncType, ineq_func),
            eq   = cast(EqFuncType, eq_func),
            cst  = cast(CstFuncType, cst_func),
            lagr = cast(LagrFuncType, lagr_func),
            kkt  = cast(KKTFuncType, kkt_func),
            ift  = cast(IFTFuncType, ift_func),
            lbg  = lbg, ubg = ubg
        )

        # 5. Initialize Solver Configuration
        self.opts = dict(solver_opts) if solver_opts else {}
        self.plugin = self.opts.pop("plugin", "ipopt")

        nlp_dict = {
            'x': self.sym.primal,
            'p': self.sym.param,
            'f': self.sym.cost,
            'g': self.sym.cst
        }
        
        self.solver = ca.nlpsol('solver', self.plugin, nlp_dict, self.opts)

    def solve(
        self,
        theta: ArrayLike,
        x_warmstart: Optional[ArrayLike] = None,
        lam_warmstart: Optional[ArrayLike] = None,
        nu_warmstart: Optional[ArrayLike] = None
    ) -> NLPSolution:
        """
        Solves the NLP for a given parameter vector theta and optional warmstart values.

        Parameters
        ----------
        theta : ArrayLike
            Numerical value for parameter theta of shape (n_theta, 1).
        x_warmstart : Optional[ArrayLike], optional
            Warmstart guess for primal decision variable x (n_x, 1).
        lam_warmstart : Optional[ArrayLike], optional
            Warmstart guess for inequality multipliers lambda (n_ineq, 1).
        nu_warmstart : Optional[ArrayLike], optional
            Warmstart guess for equality multipliers nu (n_eq, 1).

        Returns
        -------
        NLPSolution
            Object containing the numerical solution, evaluated parameters,
            dual multipliers, and exact KKT residual analysis.
        """

        theta_DM = ca.DM(theta)

        args: Dict[str, Any] = {
            'p': theta_DM,
            'lbg': self.funcs.lbg,
            'ubg': self.funcs.ubg
        }

        # Handle warmstarts
        if x_warmstart is not None:
            args['x0'] = x_warmstart

        if lam_warmstart is not None or nu_warmstart is not None:
            # Handle partial warmstarts by filling missing multiplier components with zeros
            lam_0 = lam_warmstart if lam_warmstart is not None else ca.DM(self.n_ineq,1)
            nu_0  = nu_warmstart  if nu_warmstart  is not None else ca.DM(self.n_eq,1)
            args['lam_g0'] = ca.vertcat(lam_0, nu_0)

        # Solve the compiled NLP
        res = self.solver(**args)

        # Extract solution components
        x_opt   = ca.DM(res['x'])
        lam_c   = ca.DM(res['lam_g'])
        lam_opt = ca.DM(lam_c[:self.n_ineq])
        nu_opt  = ca.DM(lam_c[self.n_ineq:])

        # Evaluate KKT residuals using the eagerly compiled function
        stat, ineq, eq, slack, dual_feas = self.funcs.kkt(x_opt, lam_opt, nu_opt, theta_DM)

        kkt_res = {
            'stationarity':     stat,
            'inequality':       ineq,
            'equality':         eq,
            'slackness':        slack,
            'dual_feasibility': dual_feas
        }

        return NLPSolution(
            x=x_opt,
            theta=theta_DM,
            lam=lam_opt,
            nu=nu_opt,
            kkt_residuals=kkt_res,
            properties={},
        )

    def differentiate(
        self,
        theta: ArrayLike,
        x: ArrayLike,
        lam: ArrayLike,
        nu: ArrayLike,
        mode: str = "full",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ca.DM]:
        """
        Computes the parametric sensitivities (dx/dtheta, dlam/dtheta, dnu/dtheta)
        by evaluating the Implicit Function Theorem (IFT) on the KKT system.

        Parameters
        ----------
        theta : ArrayLike
            Numerical value for parameter theta (n_theta, 1).
        x : ArrayLike
            Primal decision variable vector at solution (n_x, 1).
        lam : ArrayLike
            Inequality dual multipliers at solution (n_ineq, 1).
        nu : ArrayLike
            Equality dual multipliers at solution (n_eq, 1).
        mode : str, default="full"
            Sensitivity computation mode:
            - "full": Differentiates the full KKT system directly (non-symmetric).
            - "active_set": Identifies active inequalities and solves a reduced, 
                            symmetric KKT system.
        options : Optional[Dict[str, Any]], optional
            Dictionary of options (e.g., {'tol': 1e-5} for active set thresholding).

        Returns
        -------
        Dict[str, ca.DM]
            Sensitivity matrices containing:
            - 'x'   : dx/dtheta matrix of shape (n_x, n_theta)
            - 'lam' : dlam/dtheta matrix of shape (n_ineq, n_theta)
            - 'nu'  : dnu/dtheta matrix of shape (n_eq, n_theta)
        """
        opts = options or {}
        tol: float = opts.get("tol", 1e-5)

        # 1. Evaluate IFT Jacobians & Hessians using precompiled functions
        H_xx_eval, H_xtheta_eval, J_gx_eval, J_gtheta_eval, J_hx_eval, J_htheta_eval = \
            self.funcs.ift(x, lam, nu, theta)

        g_eval = self.funcs.ineq(x, theta) if self.n_ineq > 0 else np.zeros((0, 1))

        # 2. Convert CasADi matrices to NumPy arrays for assembly and slicing
        H_xx_np = np.array(ca.DM(H_xx_eval).full())
        H_xtheta_np = np.array(ca.DM(H_xtheta_eval).full())

        J_gx_np = np.array(ca.DM(J_gx_eval).full())
        J_gtheta_np = np.array(ca.DM(J_gtheta_eval).full())

        J_hx_np = np.array(ca.DM(J_hx_eval).full())
        J_htheta_np = np.array(ca.DM(J_htheta_eval).full())

        g_np = np.array(ca.DM(g_eval).full()).flatten()
        lam_np = np.array(ca.DM(lam).full()).flatten()

        n_x, n_theta = self.n_x, self.n_theta
        n_ineq, n_eq = self.n_ineq, self.n_eq

        # 3. Assemble and Solve Linear System M * d = -N_rhs
        if mode == "full":
            N_full = n_x + n_ineq + n_eq
            M = np.zeros((N_full, N_full))
            N_rhs = np.zeros((N_full, n_theta))

            # Block 1: Stationarity
            M[:n_x, :n_x] = H_xx_np
            N_rhs[:n_x, :] = H_xtheta_np
            if n_ineq > 0:
                M[:n_x, n_x:n_x + n_ineq] = J_gx_np.T
            if n_eq > 0:
                M[:n_x, n_x + n_ineq:] = J_hx_np.T

            # Block 2: Complementary Slackness (lam * g = 0)
            if n_ineq > 0:
                M[n_x:n_x + n_ineq, :n_x] = np.diag(lam_np) @ J_gx_np
                M[n_x:n_x + n_ineq, n_x:n_x + n_ineq] = np.diag(g_np)
                N_rhs[n_x:n_x + n_ineq, :] = np.diag(lam_np) @ J_gtheta_np

            # Block 3: Equality Constraints
            if n_eq > 0:
                M[n_x + n_ineq:, :n_x] = J_hx_np
                N_rhs[n_x + n_ineq:, :] = J_htheta_np

            # Solve linear system
            # TODO: switch to scipy sparse?
            d_full = np.linalg.solve(M, -N_rhs)

            dx = d_full[:n_x, :]
            dlam = d_full[n_x:n_x + n_ineq, :] if n_ineq > 0 else np.zeros((0, n_theta))
            dnu = d_full[n_x + n_ineq:, :] if n_eq > 0 else np.zeros((0, n_theta))

        elif mode == "active_set":
            # Identify active inequalities (g <= 0, so g >= -tol means active)
            active_idx = np.where(g_np >= -tol)[0] if n_ineq > 0 else np.array([], dtype=int)
            n_act = len(active_idx)

            # Assemble reduced symmetric KKT system
            N_act = n_x + n_act + n_eq
            M = np.zeros((N_act, N_act))
            N_rhs = np.zeros((N_act, n_theta))

            # Block 1: Stationarity
            M[:n_x, :n_x] = H_xx_np
            N_rhs[:n_x, :] = H_xtheta_np

            # Block 2: Active Inequalities
            if n_act > 0:
                J_gx_act = J_gx_np[active_idx, :]
                M[:n_x, n_x:n_x + n_act] = J_gx_act.T
                M[n_x:n_x + n_act, :n_x] = J_gx_act
                N_rhs[n_x:n_x + n_act, :] = J_gtheta_np[active_idx, :]

            # Block 3: Equality Constraints
            if n_eq > 0:
                M[:n_x, n_x + n_act:] = J_hx_np.T
                M[n_x + n_act:, :n_x] = J_hx_np
                N_rhs[n_x + n_act:, :] = J_htheta_np

            # Solve symmetric linear system
            #TODO: will this be fast, or should I convert M to a sparse scipy type?
            d_act = solve(M, -N_rhs, assume_a='sym')

            dx = d_act[:n_x, :]
            dlam = np.zeros((n_ineq, n_theta))
            if n_act > 0:
                dlam[active_idx, :] = d_act[n_x:n_x + n_act, :]
            dnu = d_act[n_x + n_act:, :] if n_eq > 0 else np.zeros((0, n_theta))

        elif mode == "regularized":

            # get regularization parameters
            rho_1 = opts.get("rho_1")
            rho_2 = opts.get("rho_2")

            if rho_1 is None or rho_2 is None:
                raise ValueError("Options 'rho_1' and 'rho_2' must be provided for 'regularized' mode.")

            N_full = n_x + n_ineq + n_eq
            M = np.zeros((N_full, N_full))
            N_rhs = np.zeros((N_full, n_theta))

            # Block 1: Regularized Stationarity
            M[:n_x, :n_x] = H_xx_np + rho_1 * np.eye(n_x)
            N_rhs[:n_x, :] = H_xtheta_np
            if n_ineq > 0: 
                M[:n_x, n_x:n_x + n_ineq] = J_gx_np.T
            if n_eq > 0: 
                M[:n_x, n_x + n_ineq:] = J_hx_np.T
            
            # Block 2: Regularized Inequality Constraints
            if n_ineq > 0:
                M[n_x:n_x + n_ineq, :n_x] = J_gx_np
                
                # Compute denominator (gamma - rho * z) with safety clamping
                eps = 1e-10
                denom = lam_np - rho_1 * g_np
                sgn = np.where(denom < 0.0, -1.0, 1.0)
                denom = sgn * np.maximum(np.abs(denom), eps)
                
                # Compute diagonal modifier term
                diag_val = g_np / denom - rho_2
                M[n_x:n_x + n_ineq, n_x:n_x + n_ineq] = np.diag(diag_val)
                
                # Compute RHS multiplier
                rhs_mult = lam_np / denom
                N_rhs[n_x:n_x + n_ineq, :] = np.diag(rhs_mult) @ J_gtheta_np
                
            # Block 3: Regularized Equality Constraints
            if n_eq > 0:
                M[n_x + n_ineq:, :n_x] = J_hx_np
                M[n_x + n_ineq:, n_x + n_ineq:] = -rho_2 * np.eye(n_eq)
                N_rhs[n_x + n_ineq:, :] = J_htheta_np
                
            # Solve symmetric linear system using scipy
            #TODO: will this be fast, or should I convert M to a sparse scipy type?
            d_full = solve(M, -N_rhs, assume_a='sym')
            
            dx = d_full[:n_x, :]
            dlam = d_full[n_x:n_x + n_ineq, :] if n_ineq > 0 else np.zeros((0, n_theta))
            dnu = d_full[n_x + n_ineq:, :] if n_eq > 0 else np.zeros((0, n_theta))

        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'full', 'active_set', or 'regularized'.")

        return {
            "x": ca.DM(dx),
            "lam": ca.DM(dlam),
            "nu": ca.DM(dnu)
        }

    def check_properties(
        self,
        theta: "ArrayLike",
        x: "ArrayLike",
        lam: "ArrayLike",
        nu: "ArrayLike",
        active_tol: float = 1e-6,
        rank_tol: Optional[float] = None,
        pd_tol: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        Evaluates LICQ, SCS, SOSC and SSOSC at the primal-dual point (x, lam, nu).

        Works unmodified for both the base NLP and the augmented problem returned by
        `build_regularized_nlp`, since the latter is itself a DifferentiableNLP whose
        primal is xi, parameter is p_reg, inequality is z <= 0 and equality is h_reg.

        Sign convention follows the rest of the class: L = f + lam^T g + nu^T h with
        g(x) <= 0 and lam >= 0.

        Parameters
        ----------
        active_tol : float
            Threshold on g for declaring an inequality active, and on lam for
            declaring it strongly active.
        rank_tol : Optional[float]
            Passed to `np.linalg.matrix_rank`. None uses NumPy's scale-aware default
            (max(M, N) * eps * sigma_max), which is what you want unless you have a
            specific absolute threshold in mind.
        pd_tol : float
            Relative threshold on the smallest eigenvalue of the reduced Hessian.

        Returns
        -------
        Dict[str, Any]
            licq, scs, sosc, ssosc (bool), plus diagnostics: n_active,
            n_weakly_active, sosc_exact, and kkt_residual.
        """
        # --- 1. Reuse the eagerly compiled ift function -------------------------
        H_L_dm, _, J_g_dm, _, J_h_dm, _ = self.funcs.ift(x, theta, lam, nu)

        H_L = np.array(ca.DM(H_L_dm).full()).reshape(self.n_x, self.n_x)
        J_g = np.array(ca.DM(J_g_dm).full()).reshape(self.n_ineq, self.n_x)
        J_h = np.array(ca.DM(J_h_dm).full()).reshape(self.n_eq, self.n_x)

        g_np = np.array(ca.DM(self.funcs.ineq(x, theta)).full()).flatten()
        lam_np = np.array(ca.DM(lam).full()).flatten()

        # --- 2. Guard: these conditions are only meaningful at a KKT point ------
        stat, _, eq, _, dual_feas = self.funcs.kkt(theta, x, lam, nu)
        kkt_res = float(max(
            np.max(np.abs(np.array(ca.DM(stat).full()))) if self.n_x else 0.0,
            np.max(np.abs(np.array(ca.DM(eq).full()))) if self.n_eq else 0.0,
            np.max(np.abs(np.array(ca.DM(dual_feas).full()))) if self.n_ineq else 0.0,
        ))

        # --- 3. Active / strongly active / weakly active sets -------------------
        active = np.where(g_np >= -active_tol)[0]
        strict = active[lam_np[active] > active_tol] if active.size else active
        weak = np.setdiff1d(active, strict)

        def stacked(idx: np.ndarray) -> np.ndarray:
            return np.vstack([J_h, J_g[idx, :]])  # both are (k, n_x), empty is fine

        J_act = stacked(active)
        J_str = stacked(strict)

        # --- 4. LICQ: active constraint gradients linearly independent ----------
        licq = True
        if J_act.shape[0] > 0:
            licq = bool(np.linalg.matrix_rank(J_act, tol=rank_tol) == J_act.shape[0])

        # --- 5. SCS: no weakly active inequality --------------------------------
        scs = bool(weak.size == 0)

        # --- 6. Second order conditions -----------------------------------------
        def pd_on_nullspace(H: np.ndarray, J: np.ndarray) -> bool:
            Z = np.eye(H.shape[0]) if J.shape[0] == 0 else null_space(J, rcond=rank_tol)
            if Z.shape[1] == 0:
                return True  # vacuously: the critical subspace is trivial
            H_red = Z.T @ H @ Z
            H_red = 0.5 * (H_red + H_red.T)
            scale = max(1.0, float(np.max(np.abs(H_red))))
            return bool(np.min(np.linalg.eigvalsh(H_red)) > pd_tol * scale)

        # SSOSC drops the weakly active rows, so null(J_str) contains null(J_act):
        # it is the strictly stronger test, and SSOSC => SOSC.
        sosc_subspace = pd_on_nullspace(H_L, J_act)
        ssosc = pd_on_nullspace(H_L, J_str)

        # The true SOSC critical cone is
        #   C = {d : J_h d = 0, J_g[i] d = 0 for strongly active i,
        #                       J_g[i] d <= 0 for weakly active i}
        # which is a cone, not a subspace. null(J_act) is a strict subset of C
        # whenever a weakly active constraint exists, so the subspace test is
        # optimistic there. It is exact iff SCS holds; SSOSC is a rigorous
        # sufficient certificate in all cases.
        sosc = bool(ssosc or sosc_subspace)
        sosc_exact = bool(scs or ssosc)

        return {
            "licq": licq,
            "scs": scs,
            "sosc": sosc,
            "ssosc": ssosc,
            "sosc_exact": sosc_exact,
            "n_active": int(active.size),
            "n_weakly_active": int(weak.size),
            "kkt_residual": kkt_res,
        }

    def annotate_solution(self, sol: "NLPSolution", **kwargs) -> Dict[str, Any]:
        """Thin wrapper: evaluate properties at `sol` and store them on it."""
        props = self.check_properties(sol.theta, sol.x, sol.lam, sol.nu, **kwargs)
        if sol.properties is None:
            sol.properties = {}
        sol.properties.update(props)
        return props