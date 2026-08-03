import casadi as ca
import numpy as np
from typing import Dict, Optional, Tuple, Union

from dataclasses import dataclass
from typing import Dict, Optional, Any
import casadi as ca
import numpy as np
import scipy.linalg

@dataclass
class NLPSolution:
    x: ca.DM
    theta: ca.DM
    lam: ca.DM
    nu: ca.DM
    kkt_residuals: Dict[str, ca.DM]
    properties: Optional[Dict[str, bool]] = None
    raw: Optional[Dict[str, Any]] = None

# Generic type hints for CasADi and NumPy array compatibility
ArrayLike = Union[np.ndarray, ca.DM, ca.MX, ca.SX]
Symbolic = Union[ca.MX, ca.SX]


class DifferentiableNLP:
    """
    A class to solve and differentiate non-linear programs (NLPs) in CasADi.
    
    Utilizes the Implicit Function Theorem (IFT) on the KKT system to compute 
    solution sensitivities (dx/dtheta). Supports a custom Tikhonov-like 
    regularization to ensure differentiability even when strict complementarity 
    or constraint qualifications (LICQ) fail.
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
        Constructs the unregularized and regularized NLP solvers.

        Parameters
        ----------
        x : Symbolic
            Symbolic decision variables vector of shape (n_x, 1).
        theta : Symbolic
            Symbolic parameters vector of shape (n_theta, 1).
        f : Symbolic
            Scalar objective function expression f(x, theta).
        g : Optional[Symbolic], optional
            Inequality constraint expressions g(x, theta) <= 0 of shape (n_in, 1).
        h : Optional[Symbolic], optional
            Equality constraint expressions h(x, theta) == 0 of shape (n_eq, 1).
        solver_opts : Optional[Dict], optional
            Dictionary of options passed to `casadi.nlpsol` (e.g., IPOPT settings).
        """
        self.x = x
        self.theta = theta
        self.f = f

        # Default to empty symbolic vectors if constraints are not provided
        self.g = g if g is not None else ca.MX(0, 1)
        self.h = h if h is not None else ca.MX(0, 1)

        # store functions
        self._f_func = ca.Function('f_func', [self.x, self.theta], [self.f])
        self._g_func = ca.Function('g_func', [self.x, self.theta], [self.g])
        self._h_func = ca.Function('h_func', [self.x, self.theta], [self.h])

        # Dimensions
        self.n_x = self.x.size1()
        self.n_theta = self.theta.size1()
        self.n_g = self.g.size1()
        self.n_h = self.h.size1()

        self.opts = solver_opts if solver_opts is not None else {}
        self.plugin = self.opts.pop("plugin","ipopt")

        # ------------------------------------------------------------------
        # 1. Original NLP Solver: P_1(theta)
        # ------------------------------------------------------------------
        # Stack constraints: c = [g; h]
        self.c_orig = ca.vertcat(self.g, self.h)
        self.lbc_orig = np.concatenate([-np.inf * np.ones(self.n_g), np.zeros(self.n_h)])
        self.ubc_orig = np.concatenate([np.zeros(self.n_g), np.zeros(self.n_h)])

        self._nlp_orig = {
            'x': self.x,
            'p': self.theta,
            'f': self.f,
            'g': self.c_orig
        }
        self.solver_orig = None

        # ------------------------------------------------------------------
        # 2. Regularized Primal-Dual Solver: P_2(theta, phi_bar)
        # ------------------------------------------------------------------
        # Symbolic parameters for P_2: rho and anchor points phi_bar = (x_bar, lam_bar, nu_bar)
        self.rho = ca.MX.sym('rho')
        self.x_bar = ca.MX.sym('x_bar', self.n_x)
        self.lam_bar = ca.MX.sym('lam_bar', self.n_g)
        self.nu_bar = ca.MX.sym('nu_bar', self.n_h)

        # Decision variables of P_2: xi = (x, lambda, nu, z)
        self.lam_reg = ca.MX.sym('lam_reg', self.n_g)
        self.nu_reg = ca.MX.sym('nu_reg', self.n_h)
        self.z_reg = ca.MX.sym('z_reg', self.n_g)

        self.xi = ca.vertcat(self.x, self.lam_reg, self.nu_reg, self.z_reg)
        self.n_xi = self.xi.size1()

        # Evaluate g_theta(x_bar) for penalty term
        g_x_bar = ca.substitute(self.g, self.x, self.x_bar)

        # Regularized Objective f_reg(xi)
        self.f_reg = self.f + (self.rho / 2.0) * (
            ca.sumsqr(self.x - self.x_bar) +
            ca.sumsqr(self.lam_reg) +
            ca.sumsqr(self.nu_reg) +
            ca.sumsqr(self.z_reg - g_x_bar)
        )

        # Constraints for P_2:
        # Eq 1: g_theta(x) - z + rho*(lam_bar - lambda) == 0
        # Eq 2: h_theta(x) + rho*(nu_bar - nu) == 0
        self.c_reg_eq = ca.vertcat(
            self.g - self.z_reg + self.rho * (self.lam_bar - self.lam_reg),
            self.h + self.rho * (self.nu_bar - self.nu_reg)
        )

        # Bounds on decision variable xi = (x, lambda, nu, z)
        # Note: z <= 0
        self.lb_xi = np.concatenate([
            -np.inf * np.ones(self.n_x + self.n_g + self.n_h),
            -np.inf * np.ones(self.n_g)
        ])
        self.ub_xi = np.concatenate([
            np.inf * np.ones(self.n_x + self.n_g + self.n_h),
            np.zeros(self.n_g)  # z <= 0
        ])

        # Bounds on equality constraints (all 0)
        self.lbc_reg = np.zeros(self.n_g + self.n_h)
        self.ubc_reg = np.zeros(self.n_g + self.n_h)

        # Augmented parameter vector: p_reg = [theta; rho; x_bar; lam_bar; nu_bar]
        self.p_reg = ca.vertcat(self.theta, self.rho, self.x_bar, self.lam_bar, self.nu_bar)

        self._nlp_reg = {
            'x': self.xi,
            'p': self.p_reg,
            'f': self.f_reg,
            'g': self.c_reg_eq
        }
        self.solver_reg = None

        # ------------------------------------------------------------------
        # 3. Lazy Initialization States
        # ------------------------------------------------------------------
        self._differentiator_built = False
        self._differentiator_func = None
        self._kkt_func = None

    def solve(
        self,
        theta: ArrayLike,
        x_warmstart: Optional[ArrayLike] = None,
        lam_warmstart: Optional[ArrayLike] = None,
        nu_warmstart: Optional[ArrayLike] = None
    ) -> NLPSolution:
        """
        Solves the original unregularized NLP P_1(theta).

        Parameters
        ----------
        theta : ArrayLike
            Numerical value for parameter theta.
        x_warmstart : Optional[ArrayLike], optional
            Warmstart for primal variable x.
        lam_warmstart : Optional[ArrayLike], optional
            Warmstart for inequality multipliers.
        nu_warmstart : Optional[ArrayLike], optional
            Warmstart for equality multipliers.

        Returns
        -------
        NLPSolution
            Object containing the solution of the problem, its kkt
            residual, and some properties.
        """
        if self.solver_orig is None:
            self.solver_orig = ca.nlpsol('solver_orig', self.plugin, self._nlp_orig, self.opts)

        args = {
            'p': theta,
            'lbg': self.lbc_orig,
            'ubg': self.ubc_orig
        }

        # Handle warmstarts
        if x_warmstart is not None:
            args['x0'] = x_warmstart
            
        if lam_warmstart is not None or nu_warmstart is not None:
            # Handle partial warmstarts by filling missing ones with zeros
            lam = lam_warmstart if lam_warmstart is not None else np.zeros(self.n_g)
            nu = nu_warmstart if nu_warmstart is not None else np.zeros(self.n_h)
            
            # nlpsol expects a single vector for all constraint multipliers
            args['lam_g0'] = ca.vertcat(lam, nu)

        # Solve
        res = self.solver_orig(**args)

        # Extract constraint multipliers
        lam_c = res['lam_g']
        lam_opt = lam_c[:self.n_g]
        nu_opt = lam_c[self.n_g:]

        # Automatically evaluate KKT residuals for the solution
        kkt_res = self.kkt_eval(theta, res["x"], lam_opt, nu_opt)

        return NLPSolution(
            x=res["x"],
            theta=ca.DM(theta),
            lam=lam_opt,
            nu=nu_opt,
            kkt_residuals=kkt_res,
            properties={}
        )

    def solve_reg(
        self,
        theta: ArrayLike,
        rho: float,
        x_bar: ArrayLike,
        lam_bar: ArrayLike,
        nu_bar: ArrayLike
    ) -> NLPSolution:
        """
        Solves the regularized primal-dual NLP P_2(theta, phi_bar).

        Automatically initializes decision vector xi = (x, lambda, nu, z)
        using the provided anchor variables:
          - x_0 = x_bar
          - lambda_0 = lam_bar
          - nu_0 = nu_bar
          - z_0 = g(x_bar, theta)

        Parameters
        ----------
        theta : ArrayLike
            Numerical parameter theta.
        rho : float
            Regularization factor rho > 0.
        x_bar : ArrayLike
            Anchor point x_bar.
        lam_bar : ArrayLike
            Anchor point lam_bar.
        nu_bar : ArrayLike
            Anchor point nu_bar.

        Returns
        -------
        NLPSolution
            Object containing the solution of the problem, its kkt
            residual, and some properties. It additionally contains
            (in the 'raw' key)
            - 'xi': Full decision vector (x, lambda, nu, z).
            - 'psi': Dual multiplier for modified inequality constraint.
            - 'mu': Dual multiplier for modified equality constraint.
            - 'gamma': Dual multiplier for z <= 0.
        """
        if self.solver_reg is None:
            self.solver_reg = ca.nlpsol('solver_reg', self.plugin, self._nlp_reg, self.opts)

        # 1. Automatic Warmstart Construction
        # Evaluate z_0 = g(x_bar, theta)
        z_bar = self._g_func(x_bar, theta)
        
        # Initial guess xi_0 = [x_bar; lam_bar; nu_bar; z_bar]
        xi_0 = ca.vertcat(x_bar, lam_bar, nu_bar, z_bar)

        # 2. Pack Augmented Parameters: p_reg = [theta; rho; x_bar; lam_bar; nu_bar]
        p = ca.vertcat(theta, rho, x_bar, lam_bar, nu_bar)

        args = {
            'x0': xi_0,
            'p': p,
            'lbx': self.lb_xi,
            'ubx': self.ub_xi,
            'lbg': self.lbc_reg,
            'ubg': self.ubc_reg
        }

        # 3. Solve P_2
        res = self.solver_reg(**args)
        xi_opt = res['x']

        # 4. Extract Primal & Multiplier components from xi
        idx_x = self.n_x
        idx_lam = idx_x + self.n_g
        idx_nu = idx_lam + self.n_h

        x_opt = xi_opt[:idx_x]
        lam_opt = xi_opt[idx_x:idx_lam]
        nu_opt = xi_opt[idx_lam:idx_nu]

        # 5. Extract P_2 Dual Multipliers
        lam_c = res['lam_g']
        psi_opt = lam_c[:self.n_g]
        mu_opt = lam_c[self.n_g:]

        # Gamma is multiplier for z <= 0, located in lam_x at the tail of xi
        lam_xi = res['lam_x']
        gamma_opt = lam_xi[-self.n_g:]

        # Store internal state required for property checks
        raw_state = {
            'xi': xi_opt,
            'psi': psi_opt,
            'mu': mu_opt,
            'gamma': gamma_opt,
            'p_reg': p
        }

        # Evaluate the KKT residuals of the original problem at this regularized point
        kkt_res = self.kkt_eval(theta, x_opt, lam_opt, nu_opt)

        return NLPSolution(
            x=x_opt,
            theta=ca.DM(theta),
            lam=lam_opt,
            nu=nu_opt,
            kkt_residuals=kkt_res,
            properties={},
            raw=raw_state
        )

    def kkt_eval(
        self,
        theta: ArrayLike,
        x: ArrayLike,
        lam: ArrayLike,
        nu: ArrayLike
    ) -> Dict[str, ca.DM]:
        """
        Evaluates the KKT residual vectors of the original, unregularized NLP.

        Lazily builds a CasADi function to compute the exact KKT conditions:
        1. Stationarity: ∇_x L(x, λ, ν, θ) = 0
        2. Inequality: g(x, θ) <= 0
        3. Equality: h(x, θ) = 0
        4. Complementary Slackness: λ ⊙ g(x, θ) = 0
        5. Dual Feasibility: λ >= 0

        Parameters
        ----------
        theta : ArrayLike
            Parameter vector theta.
        x : ArrayLike
            Primal variable vector x.
        lam : ArrayLike
            Dual multipliers for inequality constraints g.
        nu : ArrayLike
            Dual multipliers for equality constraints h.

        Returns
        -------
        Dict[str, ca.DM]
            Dictionary containing the numerical evaluation of KKT vectors:
            - 'stationarity': Gradient of Lagrangian w.r.t. x (shape: n_x x 1).
            - 'inequality': Evaluated inequality constraints g(x, theta) (shape: n_g x 1).
            - 'equality': Evaluated equality constraints h(x, theta) (shape: n_h x 1).
            - 'slackness': Element-wise product lam * g(x, theta) (shape: n_g x 1).
            - 'dual_feasibility': Evaluated inequality multipliers lam (shape: n_g x 1).
        """
        if self._kkt_func is None:
            # Create local symbolic variables for the multipliers
            lam_sym = ca.MX.sym('lam', self.n_g)
            nu_sym = ca.MX.sym('nu', self.n_h)

            # Construct the Lagrangian L(x, lam, nu, theta)
            # L = f + lam^T * g + nu^T * h
            lagrangian = self.f + ca.dot(lam_sym, self.g) + ca.dot(nu_sym, self.h)

            # 1. Stationarity (Gradient of Lagrangian w.r.t x)
            # ca.jacobian returns a row vector (1 x n_x), so we transpose it
            stationarity = ca.jacobian(lagrangian, self.x).T

            # 2. Inequality: just g(x, theta)
            inequality = ca.fmax(self.g,0.0)

            # 3. Equality: just h(x, theta)
            equality = self.h

            # 4. Complementary Slackness: element-wise multiplication
            slackness = lam_sym * self.g

            # 5. Dual Feasibility: just the inequality multipliers
            dual_feasibility = ca.fmin(lam_sym,0.0)

            # Compile into a CasADi Function with named inputs and outputs
            self._kkt_func = ca.Function(
                'kkt_eval',
                [self.theta, self.x, lam_sym, nu_sym],               # Inputs
                [stationarity, inequality, equality, slackness, dual_feasibility], # Outputs
                ['theta', 'x', 'lam', 'nu'],                         # Input names
                ['stationarity', 'inequality', 'equality', 'slackness', 'dual_feasibility'] # Output names
            )

        # Evaluate the compiled function
        # Calling with kwargs returns a dictionary mapped to the output names
        res = self._kkt_func(
            theta=theta, 
            x=x, 
            lam=lam, 
            nu=nu
        )

        return res

    def check_properties(self, sol: NLPSolution, is_regularized: bool = False, tol: float = 1e-5) -> Dict[str, bool]:
        """
        Checks LICQ, SCS, SOSC, and SSOSC for an NLP solution.
        Converts CasADi DM objects to numpy arrays for linear algebra operations.
        """
        # ------------------------------------------------------------------
        # 1. Lazily Evaluate Jacobians and Hessians via CasADi
        # ------------------------------------------------------------------
        if not is_regularized:
            if getattr(self, '_prop_func_orig', None) is None:
                lam_sym = ca.MX.sym('lam', self.n_g)
                nu_sym = ca.MX.sym('nu', self.n_h)
                
                L = self.f + ca.dot(lam_sym, self.g) + ca.dot(nu_sym, self.h)
                
                J_g = ca.jacobian(self.g, self.x)
                J_h = ca.jacobian(self.h, self.x)
                H_L = ca.hessian(L, self.x)[0]
                
                self._prop_func_orig = ca.Function(
                    'prop_orig', [self.x, self.theta, lam_sym, nu_sym], [self.g, J_g, J_h, H_L]
                )
                
            g_eval, J_g_eval, J_h_eval, H_L_eval = self._prop_func_orig(sol.x, sol.theta, sol.lam, sol.nu)
            lam_eval = sol.lam
            
        else:
            if getattr(self, '_prop_func_reg', None) is None:
                psi_sym = ca.MX.sym('psi', self.n_g)
                nu_sym = ca.MX.sym('nu', self.n_h)
                gamma_sym = ca.MX.sym('gamma', self.n_g)
                
                L_reg = self.f_reg + ca.dot(psi_sym, self.c_reg_eq[:self.n_g]) + \
                        ca.dot(nu_sym, self.c_reg_eq[self.n_g:]) + ca.dot(gamma_sym, self.z_reg)
                        
                J_g = ca.jacobian(self.z_reg, self.xi)  # P2 inequalities: z <= 0
                J_h = ca.jacobian(self.c_reg_eq, self.xi) # P2 equalities
                H_L = ca.hessian(L_reg, self.xi)[0]
                
                self._prop_func_reg = ca.Function(
                    'prop_reg', [self.xi, self.p_reg, psi_sym, nu_sym, gamma_sym], [self.z_reg, J_g, J_h, H_L]
                )
            
            raw = sol.raw
            g_eval, J_g_eval, J_h_eval, H_L_eval = self._prop_func_reg(
                raw['xi'], raw['p_reg'], raw['psi'], raw['nu'], raw['gamma']
            )
            lam_eval = raw['gamma']

        # ------------------------------------------------------------------
        # 2. Convert to numpy for analysis
        # ------------------------------------------------------------------
        g_np = np.array(g_eval.full()).flatten()
        J_g_np = np.array(J_g_eval.full())
        J_h_np = np.array(J_h_eval.full())
        H_L_np = np.array(H_L_eval.full())
        lam_np = np.array(lam_eval.full()).flatten()
        
        # 1. Identify Active Inequality Constraints (g(x) >= -tol, since bounds are g <= 0)
        active_indices = np.where(g_np >= -tol)[0]
        
        # 2. LICQ (Linear Independence Constraint Qualification)
        J_g_act = J_g_np[active_indices, :] if len(active_indices) > 0 else np.empty((0, J_g_np.shape[1]))
        J_act = np.vstack([J_h_np, J_g_act]) if J_h_np.size > 0 else J_g_act
        
        if J_act.shape[0] > 0:
            rank = np.linalg.matrix_rank(J_act, tol=tol)
            licq = bool(rank == J_act.shape[0])
        else:
            licq = True
            
        # 3. SCS (Strict Complementary Slackness)
        if len(active_indices) > 0:
            lam_act = lam_np[active_indices]
            scs = bool(np.all(lam_act > tol))
            strict_indices = active_indices[lam_act > tol]
        else:
            scs = True
            strict_indices = np.array([], dtype=int)
            
        # Helper for checking positive definiteness on nullspace
        def check_pd_nullspace(H, J):
            if J.shape[0] == 0:
                Z = np.eye(H.shape[0])
            else:
                Z = scipy.linalg.null_space(J)
            if Z.shape[1] == 0:
                return True
            H_red = Z.T @ H @ Z
            # Using eigh since reduced Hessian is symmetric
            eigenvalues = np.linalg.eigvalsh((H_red + H_red.T) / 2) 
            return bool(np.min(eigenvalues) > tol)

        # 4. SOSC (Second Order Sufficient Condition using ALL active constraints)
        sosc = check_pd_nullspace(H_L_np, J_act)
        
        # 5. SSOSC (Strong Second Order Sufficient Condition using STRICTLY active constraints)
        J_g_strict = J_g_np[strict_indices, :] if len(strict_indices) > 0 else np.empty((0, J_g_np.shape[1]))
        J_strict = np.vstack([J_h_np, J_g_strict]) if J_h_np.size > 0 else J_g_strict
        ssosc = check_pd_nullspace(H_L_np, J_strict)

        # Update the properties dict in place
        props = {
            'licq': licq,
            'scs': scs,
            'sosc': sosc,
            'ssosc': ssosc
        }
        if sol.properties is None:
            sol.properties = {}
        sol.properties.update(props)
        
        return props

    def differentiate(
        self,
        theta: ca.DM,
        x: ca.DM,
        lam: ca.DM,
        nu: ca.DM,
        mode: str = "full",
        options: Optional[Dict] = None
    ) -> Dict[str, ca.DM]:
        """
        Computes the parametric sensitivities (dx/dtheta, dlam/dtheta, dnu/dtheta)
        by differentiating the KKT conditions.
        
        Modes:
        - "full": Differentiates the full KKT system directly (non-symmetric).
        - "active_set": Assumes strict complementary slackness, drops inactive 
                        inequalities, and solves a symmetric KKT system.
        """
        options = options or {}
        tol = options.get("tol", 1e-5)
        
        # ------------------------------------------------------------------
        # 1. Lazily Evaluate Jacobians and Hessians via CasADi
        # ------------------------------------------------------------------
        if getattr(self, '_diff_func_orig', None) is None:
            lam_sym = ca.MX.sym('lam', self.n_g)
            nu_sym = ca.MX.sym('nu', self.n_h)
            
            # Lagrangian: L = f(x, theta) + lam^T g(x, theta) + nu^T h(x, theta)
            # Note: We safely handle cases where g or h might be empty
            L = self.f
            if self.n_g > 0: L += ca.dot(lam_sym, self.g)
            if self.n_h > 0: L += ca.dot(nu_sym, self.h)
            
            # Derivatives w.r.t x
            H_xx = ca.hessian(L, self.x)[0]
            J_gx = ca.jacobian(self.g, self.x) if self.n_g > 0 else ca.MX(0, self.n_x)
            J_hx = ca.jacobian(self.h, self.x) if self.n_h > 0 else ca.MX(0, self.n_x)
            
            # Cross derivatives w.r.t theta
            dL_dx = ca.jacobian(L, self.x).T
            H_xtheta = ca.jacobian(dL_dx, self.theta)
            J_gtheta = ca.jacobian(self.g, self.theta) if self.n_g > 0 else ca.MX(0, self.n_theta)
            J_htheta = ca.jacobian(self.h, self.theta) if self.n_h > 0 else ca.MX(0, self.n_theta)
            
            self._diff_func_orig = ca.Function(
                'diff_orig', 
                [self.x, self.theta, lam_sym, nu_sym], 
                [self.g if self.n_g > 0 else ca.MX(0,1), 
                 H_xx, H_xtheta, J_gx, J_gtheta, J_hx, J_htheta]
            )
            
        # Evaluate numerical matrices
        g_eval, H_xx_eval, H_xtheta_eval, J_gx_eval, J_gtheta_eval, J_hx_eval, J_htheta_eval = \
            self._diff_func_orig(x, theta, lam, nu)
            
        # ------------------------------------------------------------------
        # 2. Convert to NumPy for assembly and slicing
        # ------------------------------------------------------------------
        g_np = np.array(g_eval.full()).flatten()
        lam_np = np.array(lam.full()).flatten()
        
        H_xx_np = np.array(H_xx_eval.full())
        H_xtheta_np = np.array(H_xtheta_eval.full())
        
        J_gx_np = np.array(J_gx_eval.full())
        J_gtheta_np = np.array(J_gtheta_eval.full())
        
        J_hx_np = np.array(J_hx_eval.full())
        J_htheta_np = np.array(J_htheta_eval.full())

        n_x, n_theta = self.n_x, self.n_theta
        n_g, n_h = self.n_g, self.n_h

        # ------------------------------------------------------------------
        # 3. Assemble and Solve Linear System
        # ------------------------------------------------------------------
        if mode == "full":
            # Assemble non-symmetric KKT matrix
            N_full = n_x + n_g + n_h
            M = np.zeros((N_full, N_full))
            N_rhs = np.zeros((N_full, n_theta))
            
            # Block 1: Stationarity
            M[:n_x, :n_x] = H_xx_np
            N_rhs[:n_x, :] = H_xtheta_np
            if n_g > 0: M[:n_x, n_x:n_x+n_g] = J_gx_np.T
            if n_h > 0: M[:n_x, n_x+n_g:] = J_hx_np.T
            
            # Block 2: Complementary Slackness (lam * g = 0) -> diag(lam)*J_gx*dx + diag(g)*dlam
            if n_g > 0:
                M[n_x:n_x+n_g, :n_x] = np.diag(lam_np) @ J_gx_np
                M[n_x:n_x+n_g, n_x:n_x+n_g] = np.diag(g_np)
                N_rhs[n_x:n_x+n_g, :] = np.diag(lam_np) @ J_gtheta_np
                
            # Block 3: Equality Constraints
            if n_h > 0:
                M[n_x+n_g:, :n_x] = J_hx_np
                N_rhs[n_x+n_g:, :] = J_htheta_np
                
            # Solve M * d = -N_rhs
            d_full = np.linalg.solve(M, -N_rhs)
            
            dx = d_full[:n_x, :]
            dlam = d_full[n_x:n_x+n_g, :] if n_g > 0 else np.zeros((0, n_theta))
            dnu = d_full[n_x+n_g:, :] if n_h > 0 else np.zeros((0, n_theta))

        elif mode == "active_set":
            # Identify active inequalities (g <= 0, so g >= -tol means active)
            active_idx = np.where(g_np >= -tol)[0] if n_g > 0 else np.array([], dtype=int)
            n_act = len(active_idx)
            
            # Assemble symmetric KKT matrix
            N_act = n_x + n_act + n_h
            M = np.zeros((N_act, N_act))
            N_rhs = np.zeros((N_act, n_theta))
            
            # Block 1: Stationarity
            M[:n_x, :n_x] = H_xx_np
            N_rhs[:n_x, :] = H_xtheta_np
            
            # Block 2: Active Inequalities (g_act = 0)
            if n_act > 0:
                J_gx_act = J_gx_np[active_idx, :]
                M[:n_x, n_x:n_x+n_act] = J_gx_act.T
                M[n_x:n_x+n_act, :n_x] = J_gx_act
                N_rhs[n_x:n_x+n_act, :] = J_gtheta_np[active_idx, :]
                
            # Block 3: Equality Constraints
            if n_h > 0:
                M[:n_x, n_x+n_act:] = J_hx_np.T
                M[n_x+n_act:, :n_x] = J_hx_np
                N_rhs[n_x+n_act:, :] = J_htheta_np
                
            # Solve M * d = -N_rhs (Symmetric, could use scipy.linalg.solve with assume_a='sym')
            d_act = np.linalg.solve(M, -N_rhs)
            
            # Extract and map back to full dimensions
            dx = d_act[:n_x, :]
            
            dlam = np.zeros((n_g, n_theta))
            if n_act > 0:
                dlam[active_idx, :] = d_act[n_x:n_x+n_act, :]
                
            dnu = d_act[n_x+n_act:, :] if n_h > 0 else np.zeros((0, n_theta))

        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'full' or 'active_set'.")

        return {
            "dx": ca.DM(dx),
            "dlam": ca.DM(dlam),
            "dnu": ca.DM(dnu)
        }

    def differentiate_reg(
        self,
        theta: ca.DM,
        x: ca.DM,
        lam: ca.DM,
        nu: ca.DM,
        rho: float,
        options: Optional[Dict] = None
    ) -> Dict[str, ca.DM]:
        """
        Computes the parametric sensitivities (dx/dtheta, dlam/dtheta, dnu/dtheta)
        for the regularized problem by solving the corresponding regularized KKT system.
        """
        options = options or {}
        
        # ------------------------------------------------------------------
        # 1. Lazily Evaluate Jacobians and Hessians of the Base Lagrangian
        # ------------------------------------------------------------------
        if getattr(self, '_diff_func_orig', None) is None:
            lam_sym = ca.MX.sym('lam', self.n_g)
            nu_sym = ca.MX.sym('nu', self.n_h)
            
            L = self.f
            if self.n_g > 0: L += ca.dot(lam_sym, self.g)
            if self.n_h > 0: L += ca.dot(nu_sym, self.h)
            
            H_xx = ca.hessian(L, self.x)[0]
            J_gx = ca.jacobian(self.g, self.x) if self.n_g > 0 else ca.MX(0, self.n_x)
            J_hx = ca.jacobian(self.h, self.x) if self.n_h > 0 else ca.MX(0, self.n_x)
            
            dL_dx = ca.jacobian(L, self.x).T
            H_xtheta = ca.jacobian(dL_dx, self.theta)
            J_gtheta = ca.jacobian(self.g, self.theta) if self.n_g > 0 else ca.MX(0, self.n_theta)
            J_htheta = ca.jacobian(self.h, self.theta) if self.n_h > 0 else ca.MX(0, self.n_theta)
            
            self._diff_func_orig = ca.Function(
                'diff_orig', 
                [self.x, self.theta, lam_sym, nu_sym], 
                [self.g if self.n_g > 0 else ca.MX(0,1), 
                 H_xx, H_xtheta, J_gx, J_gtheta, J_hx, J_htheta]
            )
            
        g_eval, H_xx_eval, H_xtheta_eval, J_gx_eval, J_gtheta_eval, J_hx_eval, J_htheta_eval = \
            self._diff_func_orig(x, theta, lam, nu)
            
        # ------------------------------------------------------------------
        # 2. Convert to NumPy arrays
        # ------------------------------------------------------------------
        z_np = np.array(g_eval.full()).flatten()      # z = g(x)
        gamma_np = np.array(lam.full()).flatten()     # gamma = lam
        
        H_xx_np = np.array(H_xx_eval.full())
        H_xtheta_np = np.array(H_xtheta_eval.full())
        
        J_gx_np = np.array(J_gx_eval.full())
        J_gtheta_np = np.array(J_gtheta_eval.full())
        
        J_hx_np = np.array(J_hx_eval.full())
        J_htheta_np = np.array(J_htheta_eval.full())

        n_x, n_theta = self.n_x, self.n_theta
        n_g, n_h = self.n_g, self.n_h

        # ------------------------------------------------------------------
        # 3. Assemble and Solve the Regularized Linear System
        # ------------------------------------------------------------------
        N_full = n_x + n_g + n_h
        M = np.zeros((N_full, N_full))
        N_rhs = np.zeros((N_full, n_theta))
        
        # Block 1: Stationarity (Row 1)
        # H_xx * dx + J_gx^T * dlam + J_hx^T * dnu = -H_xtheta
        M[:n_x, :n_x] = H_xx_np
        N_rhs[:n_x, :] = H_xtheta_np
        if n_g > 0: M[:n_x, n_x:n_x+n_g] = J_gx_np.T
        if n_h > 0: M[:n_x, n_x+n_g:] = J_hx_np.T
        
        # Block 2: Inequality Constraints (Row 2)
        # J_gx * dx + diag(...) * dlam = -diag(...) * J_gtheta
        if n_g > 0:
            M[n_x:n_x+n_g, :n_x] = J_gx_np
            
            # Compute denominator (gamma - rho * z) with safety clamping
            denom = gamma_np - rho * z_np
            denom[np.abs(denom) < 1e-14] = 1e-14
            
            # Compute diagonal modifier term
            diag_val = (z_np - rho * gamma_np + (rho**2) * z_np) / denom
            M[n_x:n_x+n_g, n_x:n_x+n_g] = np.diag(diag_val)
            
            # Compute RHS multiplier
            rhs_mult = gamma_np / denom
            N_rhs[n_x:n_x+n_g, :] = np.diag(rhs_mult) @ J_gtheta_np
            
        # Block 3: Equality Constraints (Row 3)
        # J_hx * dx - rho * I * dnu = -J_htheta
        if n_h > 0:
            M[n_x+n_g:, :n_x] = J_hx_np
            M[n_x+n_g:, n_x+n_g:] = -rho * np.eye(n_h)
            N_rhs[n_x+n_g:, :] = J_htheta_np
            
        # Solve M * d = -N_rhs
        d_full = np.linalg.solve(M, -N_rhs)
        
        # Extract sensitivities
        dx = d_full[:n_x, :]
        dlam = d_full[n_x:n_x+n_g, :] if n_g > 0 else np.zeros((0, n_theta))
        dnu = d_full[n_x+n_g:, :] if n_h > 0 else np.zeros((0, n_theta))

        return {
            "dx": ca.DM(dx),
            "dlam": ca.DM(dlam),
            "dnu": ca.DM(dnu)
        }