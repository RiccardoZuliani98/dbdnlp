import casadi as ca
import numpy as np
from typing import Dict, Optional, Tuple, Union

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

        self.xi = ca.vertcat(self.x, self.z_reg, self.lam_reg, self.nu_reg)
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
        lam_g_warmstart: Optional[ArrayLike] = None,
        lam_h_warmstart: Optional[ArrayLike] = None
    ) -> Dict[str, ca.DM]:
        """
        Solves the original unregularized NLP P_1(theta).

        Parameters
        ----------
        theta : ArrayLike
            Numerical value for parameter theta.
        x_warmstart : Optional[ArrayLike], optional
            Warmstart for primal variable x.
        lam_g_warmstart : Optional[ArrayLike], optional
            Warmstart for inequality multipliers.
        lam_h_warmstart : Optional[ArrayLike], optional
            Warmstart for equality multipliers.

        Returns
        -------
        Dict[str, ca.DM]
            Dictionary containing optimal 'x', 'lam_g', and 'lam_h'.
        """
        if self.solver_orig is None:
            self.solver_orig = ca.nlpsol('solver_orig', 'ipopt', self._nlp_orig, self.opts)

        args = {
            'p': theta,
            'lbg': self.lbc_orig,
            'ubg': self.ubc_orig
        }

        # Handle warmstarts
        if x_warmstart is not None:
            args['x0'] = x_warmstart
            
        if lam_g_warmstart is not None or lam_h_warmstart is not None:
            # Handle partial warmstarts by filling missing ones with zeros
            l_g = lam_g_warmstart if lam_g_warmstart is not None else np.zeros(self.n_g)
            l_h = lam_h_warmstart if lam_h_warmstart is not None else np.zeros(self.n_h)
            
            # nlpsol expects a single vector for all constraint multipliers
            args['lam_g0'] = ca.vertcat(l_g, l_h)

        # Solve
        res = self.solver_orig(**args)

        # Extract constraint multipliers
        lam_c = res['lam_g']
        lam_g_opt = lam_c[:self.n_g]
        lam_h_opt = lam_c[self.n_g:]

        return {
            'x': res['x'],
            'lam_g': lam_g_opt,
            'lam_h': lam_h_opt,
            'lam_c': lam_c
        }

    def solve_reg(
        self,
        theta: ArrayLike,
        rho: float,
        x_bar: ArrayLike,
        lam_bar: ArrayLike,
        nu_bar: ArrayLike
    ) -> Dict[str, Union[ca.DM, Dict[str, ca.DM]]]:
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
        Dict[str, Union[ca.DM, Dict[str, ca.DM]]]
            Dictionary containing standard solution vectors:
            - 'x': Regularized primal solution x.
            - 'lam_g': Regularized inequality multipliers lambda.
            - 'lam_h': Regularized equality multipliers nu.
            - 'lam_c': Full stacked dual vector [lam_g; lam_h].
            - 'raw': Dictionary containing internal P_2 variables:
                - 'xi': Full decision vector (x, lambda, nu, z).
                - 'psi': Dual multiplier for modified inequality constraint.
                - 'mu': Dual multiplier for modified equality constraint.
                - 'gamma': Dual multiplier for z <= 0.
        """
        if self.solver_reg is None:
            self.solver_reg = ca.nlpsol('solver_reg', 'ipopt', self._nlp_reg, self.opts)

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
        lam_g_opt = xi_opt[idx_x:idx_lam]
        lam_h_opt = xi_opt[idx_lam:idx_nu]

        # 5. Extract P_2 Dual Multipliers
        lam_c = res['lam_g']
        psi_opt = lam_c[:self.n_g]
        mu_opt = lam_c[self.n_g:]

        # Gamma is multiplier for z <= 0, located in lam_x at the tail of xi
        lam_xi = res['lam_x']
        gamma_opt = lam_xi[-self.n_g:]

        return {
            'x': x_opt,
            'lam_g': lam_g_opt,
            'lam_h': lam_h_opt,
            'lam_c': ca.vertcat(lam_g_opt, lam_h_opt),
            'raw': {
                'xi': xi_opt,
                'psi': psi_opt,
                'mu': mu_opt,
                'gamma': gamma_opt
            }
        }

    def kkt_eval(
        self,
        theta_val: ArrayLike,
        x_val: ArrayLike,
        lam_g_val: ArrayLike,
        lam_h_val: ArrayLike
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
        theta_val : ArrayLike
            Parameter vector theta.
        x_val : ArrayLike
            Primal variable vector x.
        lam_g_val : ArrayLike
            Dual multipliers for inequality constraints g.
        lam_h_val : ArrayLike
            Dual multipliers for equality constraints h.

        Returns
        -------
        Dict[str, ca.DM]
            Dictionary containing the numerical evaluation of KKT vectors:
            - 'stationarity': Gradient of Lagrangian w.r.t. x (shape: n_x x 1).
            - 'inequality': Evaluated inequality constraints g(x, theta) (shape: n_g x 1).
            - 'equality': Evaluated equality constraints h(x, theta) (shape: n_h x 1).
            - 'slackness': Element-wise product lam_g * g(x, theta) (shape: n_g x 1).
            - 'dual_feasibility': Evaluated inequality multipliers lam_g (shape: n_g x 1).
        """
        if self._kkt_func is None:
            # Create local symbolic variables for the multipliers
            lam_g_sym = ca.MX.sym('lam_g', self.n_g)
            lam_h_sym = ca.MX.sym('lam_h', self.n_h)

            # Construct the Lagrangian L(x, lam_g, lam_h, theta)
            # L = f + lam_g^T * g + lam_h^T * h
            lagrangian = self.f + ca.dot(lam_g_sym, self.g) + ca.dot(lam_h_sym, self.h)

            # 1. Stationarity (Gradient of Lagrangian w.r.t x)
            # ca.jacobian returns a row vector (1 x n_x), so we transpose it
            stationarity = ca.jacobian(lagrangian, self.x).T

            # 2. Inequality: just g(x, theta)
            inequality = self.g

            # 3. Equality: just h(x, theta)
            equality = self.h

            # 4. Complementary Slackness: element-wise multiplication
            slackness = lam_g_sym * self.g

            # 5. Dual Feasibility: just the inequality multipliers
            dual_feasibility = lam_g_sym

            # Compile into a CasADi Function with named inputs and outputs
            self._kkt_func = ca.Function(
                'kkt_eval',
                [self.theta, self.x, lam_g_sym, lam_h_sym],               # Inputs
                [stationarity, inequality, equality, slackness, dual_feasibility], # Outputs
                ['theta', 'x', 'lam_g', 'lam_h'],                         # Input names
                ['stationarity', 'inequality', 'equality', 'slackness', 'dual_feasibility'] # Output names
            )

        # Evaluate the compiled function
        # Calling with kwargs returns a dictionary mapped to the output names
        res = self._kkt_func(
            theta=theta_val, 
            x=x_val, 
            lam_g=lam_g_val, 
            lam_h=lam_h_val
        )

        return res

    def differentiate(
        self,
        theta_val: ArrayLike,
        x_val: ArrayLike,
        lam_g_val: ArrayLike,
        lam_h_val: ArrayLike,
        rho_val: float = 0.0,
        x_anchor_val: Optional[ArrayLike] = None
    ) -> Dict[str, ca.DM]:
        """
        Computes the sensitivity of the solution vector with respect to theta (dx/dtheta).

        Uses the Implicit Function Theorem (IFT) on the KKT condition matrix.
        If rho_val == 0, differentiates the original NLP system.
        If rho_val > 0, differentiates the regularized NLP system.

        Parameters
        ----------
        theta_val : ArrayLike
            Parameter vector theta.
        x_val : ArrayLike
            Primal solution point x at which to compute derivatives.
        lam_g_val : ArrayLike
            Inequality dual multipliers solution point.
        lam_h_val : ArrayLike
            Equality dual multipliers solution point.
        rho_val : float, optional
            Regularization factor rho >= 0. Default is 0.0 (original problem).
        x_anchor_val : Optional[ArrayLike], optional
            Anchor point x_anchor (required if rho_val > 0).

        Returns
        -------
        Dict[str, ca.DM]
            Dictionary containing:
            - 'dx_dtheta': Sensitivity matrix of x with respect to theta (shape: n_x x n_theta).
            - 'dlam_g_dtheta': Sensitivity matrix of inequality duals w.r.t theta.
            - 'dlam_h_dtheta': Sensitivity matrix of equality duals w.r.t theta.
        """
        pass