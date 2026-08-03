import casadi as ca
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import cvxpy as cp

@dataclass
class QPData:
    Q: ca.DM
    q: ca.DM
    A: ca.DM
    b: ca.DM
    G: ca.DM
    h: ca.DM

@dataclass
class QPSolution:
    x: ca.DM
    lam: ca.DM  # Inequality multipliers
    mu: ca.DM   # Equality multipliers
    residual: Dict[str, ca.DM]
    properties: Dict[str, bool]

def check_properties(qp: QPData, x: ca.DM, lam:ca.DM, tol: float = 1e-5) -> Dict[str, bool]:
    """
    Checks Strong Convexity, LICQ, and SCS.
    Converts CasADi DM objects to numpy arrays for linear algebra operations.
    """
    # Convert to numpy for analysis
    Q_np = np.array(qp.Q.full())
    A_np = np.array(qp.A.full())
    G_np = np.array(qp.G.full())
    h_np = np.array(qp.h.full()).flatten()
    
    x_np = np.array(x.full()).flatten()
    lam_np = np.array(lam.full()).flatten()
    
    # 1. Strong Convexity (Q > 0)
    # Using eigh since Q is symmetric
    eigenvalues = np.linalg.eigvalsh(Q_np)
    strong_convexity = bool(np.min(eigenvalues) > tol)
    
    # 2. Identify Active Inequality Constraints (Gx - h == 0)
    ineq_eval = G_np @ x_np - h_np
    active_indices = np.where(np.abs(ineq_eval) <= tol)[0]
    
    # 3. LICQ (Linear Independence Constraint Qualification)
    # Stack A and the active rows of G
    G_act = G_np[active_indices, :] if len(active_indices) > 0 else np.empty((0, Q_np.shape[0]))
    J_act = np.vstack([A_np, G_act])
    
    # If the rank equals the number of rows, they are linearly independent
    rank = np.linalg.matrix_rank(J_act, tol=tol)
    licq = bool(rank == J_act.shape[0])
    
    # 4. SCS (Strict Complementary Slackness)
    # Every active inequality must have a strictly positive multiplier
    if len(active_indices) > 0:
        lam_act = lam_np[active_indices]
        scs = bool(np.all(lam_act > tol))
    else:
        # If there are no active inequalities, SCS holds vacuously
        scs = True
        
    return {
        'strong_convexity': strong_convexity,
        'licq': licq,
        'scs': scs
    }

def evaluate_kkt(qp: QPData, x: ca.DM, lam:ca.DM, mu:ca.DM) -> Dict[str, ca.DM]:
    """
    Evaluates the KKT conditions for the QP at the given solution.
    Lagrangian: L = 0.5 x^T Q x + q^T x + mu^T (Ax - b) + lam^T (Gx - h)
    """
    # 1. Stationarity: Qx + q + A^T mu + G^T lam = 0
    stationarity = ca.mtimes(qp.Q, x) + qp.q + \
                   ca.mtimes(qp.A.T, mu) + ca.mtimes(qp.G.T, lam)
    
    # 2. Equality constraints: Ax - b = 0
    equality = ca.mtimes(qp.A, x) - qp.b
    
    # 3. Inequality constraints: Gx - h <= 0
    inequality = ca.fmax(0.0, ca.mtimes(qp.G, x) - qp.h)
    
    # 4. Complementary slackness: lam * (Gx - h) = 0
    slackness = lam * inequality  # Element-wise multiplication
    
    # 5. Multiplier non-negativity: lam >= 0 (represented as violation: min(0, lam))
    non_negativity = ca.fmin(0, lam)
    
    return {
        'stationarity': stationarity,
        'equality': equality,
        'inequality': inequality,
        'slackness': slackness,
        'non_negativity': non_negativity
    }

def generate_qp(
    nx: int, neq: int, nineq: int, seed: int, 
    licq: bool = True, scs: bool = True, sosc: bool = True
):
    """
    Reverse-engineers a QP that strictly adheres to the requested theoretical properties.
    Uses SVD during generation to prevent accidental ill-conditioning.
    """
    np.random.seed(seed)
    
    # 1. Define the optimal solution x*
    x_star = np.random.randn(nx, 1)
    
    # 2. Determine active/inactive inequality counts
    n_act = max(1, nineq // 2)
    n_inact = nineq - n_act
    n_active_total = neq + n_act
    
    # 3. Generate well-conditioned Active Jacobian (J_act = [A; G_act])
    # We enforce a condition number of ~10.0 to prevent bad random scaling
    J_raw = np.random.randn(n_active_total, nx)
    U_rand, _, Vh_rand = np.linalg.svd(J_raw, full_matrices=False)
    
    min_dim = min(n_active_total, nx)
    S_good = np.linspace(1.0, 10.0, min_dim)
    J_act = U_rand @ np.diag(S_good) @ Vh_rand
    
    # Extract A and G_act from the well-conditioned J_act
    A = J_act[:neq, :]
    G_act = J_act[neq:, :]
    
    b = A @ x_star
    h_act = G_act @ x_star
    
    # 4. Multipliers
    mu_star = np.random.randn(neq, 1)
    lam_star = np.zeros((nineq, 1))
    
    if scs:
        lam_act = np.random.uniform(0.1, 2.0, (n_act, 1)) # Strictly positive
    else:
        lam_act = np.random.uniform(0.1, 2.0, (n_act, 1))
        lam_act[0] = 0.0 # Break strict complementary slackness
        
    lam_star[:n_act, :] = lam_act
    
    # 5. Populate full G and h matrices
    G = np.zeros((nineq, nx))
    h = np.zeros((nineq, 1))
    
    G[:n_act, :] = G_act
    h[:n_act, :] = h_act
    
    # Inactive inequalities (Gx < h)
    if n_inact > 0:
        # These can be random since they don't affect the active KKT system
        G_inact = np.random.randn(n_inact, nx)
        U_rand, _, Vh_rand = np.linalg.svd(G_inact, full_matrices=False)
        min_dim = min(n_inact, nx)
        S_good = np.linspace(1.0, 10.0, min_dim)
        G_inact_good = U_rand @ np.diag(S_good) @ Vh_rand
        slack = np.random.uniform(0.5, 2.0, (n_inact, 1))
        h_inact = G_inact_good @ x_star + slack
        
        G[n_act:, :] = G_inact_good
        h[n_act:, :] = h_inact
        
    # ====================================================================
    # 6. Handle LICQ (Modified to overwrite an inactive constraint)
    # ====================================================================
    if not licq:
        if n_inact == 0:
            raise ValueError("Cannot break LICQ: 'n_inact' is 0, \
                 so there are no inactive constraints to overwrite.")
            
        # Overwrite the first inactive constraint (which lives at index n_act)
        # to be a duplicate of an active constraint, pulling it into the active set.
        if neq > 0:
            G[n_act, :] = A[0, :]
            h[n_act, :] = b[0, :]
        else:
            G[n_act, :] = G[0, :]
            h[n_act, :] = h[0, :]
            
        # To maintain Strict Complementary Slackness (if requested), 
        # this newly active constraint must have a strictly positive multiplier.
        if scs:
            lam_star[n_act, 0] = np.random.uniform(0.1, 2.0)
            
        # Update the active Jacobian so the SOSC logic knows about this new dependent row
        J_act = np.vstack([J_act, G[n_act, :]])
    
    # ====================================================================
    # 7. Handle SOSC
    # ====================================================================
    # We need the Hessian Q to be positive definite on the null space of J_act.
    U, S, Vh = np.linalg.svd(J_act, full_matrices=True)
    
    # Safely compute rank (ignoring numerical noise near zero)
    tol = max(J_act.shape) * np.finfo(float).eps * (S[0] if len(S) > 0 else 1.0)
    rank = np.sum(S > tol)
    
    Z = Vh[rank:].T # Null space basis
    Y = Vh[:rank].T # Range space basis
    
    D_y = np.eye(Y.shape[1]) # Always positive definite in the range space
    
    if sosc or Z.shape[1] == 0:
        # Strictly positive eigenvalues in the null space
        D_z = np.eye(Z.shape[1]) 
    else:
        # Break SOSC by making at least one eigenvalue zero
        D_z = np.eye(Z.shape[1])
        D_z[0, 0] = 0.0
        
    Q_np = Z @ D_z @ Z.T + Y @ D_y @ Y.T
    Q_np = 0.5 * (Q_np + Q_np.T) # Ensure symmetry due to numerical noise
    
    # ====================================================================
    # 8. Solve for q using stationarity: Qx + q + A^T mu + G^T lam = 0
    # ====================================================================
    q_np = - (Q_np @ x_star + A.T @ mu_star + G.T @ lam_star)
    
    # ====================================================================
    # 9. Pack QP and its solution
    # ====================================================================

    data = QPData(
        Q=ca.DM(Q_np), q=ca.DM(q_np),
        A=ca.DM(A), b=ca.DM(b),
        G=ca.DM(G), h=ca.DM(h)
    )

    residual = evaluate_kkt(
        qp=data,
        x=ca.DM(x_star),
        lam=ca.DM(lam_star),
        mu=ca.DM(mu_star)
    )

    solution = QPSolution(
        x=ca.DM(x_star),
        mu=ca.DM(mu_star),
        lam=ca.DM(lam_star),
        properties={
            "strong_convexity":sosc,
            "licq":licq,
            "scs":scs
        },
        residual=residual
    )

    return data, solution

def generate_dqp(nx: int, neq: int, nineq: int, seed: int) -> QPData:
    np.random.seed(seed)
    
    # 1. Generate dQ (Positive Semidefinite) and normalize
    M = np.random.randn(nx, nx)
    dQ = M @ M.T
    dQ = dQ / np.linalg.norm(dQ, ord='fro')
    
    # 2. Generate dq and normalize (L2 norm)
    dq = np.random.randn(nx, 1)
    dq = dq / np.linalg.norm(dq)
    
    # 3. Generate dA and db, handle empty cases
    if neq > 0:
        dA = np.random.randn(neq, nx)
        dA = dA / np.linalg.norm(dA, ord='fro')
        
        db = np.random.randn(neq, 1)
        db = db / np.linalg.norm(db)
    else:
        dA = np.empty((0, nx))
        db = np.empty((0, 1))
        
    # 4. Generate dG and dh, handle empty cases
    if nineq > 0:
        dG = np.random.randn(nineq, nx)
        dG = dG / np.linalg.norm(dG, ord='fro')
        
        dh = np.random.randn(nineq, 1)
        dh = dh / np.linalg.norm(dh)
    else:
        dG = np.empty((0, nx))
        dh = np.empty((0, 1))
    
    return QPData(
        Q=ca.DM(dQ), 
        q=ca.DM(dq),
        A=ca.DM(dA), 
        b=ca.DM(db),
        G=ca.DM(dG), 
        h=ca.DM(dh)
    )

def perturb_qp(qp: QPData, dqp: QPData, eps: float) -> QPData:
    return QPData(
        Q = qp.Q + eps * dqp.Q,
        q = qp.q + eps * dqp.q,
        A = qp.A + eps * dqp.A,
        b = qp.b + eps * dqp.b,
        G = qp.G + eps * dqp.G,
        h = qp.h + eps * dqp.h
    )

def regularize_qp(qp: QPData, sol: QPSolution, rho: float) -> QPData:
    """
    Constructs a regularized QP by expanding the decision variables to include
    primal variables, dual variables, and slack variables.
    
    New variable vector: X = [x, lam, mu, z]^T
    """
    nx = qp.Q.shape[0]
    neq = qp.A.shape[0]
    nineq = qp.G.shape[0]
    
    # 1. Define baseline z_bar to satisfy the unregularized equality: Gx - h - z = 0
    # SIGN CHANGED: z = Gx - h
    z_bar = ca.mtimes(qp.G, sol.x) - qp.h
    
    # 2. Build augmented Q_new
    # Q_new is block diagonal: diag(Q + rho*I, rho*I, rho*I, rho*I)
    Q_new = ca.diagcat(
        qp.Q + rho * ca.DM.eye(nx),
        rho * ca.DM.eye(nineq),
        rho * ca.DM.eye(neq),
        rho * ca.DM.eye(nineq)
    )
    
    # 3. Build augmented q_new
    # From expansion of rho/2 * ||x - x_bar||^2 and rho/2 * ||z - z_bar||^2
    q_new = ca.vertcat(
        qp.q - rho * sol.x,
        ca.DM.zeros(nineq, 1),# type: ignore
        ca.DM.zeros(neq, 1),# type: ignore
        -rho * z_bar
    )
    
    # 4. Build augmented Equality Constraints (A_new * X = b_new)
    # Eq 1: A x - rho * mu = b - rho * mu_bar
    A_eq1 = ca.horzcat(
        qp.A, 
        ca.DM.zeros(neq, nineq), 
        -rho * ca.DM.eye(neq), 
        ca.DM.zeros(neq, nineq)
    )
    
    # Eq 2: G x - rho * lam - z = h - rho * lam_bar
    # SIGN CHANGED: -I block for z
    A_eq2 = ca.horzcat(
        qp.G, 
        -rho * ca.DM.eye(nineq), 
        ca.DM.zeros(nineq, neq), 
        -ca.DM.eye(nineq) 
    )
    
    A_new = ca.vertcat(A_eq1, A_eq2)
    b_new = ca.vertcat(
        qp.b - rho * sol.mu,
        qp.h - rho * sol.lam
    )
    
    # 5. Build augmented Inequality Constraints (G_new * X <= h_new)
    # Ineq: z <= 0
    G_new = ca.horzcat(
        ca.DM.zeros(nineq, nx), 
        ca.DM.zeros(nineq, nineq), 
        ca.DM.zeros(nineq, neq), 
        ca.DM.eye(nineq)
    )
    h_new = ca.DM.zeros(nineq, 1) # type: ignore
    
    return QPData(
        Q=Q_new, q=q_new, 
        A=A_new, b=b_new, 
        G=G_new, h=h_new
    )

# class QPSolver:
#     def __init__(self, qp_data: QPData, options: Optional[dict] = None, tol : Optional[float] = None):
#         if options is None:
#             options = {}
            
#         # Extract the solver plugin (defaulting to qpoases if not provided)
#         # CasADi needs this to know which underlying C++ solver to invoke.
#         plugin = options.pop('plugin', 'qpoases')
        
#         self.default_qp = qp_data
#         self.nx = qp_data.Q.shape[0]
#         self.neq = qp_data.A.shape[0]
#         self.nineq = qp_data.G.shape[0]
        
#         # Define the structural sparsity for the solver.
#         # We stack A (neq rows) and G (nineq rows) into a single matrix.
#         h_sp = ca.Sparsity.dense(self.nx, self.nx)
#         a_sp = ca.Sparsity.dense(self.neq + self.nineq, self.nx)
        
#         # Initialize the CasADi solver object by passing the sparsity patterns.
#         self.solver = ca.conic('qp_solver', plugin, {'h': h_sp, 'a': a_sp}, options)

#         # Tolerance when checking for LICQ violations
#         self.tol = tol if tol is not None else 1e-5

#     def solve(self, qp: Optional[QPData] = None, warmstart: Optional[QPSolution] = None) -> QPSolution:
#         if qp is None:
#             qp = self.default_qp
            
#         # Ensure dimensions haven't changed, as the solver object's sparsity is already compiled
#         assert qp.Q.shape[0] == self.nx, "Updated QP nx must match original."
#         assert qp.A.shape[0] == self.neq, "Updated QP neq must match original."
#         assert qp.G.shape[0] == self.nineq, "Updated QP nineq must match original."
        
#         # Stack matrices to match CasADi's standard form: lba <= a*x <= uba
#         a_val = ca.vertcat(qp.A, qp.G)
        
#         # Equalities: Ax = b   =>   lba = b, uba = b
#         # Inequalities: Gx <= h  =>   lba = -inf, uba = h
#         lba = ca.vertcat(qp.b, ca.DM.ones(self.nineq, 1) * -ca.inf) # type:ignore
#         uba = ca.vertcat(qp.b, qp.h)
        
#         # Prepare inputs
#         solver_in = {
#             'h': qp.Q,
#             'g': qp.q,
#             'a': a_val,
#             'lba': lba,
#             'uba': uba
#         }
        
#         # Inject warmstart if provided
#         if warmstart is not None:
#             solver_in['x0'] = warmstart.x
#             # Stack the dual variables (multipliers) the same way we stacked 'a'
#             solver_in['lam_a0'] = ca.vertcat(warmstart.mu, warmstart.lam)
            
#         # Solve the QP
#         res = self.solver(**solver_in)
        
#         # Extract primal solution
#         x_opt = res['x']
        
#         # Extract and slice dual solutions
#         lam_a_opt = res['lam_a']
#         mu_opt = lam_a_opt[:self.neq]
#         lam_opt = lam_a_opt[self.neq:]
        
#         # form residual
#         residual = evaluate_kkt(qp, x_opt, lam_opt, mu_opt)

#         # get properties
#         properties = check_properties(qp, x_opt, lam_opt, self.tol)
        
#         return QPSolution(
#             x=x_opt,
#             mu=mu_opt,
#             lam=lam_opt,
#             residual=residual,
#             properties=properties
#         )

class QPSolver:
    def __init__(self, qp_data, options: Optional[dict] = None, tol: Optional[float] = None):
        if options is None:
            options = {}
            
        self.default_qp = qp_data
        self.nx = qp_data.Q.shape[0]
        self.neq = qp_data.A.shape[0]
        self.nineq = qp_data.G.shape[0]
        
        # Store solver options for CVXPY (e.g., {'solver': cp.OSQP})
        self.options = options
        self.tol = tol if tol is not None else 1e-5

    def solve(self, qp = None, warmstart = None):
        if qp is None:
            qp = self.default_qp
            
        # Ensure dimensions haven't changed
        assert qp.Q.shape[0] == self.nx, "Updated QP nx must match original."
        assert qp.A.shape[0] == self.neq, "Updated QP neq must match original."
        assert qp.G.shape[0] == self.nineq, "Updated QP nineq must match original."
        
        # 1. Convert CasADi matrices to NumPy arrays for CVXPY
        # Use .flatten() on vectors to ensure they are 1D arrays
        Q_np = np.array(qp.Q)
        q_np = np.array(qp.q).flatten()
        
        A_np = np.array(qp.A) if self.neq > 0 else None
        b_np = np.array(qp.b).flatten() if self.neq > 0 else None
        
        G_np = np.array(qp.G) if self.nineq > 0 else None
        h_np = np.array(qp.h).flatten() if self.nineq > 0 else None

        # 2. Define CVXPY Problem
        x = cp.Variable(self.nx)
        
        # Ensure Q is symmetric for CVXPY (sometimes numerical noise prevents strict symmetry)
        Q_sym = (Q_np + Q_np.T) / 2.0
        objective = cp.Minimize(0.5 * cp.quad_form(x, Q_sym) + q_np.T @ x)
        
        constraints = []
        eq_idx = -1
        ineq_idx = -1
        
        if self.neq > 0:
            eq_constr = (A_np @ x == b_np)
            constraints.append(eq_constr)
            eq_idx = len(constraints) - 1
            
        if self.nineq > 0:
            ineq_constr = (G_np @ x <= h_np)
            constraints.append(ineq_constr)
            ineq_idx = len(constraints) - 1
            
        prob = cp.Problem(objective, constraints)
        
        # 3. Apply Warmstart if provided
        if warmstart is not None:
            x.value = np.array(warmstart.x).flatten()
            # if self.neq > 0 and warmstart.mu is not None:
            #     constraints[eq_idx].dual_value = np.array(warmstart.mu).flatten()
            # if self.nineq > 0 and warmstart.lam is not None:
            #     constraints[ineq_idx].dual_value = np.array(warmstart.lam).flatten()
                
        # 4. Solve the QP
        # warm_start flag must be passed to .solve() to utilize the injected .value attributes
        prob.solve(warm_start=(warmstart is not None), **self.options)
        
        if x.value is None:
            raise RuntimeError(f"CVXPY failed to solve the QP. Solver status: {prob.status}")

        # 5. Extract solutions and convert back to CasADi DM types
        x_opt = ca.DM(x.value)
        
        # Extract duals (default to empty CasADi vectors if no constraints exist)
        if self.neq > 0 and constraints[eq_idx].dual_value is not None:
            mu_opt = ca.DM(constraints[eq_idx].dual_value)
        else:
            mu_opt = ca.DM(np.zeros(0))
            
        if self.nineq > 0 and constraints[ineq_idx].dual_value is not None:
            lam_opt = ca.DM(constraints[ineq_idx].dual_value)
        else:
            lam_opt = ca.DM(np.zeros(0))
            
        # Note: Depending on the backend solver CVXPY chooses, the sign convention 
        # of the equality dual variables (`mu_opt`) might be inverted compared to CasADi's `qpoases`.
        # If evaluate_kkt fails, you may need to flip the sign: mu_opt = -ca.DM(...)

        # Form residual
        residual = evaluate_kkt(qp, x_opt, lam_opt, mu_opt)

        # Get properties
        properties = check_properties(qp, x_opt, lam_opt, self.tol)
        
        return QPSolution(
            x=x_opt,
            mu=mu_opt,
            lam=lam_opt,
            residual=residual,
            properties=properties
        )

    def differentiate(self, sol: QPSolution, dqp: QPData, qp: Optional[QPData] = None, lstsq: bool = False) -> QPSolution:
        """
        Differentiates through the KKT conditions using the Implicit Function Theorem (IFT).
        
        Args:
            sol: The optimal solution at which to evaluate the KKT Jacobian.
            dqp: A QPData object containing the differentials (perturbations) of the QP data.
            qp: The base QP data. Defaults to self.default_qp.
            lstsq: Whether to use least squares solver (useful if KKT Jacobian is singular).
            
        Returns:
            A QPSolution object containing the differentials (dx, dmu, dlam) and solution properties.
        """
        import numpy as np # Ensure numpy is available
        
        if qp is None:
            qp = self.default_qp
            
        # 1. Build the asymmetric KKT Jacobian (J_z)
        # Block Row 1: Stationarity w.r.t (x, mu, lam) -> [Q, A^T, G^T]
        row1 = ca.horzcat(qp.Q, qp.A.T, qp.G.T)
        
        # Block Row 2: Equality w.r.t (x, mu, lam) -> [A, 0, 0]
        row2 = ca.horzcat(qp.A, ca.DM.zeros(self.neq, self.neq), ca.DM.zeros(self.neq, self.nineq))
        
        # Block Row 3: Slackness w.r.t (x, mu, lam) -> [diag(lam)*G, 0, diag(Gx - h)]
        ineq_eval = ca.mtimes(qp.G, sol.x) - qp.h
        diag_lam = ca.diag(sol.lam)
        diag_ineq = ca.diag(ineq_eval)
        
        row3 = ca.horzcat(ca.mtimes(diag_lam, qp.G), 
                          ca.DM.zeros(self.nineq, self.neq), 
                          diag_ineq)
        
        J_z = ca.vertcat(row1, row2, row3)
        
        # 2. Build the right-hand side (-J_theta * dtheta)
        d_stationarity = ca.mtimes(dqp.Q, sol.x) + dqp.q + \
                         ca.mtimes(dqp.A.T, sol.mu) + ca.mtimes(dqp.G.T, sol.lam)
                         
        d_equality = ca.mtimes(dqp.A, sol.x) - dqp.b
        
        d_slackness = sol.lam * (ca.mtimes(dqp.G, sol.x) - dqp.h)
        
        rhs = -ca.vertcat(d_stationarity, d_equality, d_slackness)
        
        # 3. Analyze the linear system for uniqueness (Rouché-Capelli Theorem)
        J_np = np.array(J_z.full())
        rhs_np = np.array(rhs.full())
        
        n_vars = self.nx + self.neq + self.nineq
        tol = 1e-8 # Tolerance for numerical rank computation
        
        rank_J = np.linalg.matrix_rank(J_np, tol=tol)
        rank_aug = np.linalg.matrix_rank(np.hstack([J_np, rhs_np]), tol=tol)
        
        # A square system has a unique solution iff it is full rank.
        unique_solution = bool(rank_J == n_vars)
        # If rank(J) == rank([J | rhs]) but < n_vars, there are infinite solutions.
        infinite_solutions = bool((rank_J == rank_aug) and (rank_J < n_vars))
        # If rank(J) < rank([J | rhs]), the system is inconsistent (no exact solution).
        no_solution = bool(rank_J < rank_aug)
        
        system_properties = {
            'unique_solution': unique_solution,
            'infinite_solutions': infinite_solutions,
            'no_solution': no_solution,
            'rank_J': int(rank_J),
            'expected_rank': int(n_vars)
        }
        
        # 4. Solve the linear system J_z * dz = rhs
        if lstsq:
            # Using lstsq with rcond=None handles singular matrices by returning the 
            # minimum-norm solution if infinite solutions exist, or best-fit if no solution exists.
            d_sol = ca.DM(np.linalg.lstsq(J_np, rhs_np, rcond=None)[0])
        else:
            d_sol = ca.solve(J_z, rhs, "csparse")
        
        # 5. Unpack the differentials
        dx = d_sol[:self.nx]
        dmu = d_sol[self.nx : self.nx + self.neq]
        dlam = d_sol[self.nx + self.neq :]
        
        return QPSolution(
            x=dx, 
            mu=dmu, 
            lam=dlam, 
            residual={}, 
            properties=system_properties
        )
    
    def differentiate_alt(self, sol: QPSolution, dqp: QPData, qp: Optional[QPData] = None, lstsq: bool = False) -> QPSolution:
        """
        Differentiates through the KKT conditions using the Implicit Function Theorem (IFT) 
        on the active set, yielding a symmetric linear system.
        """
        import numpy as np
        
        if qp is None:
            qp = self.default_qp
            
        # 1. Identify active constraints to form the reduced symmetric system
        tol = getattr(self, 'tol', 1e-5)
        ineq_eval = np.array((ca.mtimes(qp.G, sol.x) - qp.h).full()).flatten()
        act_idx = np.where(np.abs(ineq_eval) <= tol)[0].tolist()
        n_act = len(act_idx)
        
        # 2. Build the symmetric KKT Jacobian (J_sym)
        G_act = qp.G[act_idx, :] if n_act > 0 else ca.DM(0, self.nx)
        
        # Block Row 1: Stationarity -> [Q, A^T, G_act^T]
        row1 = ca.horzcat(qp.Q, qp.A.T, G_act.T)
        
        # Block Row 2: Equality -> [A, 0, 0]
        row2 = ca.horzcat(qp.A, ca.DM.zeros(self.neq, self.neq), ca.DM.zeros(self.neq, n_act))
        
        # Block Row 3: Active Inequality Slackness -> [G_act, 0, 0]
        row3 = ca.horzcat(G_act, ca.DM.zeros(n_act, self.neq), ca.DM.zeros(n_act, n_act))
        
        J_sym = ca.vertcat(row1, row2, row3)
        
        # 3. Build the right-hand side
        d_stationarity = ca.mtimes(dqp.Q, sol.x) + dqp.q + \
                         ca.mtimes(dqp.A.T, sol.mu) + ca.mtimes(dqp.G.T, sol.lam)
                         
        d_equality = ca.mtimes(dqp.A, sol.x) - dqp.b
        
        if n_act > 0:
            dG_act = dqp.G[act_idx, :]
            dh_act = dqp.h[act_idx, :]
            d_act = ca.mtimes(dG_act, sol.x) - dh_act
        else:
            d_act = ca.DM(0, 1) # type: ignore
            
        rhs = -ca.vertcat(d_stationarity, d_equality, d_act)
        
        # 4. Analyze the linear system for uniqueness
        J_np = np.array(J_sym.full())
        rhs_np = np.array(rhs.full())
        
        n_vars = self.nx + self.neq + n_act
        rank_tol = 1e-8
        
        rank_J = np.linalg.matrix_rank(J_np, tol=rank_tol)
        rank_aug = np.linalg.matrix_rank(np.hstack([J_np, rhs_np]), tol=rank_tol)
        
        system_properties = {
            'unique_solution': bool(rank_J == n_vars),
            'infinite_solutions': bool((rank_J == rank_aug) and (rank_J < n_vars)),
            'no_solution': bool(rank_J < rank_aug),
            'rank_J': int(rank_J),
            'expected_rank': int(n_vars)
        }
        
        # 5. Solve the symmetric linear system
        if lstsq:
            d_sol_act = ca.DM(np.linalg.lstsq(J_np, rhs_np, rcond=None)[0])
        else:
            d_sol_act = ca.solve(J_sym, rhs, "csparse")
        
        # 6. Unpack differentials and map active multipliers back to full dual dimension
        dx = d_sol_act[:self.nx]
        dmu = d_sol_act[self.nx : self.nx + self.neq]
        
        dlam = ca.DM.zeros(self.nineq, 1)
        if n_act > 0:
            dlam[act_idx, 0] = d_sol_act[self.nx + self.neq :]
            
        return QPSolution(
            x=dx, 
            mu=dmu, 
            lam=dlam, 
            residual={}, 
            properties=system_properties
        )