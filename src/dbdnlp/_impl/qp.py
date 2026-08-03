import casadi as ca
import numpy as np
from dataclasses import dataclass
from typing import Dict
from dbdnlp import NLPSolution

@dataclass
class QPData:
    Q: ca.DM
    q: ca.DM
    A: ca.DM
    b: ca.DM
    G: ca.DM
    h: ca.DM

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
    slackness = lam * (ca.mtimes(qp.G, x) - qp.h)
    
    # 5. Multiplier non-negativity: lam >= 0 (represented as violation: min(0, lam))
    non_negativity = ca.fmin(0, lam)
    
    return {
        'stationarity': stationarity,
        'equality': equality,
        'inequality': inequality,
        'slackness': slackness,
        'dual_feasibility': non_negativity
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

    solution = NLPSolution(
        theta=ca.DM(0,0),
        x=ca.DM(x_star),
        mu=ca.DM(mu_star),
        lam=ca.DM(lam_star),
        properties={
            "ssosc":sosc,
            "licq":licq,
            "scs":scs
        },
        kkt_residuals=residual
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