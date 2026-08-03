import casadi as ca
from dbd_opti import CasadiSolver
import numpy as np
import scipy.sparse.linalg as spla
from pypardiso import PyPardisoSolver

# =====================================================================
# 2. PROBLEM FORMULATION (Translating utils.py)
# =====================================================================

def rk4(fc, x, u, m, T):
    """Runge-Kutta 4 integration step."""
    k1 = fc(x, u, m)
    k2 = fc(x + T/2 * k1, u, m)
    k3 = fc(x + T/2 * k2, u, m)
    k4 = fc(x + T * k3, u, m)
    return x + (T/6) * (k1 + 2*k2 + 2*k3 + k4)

def build_dubins_problem(N=10, a_max=1.0, s_max=1.0, alpha=0.5, v_max=None):
    """Constructs the Opti instance for the time-dilated Dubins car."""
    opti = ca.Opti()
    
    # 1. Variables and Parameters
    X = opti.variable(6, N + 1)  # State: [px, py, vx, vy, theta, sigma]
    U = opti.variable(2, N)      # Input: [a, s]
    m_param = opti.parameter(1, 1)
    
    # 2. Continuous Dynamics
    x_sym = ca.MX.sym('x', 6, 1) # type: ignore
    u_sym = ca.MX.sym('u', 2, 1) # type: ignore
    m_sym = ca.MX.sym('m', 1, 1) # type: ignore
    
    vx, vy, theta, sigma = x_sym[2], x_sym[3], x_sym[4], x_sym[5]
    a, s = u_sym[0], u_sym[1]
    
    RHS_f = ca.vcat([
        vx, 
        vy, 
        a * ca.cos(theta), 
        a * ca.sin(theta), 
        s * (ca.cos(theta) * vx + ca.sin(theta) * vy)
    ])
    
    # Sigma is a time dilation state with derivative 0
    fc = ca.Function('fc', [x_sym, u_sym, m_sym], [ca.vcat([sigma * RHS_f, 0])])
    
    # 3. Constraints
    # Initial Condition (sigma is free to act as the time horizon scalar)
    opti.subject_to(X[0:5, 0] == 0)
    
    # Dynamics (Multiple Shooting)
    dt = 1 / (N + 1)
    for k in range(N):
        opti.subject_to(X[:, k+1] == rk4(fc, X[:, k], U[:, k], m_param, dt))
        
    # Terminal Constraints
    opti.subject_to(X[0:2, N] == ca.vertcat(0.5, 0.25))
    opti.subject_to(X[2:4, N] == 0)
    
    # Input/Path Bounds
    for k in range(N):
        opti.subject_to(U[0, k] <= a_max)
        opti.subject_to(U[0, k] >= -a_max)
        
        opti.subject_to(U[1, k] <= s_max * m_param)
        opti.subject_to(U[1, k] >= -s_max * m_param)
        
    if v_max is not None:
        for k in range(N+1):
            opti.subject_to(X[2:4, k] <= v_max)
            opti.subject_to(X[2:4, k] >= -v_max)
            
    # 4. Cost Function
    e1 = X[0:2, N] - ca.vertcat(0.5, 0.25)
    e2 = X[2:4, N]
    cost = X[-1, -1] + alpha * ca.sumsqr(e1) + alpha * ca.sumsqr(e2)
    opti.minimize(cost)
    
    return opti, X, U, m_param

# =====================================================================
# 3. MAIN EXECUTION SCRIPT
# =====================================================================

if __name__ == "__main__":
    # Settings
    N_horizon = 100
    mass_value = 1.2
    rho_reg = 1e-1  # Lowered slightly for more accurate derivatives
    fd_eps = 1e-5   # Finite difference perturbation step
    
    # 1. Build the problem
    print("Building Problem Formulation...")
    opti, X_var, U_var, m_param = build_dubins_problem(N=N_horizon)
    opti.set_value(m_param, mass_value)
    
    # Initial guess
    opti.set_initial(X_var, 0.075)
    opti.set_initial(U_var, 0.075)
    
    # 2. Instantiate solver
    print("Initializing Solver and Differentiator...")
    # Suppress IPOPT output for finite difference loop to keep terminal clean
    opts = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': False}
    solver = CasadiSolver(opti, solver_opts=opts, tolerance=1e-6)
    
    # 3. Solve the nominal NLP
    print("Solving nominal NLP...")
    sol = solver.solve()
    print(f"Solve Successful! Final Cost: {sol.value(opti.f):.4f}")
    
    x_opt = sol.value(opti.x)
    p_opt = sol.value(opti.p)
    lam_opt = sol.value(opti.lam_g) 
    
    # 4. Evaluate the KKT Matrix & RHS analytically

    from time import perf_counter
    
    lin_solver = PyPardisoSolver()

    start = perf_counter()
    print("Evaluating custom KKT sensitivities...")
    result = solver.kkt_fn(
        x=x_opt, 
        p=p_opt, 
        lam_g=lam_opt, 
        rho=rho_reg
    )
    
    KKT_matrix = result["KKT"].sparse()
    RHS_matrix = result["RHS"].full()
    d_vars_dp = spla.spsolve(KKT_matrix, RHS_matrix)
    d_vars_dp_pardiso = lin_solver.solve(KKT_matrix, RHS_matrix).squeeze()
        
    print(f"Elapsed: {perf_counter()-start}")
    
    # Extract primal sensitivities (first nx elements)
    nx = x_opt.shape[0]
    dx_dp_analytical = d_vars_dp[:nx]

    # --- FINITE DIFFERENCE APPROXIMATION ---
    print(f"Running central finite differences (eps = {fd_eps})...")
    
    # Forward solve (p + eps)
    opti.set_value(m_param, mass_value + fd_eps)
    opti.set_initial(opti.x, x_opt)      # Warm start with nominal solution
    opti.set_initial(opti.lam_g, lam_opt)
    sol_plus = solver.solve()
    x_plus = sol_plus.value(opti.x)
    
    # Backward solve (p - eps)
    opti.set_value(m_param, mass_value - fd_eps)
    opti.set_initial(opti.x, x_opt)      # Warm start with nominal solution
    opti.set_initial(opti.lam_g, lam_opt)
    sol_minus = solver.solve()
    x_minus = sol_minus.value(opti.x)
    
    # Compute central difference
    dx_dp_fd = (x_plus - x_minus) / (2 * fd_eps)

    # --- COMPARISON ---
    print("\n=========================================")
    print(" SENSITIVITY VERIFICATION RESULTS")
    print("=========================================")
    
    abs_errors = np.abs(dx_dp_analytical - dx_dp_fd)
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)
    
    print(f"Max Absolute Error:  {max_error:.2e}")
    print(f"Mean Absolute Error: {mean_error:.2e}")
    print(f"FD derivative norm: {np.linalg.norm(dx_dp_fd)}")
    
    if max_error < 1e-4:
        print("\nSUCCESS: Analytical derivatives match finite differences!")
    else:
        print("\nWARNING: Large discrepancy detected. Check KKT formulation, active set strict complementarity, or solver tolerances.")