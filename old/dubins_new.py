import casadi as ca
import numpy as np
from dbd_opti_new import RegularizedOpti

# =====================================================================
# 2. PROBLEM FORMULATION 
# =====================================================================

def rk4(fc, x, u, m, T):
    k1 = fc(x, u, m)
    k2 = fc(x + T/2 * k1, u, m)
    k3 = fc(x + T/2 * k2, u, m)
    k4 = fc(x + T * k3, u, m)
    return x + (T/6) * (k1 + 2*k2 + 2*k3 + k4)

def build_dubins_problem(N=10, a_max=1.0, s_max=1.0, alpha=0.5, v_max=None):
    opti = ca.Opti()
    
    X = opti.variable(6, N + 1)
    U = opti.variable(2, N)
    m_param = opti.parameter(1, 1)
    
    x_sym = ca.MX.sym('x', 6, 1)
    u_sym = ca.MX.sym('u', 2, 1)
    m_sym = ca.MX.sym('m', 1, 1)
    
    vx, vy, theta, sigma = x_sym[2], x_sym[3], x_sym[4], x_sym[5]
    a, s = u_sym[0], u_sym[1]
    
    RHS_f = ca.vcat([
        vx, 
        vy, 
        a * ca.cos(theta), 
        a * ca.sin(theta), 
        s * (ca.cos(theta) * vx + ca.sin(theta) * vy)
    ])
    
    fc = ca.Function('fc', [x_sym, u_sym, m_sym], [ca.vcat([sigma * RHS_f, 0])])
    
    # ---------------------------------------------------------
    # EQUALITIES: Formulated as expression == 0
    # ---------------------------------------------------------
    opti.subject_to(X[0:5, 0] == 0)
    
    dt = 1 / (N + 1)
    for k in range(N):
        opti.subject_to(X[:, k+1] - rk4(fc, X[:, k], U[:, k], m_param, dt) == 0)
        
    opti.subject_to(X[0:2, N] - ca.vertcat(0.5, 0.25) == 0)
    opti.subject_to(X[2:4, N] == 0)
    
    # ---------------------------------------------------------
    # INEQUALITIES: Formulated as expression <= 0
    # ---------------------------------------------------------
    for k in range(N):
        # Acceleration bounds
        opti.subject_to(U[0, k] - a_max <= 0)           # U <= a_max
        opti.subject_to(-U[0, k] - a_max <= 0)          # U >= -a_max
        
        # Steering bounds
        opti.subject_to(U[1, k] - s_max * m_param <= 0)
        opti.subject_to(-U[1, k] - s_max * m_param <= 0)
        
    if v_max is not None:
        for k in range(N+1):
            opti.subject_to(X[2:4, k] - v_max <= 0)
            opti.subject_to(-X[2:4, k] - v_max <= 0)
            
    e1 = X[0:2, N] - ca.vertcat(0.5, 0.25)
    e2 = X[2:4, N]
    cost = X[-1, -1] + alpha * ca.sumsqr(e1) + alpha * ca.sumsqr(e2)
    opti.minimize(cost)
    
    return opti, X, U, m_param


# =====================================================================
# 3. MAIN EXECUTION & VERIFICATION
# =====================================================================

if __name__ == "__main__":
    print("--- 1. Solving Original Unregularized Problem ---")
    opti, X, U, m_param = build_dubins_problem(N=20)
    
    m_val = 1.0
    opti.set_value(m_param, m_val)
    opti.set_initial(opti.x,0.075)
    opti.solver('ipopt', {'print_time': False}, {'print_level': 3})
    sol_orig = opti.solve()
    
    x_opt = sol_orig.value(opti.x)
    
    print("\n--- 2. Setting up and Solving Regularized Problem ---")
    reg_opti = RegularizedOpti(opti, p_opts={'print_time': False}, s_opts={'print_level': 3})
    
    theta_val = sol_orig.value(opti.p)
    z_opt = sol_orig.value(reg_opti.z_orig_expr)
    mu_opt = sol_orig.value(reg_opti.mu_orig_expr)
    rho_val = 1e-5
    
    # Solve the regularized version
    sol_reg = reg_opti.solve(theta_val, x_opt, z_opt, mu_opt, rho_val)
    x_reg_opt = sol_reg.value(reg_opti.x)
    
    print("\n--- 3. Verification ---")
    max_diff = np.max(np.abs(x_reg_opt - x_opt))
    print(f"Max absolute difference in primal variables 'x' after solve: {max_diff:.3e}")
    
    if max_diff < 1e-6:
        print("Success! The regularized solver holds perfectly at the original solution.")
    else:
        print("Warning: The solutions differ.")