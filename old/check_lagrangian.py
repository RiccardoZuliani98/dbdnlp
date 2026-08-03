import casadi as ca
import numpy as np

# =====================================================================
# 1. PROBLEM FORMULATION (STRICTLY <= 0 or == 0)
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

if __name__ == "__main__":
    print("--- 1. Solving Standardized Problem ---")
    opti, X, U, m_param = build_dubins_problem(N=15)

    opti.set_value(m_param, 1.0)
    opti.set_initial(opti.x, 0.075)
    opti.solver('ipopt', {'print_time': False}, {'print_level': 3})
    sol = opti.solve()

    # ---------------------------------------------------------
    # 2. Extract Base Symbolic Variables and Constraints
    # ---------------------------------------------------------
    x_sym = opti.x
    theta_sym = opti.p
    f_sym = opti.f

    g_orig = opti.g if hasattr(opti, 'g') else opti.advanced.g
    lbg_orig = opti.lbg if hasattr(opti, 'lbg') else opti.advanced.lbg
    ubg_orig = opti.ubg if hasattr(opti, 'ubg') else opti.advanced.ubg
    lam_orig = opti.lam_g if hasattr(opti, 'lam_g') else opti.advanced.lam_g

    # ---------------------------------------------------------
    # 3. Clean Parsing (No ambiguous bounds)
    # ---------------------------------------------------------
    h_sym_list, nu_list = [], []
    g_sym_list, lam_list = [], []

    for i in range(g_orig.shape[0]):
        g_i = g_orig[i]
        l_i = float(lbg_orig[i])
        u_i = float(ubg_orig[i])
        lam_i = lam_orig[i]
        
        # In our explicit formulation, upper bound is ALWAYS 0.0
        assert u_i == 0.0, f"Expected upper bound 0.0, got {u_i} at index {i}"

        if l_i == 0.0: 
            # EQUALITY (0 <= g <= 0)
            h_sym_list.append(g_i)
            nu_list.append(lam_i)
        elif l_i == -np.inf:
            # UPPER-BOUND INEQUALITY (-inf <= g <= 0)
            g_sym_list.append(g_i)
            lam_list.append(lam_i) 
        else:
            raise ValueError(f"Unexpected constraint bounds: [{l_i}, {u_i}]")

    h_sym = ca.vcat(h_sym_list) if h_sym_list else ca.MX(0, 1)
    g_sym = ca.vcat(g_sym_list) if g_sym_list else ca.MX(0, 1)

    nu_sym = ca.vcat(nu_list)   if nu_list    else ca.MX(0, 1)
    lam_sym = ca.vcat(lam_list) if lam_list   else ca.MX(0, 1)

    # ---------------------------------------------------------
    # 4. Create Functions
    # ---------------------------------------------------------
    f_func = ca.Function("f", [x_sym, theta_sym], [f_sym])
    h_func = ca.Function("h", [x_sym, theta_sym], [h_sym])
    g_func = ca.Function("g", [x_sym, theta_sym], [g_sym])

    # ---------------------------------------------------------
    # 5. Form Lagrangian & Gradient
    # ---------------------------------------------------------
    nu_var = ca.MX.sym('nu', h_sym.size1()) if h_sym.size1() > 0 else ca.MX(0, 1)
    lam_var = ca.MX.sym('lam', g_sym.size1()) if g_sym.size1() > 0 else ca.MX(0, 1)

    # L = f(x) + nu^T * h(x) + lam^T * g(x)
    L_expr = f_func(x_sym, theta_sym)
    if h_sym.size1() > 0:
        L_expr += ca.dot(nu_var, h_func(x_sym, theta_sym))
    if g_sym.size1() > 0:
        L_expr += ca.dot(lam_var, g_func(x_sym, theta_sym))

    grad_L_expr = ca.jacobian(L_expr, x_sym).T
    grad_L_func = ca.Function('grad_L', [x_sym, theta_sym, nu_var, lam_var], [grad_L_expr])

    # ---------------------------------------------------------
    # 6. Evaluate and Verify
    # ---------------------------------------------------------
    x_val = sol.value(x_sym)
    theta_val = sol.value(theta_sym)
    
    # Map raw lam_orig directly to parsed nu and lam
    fcn_nu_val = ca.Function('fcn_nu', [lam_orig], [nu_sym])
    fcn_lam_val = ca.Function('fcn_lam', [lam_orig], [lam_sym])
    
    lam_orig_val = sol.value(lam_orig)
    nu_val = fcn_nu_val(lam_orig_val).full().flatten() if h_sym.size1() > 0 else []
    lam_val = fcn_lam_val(lam_orig_val).full().flatten() if g_sym.size1() > 0 else []
    
    grad_val = grad_L_func(x_val, theta_val, nu_val, lam_val).full().flatten()
    grad_norm = np.linalg.norm(grad_val, ord=np.inf)
    
    print("\n--- 2. Validation Results ---")
    print(f"Infinity norm of parsed Lagrangian gradient: {grad_norm:.3e}")
    
    if grad_norm < 1e-6:
        print("\nSuccess! By forcing standard forms (<= 0 or == 0), the multipliers align perfectly.")