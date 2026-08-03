import casadi as ca
import numpy as np
import scipy.sparse.linalg as spla
from time import perf_counter


# =====================================================================
# 1. PROBLEM FORMULATION 
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
    
    x_sym, u_sym, m_sym = ca.MX.sym('x', 6, 1), ca.MX.sym('u', 2, 1), ca.MX.sym('m', 1, 1)
    vx, vy, theta, sigma = x_sym[2], x_sym[3], x_sym[4], x_sym[5]
    a, s = u_sym[0], u_sym[1]
    
    RHS_f = ca.vcat([
        vx, vy, 
        a * ca.cos(theta), a * ca.sin(theta), 
        s * (ca.cos(theta) * vx + ca.sin(theta) * vy)
    ])
    fc = ca.Function('fc', [x_sym, u_sym, m_sym], [ca.vcat([sigma * RHS_f, 0])])
    
    opti.subject_to(X[0:5, 0] == 0)
    dt = 1 / (N + 1)
    for k in range(N):
        opti.subject_to(X[:, k+1] - rk4(fc, X[:, k], U[:, k], m_param, dt) == 0)
        
    opti.subject_to(X[0:2, N] - ca.vertcat(0.5, 0.25) == 0)
    opti.subject_to(X[2:4, N] == 0)
    
    for k in range(N):
        opti.subject_to(U[0, k] - a_max <= 0)
        opti.subject_to(-U[0, k] - a_max <= 0)
        opti.subject_to(U[1, k] - s_max * m_param <= 0)
        opti.subject_to(-U[1, k] - s_max * m_param <= 0)
        
    if v_max is not None:
        for k in range(N+1):
            opti.subject_to(X[2:4, k] - v_max <= 0)
            opti.subject_to(-X[2:4, k] - v_max <= 0)
            
    e1, e2 = X[0:2, N] - ca.vertcat(0.5, 0.25), X[2:4, N]
    cost = X[-1, -1] #+ alpha * ca.sumsqr(e1) + alpha * ca.sumsqr(e2)
    opti.minimize(cost)
    
    return opti, X, U, m_param

def build_standard_differentiator_orig(opti: ca.Opti) -> ca.Function:
    """
    Builds the classic KKT differentiator for an unmodified CasADi Opti problem.
    """
    x = opti.x
    p = opti.p
    f = opti.f
    
    g = opti.g if hasattr(opti, 'g') else opti.advanced.g
    lbg = opti.lbg if hasattr(opti, 'lbg') else opti.advanced.lbg
    ubg = opti.ubg if hasattr(opti, 'ubg') else opti.advanced.ubg
    
    lam_g = ca.MX.sym('lam_g', g.shape[0]) if g.shape[0] > 0 else ca.MX(0, 1)
    
    L = f + ca.dot(lam_g, g)
    R_x = ca.gradient(L, x)
    
    R_g_list = []
    for i in range(g.shape[0]):
        g_i, l_i, u_i, lam_i = g[i], float(lbg[i]), float(ubg[i]), lam_g[i]
        if l_i == u_i:
            R_g_list.append(g_i - u_i)
        elif u_i != np.inf and l_i == -np.inf:
            R_g_list.append(lam_i * (g_i - u_i))
        elif l_i != -np.inf and u_i == np.inf:
            R_g_list.append(lam_i * (g_i - l_i))
        else:
            raise ValueError(f"Two-sided bounds not supported: [{l_i}, {u_i}]")
            
    R_g = ca.vertcat(*R_g_list) if R_g_list else ca.MX(0, 1)
    
    R = ca.vertcat(R_x, R_g)
    y = ca.vertcat(x, lam_g) if lam_g.shape[0] > 0 else x
    
    KKT = ca.jacobian(R, y)
    RHS = -ca.jacobian(R, p)
    
    return ca.Function('kkt_std_orig', [x, p, lam_g], [KKT, RHS], ['x', 'p', 'lam_g'], ['KKT', 'RHS'])


# =====================================================================
# 2. REGULARIZED OPTI WRAPPER
# =====================================================================

class RegularizedOpti:
    def __init__(self, original_opti, p_opts=None, s_opts=None):
        self.original_opti = original_opti
        self.opti = ca.Opti()
        
        x_orig = original_opti.x
        theta_orig = original_opti.p
        f_orig = original_opti.f
        
        g_orig = original_opti.g if hasattr(original_opti, 'g') else original_opti.advanced.g
        lbg_orig = original_opti.lbg if hasattr(original_opti, 'lbg') else original_opti.advanced.lbg
        ubg_orig = original_opti.ubg if hasattr(original_opti, 'ubg') else original_opti.advanced.ubg
        lam_orig = original_opti.lam_g if hasattr(original_opti, 'lam_g') else original_opti.advanced.lam_g
        
        h_sym_list, nu_list, g_sym_list, lam_list = [], [], [], []

        for i in range(g_orig.shape[0]):
            g_i, l_i, u_i, lam_i = g_orig[i], float(lbg_orig[i]), float(ubg_orig[i]), lam_orig[i]
            assert u_i == 0.0, f"Expected upper bound 0.0, got {u_i} at index {i}"
            if l_i == 0.0:
                h_sym_list.append(g_i); nu_list.append(lam_i)
            elif l_i == -np.inf:
                g_sym_list.append(g_i); lam_list.append(lam_i)
            else:
                raise ValueError(f"Unexpected bounds: [{l_i}, {u_i}]")
        
        g_expr = ca.vertcat(*g_sym_list) if g_sym_list else ca.MX(0, 1)
        h_expr = ca.vertcat(*h_sym_list) if h_sym_list else ca.MX(0, 1)
        
        self.n_g, self.n_h = g_expr.size1(), h_expr.size1()
        self.n_c = self.n_g + self.n_h
        
        self.z_orig_expr = g_expr
        self.mu_orig_expr = ca.vertcat(*(lam_list + nu_list)) if self.n_c > 0 else ca.MX(0, 1)
        
        self.x = self.opti.variable(x_orig.size1(), x_orig.size2())
        self.z = self.opti.variable(self.n_g) if self.n_g > 0 else ca.MX(0, 1)
        self.mu = self.opti.variable(self.n_c) if self.n_c > 0 else ca.MX(0, 1)
        
        self.theta = self.opti.parameter(theta_orig.size1(), theta_orig.size2())
        self.x_bar = self.opti.parameter(x_orig.size1(), x_orig.size2())
        self.z_bar = self.opti.parameter(self.n_g) if self.n_g > 0 else ca.MX(0, 1)
        self.mu_bar = self.opti.parameter(self.n_c) if self.n_c > 0 else ca.MX(0, 1)
        self.rho = self.opti.parameter()
        
        v_old = ca.vertcat(ca.vec(x_orig), ca.vec(theta_orig))
        v_new = ca.vertcat(ca.vec(self.x), ca.vec(self.theta))
        
        self.f_new = ca.substitute(f_orig, v_old, v_new)
        self.g_new = ca.substitute(g_expr, v_old, v_new) if self.n_g > 0 else ca.MX(0, 1)
        self.h_new = ca.substitute(h_expr, v_old, v_new) if self.n_h > 0 else ca.MX(0, 1)
        
        reg_x = ca.sumsqr(self.x - self.x_bar)
        reg_z = ca.sumsqr(self.z - self.z_bar) if self.n_g > 0 else 0
        reg_mu = ca.sumsqr(self.mu) if self.n_c > 0 else 0
        
        self.opti.minimize(self.f_new + (self.rho / 2) * (reg_x + reg_z + reg_mu))
        
        if self.n_c > 0:
            c_components = []
            if self.n_g > 0: c_components.append(self.g_new - self.z)
            if self.n_h > 0: c_components.append(self.h_new)
            self.opti.subject_to(ca.vertcat(*c_components) + self.rho * (self.mu_bar - self.mu) == 0)
            
        if self.n_g > 0: self.opti.subject_to(self.z <= 0)
            
        self.opti.solver('ipopt', p_opts or {}, s_opts or {})

    def solve(self, theta_val, x_bar_val, z_bar_val, mu_bar_val, rho_val):
        self.opti.set_value(self.theta, theta_val)
        self.opti.set_value(self.rho, rho_val)
        
        self.opti.set_value(self.x_bar, x_bar_val)
        self.opti.set_initial(self.x, x_bar_val)
        
        if self.z.size1() > 0:
            self.opti.set_value(self.z_bar, z_bar_val)
            self.opti.set_initial(self.z, z_bar_val)
        if self.mu.size1() > 0:
            self.opti.set_value(self.mu_bar, mu_bar_val)
            self.opti.set_initial(self.mu, mu_bar_val)
            
        return self.opti.solve()

    def build_standard_differentiator(self) -> ca.Function:
        """
        Builds the classic KKT differentiator for the Regularized Opti problem.
        """
        x = self.opti.x
        p = self.opti.p
        f = self.opti.f
        
        g = self.opti.g 
        lbg = self.opti.lbg 
        ubg = self.opti.ubg 
        
        lam_g = ca.MX.sym('lam_g', g.shape[0]) if g.shape[0] > 0 else ca.MX(0, 1)
        
        L = f + ca.dot(lam_g, g)
        R_x = ca.gradient(L, x)
        
        R_g_list = []
        for i in range(g.shape[0]):
            g_i, l_i, u_i, lam_i = g[i], float(lbg[i]), float(ubg[i]), lam_g[i]
            
            if l_i == u_i:
                R_g_list.append(g_i - u_i)
            elif u_i != np.inf and l_i == -np.inf:
                R_g_list.append(lam_i * (g_i - u_i))
            elif l_i != -np.inf and u_i == np.inf:
                R_g_list.append(lam_i * (g_i - l_i))
            else:
                raise ValueError(f"Two-sided bounds not supported: [{l_i}, {u_i}]")
                
        R_g = ca.vertcat(*R_g_list) if R_g_list else ca.MX(0, 1)
        
        R = ca.vertcat(R_x, R_g)
        y = ca.vertcat(x, lam_g) if lam_g.shape[0] > 0 else x
        
        KKT = ca.jacobian(R, y)
        RHS = -ca.jacobian(R, self.theta)
        
        return ca.Function('kkt_std_reg', [x, p, lam_g], [KKT, RHS], ['x', 'p', 'lam_g'], ['KKT', 'RHS'])


# =====================================================================
# 3. MAIN SCRIPT
# =====================================================================
if __name__ == "__main__":
    # Settings
    N_horizon = 125
    mass_value = 1.5
    rho_reg = 1e-6
    fd_eps = 1e-5
    
    # -----------------------------------------------------------------
    # 1. Build & Solve Original Problem
    # -----------------------------------------------------------------
    print("Building Original Problem Formulation...")
    opti_orig, X_var, U_var, m_param = build_dubins_problem(N=N_horizon,v_max=1.0)
    opti_orig.set_value(m_param, mass_value)
    opti_orig.set_initial(X_var, 0.075)
    opti_orig.set_initial(U_var, 0.075)
    
    print("Solving Original Nominal NLP...")
    opts = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': False}
    opti_orig.solver('ipopt', opts, opts['ipopt'])
    sol_orig = opti_orig.solve()
    print(f"Original Solve Successful! Final Cost: {sol_orig.value(opti_orig.f):.4f}")

    # Extract original variables safely
    x_bar_val = sol_orig.value(opti_orig.x)
    try:
        lam_orig_val = sol_orig.value(opti_orig.advanced.lam_g)
    except AttributeError:
        lam_orig_val = sol_orig.value(opti_orig.lam_g)

    # -----------------------------------------------------------------
    # 2. Evaluate Analytical KKT Sensitivities (Original NLP)
    # -----------------------------------------------------------------
    print("\nEvaluating Standard KKT sensitivities on Original NLP...")
    kkt_orig_fn = build_standard_differentiator_orig(opti_orig)
    result_orig = kkt_orig_fn(x=x_bar_val, p=mass_value, lam_g=lam_orig_val)
    
    orig_kkt_success = False
    dx_dp_orig_analytical = None
    try:
        start = perf_counter()
        # d_vars_dp_orig = ca.solve(result_orig["KKT"], result_orig["RHS"], "csparse")
        d_vars_dp_orig = ca.DM(np.linalg.lstsq(result_orig["KKT"].full(), result_orig["RHS"].full())[0])
        dx_dp_orig_analytical = d_vars_dp_orig[:opti_orig.x.shape[0]].full().flatten()
        print(f"Elapsed time (Original Analytical KKT): {perf_counter()-start:.4f} sec")
        orig_kkt_success = True
    except Exception as e:
        print("WARNING: Original KKT Matrix is mathematically SINGULAR at the optimum!")
        print("This is exactly why the regularized approximation is necessary.")
    
    # -----------------------------------------------------------------
    # 3. Instantiate Regularized Solver & Map Reference Points
    # -----------------------------------------------------------------
    print("\nInitializing Regularized Opti...")
    reg_opti = RegularizedOpti(opti_orig, p_opts={'print_time': False}, s_opts={'print_level': 0, 'sb': 'yes'})
    
    fcn_z = ca.Function('fcn_z', [opti_orig.x, opti_orig.p], [reg_opti.z_orig_expr])
    fcn_mu = ca.Function('fcn_mu', [opti_orig.advanced.lam_g if hasattr(opti_orig.advanced, 'lam_g') else opti_orig.lam_g], [reg_opti.mu_orig_expr])
    
    z_bar_val = fcn_z(x_bar_val, mass_value).full().flatten() if reg_opti.n_g > 0 else []
    mu_bar_val = fcn_mu(lam_orig_val).full().flatten() if reg_opti.n_c > 0 else []

    print("Solving Regularized NLP at nominal point...")
    sol_reg = reg_opti.solve(mass_value, x_bar_val, z_bar_val, mu_bar_val, rho_reg)
    
    # -----------------------------------------------------------------
    # 4. Evaluate Analytical KKT Sensitivities (Regularized NLP)
    # -----------------------------------------------------------------
    print("Evaluating Standard KKT sensitivities on Regularized NLP...")
    kkt_reg_fn = reg_opti.build_standard_differentiator()
    start = perf_counter()
    
    x_full = sol_reg.value(reg_opti.opti.x)
    p_full = sol_reg.value(reg_opti.opti.p)
    try:
        lam_full = sol_reg.value(reg_opti.opti.advanced.lam_g)
    except AttributeError:
        lam_full = sol_reg.value(reg_opti.opti.lam_g)
        
    result_reg = kkt_reg_fn(x=x_full, p=p_full, lam_g=lam_full)
    d_vars_dp_reg = ca.solve(result_reg["KKT"], result_reg["RHS"], "csparse")
    print(f"Elapsed time (Regularized Analytical KKT): {perf_counter()-start:.4f} sec")
    
    nx = x_bar_val.shape[0]
    dx_dp_analytical = d_vars_dp_reg[:nx].full().flatten()

    # -----------------------------------------------------------------
    # 5. Central Finite Difference Approximation (Regularized)
    # -----------------------------------------------------------------
    print(f"\nRunning central FD on Regularized NLP (eps = {fd_eps})...")
    
    sol_plus = reg_opti.solve(mass_value + fd_eps, x_bar_val, z_bar_val, mu_bar_val, rho_reg)
    x_plus = sol_plus.value(reg_opti.x)
    
    sol_minus = reg_opti.solve(mass_value - fd_eps, x_bar_val, z_bar_val, mu_bar_val, rho_reg)
    x_minus = sol_minus.value(reg_opti.x)[:nx]  # Note: only slice primal x
    
    dx_dp_fd_reg = (x_plus[:nx] - x_minus[:nx]) / (2 * fd_eps)

    # -----------------------------------------------------------------
    # 6. Central Finite Difference Approximation (Original Problem)
    # -----------------------------------------------------------------
    print(f"Running central FD on Original NLP (eps = {fd_eps})...")

    # Forward solve (Original)
    opti_orig.set_value(m_param, mass_value + fd_eps)
    opti_orig.set_initial(opti_orig.x, x_bar_val)
    if hasattr(opti_orig.advanced, 'lam_g'):
        opti_orig.set_initial(opti_orig.advanced.lam_g, lam_orig_val)
    else:
        opti_orig.set_initial(opti_orig.lam_g, lam_orig_val)
    sol_orig_plus = opti_orig.solve()
    x_orig_plus = sol_orig_plus.value(opti_orig.x)
    
    # Backward solve (Original)
    opti_orig.set_value(m_param, mass_value - fd_eps)
    opti_orig.set_initial(opti_orig.x, x_bar_val)
    if hasattr(opti_orig.advanced, 'lam_g'):
        opti_orig.set_initial(opti_orig.advanced.lam_g, lam_orig_val)
    else:
        opti_orig.set_initial(opti_orig.lam_g, lam_orig_val)
    sol_orig_minus = opti_orig.solve()
    x_orig_minus = sol_orig_minus.value(opti_orig.x)

    dx_dp_fd_orig = (x_orig_plus - x_orig_minus) / (2 * fd_eps)

    # -----------------------------------------------------------------
    # 7. Verification & Comparison
    # -----------------------------------------------------------------
    print("\n=========================================")
    print(" SENSITIVITY VERIFICATION RESULTS")
    print("=========================================")
    
    abs_err_reg = np.abs(dx_dp_analytical - dx_dp_fd_reg)
    abs_err_orig = np.abs(dx_dp_analytical - dx_dp_fd_orig)
    abs_err_fds = np.abs(dx_dp_fd_reg - dx_dp_fd_orig)
    
    print("[1] Analytical Reg vs Regularized FD:")
    print(f"    Max Absolute Error:  {np.max(abs_err_reg):.2e}")
    print(f"    Mean Absolute Error: {np.mean(abs_err_reg):.2e}")
    
    print("\n[2] Analytical Reg vs Original FD:")
    print(f"    Max Absolute Error:  {np.max(abs_err_orig):.2e}")
    print(f"    Mean Absolute Error: {np.mean(abs_err_orig):.2e}")

    print("\n[3] Regularized FD vs Original FD:")
    print(f"    Max Absolute Error:  {np.max(abs_err_fds):.2e}")
    print(f"    Mean Absolute Error: {np.mean(abs_err_fds):.2e}")

    print("\n[4] Original Analytical vs Original FD:")
    if orig_kkt_success:
        abs_err_orig_ana_vs_orig_fd = np.abs(dx_dp_orig_analytical - dx_dp_fd_orig)
        print(f"    Max Absolute Error:  {np.max(abs_err_orig_ana_vs_orig_fd):.2e}")
        print(f"    Mean Absolute Error: {np.mean(abs_err_orig_ana_vs_orig_fd):.2e}")
    else:
        print("    (Skipped: Original unregularized KKT system was mathematically singular)")
    
    print("\n[5] Derivative Norms:")
    print(f"    Regularized Analytical: {np.linalg.norm(dx_dp_analytical):.4f}")
    if orig_kkt_success:
        print(f"    Original Analytical:    {np.linalg.norm(dx_dp_orig_analytical):.4f}")
    else:
        print("    Original Analytical:    [FAILED/SINGULAR]")
    print(f"    Regularized FD norm:    {np.linalg.norm(dx_dp_fd_reg):.4f}")
    print(f"    Original FD norm:       {np.linalg.norm(dx_dp_fd_orig):.4f}")