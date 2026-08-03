import casadi as ca
import numpy as np

class RegularizedOpti:
    def __init__(self, original_opti, p_opts=None, s_opts=None):
        self.original_opti = original_opti
        self.opti = ca.Opti()
        
        # 1. Extract base components
        x_orig = original_opti.x
        theta_orig = original_opti.p
        f_orig = original_opti.f
        
        g_orig = original_opti.g if hasattr(original_opti, 'g') else original_opti.advanced.g
        lbg_orig = original_opti.lbg if hasattr(original_opti, 'lbg') else original_opti.advanced.lbg
        ubg_orig = original_opti.ubg if hasattr(original_opti, 'ubg') else original_opti.advanced.ubg
        lam_orig = original_opti.lam_g if hasattr(original_opti, 'lam_g') else original_opti.advanced.lam_g
        
        # 2. Parse constraints symbolically (Strictly <= 0 or == 0)
        h_sym_list = []
        nu_list = []
        g_sym_list = []
        lam_list = []

        for i in range(g_orig.shape[0]):
            g_i = g_orig[i]
            l_i = float(lbg_orig[i])
            u_i = float(ubg_orig[i])
            lam_i = lam_orig[i]

            assert u_i == 0.0, f"Expected upper bound 0.0, got {u_i} at index {i}. Ensure original constraints are formulated as <= 0 or == 0."

            if l_i == 0.0:
                # Equality constraint: 0 <= g_i <= 0
                h_sym_list.append(g_i)
                nu_list.append(lam_i)
            elif l_i == -np.inf:
                # Inequality constraint: -inf <= g_i <= 0
                g_sym_list.append(g_i)
                lam_list.append(lam_i)
            else:
                raise ValueError(f"Unexpected constraint bounds: [{l_i}, {u_i}].")
        
        g_expr = ca.vertcat(*g_sym_list) if g_sym_list else ca.MX(0, 1)
        h_expr = ca.vertcat(*h_sym_list) if h_sym_list else ca.MX(0, 1)
        
        n_g = g_expr.size1()
        n_h = h_expr.size1()
        n_c = n_g + n_h
        
        # Save original symbolic expressions so we can extract exact values from the original solution
        self.z_orig_expr = g_expr
        self.mu_orig_expr = ca.vertcat(*(lam_list + nu_list)) if n_c > 0 else ca.MX(0, 1)
        
        # 3. Define new Decision Variables
        self.x = self.opti.variable(x_orig.size1(), x_orig.size2())
        self.z = self.opti.variable(n_g) if n_g > 0 else ca.MX(0, 1)
        self.mu = self.opti.variable(n_c) if n_c > 0 else ca.MX(0, 1)
        
        # 4. Define new Parameters
        self.theta = self.opti.parameter(theta_orig.size1(), theta_orig.size2())
        self.x_bar = self.opti.parameter(x_orig.size1(), x_orig.size2())
        self.z_bar = self.opti.parameter(n_g) if n_g > 0 else ca.MX(0, 1)
        self.mu_bar = self.opti.parameter(n_c) if n_c > 0 else ca.MX(0, 1)
        self.rho = self.opti.parameter()
        
        # 5. Substitute original variables
        v_old = ca.vertcat(ca.vec(x_orig), ca.vec(theta_orig))
        v_new = ca.vertcat(ca.vec(self.x), ca.vec(self.theta))
        
        f_new = ca.substitute(f_orig, v_old, v_new)
        g_new = ca.substitute(g_expr, v_old, v_new) if n_g > 0 else ca.MX(0, 1)
        h_new = ca.substitute(h_expr, v_old, v_new) if n_h > 0 else ca.MX(0, 1)
        
        # 6. Formulate Objective
        reg_x = ca.sumsqr(self.x - self.x_bar)
        reg_z = ca.sumsqr(self.z - self.z_bar) if n_g > 0 else 0
        reg_mu = ca.sumsqr(self.mu) if n_c > 0 else 0
        
        self.opti.minimize(f_new + (self.rho / 2) * (reg_x + reg_z + reg_mu))
        
        # 7. Formulate Constraints
        if n_c > 0:
            c_components = []
            if n_g > 0:
                c_components.append(g_new - self.z)
            if n_h > 0:
                c_components.append(h_new)
            
            c_theta = ca.vertcat(*c_components)
            self.opti.subject_to(c_theta + self.rho * (self.mu_bar - self.mu) == 0)
            
        if n_g > 0:
            self.opti.subject_to(self.z <= 0)
            
        # 8. Apply Solver Options
        p_opts = p_opts if p_opts is not None else {}
        s_opts = s_opts if s_opts is not None else {}
        self.opti.solver('ipopt', p_opts, s_opts)

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