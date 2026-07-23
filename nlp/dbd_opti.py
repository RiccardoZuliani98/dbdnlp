import casadi as ca
import numpy as np
from typing import Any, Dict, Optional

class CasadiSolver:
    def __init__(
        self, 
        opti: ca.Opti, 
        tolerance: float = 1e-6,  # <-- Added tolerance argument here
        solver_opts: Optional[Dict[str, Any]] = None
    ):
        self.opti = opti
        self.tolerance = tolerance  # <-- Store tolerance for the differentiator
        
        if solver_opts is None:
            solver_opts = {
                'ipopt': {'print_level': 5, 'tol': 1e-7},
                'print_time': True
            }
            
        self.opti.solver('ipopt', solver_opts)
        self.kkt_fn = self._build_differentiator()

    def _build_differentiator(self) -> ca.Function:
        x = self.opti.x
        p = self.opti.p
        f = self.opti.f
        
        g_all = self.opti.g
        lbg = self.opti.lbg
        ubg = self.opti.ubg
        
        # The pure symbol for all multipliers
        lam_all = self.opti.lam_g
        
        # --- Symbolically Classify and Extract Constraints ---
        h_sym_list = []
        nu_list = []
        
        g_sym_list = []
        lam_list = []
        
        for i in range(g_all.shape[0]):
            g_i = g_all[i]
            l_i = lbg[i]
            u_i = ubg[i]
            lam_i = lam_all[i]
            
            # 1. Check for Equality Constraints (lbg == ubg)
            is_sym_eq = ca.is_equal(l_i, u_i)
            is_num_eq = (l_i.is_constant() and u_i.is_constant() and 
                         float(l_i) == float(u_i))
            
            if is_sym_eq or is_num_eq:
                # Discard pathological cases like -inf == -inf
                if l_i.is_constant() and np.isinf(float(l_i)):
                    continue
                h_sym_list.append(g_i - u_i)
                nu_list.append(lam_i)
                
            else:
                # 2. Check for Inequality Constraints
                # Identify if bounds are infinity
                l_is_minf = l_i.is_constant() and float(l_i) == -np.inf
                u_is_pinf = u_i.is_constant() and float(u_i) == np.inf
                
                # Upper bound active: g(x) <= ubg  =>  g(x) - ubg <= 0
                if not u_is_pinf:
                    g_sym_list.append(g_i - u_i)
                    lam_list.append(lam_i)
                    
                # Lower bound active: lbg <= g(x)  =>  lbg - g(x) <= 0
                if not l_is_minf:
                    g_sym_list.append(l_i - g_i)
                    lam_list.append(-lam_i)  # Flip sign for standard G(x) <= 0 form
                    
        # --- Assemble Sub-Vectors ---
        if len(h_sym_list) > 0:
            h_sym = ca.vertcat(*h_sym_list)
            nu = ca.vertcat(*nu_list)
        else:
            h_sym, nu = ca.MX(0, 1), ca.MX(0, 1)
            
        if len(g_sym_list) > 0:
            g_sym = ca.vertcat(*g_sym_list)
            lam = ca.vertcat(*lam_list)
        else:
            g_sym, lam = ca.MX(0, 1), ca.MX(0, 1)

        nx = x.shape[0]
        rho = ca.MX.sym("rho", 1, 1)
        
        # --- Construct Lagrangian & Derivatives ---
        L = f
        if g_sym.shape[0] > 0: L += ca.dot(lam, g_sym)
        if h_sym.shape[0] > 0: L += ca.dot(nu, h_sym)
            
        H_xx, dL_dx = ca.hessian(L, x)
        
        ng = g_sym.shape[0]
        nh = h_sym.shape[0]
        
        J_g_x = ca.jacobian(g_sym, x) if ng > 0 else ca.MX(0, nx)
        J_h_x = ca.jacobian(h_sym, x) if nh > 0 else ca.MX(0, nx)
        
        H_xp = ca.jacobian(dL_dx, p)
        J_g_p = ca.jacobian(g_sym, p) if ng > 0 else ca.MX(0, p.shape[0])
        J_h_p = ca.jacobian(h_sym, p) if nh > 0 else ca.MX(0, p.shape[0])

        # --- Build KKT Matrix Blocks ---
        # Row 1: [H + rho*I, J_g^T, J_h^T]
        block11 = H_xx + rho * ca.MX.eye(nx)
        row1 = [block11]
        if ng > 0: row1.append(J_g_x.T)
        if nh > 0: row1.append(J_h_x.T)
        row1 = ca.horzcat(*row1)
        
        # Row 2: [J_g, diag(...), 0]
        if ng > 0:
            # <-- Modified section below to split based on constraint activity
            is_active = ca.fabs(g_sym) <= self.tolerance
            diag_elements = ca.if_else(is_active, -rho, -1.0 / rho - rho)
            block22 = ca.diag(diag_elements)
            # <-- End modified section
            
            row2 = [J_g_x, block22]
            if nh > 0: row2.append(ca.MX.zeros(ng, nh))
            row2 = ca.horzcat(*row2)
        else:
            row2 = None
            
        # Row 3: [J_h, 0, -rho*I]
        if nh > 0:
            block33 = -rho * ca.MX.eye(nh)
            row3 = [J_h_x]
            if ng > 0: row3.append(ca.MX.zeros(nh, ng))
            row3.append(block33)
            row3 = ca.horzcat(*row3)
        else:
            row3 = None

        # Assemble Full LHS Matrix
        matrix_blocks = [row1]
        if row2 is not None: matrix_blocks.append(row2)
        if row3 is not None: matrix_blocks.append(row3)
        KKT = ca.vertcat(*matrix_blocks)
        
        # --- Build RHS Vector ---
        rhs_blocks = [H_xp]
        if ng > 0: rhs_blocks.append(J_g_p)
        if nh > 0: rhs_blocks.append(J_h_p)
        RHS = -ca.vertcat(*rhs_blocks)
        
        # Return function using the unaltered lam_all symbol as input
        return ca.Function(
            "kkt_system",
            [x, p, lam_all, rho], [KKT, RHS],
            ["x", "p", "lam_g", "rho"], ["KKT", "RHS"]
        )

    def solve(self) -> ca.OptiSol:
        return self.opti.solve()