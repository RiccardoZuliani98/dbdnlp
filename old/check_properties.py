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