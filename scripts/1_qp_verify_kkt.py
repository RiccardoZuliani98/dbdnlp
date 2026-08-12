import casadi as ca
import numpy as np

# Assuming the provided code is saved in modules named `qp` and `nlp`
from dbdnlp.qp import generate_qp, generate_dqp
from dbdnlp import DifferentiableNLP

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================
N_QPS = 25
NX = 20
NEQ = 5
NINEQ = 5
SEED = 42

# Theoretical properties for the base QP
LICQ = False
SCS = False
SOSC = False

# Hyperparameters
RHO_1 = 1   # Regularization parameter primal
RHO_2 = 1.0   # Regularization parameter dual
EPS_VALS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  # Perturbation magnitudes

# Data storage for the summary table
results = {
    "Reg KKT Res (Base)": []
}
# Add dynamic keys for both nominal and approximated evaluations
for eps in EPS_VALS:
    results[f"Nominal KKT (e={eps:.0e})"] = []
    results[f"Approx KKT (e={eps:.0e})"] = []

# Suppress solver printing for clean output
solver_opts = {
    # 'plugin': 'clarabel',
    # 'clarabel': {'verbose': False}
}

for i in range(N_QPS):
    # ==========================================
    # 1. BASE QP: Generate True Solution & Perturbation
    # ==========================================
    qp_base, true_sol = generate_qp(
        NX, NEQ, NINEQ, seed=SEED + i, 
        licq=LICQ, scs=SCS, sosc=SOSC
    )
    
    # Generate the perturbation direction matrices (using BASE dimensions only!)
    dqp_base = generate_dqp(NX, NEQ, NINEQ, seed=SEED + i + 2000)

    # ==========================================
    # 2. SYMBOLIC FORMULATION via theta (eps)
    # ==========================================
    x_sym = ca.MX.sym('x', NX)
    theta_sym = ca.MX.sym('eps', 1)  # Theta now represents the perturbation scalar eps

    # Parameterize the base matrices: M(eps) = M + eps * dM
    Q_sym = qp_base.Q + theta_sym * dqp_base.Q
    q_sym = qp_base.q + theta_sym * dqp_base.q
    
    A_sym = qp_base.A + theta_sym * dqp_base.A if NEQ > 0 else None
    b_sym = qp_base.b + theta_sym * dqp_base.b if NEQ > 0 else None
    
    G_sym = qp_base.G + theta_sym * dqp_base.G if NINEQ > 0 else None
    h_sym = qp_base.h + theta_sym * dqp_base.h if NINEQ > 0 else None

    # Construct parametric symbolic objective and constraints
    f_sym = 0.5 * ca.mtimes([x_sym.T, Q_sym, x_sym]) + ca.mtimes(q_sym.T, x_sym)
    g_sym = ca.mtimes(G_sym, x_sym) - h_sym if NINEQ > 0 else None
    h_eq_sym = ca.mtimes(A_sym, x_sym) - b_sym if NEQ > 0 else None

    # Instantiate the solver. The regularized problem P_2(theta, p_reg) is built automatically.
    nlp = DifferentiableNLP(
        x=x_sym, theta=theta_sym, f=f_sym, g=g_sym, h=h_eq_sym,
        solver_opts=solver_opts
    )

    # ==========================================
    # 3. REGULARIZED QP: Solve and Differentiate
    # ==========================================
    # Solve the regularized problem at the base point (eps = 0)
    eps_0 = ca.DM([0.0])
    sol_reg = nlp.solve_reg(
        theta=eps_0,
        rho_1=RHO_1,
        rho_2=RHO_2,
        x_bar=true_sol.x,
        lam_bar=true_sol.lam,
        nu_bar=true_sol.nu
    )

    # Base residual for the unperturbed regularized problem (using the original KKT!)
    max_res_base = max([float(ca.norm_inf(v)) for v in sol_reg.kkt_residuals.values()])
    results["Reg KKT Res (Base)"].append(f"{max_res_base:.2e}")
    
    # Compute the analytical directional derivative with respect to theta (eps)
    # Assuming `differentiate` returns an NLPSolution containing dx/dtheta evaluated at eps=0
    sol_reg_diff = nlp.differentiate_reg(
        theta=eps_0,
        x=sol_reg.x,
        lam=sol_reg.lam,
        nu=sol_reg.nu,
        rho_1=RHO_1,
        rho_2=RHO_2
    )
    
    # ==========================================
    # 4. TAYLOR EXPANSION VERIFICATION
    # ==========================================
    for eps in EPS_VALS:
        eps_val = ca.DM([eps])
        
        # B. 0th-Order Evaluation: Evaluate parametric original KKT at eps using NOMINAL solution
        kkt_eval_nom = nlp.kkt_eval(
            theta=eps_val, 
            x=sol_reg.x, 
            lam=sol_reg.lam, 
            nu=sol_reg.nu
        )
        max_kkt_res_nom = max([float(ca.norm_inf(v)) for v in kkt_eval_nom.values()])
        results[f"Nominal KKT (e={eps:.0e})"].append(f"{max_kkt_res_nom:.2e}")
        
        # C. 1st-Order Evaluation: Evaluate parametric original KKT at eps using APPROXIMATE solution
        x_approx   = sol_reg.x   + eps * sol_reg_diff["x"]
        lam_approx = sol_reg.lam + eps * sol_reg_diff["lam"]
        nu_approx  = sol_reg.nu  + eps * sol_reg_diff["nu"]
        
        kkt_eval_approx = nlp.kkt_eval(
            theta=eps_val, 
            x=x_approx, 
            lam=lam_approx, 
            nu=nu_approx
        )
        max_kkt_res_approx = max([float(ca.norm_inf(v)) for v in kkt_eval_approx.values()])
        results[f"Approx KKT (e={eps:.0e})"].append(f"{max_kkt_res_approx:.2e}")

# =====================================================================
# PRINT SUMMARY TABLE
# =====================================================================
print(f"Taylor Expansion Verification on {N_QPS} Regularized QPs")
print(f"Base Dimensions: NX={NX}, NEQ={NEQ}, NINEQ={NINEQ} | Regularization RHO_1={RHO_1} | Regularization RHO_2={RHO_2}")
print("=" * (25 + 12 * N_QPS))

# Build header
header = f"{'Metric':<22} |"
for i in range(N_QPS):
    header += f" {'QP ' + str(i+1):<9} |"
print(header)
print("-" * len(header))

# Print rows
for metric, values in results.items():
    row_str = f"{metric:<22} |"
    for val in values:
        row_str += f" {val:<9} |"
    print(row_str)
    
    # Add a visual divider after the base residual and after each epsilon block
    if metric == "Reg KKT Res (Base)" or metric.startswith("Approx"):
        print("-" * len(header))

print("=" * len(header))