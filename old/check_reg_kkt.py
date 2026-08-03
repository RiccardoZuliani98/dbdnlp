import casadi as ca
from src import QPSolver, generate_qp, generate_dqp, perturb_qp, regularize_qp, evaluate_kkt, QPSolution

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
RHO = 1  # Regularization parameter
EPS_VALS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  # Perturbation magnitudes

# Data storage for the summary table
results = {
    "Reg KKT Res (Base)": []
}
# Add dynamic keys for both nominal and approximated evaluations
for eps in EPS_VALS:
    results[f"Nominal KKT (e={eps:.0e})"] = []
    results[f"Approx KKT (e={eps:.0e})"] = []

# Supress solver printing for clean output
# opts = {'printLevel': 'none'}
opts = {"solver":"piqp"}

for i in range(N_QPS):
    # ==========================================
    # 1. BASE QP: Generate and regularize
    # ==========================================
    qp_base, true_sol = generate_qp(
        NX, NEQ, NINEQ, seed=SEED + i, 
        licq=LICQ, scs=SCS, sosc=SOSC
    )
    
    # Construct the regularized problem
    qp_reg = regularize_qp(qp_base, true_sol, RHO)
    
    # ==========================================
    # 2. REGULARIZED QP: Solve and Differentiate
    # ==========================================
    solver_reg = QPSolver(qp_reg, options=opts)
    
    # Construct Warmstart for the regularized problem
    # Augmented Primal: X = [x, lam, mu, z]
    z_bar = ca.mtimes(qp_base.G, true_sol.x) - qp_base.h
    x_warm = ca.vertcat(true_sol.x, true_sol.lam, true_sol.mu, z_bar)
    
    # Dual variables for the regularized problem (initialized to zero)
    mu_warm = ca.DM.zeros(NEQ + NINEQ, 1)
    lam_warm = ca.DM.zeros(NINEQ, 1)
    
    warmstart_reg = QPSolution(
        x=x_warm,
        mu=mu_warm,
        lam=lam_warm,
        residual={},
        properties={}
    )
    
    # Solve with the warmstart
    sol_reg = solver_reg.solve(warmstart=warmstart_reg)

    
    # Base residual for the unperturbed regularized problem
    max_res_base = max([float(ca.norm_inf(v)) for v in sol_reg.residual.values()])
    results["Reg KKT Res (Base)"].append(f"{max_res_base:.2e}")
    
    # Dimensions of the augmented regularized problem
    NX_reg = NX + NINEQ + NEQ + NINEQ
    NEQ_reg = NEQ + NINEQ
    NINEQ_reg = NINEQ
    
    # Generate the perturbation matrices for the regularized problem
    dqp_reg = generate_dqp(NX_reg, NEQ_reg, NINEQ_reg, seed=SEED + i + 2000)
    
    # Compute the analytical directional derivative
    # sol_reg_diff = solver_reg.differentiate_alt(sol_reg, dqp_reg, qp_reg, lstsq=True)
    sol_reg_diff = solver_reg.differentiate(sol_reg, dqp_reg, qp_reg, lstsq=True)
    
    # ==========================================
    # 3. TAYLOR EXPANSION VERIFICATION
    # ==========================================
    for eps in EPS_VALS:
        # A. Create the strictly modified (perturbed) QP
        qp_mod = perturb_qp(qp_reg, dqp_reg, eps)
        
        # B. 0th-Order Evaluation: Evaluate modified QP using NOMINAL solution
        kkt_eval_nom = evaluate_kkt(qp_mod, sol_reg.x, sol_reg.lam, sol_reg.mu)
        max_kkt_res_nom = max([float(ca.norm_inf(v)) for v in kkt_eval_nom.values()])
        results[f"Nominal KKT (e={eps:.0e})"].append(f"{max_kkt_res_nom:.2e}")
        
        # C. 1st-Order Evaluation: Evaluate modified QP using APPROXIMATE solution
        x_approx   = sol_reg.x + eps * sol_reg_diff.x
        mu_approx  = sol_reg.mu + eps * sol_reg_diff.mu
        lam_approx = sol_reg.lam + eps * sol_reg_diff.lam
        
        kkt_eval_approx = evaluate_kkt(qp_mod, x_approx, lam_approx, mu_approx)
        max_kkt_res_approx = max([float(ca.norm_inf(v)) for v in kkt_eval_approx.values()])
        results[f"Approx KKT (e={eps:.0e})"].append(f"{max_kkt_res_approx:.2e}")

# =====================================================================
# PRINT SUMMARY TABLE
# =====================================================================
print(f"Taylor Expansion Verification on {N_QPS} Regularized QPs")
print(f"Base Dimensions: NX={NX}, NEQ={NEQ}, NINEQ={NINEQ} | Regularization RHO={RHO}")
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