import casadi as ca
from src import QPSolver, generate_qp, generate_dqp, perturb_qp, regularize_qp

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================
N_QPS = 5
NX = 40
NEQ = 10
NINEQ = 10
SEED = 42

# Theoretical properties
LICQ = False
SCS = False
SOSC = False

# Hyperparameters
EPS = 1e-8         # Finite difference step size
RHO = 1e-6         # Regularization parameter
TOLERANCE = 1e-12   # Tolerance for checking constraints

# Data storage for the summary table
results = {
    "Base KKT Res": [],
    "Reg KKT Res": [],
    "Sol Diff (max)": [],
    "Base Err dx": [],
    "Base Err dmu": [],
    "Base Err dlam": [],
    "Reg Err dx": [],
    "Reg Err dmu": [],
    "Reg Err dlam": [],
    "FD Diff dx": [],    # <-- NEW
    "FD Diff dmu": [],   # <-- NEW
    "FD Diff dlam": [],  # <-- NEW
    "Base Diff Unique": [],
    "Reg Diff Unique": [],
    "Strong Convex": [],
    "LICQ": [],
    "SCS": []
}

# Supress solver printing for clean output
opts = {
    # 'printLevel': 'none',
    "plugin":"daqp"
}

for i in range(N_QPS):
    # ==========================================
    # 1. BASE QP: Generate, Solve, Differentiate
    # ==========================================
    qp, sol = generate_qp(
        NX, NEQ, NINEQ, seed=SEED + i, 
        licq=LICQ, scs=SCS, sosc=SOSC
    )
    
    solver = QPSolver(qp, options=opts)
    # sol = solver.solve(warmstart=sol0)
    
    max_res = max([float(ca.norm_inf(v)) for v in sol.residual.values()])
    
    dqp = generate_dqp(NX, NEQ, NINEQ, seed=SEED + i + 1000)
    sol_diff = solver.differentiate(sol, dqp, qp, lstsq=True)
    
    # -------------------------------------------------------------
    # Two-Sided Finite Difference for Base QP
    # -------------------------------------------------------------
    qp_pert_plus = perturb_qp(qp, dqp, EPS)
    sol_pert_plus = solver.solve(qp=qp_pert_plus, warmstart=sol)
    
    qp_pert_minus = perturb_qp(qp, dqp, -EPS)
    sol_pert_minus = solver.solve(qp=qp_pert_minus, warmstart=sol)
    
    dx_fd = (sol_pert_plus.x - sol_pert_minus.x) / (2 * EPS)
    dmu_fd = (sol_pert_plus.mu - sol_pert_minus.mu) / (2 * EPS)
    dlam_fd = (sol_pert_plus.lam - sol_pert_minus.lam) / (2 * EPS)
    
    err_x = float(ca.norm_inf(sol_diff.x - dx_fd))
    err_mu = float(ca.norm_inf(sol_diff.mu - dmu_fd))
    err_lam = float(ca.norm_inf(sol_diff.lam - dlam_fd))
    
    # ==========================================
    # 2. REGULARIZED QP: Generate, Solve, Diff
    # ==========================================
    qp_reg = regularize_qp(qp, sol, RHO)
    solver_reg = QPSolver(qp_reg, options=opts)
    sol_reg = solver_reg.solve()
    
    max_res_reg = max([float(ca.norm_inf(v)) for v in sol_reg.residual.values()])
    
    # Extract equivalent original variables from the augmented primal state X = [x, lam, mu, z]
    reg_x   = sol_reg.x[0 : NX]
    reg_lam = sol_reg.x[NX : NX + NINEQ]
    reg_mu  = sol_reg.x[NX + NINEQ : NX + NINEQ + NEQ]
    
    # Check how close the regularized solution is to the base solution
    diff_x = float(ca.norm_inf(sol.x - reg_x))
    diff_lam = float(ca.norm_inf(sol.lam - reg_lam))
    diff_mu = float(ca.norm_inf(sol.mu - reg_mu))
    max_sol_diff = max(diff_x, diff_lam, diff_mu)
    
    # Differentiate the regularized problem using a matching augmented perturbation
    NX_reg = NX + NINEQ + NEQ + NINEQ
    NEQ_reg = NEQ + NINEQ
    NINEQ_reg = NINEQ
    
    dqp_reg = generate_dqp(NX_reg, NEQ_reg, NINEQ_reg, seed=SEED + i + 2000)
    sol_reg_diff = solver_reg.differentiate(sol_reg, dqp_reg, qp_reg, lstsq=True)
    
    # -------------------------------------------------------------
    # Two-Sided Finite Difference for Regularized QP
    # In the regularized problem, X = [x, lam, mu, z] are ALL primal.
    # -------------------------------------------------------------
    qp_reg_pert_plus = perturb_qp(qp_reg, dqp_reg, EPS)
    sol_reg_pert_plus = solver_reg.solve(qp=qp_reg_pert_plus, warmstart=sol_reg)
    
    qp_reg_pert_minus = perturb_qp(qp_reg, dqp_reg, -EPS)
    sol_reg_pert_minus = solver_reg.solve(qp=qp_reg_pert_minus, warmstart=sol_reg)
    
    # Finite Difference of primal variables using central diff: (f(x+h) - f(x-h)) / 2h
    dx_fd_reg_x   = (sol_reg_pert_plus.x[0 : NX] - sol_reg_pert_minus.x[0 : NX]) / (2 * EPS)
    dx_fd_reg_lam = (sol_reg_pert_plus.x[NX : NX + NINEQ] - sol_reg_pert_minus.x[NX : NX + NINEQ]) / (2 * EPS)
    dx_fd_reg_mu  = (sol_reg_pert_plus.x[NX + NINEQ : NX + NINEQ + NEQ] - sol_reg_pert_minus.x[NX + NINEQ : NX + NINEQ + NEQ]) / (2 * EPS)

    # Analytical Derivative of primal variables
    diff_reg_x   = sol_reg_diff.x[0 : NX]
    diff_reg_lam = sol_reg_diff.x[NX : NX + NINEQ]
    diff_reg_mu  = sol_reg_diff.x[NX + NINEQ : NX + NINEQ + NEQ]
    
    err_x_reg   = float(ca.norm_inf(diff_reg_x - dx_fd_reg_x))
    err_mu_reg  = float(ca.norm_inf(diff_reg_mu - dx_fd_reg_mu))
    err_lam_reg = float(ca.norm_inf(diff_reg_lam - dx_fd_reg_lam))
    
    # -------------------------------------------------------------
    # Compare Base FD vs Regularized FD
    # -------------------------------------------------------------
    fd_diff_x   = float(ca.norm_inf(dx_fd - dx_fd_reg_x))
    fd_diff_mu  = float(ca.norm_inf(dmu_fd - dx_fd_reg_mu))
    fd_diff_lam = float(ca.norm_inf(dlam_fd - dx_fd_reg_lam))
    
    # ==========================================
    # 3. Store Results
    # ==========================================
    results["Base KKT Res"].append(f"{max_res:.2e}")
    results["Reg KKT Res"].append(f"{max_res_reg:.2e}")
    results["Sol Diff (max)"].append(f"{max_sol_diff:.2e}")
    
    results["Base Err dx"].append(f"{err_x:.2e}")
    results["Base Err dmu"].append(f"{err_mu:.2e}")
    results["Base Err dlam"].append(f"{err_lam:.2e}")
    
    results["Reg Err dx"].append(f"{err_x_reg:.2e}")
    results["Reg Err dmu"].append(f"{err_mu_reg:.2e}")
    results["Reg Err dlam"].append(f"{err_lam_reg:.2e}")
    
    results["FD Diff dx"].append(f"{fd_diff_x:.2e}")
    results["FD Diff dmu"].append(f"{fd_diff_mu:.2e}")
    results["FD Diff dlam"].append(f"{fd_diff_lam:.2e}")
    
    # Store Uniqueness Flags
    results["Base Diff Unique"].append(str(sol_diff.properties.get('unique_solution', 'N/A')))
    results["Reg Diff Unique"].append(str(sol_reg_diff.properties.get('unique_solution', 'N/A')))
    
    results["Strong Convex"].append(str(sol.properties['strong_convexity']))
    results["LICQ"].append(str(sol.properties['licq']))
    results["SCS"].append(str(sol.properties['scs']))

# =====================================================================
# PRINT SUMMARY TABLE
# =====================================================================
print(f"Testing {N_QPS} QPs (Two-Sided FD EPS={EPS}, REG RHO={RHO})")
print("=" * (21 + 12 * N_QPS))

# Build header
header = f"{'Metric':<18} |"
for i in range(N_QPS):
    header += f" {'QP ' + str(i+1):<9} |"
print(header)
print("-" * len(header))

# Print rows
for metric, values in results.items():
    row_str = f"{metric:<18} |"
    for val in values:
        row_str += f" {val:<9} |"
    print(row_str)
    
    # Add visual dividers between logical sections
    if metric in ["Sol Diff (max)", "Base Err dlam", "Reg Err dlam", "FD Diff dlam", "Reg Diff Unique"]:
        print("-" * len(header))

print("=" * len(header))