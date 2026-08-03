import casadi as ca
from src import QPSolver, generate_qp, generate_dqp, perturb_qp

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

# Finite difference step size
EPS = 1e-7

# Tolerance when checking for LICQ violations
TOLERANCE = 1e-8

# Data storage for the summary table
results = {
    "KKT Res (max)": [],
    "Err dx": [],
    "Err dmu": [],
    "Err dlam": [],
    "Strong Convex": [],
    "LICQ": [],
    "SCS": []
}

# Supress solver printing for clean output
opts = {'printLevel': 'none'}

for i in range(N_QPS):
    # 1. Generate Base QP
    qp = generate_qp(
        NX, NEQ, NINEQ, seed=SEED + i, 
        licq=LICQ, scs=SCS, sosc=SOSC
    )
    
    # 2. Solve Base QP
    solver = QPSolver(qp, options=opts, tol=TOLERANCE)
    sol = solver.solve()
    
    # Check KKT Residuals
    max_res = max([float(ca.norm_inf(v)) for v in sol.residual.values()])
    assert max_res < 1e-6, f"Base QP solution {i+1} is not optimal! Max res: {max_res}"
    
    # 3. Generate Perturbation
    dqp = generate_dqp(NX, NEQ, NINEQ, seed=SEED + i + 1000)
    
    # 4. Analytical Derivative (IFT)
    sol_diff = solver.differentiate(sol, dqp, qp)
    
    # 5. Numerical Derivative (Finite Differences)
    qp_pert = perturb_qp(qp, dqp, EPS)
    sol_pert = solver.solve(qp=qp_pert, warmstart=sol) # Using warmstart!
    
    dx_fd = (sol_pert.x - sol.x) / EPS
    dmu_fd = (sol_pert.mu - sol.mu) / EPS
    dlam_fd = (sol_pert.lam - sol.lam) / EPS
    
    # 6. Compare & Report Errors
    err_x = float(ca.norm_inf(sol_diff.x - dx_fd))
    err_mu = float(ca.norm_inf(sol_diff.mu - dmu_fd))
    err_lam = float(ca.norm_inf(sol_diff.lam - dlam_fd))
    
    # 7. Store iteration results
    results["KKT Res (max)"].append(f"{max_res:.2e}")
    results["Err dx"].append(f"{err_x:.2e}")
    results["Err dmu"].append(f"{err_mu:.2e}")
    results["Err dlam"].append(f"{err_lam:.2e}")
    
    # 8. Store Properties (extracted from sol.properties)
    results["Strong Convex"].append(str(sol.properties['strong_convexity']))
    results["LICQ"].append(str(sol.properties['licq']))
    results["SCS"].append(str(sol.properties['scs']))

# =====================================================================
# PRINT SUMMARY TABLE
# =====================================================================
print(f"Testing {N_QPS} QPs with FD step EPS={EPS}")
print("=" * (18 + 12 * N_QPS))

# Build header
header = f"{'Metric':<15} |"
for i in range(N_QPS):
    header += f" {'QP ' + str(i+1):<9} |"
print(header)
print("-" * len(header))

# Print rows
for metric, values in results.items():
    row_str = f"{metric:<15} |"
    for val in values:
        row_str += f" {val:<9} |"
    print(row_str)
    
    # Add a visual divider after the numerical errors, before the booleans
    if metric == "Err dlam":
        print("-" * len(header))

print("=" * len(header))