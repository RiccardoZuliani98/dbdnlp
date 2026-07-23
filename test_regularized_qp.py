import pytest
import casadi as ca
import numpy as np

# Assuming the provided code is saved in a module named `src`
from src import QPSolver, generate_qp, regularize_qp, evaluate_kkt

# =====================================================================
# TEST CONFIGURATIONS
# =====================================================================

DIMENSIONS = [
    (15, 3, 5),
    (30, 8, 12)
]

BOOLEANS = [True, False]
RHO = 1e-4
TOLERANCE = 1e-6

# =====================================================================
# TEST IMPLEMENTATION
# =====================================================================

@pytest.mark.parametrize("nx, neq, nineq", DIMENSIONS)
@pytest.mark.parametrize("licq", BOOLEANS)
@pytest.mark.parametrize("sosc", BOOLEANS)
def test_regularized_qp_restores_properties(nx, neq, nineq, licq, sosc):
    """
    Verifies that applying the regularization scheme to a potentially degenerate QP 
    yields a new QP that satisfies LICQ, SOSC, and SCS, while returning an 
    equivalent solution that zeros KKT of the base problem.
    """
    # 1. Generate the base QP and the true solution (forcing SCS=True)
    seed = 42 + nx + neq + nineq + int(licq) + int(sosc)
    qp, true_sol = generate_qp(
        nx=nx, neq=neq, nineq=nineq, seed=seed,
        licq=licq, scs=True, sosc=sosc
    )
    
    # 2. Regularize the QP using the true solution
    qp_reg = regularize_qp(qp, true_sol, RHO)
    
    # 3. Solve the regularized QP
    opts = {'printLevel': 'none'}
    solver_reg = QPSolver(qp_reg, options=opts)
    sol_reg = solver_reg.solve()
    
    # 4. Extract equivalent original variables from the augmented primal state
    # As per main_regularized.py, the regularized state is X = [x, lam, mu, z]
    reg_x = sol_reg.x[0 : nx]
    reg_lam = sol_reg.x[nx : nx + nineq]
    reg_mu = sol_reg.x[nx + nineq : nx + nineq + neq]
    
    # 5. Verify the new solution matches the true base solution
    diff_x = float(ca.norm_inf(true_sol.x - reg_x))
    diff_lam = float(ca.norm_inf(true_sol.lam - reg_lam))
    diff_mu = float(ca.norm_inf(true_sol.mu - reg_mu))
    
    assert diff_x < TOLERANCE, f"Primal x difference {diff_x:.2e} exceeds {TOLERANCE}"
    assert diff_lam < TOLERANCE, f"Dual lam difference {diff_lam:.2e} exceeds {TOLERANCE}"
    assert diff_mu < TOLERANCE, f"Dual mu difference {diff_mu:.2e} exceeds {TOLERANCE}"
    
    # 6. Verify that the extracted solution zeros the original KKT conditions
    kkt_res = evaluate_kkt(qp, reg_x, reg_lam, reg_mu)
    max_kkt = max([float(ca.norm_inf(v)) for v in kkt_res.values()])
    assert max_kkt < TOLERANCE, f"Original KKT residual {max_kkt:.2e} exceeds {TOLERANCE}"
    
    # 7. Verify that the regularized QP satisfies ALL theoretical properties
    assert sol_reg.properties['licq'] is True, "Regularized QP failed to satisfy LICQ"
    assert sol_reg.properties['scs'] is True, "Regularized QP failed to satisfy SCS"
    assert sol_reg.properties['strong_convexity'] is True, "Regularized QP failed to satisfy SOSC"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])