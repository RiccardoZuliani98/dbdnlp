import pytest
import casadi as ca
import numpy as np

# Assuming the provided code is saved in a module named `src`
from src import QPSolver, generate_qp

# =====================================================================
# TEST CONFIGURATIONS
# =====================================================================

# Define combinations of (nx, neq, nineq) to test
DIMENSIONS = [
    (15, 3, 5),
    (30, 8, 12)
]

# Boolean combinations for theoretical properties
BOOLEANS = [True, False]

# =====================================================================
# TEST IMPLEMENTATION
# =====================================================================

@pytest.mark.parametrize("nx, neq, nineq", DIMENSIONS)
@pytest.mark.parametrize("licq", BOOLEANS)
@pytest.mark.parametrize("sosc", BOOLEANS)
# @pytest.mark.parametrize("scs", BOOLEANS)
def test_qp_solver_properties_and_kkt(nx, neq, nineq, licq, sosc):
    """
    Generates a QP with specific properties, solves it using the known 
    solution as a warmstart, checks KKT residuals, and verifies properties.
    """
    # 1. Generate the QP and the known solution
    # Use a unique seed for each combination to ensure varied matrices
    scs = True
    seed = 42 + nx + neq + nineq + int(licq) + int(sosc) + int(scs)
    
    qp, true_sol = generate_qp(
        nx=nx, neq=neq, nineq=nineq, seed=seed,
        licq=licq, scs=scs, sosc=sosc
    )
    
    # 2. Initialize the solver and solve with warmstart
    opts = {'printLevel': 'none'}
    solver = QPSolver(qp, options=opts)
    
    sol = solver.solve(warmstart=true_sol)
    
    # 3. Check that the solution zeros the KKT conditions
    # Extract KKT residuals evaluated in the solver
    max_res = max([float(ca.norm_inf(v)) for v in sol.residual.values()])
    
    # Assert the max residual is sufficiently close to zero
    KKT_TOLERANCE = 1e-8
    assert max_res < KKT_TOLERANCE, f"KKT residual {max_res:.2e} exceeds tolerance {KKT_TOLERANCE}"
    
    # 4. Check that the properties correspond to what was required
    assert sol.properties['licq'] == licq, f"Expected LICQ={licq}, got {sol.properties['licq']}"
    # assert sol.properties['scs'] == scs, f"Expected SCS={scs}, got {sol.properties['scs']}"
    assert sol.properties['strong_convexity'] == sosc, f"Expected SOSC={sosc}, got {sol.properties['strong_convexity']}"

if __name__ == "__main__":
    # Allows the script to be run directly via `python test_qp_solver.py`
    pytest.main([__file__, "-v"])