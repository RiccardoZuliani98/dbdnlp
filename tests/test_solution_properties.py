# Copyright (c) 2026 ETH Zurich, Riccardo Zuliani
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See the LICENSE file in the project root.

import pytest
import casadi as ca
import numpy as np

# Assuming the provided code is saved in modules named `qp` and `nlp`
from dbdnlp.qp import generate_qp
from dbdnlp import DifferentiableNLP

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
    
    # 2. Build symbolic expressions from QPData
    x_sym = ca.MX.sym('x', nx) # type: ignore
    theta_sym = ca.MX.sym('theta', 0)  # Empty parameter vector for static QP # type: ignore

    f_sym = 0.5 * ca.mtimes([x_sym.T, qp.Q, x_sym]) + ca.mtimes(qp.q.T, x_sym)
    g_sym = ca.mtimes(qp.G, x_sym) - qp.h if nineq > 0 else None
    h_sym = ca.mtimes(qp.A, x_sym) - qp.b if neq > 0 else None
    
    # 3. Initialize the solver and solve with warmstart
    nlp = DifferentiableNLP(
        x=x_sym,
        theta=theta_sym,
        f=f_sym,
        g=g_sym,
        h=h_sym,
    )
    
    theta = ca.DM(0, 1)
    sol = nlp.solve(
        theta=theta,
        x_warmstart=true_sol.x,
        lam_warmstart=true_sol.lam,
        nu_warmstart=true_sol.nu
    )
    
    # 4. Check that the solution zeros the KKT conditions
    # Extract KKT residuals evaluated in the solver
    max_res = max([float(ca.norm_inf(v)) for v in sol.kkt_residuals.values()])
    
    # Assert the max residual is sufficiently close to zero
    KKT_TOLERANCE = 1e-7
    assert max_res < KKT_TOLERANCE, f"KKT residual {max_res:.2e} exceeds tolerance {KKT_TOLERANCE}"
    
    # 5. Check that the properties correspond to what was required
    # Evaluate properties explicitly via the DifferentiableNLP method
    props = nlp.check_properties(
        theta=sol.theta,
        x=sol.x,
        lam=sol.lam,
        nu=sol.nu
    )

    assert props['licq'] == licq, f"Expected LICQ={licq}, got {props['licq']}"
    assert props['sosc'] == sosc, f"Expected SOSC={sosc}, got {props['sosc']}"
    # assert props['scs'] == scs, f"Expected SCS={scs}, got {props['scs']}"

if __name__ == "__main__":
    # Allows the script to be run directly via `python test_qp_solver.py`
    pytest.main([__file__, "-v"])