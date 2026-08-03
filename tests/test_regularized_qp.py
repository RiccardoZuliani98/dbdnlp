import pytest
import casadi as ca

from qp import generate_qp
from nlp import DifferentiableNLP, NLPSolution

# =====================================================================
# TEST CONFIGURATIONS
# =====================================================================

DIMENSIONS = [
    (15, 3, 5),
    (30, 8, 12)
]

BOOLEANS = [True, False]
RHO = 1e-3
TOLERANCE = 1e-5


# =====================================================================
# TEST IMPLEMENTATION
# =====================================================================

@pytest.mark.parametrize("nx, neq, nineq", DIMENSIONS)
@pytest.mark.parametrize("licq", BOOLEANS)
@pytest.mark.parametrize("sosc", BOOLEANS)
def test_regularized_qp_restores_properties(nx, neq, nineq, licq, sosc):
    """
    Verifies that applying the regularization scheme P_2 to a potentially 
    degenerate QP yields a formulation that satisfies LICQ, SOSC, SSOSC, 
    and SCS, while returning a solution that zeros the original KKT system.
    """
    # 1. Generate the base QP and the ground-truth solution (forcing SCS=True)
    seed = 42 + nx + neq + nineq + int(licq) + int(sosc)
    qp, true_sol = generate_qp(
        nx=nx, neq=neq, nineq=nineq, seed=seed,
        licq=licq, scs=True, sosc=sosc
    )

    # 2. Build symbolic expressions from QPData
    x_sym = ca.MX.sym('x', nx)
    theta_sym = ca.MX.sym('theta', 0)  # Empty parameter vector for static QP

    f_sym = 0.5 * ca.mtimes([x_sym.T, qp.Q, x_sym]) + ca.mtimes(qp.q.T, x_sym)
    g_sym = ca.mtimes(qp.G, x_sym) - qp.h if nineq > 0 else None
    h_sym = ca.mtimes(qp.A, x_sym) - qp.b if neq > 0 else None

    # 3. Instantiate DifferentiableNLP configured with the Clarabel solver
    nlp = DifferentiableNLP(
        x=x_sym,
        theta=theta_sym,
        f=f_sym,
        g=g_sym,
        h=h_sym
    )

    theta_val = ca.DM.zeros(0, 1)

    # =================================================================
    # 3.5 NEW: Verify ground-truth solution against original KKT system
    # =================================================================
    
    # Check A: Does the true solution zero the QP generator's internal KKT?
    max_kkt_gen = max([float(ca.norm_inf(v)) for v in true_sol.kkt_residuals.values()])
    assert max_kkt_gen < TOLERANCE, f"Generated true_sol KKT residual {max_kkt_gen:.2e} exceeds {TOLERANCE}!"

    # Check B: Does the true solution zero the NLP class's KKT evaluator?
    kkt_orig = nlp.kkt_eval(
        theta=theta_val,
        x=true_sol.x, 
        lam_g=true_sol.lam, 
        lam_h=true_sol.mu
    )
    max_kkt_orig = max([float(ca.norm_inf(v)) for v in kkt_orig.values()])
    assert max_kkt_orig < TOLERANCE, f"NLP original KKT residual {max_kkt_orig:.2e} exceeds {TOLERANCE}!"

    # 4. Solve the regularized problem P_2 using the anchor solution
    
    sol_reg: NLPSolution = nlp.solve_reg(
        theta=theta_val,
        rho=RHO,
        x_bar=true_sol.x,
        lam_bar=true_sol.lam,
        nu_bar=true_sol.mu
    )

    # 5. Verify the regularized solution matches the ground-truth solution
    diff_x = float(ca.norm_inf(true_sol.x - sol_reg.x))
    diff_lam = float(ca.norm_inf(true_sol.lam - sol_reg.lam))
    diff_mu = float(ca.norm_inf(true_sol.mu - sol_reg.mu))

    assert diff_x < TOLERANCE, f"Primal x difference {diff_x:.2e} exceeds {TOLERANCE}"
    assert diff_lam < TOLERANCE, f"Dual lam difference {diff_lam:.2e} exceeds {TOLERANCE}"
    assert diff_mu < TOLERANCE, f"Dual mu difference {diff_mu:.2e} exceeds {TOLERANCE}"

    # 6. Verify that the regularized point zeros the original KKT conditions
    # (sol_reg.kkt_residuals is automatically computed by solve_reg)
    max_kkt = max([float(ca.norm_inf(v)) for v in sol_reg.kkt_residuals.values()])
    assert max_kkt < TOLERANCE, f"Original KKT residual {max_kkt:.2e} exceeds {TOLERANCE}"

    # 7. Evaluate and check properties of the regularized problem P_2
    props = nlp.check_properties(sol_reg, is_regularized=True, tol=TOLERANCE)

    assert props['licq'] is True, "Regularized problem P_2 failed to satisfy LICQ"
    assert props['scs'] is True, "Regularized problem P_2 failed to satisfy SCS"
    assert props['sosc'] is True, "Regularized problem P_2 failed to satisfy SOSC"
    assert props['ssosc'] is True, "Regularized problem P_2 failed to satisfy SSOSC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])