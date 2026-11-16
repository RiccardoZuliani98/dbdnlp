# Copyright (c) 2026 ETH Zurich, Riccardo Zuliani
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the project root.

import pytest
import casadi as ca
import numpy as np
import scipy.sparse as sp

from dbdnlp.qp import generate_qp
from dbdnlp import DifferentiableNLP
from dbdnlp._impl.nlp import _SparseKernelFailure

# =====================================================================
# TEST CONFIGURATIONS
# =====================================================================

# (m, n, dependent rows, dependent columns) of the sparse test matrices
SHAPES = [
    (600, 500, 30, 5),
    (500, 600, 10, 40),
    (800, 800, 60, 60),
    (1000, 900, 150, 20),
]

# null-space dimension hints: none, too small, too large
HINTS = [None, (2, 3), (50, 50)]


def rank_deficient_matrix(m, n, dep_rows, dep_cols, seed):
    """Sparse m x n matrix with `dep_rows` (`dep_cols`) rows (columns) that are
    linear combinations of two others."""
    rng = np.random.default_rng(seed)
    J = (sp.random(m, n, density=0.004, random_state=rng, format="lil")
         + sp.eye(m, n, format="lil")).tolil()
    for _ in range(dep_rows):
        i, a, b = rng.choice(m, 3, replace=False)
        J[i] = 0.7 * J[a] - 1.3 * J[b]
    for _ in range(dep_cols):
        j, a, b = rng.choice(n, 3, replace=False)
        J[:, j] = 0.5 * J[:, a] + 2.0 * J[:, b]
    return J.tocsc()

# =====================================================================
# TEST IMPLEMENTATION
# =====================================================================

@pytest.mark.parametrize("m, n, dep_rows, dep_cols", SHAPES)
@pytest.mark.parametrize("hint", HINTS)
def test_sparse_null_spaces_match_dense(m, n, dep_rows, dep_cols, hint):
    """
    The sparse null-space method returns the same null spaces as the dense SVD
    (compared through their orthogonal projectors), or gives up.

    A projector is determined up to ~ eps * s_max / s_gap (s_gap the smallest
    nonzero singular value), which sets the tolerance.
    """
    J = rank_deficient_matrix(m, n, dep_rows, dep_cols, seed=m + n + dep_rows)
    rcond = np.finfo(float).eps * max(m, n)
    Z_d, U_d = DifferentiableNLP._null_spaces_dense(J, rcond)
    s = np.linalg.svd(J.toarray(), compute_uv=False)
    tol = 1e3 * np.finfo(float).eps * s[0] / s[s > rcond * s[0]][-1]

    try:
        Z, U = DifferentiableNLP._null_spaces_sparse(J, rcond, hint=hint)
    except _SparseKernelFailure:
        pytest.skip("sparse method fell back to the dense SVD")

    assert Z.shape == Z_d.shape and U.shape == U_d.shape
    assert np.allclose(Z.T @ Z, np.eye(Z.shape[1]), atol=1e-12)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-12)
    assert np.abs(Z @ Z.T - Z_d @ Z_d.T).max() < tol
    assert np.abs(U @ U.T - U_d @ U_d.T).max() < tol


@pytest.mark.parametrize("rho", [1e-6, 1e-2, 1.0])
@pytest.mark.parametrize("method", ["dense", "sparse"])
def test_projected_derivative_solves_projected_system(rho, method):
    """
    The projected-mode derivative solves the (dense) projected KKT system
        [ H + rho Pi_P   J^T      ] [dx  ]     [H_xtheta ]
        [ J              -rho Pi_U] [dpsi] = - [J_theta  ]
    on a QP where both LICQ and SOSC fail (Pi_P and Pi_U nontrivial).
    """
    nx, neq, nineq = 30, 8, 12
    qp, true_sol = generate_qp(nx=nx, neq=neq, nineq=nineq, seed=7, licq=False, sosc=False)

    # parametric QP: theta moves the linear cost and the inequality bounds
    rng = np.random.default_rng(0)
    C = ca.DM(rng.standard_normal((nx, 2)))
    E = ca.DM(rng.standard_normal((nineq, 2)))
    x = ca.MX.sym("x", nx)  # type: ignore
    theta = ca.MX.sym("theta", 2)  # type: ignore
    f = 0.5 * ca.mtimes([x.T, qp.Q, x]) + ca.mtimes((qp.q + ca.mtimes(C, theta)).T, x)
    g = ca.mtimes(qp.G, x) - qp.h - ca.mtimes(E, theta)
    h = ca.mtimes(qp.A, x) - qp.b
    nlp = DifferentiableNLP(x=x, theta=theta, f=f, g=g, h=h)

    th = ca.DM.zeros(2)
    opts = {"rho_1": rho, "rho_2": rho, "tol": 1e-7, "null_space_method": method}
    d = nlp.differentiate(th, true_sol.x, true_sol.lam, true_sol.nu, mode="projected", options=opts)

    sub = nlp.degenerate_subspaces(th, true_sol.x, true_sol.lam, true_sol.nu, active_tol=1e-7)
    assert sub["dim_P"] > 0 and sub["dim_U"] > 0
    ev = nlp._eval_point(th, true_sol.x, true_sol.lam, true_sol.nu)
    idx = sub["sys_idx"]
    J = sp.vstack([ev["J_g"][idx], ev["J_h"]]).toarray()
    J_th = sp.vstack([ev["J_gtheta"][idx], ev["J_htheta"]]).toarray()
    M = np.block([
        [ev["H"].toarray() + rho * sub["Pi_P"], J.T],
        [J, -rho * sub["Pi_U"]],
    ])
    r = -np.vstack([ev["H_xtheta"].toarray(), J_th])
    v = np.vstack([np.array(d["x"]), np.array(d["lam"])[idx], np.array(d["nu"])])

    assert np.allclose(v, np.linalg.solve(M, r), rtol=1e-8, atol=1e-8 * np.abs(v).max())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
