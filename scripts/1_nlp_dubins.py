"""
Dubins-style minimum-time car, adapted to DifferentiableNLP.

With integrator="euler" (used in the paper) the problem is structurally degenerate:
  * steering at standstill has no effect (the car starts at rest, v_0 = 0), and
  * the last steering input only changes the final heading, which is free.
Both are exactly flat directions of the Lagrangian on the critical cone (dim P = 2),
so SSOSC fails and the unregularized KKT matrix is singular. With RK4 the intermediate
stages give both inputs some authority and the problem is nondegenerate.

The standstill steering is flat *because* it multiplies a quantity constrained to zero
(v_0 = 0). Its flat direction p therefore has H p != 0, and the least-squares solution
of the singular KKT system then depends on how the constraints are scaled, whereas the
projected regularization returns the closest-point derivative for every rho and every
scaling.

State  X_k = [x, y, vx, vy, theta, sigma],  sigma = free time scaling (the cost).
Input  U_k = [a, s].
Parameter m scales the steering bound: |s| <= s_max * m.  Differentiating w.r.t. m
therefore moves an ACTIVE constraint, which is the interesting case.

The script:
  1. builds the NLP symbolically (no ca.Opti -- DifferentiableNLP wants f, g, h),
  2. solves it and reports LICQ / SCS / SOSC / SSOSC at the solution,
  3. checks that P_2 anchored there reproduces it,
  4. Taylor-tests  phi(m + dm) ~ phi(m) + dm * dphi/dm  against re-solves.
"""
import numpy as np
import casadi as ca

from dbdnlp import DifferentiableNLP

SOLVER_OPTS = {
    "ipopt.print_level": 0, "print_time": 0,
    "ipopt.tol": 1e-12, "ipopt.constr_viol_tol": 1e-12,
    "ipopt.dual_inf_tol": 1e-10, "ipopt.max_iter": 3000,
}


# =====================================================================
# PROBLEM FORMULATION
# =====================================================================

def rk4(fc, x, u, m, T):
    """
    One classical RK4 step for state x under control u and parameter m.

    fc  : continuous dynamics function fc(x, u, m) -> xdot.
    x   : state at the start of the step.
    u   : control held constant over the step (zero-order hold).
    m   : problem parameter passed through to fc unchanged.
    T   : step length (here: dt, the fixed-size interval between shooting nodes).

    Returns the state at x + T. Using RK4 (rather than "euler") is what gives
    the last input in a stage some authority over the intermediate dynamics,
    which is why it avoids the structural degeneracy described in the module
    docstring.
    """
    k1 = fc(x, u, m)
    k2 = fc(x + T / 2 * k1, u, m)
    k3 = fc(x + T / 2 * k2, u, m)
    k4 = fc(x + T * k3, u, m)
    return x + (T / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def build_dubins_nlp(
    N=15,
    a_max=1.0,
    s_max=1.0,
    alpha=0.5,
    v_max=None,
    solver_opts=None,
    integrator="rk4",
    ic_scale=1.0
):
    """
    Returns (nlp, layout) where layout lets you unpack X and U from x.

    integrator : "rk4" (default, nondegenerate) or "euler" (structurally degenerate).
    ic_scale   : the initial condition is imposed as ic_scale * X[0:5, 0] = 0. Any
                 ic_scale != 0 describes the same problem; it only rescales constraint
                 rows (and their multipliers).
    """
    nX, nU = 6, 2
    X = ca.MX.sym("X", nX, N + 1)   # state trajectory: columns are shooting nodes 0..N
    U = ca.MX.sym("U", nU, N)       # control trajectory: columns are intervals 0..N-1
    m = ca.MX.sym("m", 1)           # scalar parameter that scales the steering bound

    # continuous dynamics: symbolic template fc(x, u, m) -> xdot, later plugged
    # into each shooting interval by rk4() or the explicit Euler step below.
    xs, us, ms = ca.MX.sym("xs", nX), ca.MX.sym("us", nU), ca.MX.sym("ms", 1)
    vx, vy, th, sg = xs[2], xs[3], xs[4], xs[5]
    a, s = us[0], us[1]
    # sigma (sg) is a free time-scaling factor: multiplying the physical rhs by it
    # lets the optimizer choose how "fast" simulated time maps to real time, and
    # sigma at the final node becomes the minimum-time cost. The dynamics for
    # sigma itself are trivial (appended 0 below): sigma is piecewise-constant
    # per shooting interval and only enters as a multiplier.
    rhs = ca.vertcat(vx, vy,
                     a * ca.cos(th), a * ca.sin(th),
                     s * (ca.cos(th) * vx + ca.sin(th) * vy))
    fc = ca.Function("fc", [xs, us, ms], [ca.vertcat(sg * rhs, 0)])

    dt = 1.0 / (N + 1)
    goal = ca.DM([0.5, 0.25])

    # ---- equalities h(x, m) = 0 ----
    h = [ic_scale * X[0:5, 0]]                        # start at rest at origin
    for k in range(N):
        if integrator == "rk4":
            x_next = rk4(fc, X[:, k], U[:, k], m, dt)
        elif integrator == "euler":
            x_next = X[:, k] + dt * fc(X[:, k], U[:, k], m)
        else:
            raise ValueError(f"unknown integrator '{integrator}'")
        h.append(X[:, k + 1] - x_next)                # collocation/shooting defect = 0
    h.append(X[0:2, N] - goal)                        # reach the goal
    h.append(X[2:4, N])                               # at rest
    h = ca.vertcat(*h)

    # ---- inequalities g(x, m) <= 0 ----
    g = []
    for k in range(N):
        g.append(U[0, k] - a_max)                     # a_k <= a_max
        g.append(-U[0, k] - a_max)                    # a_k >= -a_max
        g.append(U[1, k] - s_max * m)                 # s_k <= s_max * m (parameter enters here)
        g.append(-U[1, k] - s_max * m)                # s_k >= -s_max * m
    if v_max is not None:
        for k in range(N + 1):
            g.append(X[2:4, k] - v_max)                # |vx_k|, |vy_k| <= v_max
            g.append(-X[2:4, k] - v_max)
    g.append(-X[5, 0])                                # sigma >= 0: keep time forward
    g = ca.vertcat(*g)

    # terminal position/velocity errors, kept for reference even though the
    # tracking penalty is currently disabled (they are already enforced as
    # hard equality constraints above, via h).
    e1 = X[0:2, N] - goal
    e2 = X[2:4, N]
    f = X[5, N] #+ alpha * ca.sumsqr(e1) + alpha * ca.sumsqr(e2)

    # flatten (X, U) into a single primal vector, column-major (Fortran) order,
    # matching the reshape used by unpack() to invert this.
    x = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = DifferentiableNLP(x=x, theta=m, f=f, g=g, h=h,
                            solver_opts=dict(solver_opts or SOLVER_OPTS))
    return nlp, dict(N=N, nX=nX, nU=nU, nXtot=nX * (N + 1), integrator=integrator)


def unpack(z, L):
    """
    Inverse of the flattening done in build_dubins_nlp: splits a flat primal
    vector x (or any vector shaped like it, e.g. a sensitivity dx/dm) back
    into the state matrix X (nX x N+1) and control matrix U (nU x N), using
    the same column-major ("F") order as the original ca.reshape.
    """
    z = np.asarray(z, float).flatten()
    X = z[:L["nXtot"]].reshape(L["nX"], L["N"] + 1, order="F")
    U = z[L["nXtot"]:].reshape(L["nU"], L["N"], order="F")
    return X, U


def initial_guess(L, val=0.075):
    """Flat warm-start vector: every state/control component set to the same constant."""
    return np.full(L["nXtot"] + L["nU"] * L["N"], val)


def closest_point_solver(nlp, eps=1e-8):
    """
    argmin f + eps/2 |x - x_bar|^2 s.t. g <= 0, h == 0: same feasible set and
    cost as `nlp`, but with x_bar appended to the parameters and a tiny
    proximal term added to the cost. Used as a well-posed reference: even when
    the original problem has a degenerate/flat solution set, this variant
    picks out the point of that set closest to x_bar, giving an unambiguous
    finite-difference target for dx/dm.
    """
    S = nlp.sym
    xb = type(S.primal).sym("x_bar", nlp.n_x)
    return DifferentiableNLP(x=S.primal, theta=ca.vertcat(S.param, xb),
                             f=S.cost + eps / 2 * ca.sumsqr(S.primal - xb),
                             g=S.ineq, h=S.eq, solver_opts=dict(SOLVER_OPTS))


def dx_dm(nlp, m, sol, mode, **opts):
    """
    Sensitivity of the primal solution w.r.t. the parameter m, dx/dm, evaluated
    at (sol.x, sol.lam, sol.nu) using the requested differentiation `mode`
    (e.g. "active_set", "projected", "regularized" — see DifferentiableNLP).
    `opts` are forwarded as solver-specific options (rho_1, tol, lstsq, ...).
    Returns a flat numpy vector, same shape/order as the primal x.
    """
    d = nlp.differentiate(np.array([m]), sol.x, sol.lam, sol.nu, mode=mode, options=opts)
    return np.array(ca.DM(d["x"]).full()).flatten()


def rel(a, b):
    """Relative L2 error of a against reference b: |a - b| / |b|."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    N = 150
    M0 = 1.0
    DELTA = 1e-5
    TOL = 1e-7                     # active-set tolerance (complementarity-aware rule)
    RHOS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]

    nlp, L = build_dubins_nlp(N=N, integrator="euler")
    print(f"Dubins NLP (Euler): n_x={nlp.n_x}  n_ineq={nlp.n_ineq}  n_eq={nlp.n_eq}")

    # ---------------- 1. solve ----------------
    sol = nlp.solve(np.array([M0]), x_warmstart=initial_guess(L))
    x0 = np.array(sol.x).flatten()
    Xs, Us = unpack(x0, L)
    print(f"\n--- 1. Solution: status={nlp.solver.stats()['return_status']}, sigma={Xs[5, -1]:.6f}")

    # ---------------- 2. degeneracy ----------------
    sub = nlp.degenerate_subspaces(np.array([M0]), sol.x, sol.lam, sol.nu, active_tol=TOL)
    print(f"\n--- 2. Degeneracy: n_active={len(sub['active_idx'])}, dim P={sub['dim_P']}, "
          f"dim U={sub['dim_U']}, weakly active={len(sub['weak_idx'])}, "
          f"min eig of reduced Hessian={sub['min_reduced_eig']:.1e}")
    w, V = np.linalg.eigh(sub["Pi_P"])
    for c in np.where(w > 0.5)[0]:
        top = np.argsort(-np.abs(V[:, c]))[:2]
        print("    flat direction along index " + ", ".join(f"{i} ({V[i, c]:+.2f})" for i in top))
    try:
        dx_dm(nlp, M0, sol, "active_set", tol=TOL, active_rule="ratio")
        print("    unregularized KKT system: solved")
    except np.linalg.LinAlgError:
        print("    unregularized KKT system: singular")

    # ---------------- 3. references ----------------
    # `ref`: central finite difference of the closest-point re-solve (well-posed
    # even on the degenerate flat directions, since eps/2|x-x_bar|^2 picks a
    # unique minimizer). `warm`: central finite difference of the original
    # (degenerate) NLP re-solved from a warm start, for comparison.
    prox = closest_point_solver(nlp)
    xcp = lambda m: np.array(prox.solve(np.r_[m, x0], x_warmstart=x0).x).flatten()
    ref_plus = xcp(M0 + DELTA)
    ref_minus = xcp(M0 - DELTA)
    ref = (ref_plus - ref_minus) / (2 * DELTA)
    warm = (np.array(nlp.solve([M0 + DELTA], x_warmstart=ref_plus).x).flatten()
            - np.array(nlp.solve([M0 - DELTA], x_warmstart=ref_minus).x).flatten()) / (2 * DELTA)

    # ---------------- 4. comparison ----------------
    # Compare every differentiation mode/setting against both references, using
    # the relative error over the full vector: `ref` (closest-point, well-posed
    # even on the degenerate directions) and `warm` (plain warm-started FD on
    # the original problem, itself just one method among the ones being judged).
    print("\n--- 3. dx/dm against the closest-point reference")
    print(f"    {'method':<26} {'err vs ref':>10} {'err vs warm':>12}")

    def row(label, v):
        """Print one comparison row: relative error of v against ref and against warm, over the full vector."""
        print(f"    {label:<26} {rel(v, ref):>10.1e} {rel(v, warm):>12.1e}")

    row("closest point (reference)", ref)
    row("warm-started IPOPT (FD)", warm)
    for r in RHOS:
        row(f"projected rho={r:.0e}", dx_dm(nlp, M0, sol, "projected", rho_1=r, rho_2=r, tol=TOL))
    row("least squares", dx_dm(nlp, M0, sol, "active_set", tol=TOL, lstsq=True, active_rule="ratio"))
    for r in RHOS:
        row(f"previous (rho I) rho={r:.0e}", dx_dm(nlp, M0, sol, "regularized", rho_1=r, rho_2=r))

    # ---------------- 5. invariance to constraint scaling ----------------
    # Rebuild the NLP with the initial-condition constraint scaled by c (a
    # mathematically equivalent problem, per ic_scale's docstring) and check
    # that the projected-regularization derivative of s_0 (the flat, standstill
    # steering direction) stays ~constant across c, while the plain
    # least-squares derivative does not.
    # s_0 is control 1 ("s") at interval 0: since U is flattened column-major
    # after the nXtot state entries, a_0 sits at nXtot and s_0 right after it.
    i_s0 = L["nXtot"] + 1
    print("\n--- 4. Same problem, initial condition imposed as c * X[0:5,0] = 0")
    print(f"    {'c':>6} {'projected ds_0/dm':>18} {'err ref':>8} {'err warm':>9} "
          f"{'LS ds_0/dm':>12} {'err ref':>8} {'err warm':>9}")
    for c in [1e-2, 1e-1, 1.0, 1e1, 1e2]:
        nlp_c, _ = build_dubins_nlp(N=N, integrator="euler", ic_scale=c)
        sol_c = nlp_c.solve(np.array([M0]), x_warmstart=x0)
        vp = dx_dm(nlp_c, M0, sol_c, "projected", rho_1=1e-4, rho_2=1e-4, tol=TOL)
        vl = dx_dm(nlp_c, M0, sol_c, "active_set", tol=TOL, lstsq=True, active_rule="ratio")
        print(f"    {c:>6.0e} {vp[i_s0]:>18.2e} {rel(vp, ref):>8.1e} {rel(vp, warm):>9.1e} "
              f"{vl[i_s0]:>12.4f} {rel(vl, ref):>8.1e} {rel(vl, warm):>9.1e}")

    # ---------------- 6. Taylor test ----------------
    # For shrinking step sizes dm, check that x0 + dm * dx/dm approximates the
    # true closest-point re-solve x_cp(m + dm) to first order: the prediction
    # error should shrink faster than the nominal displacement |x_cp - x0| as
    # dm -> 0, for a correctly computed derivative.
    print("\n--- 5. First-order prediction error |x_cp(m+dm) - x0 - dm * dx/dm| (closest-point re-solves)")
    methods = {
        "projected": dx_dm(nlp, M0, sol, "projected", rho_1=1e-4, rho_2=1e-4, tol=TOL),
        "LS (c=0.1)": dx_dm(*(lambda n: (n, M0, n.solve(np.array([M0]), x_warmstart=x0)))(
            build_dubins_nlp(N=N, integrator="euler", ic_scale=0.1)[0]),
            "active_set", tol=TOL, lstsq=True, active_rule="ratio"),
        "previous 1e-5": dx_dm(nlp, M0, sol, "regularized", rho_1=1e-5, rho_2=1e-5),
        # unregularized FD on the original (degenerate) problem itself, i.e. no
        # closest-point proximal term at all -- computed above as `warm`.
        "FD unregularized": warm,
    }
    print(f"    {'dm':>8} {'|x(m+dm)-x0|':>13} " + " ".join(f"{k:>14}" for k in methods))
    taylor = []
    for dm in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        xt = xcp(M0 + dm)
        errs = [np.linalg.norm(x0 + dm * v - xt) for v in methods.values()]
        taylor.append([dm, np.linalg.norm(xt - x0)] + errs)
        print(f"    {dm:>8.0e} {np.linalg.norm(xt - x0):>13.3e} " + " ".join(f"{e:>14.3e}" for e in errs))
    np.savetxt("dubins_taylor.dat", np.array(taylor),
               header="dm nominal " + " ".join(k.replace(' ', '_') for k in methods), comments="")
    print("    (saved to dubins_taylor.dat for pgfplots)")
