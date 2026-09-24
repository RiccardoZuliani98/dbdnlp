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
  4. Taylor-tests  phi(m + dm) ~ phi(m) + dm * dphi/dm  against re-solves,
  5. writes the trajectory data used by the position.tex figure,
  6. writes the data for the remaining tables/figures of the paper (all in RESULTS_DIR):
       dubins_errors.dat   -> tab:dubins_errors  (relative errors, all methods and rho)
       dubins_scaling.dat  -> tab:scaling        (ds_0/dm and errors vs. constraint scaling c)
       dubins_taylor.dat   -> fig:dubins_taylor  (first-order prediction error vs. dm)
       dubins_theta3.dat   -> 3-parameter test   (relative error and cosine similarity)
       dubins_times.dat    -> tab:comp_times     (computation times, mean/variance)
"""
import os
import numpy as np
import casadi as ca
from time import perf_counter

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
    ic_scale=1.0,
    goal_as_param=False,
    goal=(0.5, 0.25),
):
    """
    Returns (nlp, layout) where layout lets you unpack X and U from x.

    integrator : "rk4" (default, nondegenerate) or "euler" (structurally degenerate).
    ic_scale   : the initial condition is imposed as ic_scale * X[0:5, 0] = 0. Any
                 ic_scale != 0 describes the same problem; it only rescales constraint
                 rows (and their multipliers).
    goal_as_param : if True the parameter is theta = [m, p_x(T), p_y(T)] (3 components),
                 i.e. the terminal position becomes a parameter as well; otherwise theta = m
                 and the terminal position is fixed to `goal`.
    """
    nX, nU = 6, 2
    X = ca.MX.sym("X", nX, N + 1)   # state trajectory: columns are shooting nodes 0..N
    U = ca.MX.sym("U", nU, N)       # control trajectory: columns are intervals 0..N-1
    if goal_as_param:
        theta = ca.MX.sym("theta", 3)   # [steering-bound scaling, p_x(T), p_y(T)]
        m, goal = theta[0], theta[1:3]
    else:
        theta = ca.MX.sym("m", 1)       # scalar parameter that scales the steering bound
        m, goal = theta, ca.DM(list(goal))

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
    nlp = DifferentiableNLP(x=x, theta=theta, f=f, g=g, h=h,
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


def dx_dtheta(nlp, theta, sol, mode, **opts):
    """Full sensitivity dx/dtheta as a (n_x, n_theta) numpy array (see dx_dm)."""
    d = nlp.differentiate(np.atleast_1d(np.asarray(theta, float)), sol.x, sol.lam, sol.nu,
                          mode=mode, options=opts)
    return np.array(ca.DM(d["x"]).full()).reshape(nlp.n_x, nlp.n_theta)


def fd_jacobians(nlp, prox, theta0, x_bar, delta):
    """
    Central finite-difference Jacobians dx/dtheta, one parameter component at a time:
      prox : FD of the closest-point (proximal) problem, warm-started at x_bar;
      warm : FD of the original problem, warm-started at the proximal solutions x^+/x^-.
    Both are (n_x, n_theta) arrays.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, float))
    J_prox = np.zeros((nlp.n_x, theta0.size))
    J_warm = np.zeros((nlp.n_x, theta0.size))
    for j in range(theta0.size):
        e = np.zeros(theta0.size)
        e[j] = delta
        xp = np.array(prox.solve(np.r_[theta0 + e, x_bar], x_warmstart=x_bar).x).flatten()
        xm = np.array(prox.solve(np.r_[theta0 - e, x_bar], x_warmstart=x_bar).x).flatten()
        J_prox[:, j] = (xp - xm) / (2 * delta)
        wp = np.array(nlp.solve(theta0 + e, x_warmstart=xp).x).flatten()
        wm = np.array(nlp.solve(theta0 - e, x_warmstart=xm).x).flatten()
        J_warm[:, j] = (wp - wm) / (2 * delta)
    return J_prox, J_warm


def cos_sim(a, b):
    """Cosine similarity of a and b, flattened (Frobenius inner product for matrices)."""
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def time_it(fn, reps):
    """Wall-clock times (s) of `reps` calls of fn(), after one untimed warm-up call."""
    fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = perf_counter()
        fn()
        t[i] = perf_counter() - t0
    return t


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
    RHO = 1e-4                     # regularization used everywhere a single rho is needed
    C_LIST = [1e-2, 1e-1, 1.0, 1e1, 1e2]
    THETA3 = np.array([M0, 0.5, 0.25])   # nominal [m, p_x(T), p_y(T)] for the 3-parameter test
    N_TIMING = 100                 # evaluations per method for tab:comp_times
    RESULTS_DIR = ".results"       # all .dat/.tex outputs go here
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    err_rows = []                  # (method, rho, err vs proximal FD, err vs warm-started FD)

    def row(label, v, method, rho=np.nan):
        """Print one comparison row (relative error of v against ref and warm) and store it."""
        err_rows.append((method, rho, rel(v, ref), rel(v, warm)))
        print(f"    {label:<26} {rel(v, ref):>10.1e} {rel(v, warm):>12.1e}")

    row("closest point (reference)", ref, "proxFD")
    row("warm-started IPOPT (FD)", warm, "warmFD")
    for r in RHOS:
        row(f"projected rho={r:.0e}", dx_dm(nlp, M0, sol, "projected", rho_1=r, rho_2=r, tol=TOL),
            "projected", r)
    row("least squares", dx_dm(nlp, M0, sol, "active_set", tol=TOL, lstsq=True, active_rule="ratio"),
        "lstsq")
    for r in RHOS:
        row(f"previous (rho I) rho={r:.0e}", dx_dm(nlp, M0, sol, "regularized", rho_1=r, rho_2=r),
            "uniform", r)

    # tab:dubins_errors. `nan` marks entries that are "--" in the table (no rho, or a
    # reference compared against itself). Readable with \pgfplotstableread.
    with open(os.path.join(RESULTS_DIR, "dubins_errors.dat"), "w") as fh:
        fh.write("method rho err_prox err_warm\n")
        for meth, r, ep, ew in err_rows:
            fh.write(f"{meth} {r:.1e} {ep:.3e} {ew:.3e}\n")
    print(f"    (saved to {RESULTS_DIR}/dubins_errors.dat)")

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
    scaling = []
    for c in C_LIST:
        nlp_c, _ = build_dubins_nlp(N=N, integrator="euler", ic_scale=c)
        sol_c = nlp_c.solve(np.array([M0]), x_warmstart=x0)
        vp = dx_dm(nlp_c, M0, sol_c, "projected", rho_1=RHO, rho_2=RHO, tol=TOL)
        vl = dx_dm(nlp_c, M0, sol_c, "active_set", tol=TOL, lstsq=True, active_rule="ratio")
        scaling.append([c, vp[i_s0], rel(vp, ref), rel(vp, warm),
                        vl[i_s0], rel(vl, ref), rel(vl, warm)])
        print(f"    {c:>6.0e} {vp[i_s0]:>18.2e} {rel(vp, ref):>8.1e} {rel(vp, warm):>9.1e} "
              f"{vl[i_s0]:>12.4f} {rel(vl, ref):>8.1e} {rel(vl, warm):>9.1e}")
    # tab:scaling (the table uses the *_prox columns; *_warm are for reference)
    np.savetxt(os.path.join(RESULTS_DIR, "dubins_scaling.dat"), np.array(scaling),
               header="c proj_ds0 proj_err_prox proj_err_warm ls_ds0 ls_err_prox ls_err_warm",
               comments="", fmt="%.6e")
    print(f"    (saved to {RESULTS_DIR}/dubins_scaling.dat)")

    # ---------------- 6. Taylor test ----------------
    # For shrinking step sizes dm, check that x0 + dm * dx/dm approximates the
    # true closest-point re-solve x_cp(m + dm) to first order: the prediction
    # error should shrink faster than the nominal displacement |x_cp - x0| as
    # dm -> 0, for a correctly computed derivative.
    print("\n--- 5. First-order prediction error |x_cp(m+dm) - x0 - dm * dx/dm| (closest-point re-solves)")
    methods = {
        "projected": dx_dm(nlp, M0, sol, "projected", rho_1=1e-4, rho_2=1e-4, tol=TOL),
        "LS c0.1": dx_dm(*(lambda n: (n, M0, n.solve(np.array([M0]), x_warmstart=x0)))(
            build_dubins_nlp(N=N, integrator="euler", ic_scale=0.1)[0]),
            "active_set", tol=TOL, lstsq=True, active_rule="ratio"),
        "uniform 1e-5": dx_dm(nlp, M0, sol, "regularized", rho_1=1e-5, rho_2=1e-5),
        # unregularized FD on the original (degenerate) problem itself, i.e. no
        # closest-point proximal term at all -- computed above as `warm`.
        "FD unregularized": warm,
    }
    # Each method's linear prediction x0 + dm*v is checked against two targets:
    # xt, the closest-point re-solve (well-posed reference), and x_resolve, the
    # ORIGINAL (degenerate) solver re-solved at m+dm and warm-started at xt --
    # if the original solver reliably landed back on xt, both errors would
    # coincide; instead it typically drifts to a different point of the flat
    # solution set, so the two columns per method can disagree.
    print(f"    {'dm':>8} {'|x(m+dm)-x0|':>13} {'|resolve-xt|':>13} "
          + " ".join(f"{k + ' vs xcp':>16} {k + ' vs solver':>16}" for k in methods))
    taylor = []
    for dm in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        xt = xcp(M0 + dm)
        x_resolve = np.array(nlp.solve([M0 + dm], x_warmstart=xt).x).flatten()
        resolve_err = np.linalg.norm(x_resolve - xt)
        errs, errs_state = [], []
        nXt = L["nXtot"]
        for v in methods.values():
            pred = x0 + dm * v
            errs.append(np.linalg.norm(pred - xt))
            errs.append(np.linalg.norm(pred - x_resolve))
            # same error restricted to the state trajectory (inputs excluded): for large dm
            # the full-vector error is dominated by the few steering inputs that saturate,
            # which the first-order prediction cannot capture
            errs_state.append(np.linalg.norm(pred[:nXt] - xt[:nXt]))
        taylor.append([dm, np.linalg.norm(xt - x0), resolve_err] + errs
                      + [np.linalg.norm(xt[:nXt] - x0[:nXt])] + errs_state)
        print(f"    {dm:>8.0e} {np.linalg.norm(xt - x0):>13.3e} {resolve_err:>13.3e} "
              + " ".join(f"{e:>16.3e}" for e in errs))
    header = ("dm nominal resolve_err " +
              " ".join(f"{k.replace(' ', '_')}_vs_xcp {k.replace(' ', '_')}_vs_solver" for k in methods)
              + " nominal_states " + " ".join(f"{k.replace(' ', '_')}_vs_xcp_states" for k in methods))
    np.savetxt(os.path.join(RESULTS_DIR, "dubins_taylor.dat"), np.array(taylor),
               header=header, comments="")
    print(f"    (saved to {RESULTS_DIR}/dubins_taylor.dat for pgfplots)")

    # ---------------- 7. data for the trajectory figure (position.tex) ----------------
    # Three trajectories, each written as two-column pgfplots tables (t, value) for the
    # x-position, y-position, throttle a and steering s:
    #   suffix ""           : nominal solution x(m)
    #   suffix "_m2"        : original NLP re-solved at m + dm, warm-started at x(m)
    #   suffix "_m2_approx" : first-order prediction x(m) + dm * dx/dm (projected regularization)
    # The time axis is physical time t_k = k * sigma / (N+1), with sigma the time scaling
    # of that trajectory (for the prediction, sigma is itself predicted to first order,
    # so its time axis stretches accordingly).
    PLOT_DIR = RESULTS_DIR         # copy the files to Figures/Data/ (the path used in position.tex)
    PLOT_DM = 1e-1                 # parameter perturbation delta theta shown in the figure
    PLOT_RHO = 1e-4                # regularization of the derivative used for the prediction
    os.makedirs(PLOT_DIR, exist_ok=True)

    def save_trajectory(z, suffix):
        """Write x_pos, y_pos, a, s tables for primal vector z; return the final time."""
        X, U = unpack(z, L)
        t = np.arange(L["N"] + 1) * X[5, -1] / (L["N"] + 1)
        # inputs are held constant on [t_k, t_{k+1}): repeat the last one at t_N so that
        # every curve spans [0, t_N] (use `const plot` in pgfplots for exact steps)
        series = {"x_pos": X[0], "y_pos": X[1],
                  "a": np.r_[U[0], U[0, -1]], "s": np.r_[U[1], U[1, -1]]}
        for name, v in series.items():
            np.savetxt(os.path.join(PLOT_DIR, f"{name}{suffix}.tex"), np.column_stack([t, v]),
                       header="t v", comments="", fmt="%.10e")
        return t[-1], series

    sol_p = nlp.solve(np.array([M0 + PLOT_DM]), x_warmstart=x0)
    status_p = nlp.solver.stats()["return_status"]
    x_p = np.array(sol_p.x).flatten()
    x_app = x0 + PLOT_DM * dx_dm(nlp, M0, sol, "projected", rho_1=PLOT_RHO, rho_2=PLOT_RHO, tol=TOL)

    tf, data = {}, {}
    for label, z, suffix in [("nominal", x0, ""), ("perturbed", x_p, "_m2"),
                             ("approx", x_app, "_m2_approx")]:
        tf[label], data[label] = save_trajectory(z, suffix)

    # final time for the x-axis limit (xmax=\dubinsTf in position.tex)
    with open(os.path.join(PLOT_DIR, "position_params.tex"), "w") as fh:
        fh.write(f"\\def\\dubinsTf{{{max(tf.values()):.6f}}}%\n")

    print(f"\n--- 6. Figure data (dm = {PLOT_DM:g}, rho = {PLOT_RHO:g}), saved to {PLOT_DIR}/")
    print(f"    re-solve at m + dm: status={status_p}")
    print("    final time        " + "  ".join(f"{k}={v:.4f}" for k, v in tf.items()))
    print(f"    |x(m+dm) - x(m)|              = {np.linalg.norm(x_p - x0):.3e}")
    print(f"    |x(m+dm) - x(m) - dm * dx/dm| = {np.linalg.norm(x_p - x_app):.3e}")
    print("    data ranges (for the y-axis limits):")
    for name in ("x_pos", "y_pos", "a", "s"):
        v = np.concatenate([d[name] for d in data.values()])
        print(f"      {name:<6} [{v.min():+.3f}, {v.max():+.3f}]")
    
    # ---------------- 8. three-dimensional parameter ----------------
    # theta = [m, p_x(T), p_y(T)]: the steering-bound scaling plus the terminal position.
    # The reference is again the proximal central FD (one pair of solves per component),
    # anchored at the nominal solution x0. Errors are relative Frobenius-norm errors of the
    # full Jacobian dx/dtheta (n_x x 3); cosine similarity uses the Frobenius inner product.
    # Least squares depends on the constraint scaling, so both methods are evaluated for
    # every c in C_LIST (the paper uses c = 1, the other rows show the sensitivity to c).
    print("\n--- 7. Three-dimensional parameter theta = [m, p_x(T), p_y(T)]")
    nlp3, _ = build_dubins_nlp(N=N, integrator="euler", goal_as_param=True)
    sol3 = nlp3.solve(THETA3, x_warmstart=x0)
    print(f"    nominal re-solve: status={nlp3.solver.stats()['return_status']}, "
          f"|x - x0| = {np.linalg.norm(np.array(sol3.x).flatten() - x0):.1e}")
    prox3 = closest_point_solver(nlp3)
    J_prox3, J_warm3 = fd_jacobians(nlp3, prox3, THETA3, x0, DELTA)
    print(f"    consistency: column m of the FD Jacobian vs 1-D reference: {rel(J_prox3[:, 0], ref):.1e}")

    th3_rows = [("warmFD", np.nan, rel(J_warm3, J_prox3), cos_sim(J_warm3, J_prox3), 0.0, 1.0)]
    print(f"    {'method':<10} {'c':>6} {'err prox':>9} {'cos prox':>12} {'err warm':>9} {'cos warm':>12}")
    for c in C_LIST:
        nlp3_c = nlp3 if c == 1.0 else build_dubins_nlp(N=N, integrator="euler", ic_scale=c,
                                                          goal_as_param=True)[0]
        sol3_c = sol3 if c == 1.0 else nlp3_c.solve(THETA3, x_warmstart=x0)
        for meth, J in [
            ("projected", dx_dtheta(nlp3_c, THETA3, sol3_c, "projected", rho_1=RHO, rho_2=RHO, tol=TOL)),
            ("lstsq", dx_dtheta(nlp3_c, THETA3, sol3_c, "active_set", tol=TOL, lstsq=True,
                                active_rule="ratio")),
        ]:
            th3_rows.append((meth, c, rel(J, J_prox3), cos_sim(J, J_prox3),
                             rel(J, J_warm3), cos_sim(J, J_warm3)))
    for meth, c, ep, cp, ew, cw in th3_rows:
        print(f"    {meth:<10} {c:>6.0e} {ep:>9.2e} {cp:>12.8f} {ew:>9.2e} {cw:>12.8f}")
    with open(os.path.join(RESULTS_DIR, "dubins_theta3.dat"), "w") as fh:
        fh.write("method c err_prox cos_prox err_warm cos_warm\n")
        for meth, c, ep, cp, ew, cw in th3_rows:
            fh.write(f"{meth} {c:.1e} {ep:.6e} {cp:.10f} {ew:.6e} {cw:.10f}\n")
    print(f"    (saved to {RESULTS_DIR}/dubins_theta3.dat)")

    # ---------------- 9. computation times ----------------
    # Both methods start from the same nominal primal-dual solution, whose computation is
    # not timed. "ours" = one call to the projected-regularization derivative (Hessian and
    # Jacobian evaluation, subspace computation and linear solve). "fd" = central proximal
    # finite differences as used for the reference: 2 * n_theta IPOPT solves, warm-started
    # at the nominal solution. Each is repeated N_TIMING times after one untimed warm-up.
    print(f"\n--- 8. Computation times ({N_TIMING} evaluations each, seconds)")

    def prox_fd(prox_nlp, theta0):
        """Central proximal FD Jacobian (timing only; no warm-FD companion solves)."""
        J = np.zeros((nlp.n_x, theta0.size))
        for j in range(theta0.size):
            e = np.zeros(theta0.size)
            e[j] = DELTA
            xp = np.array(prox_nlp.solve(np.r_[theta0 + e, x0], x_warmstart=x0).x).flatten()
            xm = np.array(prox_nlp.solve(np.r_[theta0 - e, x0], x_warmstart=x0).x).flatten()
            J[:, j] = (xp - xm) / (2 * DELTA)
        return J

    time_rows = []
    for nlp_t, sol_t, th0, prox_t in [(nlp, sol, np.array([M0]), prox), (nlp3, sol3, THETA3, prox3)]:
        t_ours = time_it(lambda: dx_dtheta(nlp_t, th0, sol_t, "projected",
                                           rho_1=RHO, rho_2=RHO, tol=TOL), N_TIMING)
        t_fd = time_it(lambda: prox_fd(prox_t, th0), N_TIMING)
        for meth, t in [("ours", t_ours), ("fd", t_fd)]:
            time_rows.append((meth, th0.size, t.mean(), t.var(ddof=1), t.std(ddof=1), np.median(t)))
    print(f"    {'method':<6} {'n_theta':>7} {'mean':>10} {'variance':>10} {'std':>10} {'median':>10}")
    for meth, nth, mu, var, sd, med in time_rows:
        print(f"    {meth:<6} {nth:>7d} {mu:>10.3e} {var:>10.3e} {sd:>10.3e} {med:>10.3e}")
    with open(os.path.join(RESULTS_DIR, "dubins_times.dat"), "w") as fh:
        fh.write("method n_theta mean var std median\n")
        for meth, nth, mu, var, sd, med in time_rows:
            fh.write(f"{meth} {nth:d} {mu:.6e} {var:.6e} {sd:.6e} {med:.6e}\n")
    print(f"    (saved to {RESULTS_DIR}/dubins_times.dat)")