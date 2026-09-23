"""
Over-actuated MPC: a degenerate-by-construction version of the MPC example of Section 5.

Plant (Zuliani et al.), with the single input replaced by m redundant actuators:
    x1+ = x1 + 0.4 x2
    x2+ = 0.56 x2 + 0.1 x1 x2 + 0.4 * sum_i u_i + theta * x1 * exp(-x1)
Constraints:  |u_i| <= u_max / m,  |sum_i u_i| <= u_max (redundant),  |x2| <= x2_max.
Cost:         sum_{k=0}^{N} x_k^T Q x_k   (no input penalty).

The feasible set in terms of u = sum_i u_i is exactly the one of the original MPC, so
the closed loop x(theta) is IDENTICAL to the m = 1 problem. Every MPC instance is
degenerate, however:
  * flat directions  (P != {0}): moving input between actuators at unsaturated stages;
  * redundant active constraints (U != {0}): when the total input saturates, all m
    individual bounds and the sum bound are active simultaneously.
The closed-loop sensitivity d x(theta) / d theta is nevertheless unique, so we can
compare against (i) finite differences and (ii) the m = 1 problem as a reference.

The script:
  1. simulates the nominal closed loop for the reference (m = 1) and over-actuated MPC,
  2. reports the degeneracy structure (dim P, dim U, weakly active) along the loop,
  3. propagates closed-loop sensitivities with:
       - "projected"   : current method (rho_1 Pi_P, rho_2 Pi_U), rho sweeps
       - "regularized" : previous method (rho I), rho sweep
       - "lstsq"       : unregularized KKT system solved in the least-squares sense
  4. compares them to finite differences and to the m = 1 reference,
  5. optionally (--scaling) repeats the comparison for several m.

Results are saved to mpc_overactuated_results.npz and a preview figure.
"""
import argparse
import time
import numpy as np
import casadi as ca

from dbdnlp import DifferentiableNLP

SOLVER_OPTS = {
    "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
    "ipopt.tol": 1e-12, "ipopt.constr_viol_tol": 1e-12,
    "ipopt.dual_inf_tol": 1e-10, "ipopt.max_iter": 3000,
}
ACTIVE_TOL = 1e-7


# =====================================================================
# PROBLEM FORMULATION
# =====================================================================

def plant_function(m):
    """Discrete-time plant x+ = f(x, u, theta) with m actuators."""
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", m)
    th = ca.SX.sym("th", 1)
    xn = ca.vertcat(
        x[0] + 0.4 * x[1],
        0.56 * x[1] + 0.1 * x[0] * x[1] + 0.4 * ca.sum1(u) + th * x[0] * ca.exp(-x[0]),
    )
    f = ca.Function("f", [x, u, th], [xn])
    jac = ca.Function("f_jac", [x, u, th],
                      [ca.jacobian(xn, x), ca.jacobian(xn, u), ca.jacobian(xn, th)])
    return f, jac


def build_mpc_nlp(m=2, N=20, u_max=2.0, x2_max=2.0, Q=(1e-2, 1.0), redundant=True,
                  solver_opts=None):
    """
    MPC problem as a DifferentiableNLP with parameter p = (x_init, theta) in R^3.
    Decision vector: x = [vec(X) (2 x (N+1)); vec(U) (m x N)], column-major.
    Returns (nlp, layout).
    """
    f, _ = plant_function(m)
    X = ca.SX.sym("X", 2, N + 1)
    U = ca.SX.sym("U", m, N)
    p = ca.SX.sym("p", 3)
    x_init, th = p[0:2], p[2]

    # ---- equalities h(x, p) = 0 ----
    h = [X[:, 0] - x_init]
    for k in range(N):
        h.append(X[:, k + 1] - f(X[:, k], U[:, k], th))
    h = ca.vertcat(*h)

    # ---- inequalities g(x, p) <= 0 ----
    g = []
    for k in range(N):
        for i in range(m):
            g += [U[i, k] - u_max / m, -U[i, k] - u_max / m]
        if m > 1 and redundant:
            g += [ca.sum1(U[:, k]) - u_max, -ca.sum1(U[:, k]) - u_max]
    for k in range(1, N + 1):   # k = 0 is fixed by x_init: omit to avoid trivial LICQ loss
        g += [X[1, k] - x2_max, -X[1, k] - x2_max]
    g = ca.vertcat(*g)

    Qm = ca.diag(ca.DM(Q))
    cost = sum(ca.bilin(Qm, X[:, k], X[:, k]) for k in range(N + 1))

    x = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = DifferentiableNLP(x=x, theta=p, f=cost, g=g, h=h,
                            solver_opts=dict(solver_opts or SOLVER_OPTS))
    layout = dict(N=N, m=m, nXtot=2 * (N + 1), u_max=u_max)
    return nlp, layout


def unpack(z, L):
    """Split a flat decision vector z into the state trajectory X (2, N+1) and
    input trajectory U (m, N), inverting the column-major packing used in
    build_mpc_nlp."""
    z = np.asarray(z, float).flatten()
    X = z[:L["nXtot"]].reshape(2, L["N"] + 1, order="F")
    U = z[L["nXtot"]:].reshape(L["m"], L["N"], order="F")
    return X, U


def u0_rows(L):
    """Indices of the first-stage inputs u_{0|t} inside the decision vector."""
    return L["nXtot"] + np.arange(L["m"])


def shift_warmstart(z, L):
    """Shift a previous MPC solution by one stage (repeat the last stage)."""
    X, U = unpack(z, L)
    Xs = np.hstack([X[:, 1:], X[:, -1:]])
    Us = np.hstack([U[:, 1:], U[:, -1:]])
    return np.concatenate([Xs.flatten(order="F"), Us.flatten(order="F")])


# =====================================================================
# CLOSED LOOP
# =====================================================================

def simulate(nlp, L, theta, x0, T, warm=None):
    """
    Closed-loop simulation. If `warm` (list of nominal MPC solutions) is given,
    step t is warm-started with warm[t] (used for finite differences, as in the
    paper); otherwise with the shifted previous solution.
    Returns states (T+1, 2), applied inputs (T, m), and the list of MPC solutions.
    """
    f, _ = plant_function(L["m"])
    xs = np.zeros((T + 1, 2)); xs[0] = x0
    us = np.zeros((T, L["m"]))
    sols = []
    z_ws = np.zeros(nlp.n_x)
    for t in range(T):
        ws = warm[t].x if warm is not None else z_ws
        sol = nlp.solve(np.r_[xs[t], theta], x_warmstart=ws)
        sols.append(sol)
        z = np.array(sol.x).flatten()
        us[t] = z[u0_rows(L)]
        xs[t + 1] = np.array(f(xs[t], us[t], theta)).flatten()
        z_ws = shift_warmstart(z, L)
    return xs, us, sols


def propagate(nlp, L, theta, xs, us, sols, mode, options=None):
    """
    Closed-loop sensitivity  dx_t / dtheta  (T+1, 2), propagated with
      dx_{t+1} = f_x dx_t + f_th + f_u (dMPC/dx_t dx_t + dMPC/dtheta),
    where dMPC/d(x_t, theta) is computed by nlp.differentiate(mode, options).
    """
    _, fjac = plant_function(L["m"])
    T = len(sols)
    dxs = np.zeros((T + 1, 2))
    rows = u0_rows(L)
    for t, sol in enumerate(sols):
        d = nlp.differentiate(sol.theta, sol.x, sol.lam, sol.nu, mode=mode, options=options)
        dU = np.array(ca.DM(d["x"]).full())[rows, :]          # (m, 3): d u0 / d(x_init, theta)
        fx, fu, fth = (np.array(M.full()) for M in fjac(xs[t], us[t], theta))
        dxs[t + 1] = (fx @ dxs[t] + fth.flatten()
                      + fu @ (dU[:, :2] @ dxs[t] + dU[:, 2]))
    return dxs


def finite_differences(nlp, L, theta, x0, T, sols_nom, delta):
    xp, _, _ = simulate(nlp, L, theta + delta, x0, T, warm=sols_nom)
    xm, _, _ = simulate(nlp, L, theta - delta, x0, T, warm=sols_nom)
    return (xp - xm) / (2 * delta)


def degeneracy_map(nlp, sols):
    """Compute the degeneracy structure (flat-direction dim P, redundant-active
    dim U, weakly/strongly active constraint counts, reduced-Hessian eigenvalue
    stats) at each MPC solution in `sols`, returning arrays indexed by time step."""
    out = {k: [] for k in ("dim_P", "dim_U", "n_weak", "n_active", "min_eig", "gap")}
    for sol in sols:
        s = nlp.degenerate_subspaces(sol.theta, sol.x, sol.lam, sol.nu, active_tol=ACTIVE_TOL)
        out["dim_P"].append(s["dim_P"])
        out["dim_U"].append(s["dim_U"])
        out["n_weak"].append(len(s["weak_idx"]))
        out["n_active"].append(len(s["active_idx"]))
        out["min_eig"].append(s["min_reduced_eig"])
        out["gap"].append(s["smallest_nonflat_eig"])
    return {k: np.array(v) for k, v in out.items()}


def rel_err(a, b):
    """Relative error ||a - b|| / ||b|| (Frobenius norm for 2-D arrays)."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos_sim(a, b):
    """Cosine similarity between a and b, flattened to vectors first."""
    a, b = a.flatten(), b.flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


LAST_TIME = [0.0]   # mutable box holding the per-call timing from the last safe_propagate call


def safe_propagate(*args, **kwargs):
    """Wrap propagate() to catch solver/linear-algebra failures (which happen for
    some rho values in the sweeps) and time the call, recording the average
    per-MPC-instance cost in LAST_TIME. Returns None on failure instead of raising."""
    try:
        t0 = time.perf_counter()
        out = propagate(*args, **kwargs)
        LAST_TIME[0] = (time.perf_counter() - t0) / len(args[5])   # per MPC instance
        return out
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"      failed: {type(e).__name__}: {e}")
        return None


# =====================================================================
# MAIN
# =====================================================================

def run(m, theta, x0, T, N, delta, rho_grid, rho_fixed, u_max=0.95, verbose=True):
    """
    Run the full comparison for a given number of actuators m:
    simulate the m = 1 and over-actuated closed loops, characterize their
    degeneracy, compute finite-difference and reference sensitivities, then
    sweep the "projected" and "regularized" (old) sensitivity methods over
    rho_grid and evaluate the unregularized least-squares/plain solves.
    Returns a dict of results (arrays and scalar summaries) consumed by
    plot() and by the __main__ scaling loop.
    """
    res = {"m": m}

    # ---------------- 1. reference (m = 1) and over-actuated closed loops ----------------
    t0 = time.time()
    nlp1, L1 = build_mpc_nlp(m=1, N=N, u_max=u_max)
    xs1, us1, sols1 = simulate(nlp1, L1, theta, x0, T)
    nlpm, Lm = build_mpc_nlp(m=m, N=N, u_max=u_max)
    xsm, usm, solsm = simulate(nlpm, Lm, theta, x0, T)
    if verbose:
        print(f"\n=== m = {m}: n_x = {nlpm.n_x}, n_ineq = {nlpm.n_ineq}, n_eq = {nlpm.n_eq} "
              f"({time.time() - t0:.1f}s for both closed loops)")
        print(f"  max |x_cl(m) - x_cl(1)|           = {np.max(np.abs(xsm - xs1)):.2e}")
        print(f"  max |sum u(m) - u(1)|             = {np.max(np.abs(usm.sum(1) - us1[:, 0])):.2e}")
        sat = np.where(np.abs(np.abs(us1[:, 0]) - u_max) < 1e-6)[0]
        print(f"  saturated steps                   = {sat.size}" + (f" (t = {sat[0]}..{sat[-1]})" if sat.size else ""))
    res.update(xs=xsm, us=usm, xs_ref=xs1, n_x=nlpm.n_x)

    # ---------------- 2. degeneracy structure ----------------
    deg1 = degeneracy_map(nlp1, sols1)
    degm = degeneracy_map(nlpm, solsm)
    res["deg"] = degm
    if verbose:
        print("  degeneracy along the closed loop       m = 1        m = %d" % m)
        for k, lbl in [("dim_P", "steps with dim P > 0"), ("dim_U", "steps with dim U > 0"),
                       ("n_weak", "steps with weakly active")]:
            print(f"    {lbl:<32} {np.sum(deg1[k] > 0):>6}/{T}   {np.sum(degm[k] > 0):>6}/{T}")
        print(f"    {'mean dim P / mean dim U':<32} {deg1['dim_P'].mean():5.1f}/{deg1['dim_U'].mean():<4.1f}"
              f"  {degm['dim_P'].mean():5.1f}/{degm['dim_U'].mean():<4.1f}")
        print(f"    min eigenvalue of reduced Hessian  {deg1['min_eig'].min():+.1e}   {degm['min_eig'].min():+.1e}")
        print(f"    smallest non-flat eigenvalue       {deg1['gap'].min():.1e}    {degm['gap'].min():.1e}")

    # ---------------- 3. finite differences and reference sensitivity ----------------
    dfd = finite_differences(nlpm, Lm, theta, x0, T, solsm, delta)
    dfd1 = finite_differences(nlp1, L1, theta, x0, T, sols1, delta)
    # Reference: the m = 1 problem with the projected method (exact where nondegenerate).
    dref = propagate(nlp1, L1, theta, xs1, us1, sols1, "projected",
                     {"rho_1": rho_fixed, "rho_2": rho_fixed, "tol": ACTIVE_TOL})
    res.update(dfd=dfd, dfd_ref=dfd1, dref=dref)
    if verbose:
        print(f"  FD(m) vs FD(1)          rel. err = {rel_err(dfd, dfd1):.2e}")
        print(f"  reference vs FD(1)      rel. err = {rel_err(dref, dfd1):.2e}")

    # ---------------- 4. sensitivities of the over-actuated MPC ----------------
    def report(name, d):
        """Print/collect (err vs FD, err vs m=1 reference, cos-sim vs FD, ms/instance)
        for a sensitivity result d, or NaNs if the underlying solve failed (d is None)."""
        if d is None:
            return (np.nan, np.nan, np.nan, np.nan)
        e = (rel_err(d, dfd), rel_err(d, dref), cos_sim(d, dfd), 1e3 * LAST_TIME[0])
        if verbose:
            print(f"    {name:<28} {e[0]:>11.3e} {e[1]:>11.3e} {e[2]:>9.5f} {e[3]:>9.2f}")
        return e

    if verbose:
        print(f"\n  {'method':<30} {'err vs FD':>11} {'err vs ref':>11} {'cos(FD)':>9} {'ms/inst':>9}")

    sweeps = {"proj_rho1": [], "proj_rho2": [], "old": []}
    for r in rho_grid:
        d = safe_propagate(nlpm, Lm, theta, xsm, usm, solsm, "projected",
                           {"rho_1": r, "rho_2": rho_fixed, "tol": ACTIVE_TOL})
        sweeps["proj_rho1"].append(report(f"projected rho1={r:.0e}", d))
    for r in rho_grid:
        d = safe_propagate(nlpm, Lm, theta, xsm, usm, solsm, "projected",
                           {"rho_1": rho_fixed, "rho_2": r, "tol": ACTIVE_TOL})
        sweeps["proj_rho2"].append(report(f"projected rho2={r:.0e}", d))
    for r in rho_grid:
        d = safe_propagate(nlpm, Lm, theta, xsm, usm, solsm, "regularized",
                           {"rho_1": r, "rho_2": r})
        sweeps["old"].append(report(f"old rho*I  rho={r:.0e}", d))

    d_ls = safe_propagate(nlpm, Lm, theta, xsm, usm, solsm, "active_set",
                          {"tol": ACTIVE_TOL, "lstsq": True})
    res["lstsq"] = report("unregularized, lstsq", d_ls)
    d_as = safe_propagate(nlpm, Lm, theta, xsm, usm, solsm, "active_set", {"tol": ACTIVE_TOL})
    res["plain"] = report("unregularized, solve", d_as)

    res.update({k: np.array(v) for k, v in sweeps.items()})
    res["d_proj"] = propagate(nlpm, Lm, theta, xsm, usm, solsm, "projected",
                              {"rho_1": rho_fixed, "rho_2": rho_fixed, "tol": ACTIVE_TOL})
    return res


def plot(res, rho_grid, T, fname):
    """
    Save a 3-panel preview figure to `fname`:
      1. rel. error vs the m = 1 reference for each sensitivity method, over rho_grid,
      2. degeneracy structure (dim P, dim U, # weakly active) along the closed loop,
      3. finite-difference vs projected-method state sensitivity trajectories.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    a = ax[0]
    # error against the m = 1 reference (exact); FD noise would floor all curves at ~1e-7
    a.loglog(rho_grid, res["proj_rho1"][:, 1], "o-", label=r"projected, sweep $\rho_1$")
    a.loglog(rho_grid, res["proj_rho2"][:, 1], "s--", label=r"projected, sweep $\rho_2$")
    a.loglog(rho_grid, res["old"][:, 1], "^-", label=r"previous ($\rho I$)")
    if np.isfinite(res["lstsq"][1]):
        a.axhline(res["lstsq"][1], color="k", ls=":", label="least squares")
    a.axhline(rel_err(res["dfd"], res["dref"]), color="gray", ls="-.", lw=1, label="finite differences")
    a.set_xlabel(r"$\rho$"); a.set_ylabel("rel. error vs m = 1 reference")
    a.set_title(f"closed-loop sensitivity, m = {res['m']}"); a.legend(fontsize=8); a.grid(True, which="both", alpha=.3)

    a = ax[1]
    t = np.arange(T)
    a.step(t, res["deg"]["dim_P"], where="post", label=r"dim $\mathcal{P}$")
    a.step(t, res["deg"]["dim_U"], where="post", label=r"dim $\mathcal{U}$")
    a.step(t, res["deg"]["n_weak"], where="post", label="# weakly active")
    a.set_xlabel("closed-loop step t"); a.set_title("degeneracy of each MPC instance"); a.legend(fontsize=8)

    a = ax[2]
    tt = np.arange(T + 1)
    for i in range(2):
        a.plot(tt, res["dfd"][:, i], "-", color=f"C{i}", label=rf"FD $\partial x^{i+1}/\partial\theta$")
        a.plot(tt, res["d_proj"][:, i], "--", color="k" if i == 0 else "gray",
               label=rf"projected $\partial x^{i+1}/\partial\theta$")
    a.set_xlabel("t"); a.set_title("closed-loop state sensitivity"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"\nfigure saved to {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2, help="number of redundant actuators")
    ap.add_argument("--theta", type=float, default=1.0, help="nominal parameter")
    ap.add_argument("--T", type=int, default=200, help="closed-loop horizon")
    ap.add_argument("--N", type=int, default=20, help="MPC horizon")
    ap.add_argument("--umax", type=float, default=0.95,
                    help="total input bound (the paper uses 2, which never saturates from x0=(3,0))")
    ap.add_argument("--delta", type=float, default=1e-6, help="finite-difference step")
    ap.add_argument("--rho", type=float, default=1e-4, help="fixed rho in the sweeps")
    ap.add_argument("--scaling", type=int, nargs="*", default=None,
                    help="also run for these m (e.g. --scaling 1 2 5 10)")
    ap.add_argument("--out", type=str, default="mpc_overactuated")
    args = ap.parse_args()

    x0 = np.array([3.0, 0.0])
    rho_grid = np.logspace(-8, 0, 9)

    res = run(args.m, args.theta, x0, args.T, args.N, args.delta, rho_grid, args.rho, args.umax)
    plot(res, rho_grid, args.T, args.out + ".png")

    save = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    save.update({f"deg_{k}": v for k, v in res["deg"].items()})
    save["rho_grid"] = rho_grid

    if args.scaling:
        print("\n=== scaling with the number of actuators ===")
        print(f"  {'m':>3} {'n_x':>5} {'mean dim P':>11} {'mean dim U':>11} {'projected':>11} "
              f"{'old rho I':>11} {'lstsq':>11} {'ms proj':>8} {'ms lstsq':>8}"
              f"   (rel. err vs m=1 reference, rho = {args.rho:.0e})")
        rows = []
        for m in args.scaling:
            r = run(m, args.theta, x0, args.T, args.N, args.delta, np.array([args.rho]),
                    args.rho, args.umax, verbose=False)
            row = [m, r["n_x"], r["deg"]["dim_P"].mean(), r["deg"]["dim_U"].mean(),
                   r["proj_rho1"][0, 1], r["old"][0, 1], r["lstsq"][1],
                   r["proj_rho1"][0, 3], r["lstsq"][3]]
            rows.append(row)
            print(f"  {row[0]:>3} {row[1]:>5} {row[2]:>11.1f} {row[3]:>11.1f} {row[4]:>11.3e} "
                  f"{row[5]:>11.3e} {row[6]:>11.3e} {row[7]:>8.2f} {row[8]:>8.2f}")
        save["scaling"] = np.array(rows)

    np.savez(args.out + "_results.npz", **save)
    print(f"results saved to {args.out}_results.npz")
