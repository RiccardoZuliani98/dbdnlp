# =====================================================================
# IMPORTS
# =====================================================================

import os
import numpy as np
import casadi as ca
from time import perf_counter
from typing import Callable, Literal, Optional, Tuple, cast

from dbdnlp import DifferentiableNLP


# =====================================================================
# CONSTANT PARAMETERS
# =====================================================================

# IPOPT options
SOLVER_OPTS = {
    "ipopt.print_level": 0, 
    "print_time": 0,
    "ipopt.tol": 1e-12, 
    "ipopt.constr_viol_tol": 1e-12,
    "ipopt.dual_inf_tol": 1e-10, 
    "ipopt.max_iter": 3000,
}

N = 150                     # number of time-steps
M0 = 1.0                    # nominal steering-bound scaling m
DELTA = 1e-5                # finite difference perturbation
TOL = 1e-7                  # active-set tolerance (complementarity-aware rule)
RHOS = [
    1e-8, 1e-6, 
    1e-4, 1e-2, 
    1.0
]                           # rho sweep (for both $\rho_1$ and $\rho_2$)
RHO = 1e-4                  # when a single $\rho_1=\rho_2$ is needed
C_LIST = [
    1e-2, 1e-1,
    1.0 , 1e1,
    1e2
]                           # scaling of initial condition constraint
THETA3 = np.array([
    M0, 0.5, 0.25
])                          # nominal [m, p_x(T), p_y(T)] for the 3-parameter test
TAYLOR_STEPS = [
    1e-1, 3e-2, 1e-2,
    3e-3, 1e-3, 3e-4,
    1e-4
]                           # step sizes dm for the Taylor test
N_TIMING = 100              # number of evaluations to compute timings
RESULTS_DIR = ".results"    # location of dat files
PLOT_DIR = RESULTS_DIR      # location of trajectory tables (copy to Figures/Data/)
PLOT_DM = 1e-1              # parameter perturbation shown in the trajectory figure
PLOT_RHO = 1e-4             # regularization used for the figure's first-order prediction


# helper to suppress casadi's type warnings
def MX_sym(*args) -> ca.MX:
    return ca.MX.sym(*args)

def DM_ones(*args) -> ca.DM:
    return ca.DM.ones(*args)

# =====================================================================
# NLP CREATION
# =====================================================================

def rk4(fc: ca.Function, x: ca.MX, u: ca.MX, m: ca.MX, T: ca.DM) -> ca.MX:
    """
    Take one classical fourth-order Runge-Kutta step.
 
    Integrates ``xdot = fc(x, u, m)`` over a step of length `T`, with the
    control held constant over the step (zero-order hold).
 
    Parameters
    ----------
    fc : ca.Function
        Continuous-time dynamics, ``fc(x, u, m) -> xdot``.
    x : ca.MX
        State at the start of the step, shape ``(nx, 1)``.
    u : ca.MX
        Control applied over the step, shape ``(nu, 1)``.
    m : ca.MX
        Problem parameter, passed through to `fc` unchanged.
    T : ca.DM
        Step length (here the fixed interval ``dt`` between shooting nodes).
 
    Returns
    -------
    ca.MX
        State at the end of the step, shape ``(nx, 1)``.
    """

    k1 = cast(ca.MX, fc(x, u, m))
    k2 = cast(ca.MX, fc(x + T / 2 * k1, u, m))
    k3 = cast(ca.MX, fc(x + T / 2 * k2, u, m))
    k4 = cast(ca.MX, fc(x + T * k3, u, m))
    return x + (T / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def euler(fc: ca.Function, x: ca.MX, u: ca.MX, m: ca.MX, T: ca.DM) -> ca.MX:
    """
    Take one explicit (forward) Euler step.
 
    Computes ``x + T * fc(x, u, m)``, with the control held constant over
    the step (zero-order hold).
 
    Parameters
    ----------
    fc : ca.Function
        Continuous-time dynamics, ``fc(x, u, m) -> xdot``.
    x : ca.MX
        State at the start of the step, shape ``(nx, 1)``.
    u : ca.MX
        Control applied over the step, shape ``(nu, 1)``.
    m : ca.MX
        Problem parameter, passed through to `fc` unchanged.
    T : ca.DM
        Step length (here the fixed interval ``dt`` between shooting nodes).
 
    Returns
    -------
    ca.MX
        State at the end of the step, shape ``(nx, 1)``.

    Notes
    -----
    With Euler the Dubins problem is structurally degenerate: steering at
    standstill has no effect (the car starts at rest), and the last
    steering input only changes the final heading, which is free. Both are
    flat directions of the Lagrangian on the critical cone, so SSOSC fails
    and the unregularized KKT matrix is singular. RK4's intermediate stages
    give both inputs some authority, which removes the degeneracy.
    """
    return x + T * cast(ca.MX, fc(x, u, m))

def build_dubins_nlp(
    N : int = 15,
    a_max : float = 1.0,
    s_max : float = 1.0,
    v_max : Optional[float] = None,
    solver_opts : Optional[dict] = None,
    integrator : Literal["rk4", "euler"] = "rk4",
    init_scale : float = 1.0,
    goal_as_param : bool = False,
    goal : Tuple[float, float] = (0.5, 0.25)
) -> Tuple[DifferentiableNLP, dict[str, int]]:
    """
    Build the minimum-time Dubins-car NLP as a `DifferentiableNLP`.
 
    The car starts at rest at the origin and must reach a goal position at
    rest in minimum time. Free final time is handled through a constant time
    scaling state ``sigma``, which is also the cost.
 
    Parameters
    ----------
    N : int, default 15
        Number of shooting intervals. The state trajectory has ``N + 1``
        nodes and the control trajectory has ``N`` intervals.
    a_max : float, default 1.0
        Bound on the acceleration input, ``|a_k| <= a_max``.
    s_max : float, default 1.0
        Nominal steering bound. The effective bound is
        ``|s_k| <= s_max * m``, so the parameter `m` moves an active
        constraint.
    v_max : float, optional
        If given, adds the box bounds ``|vx_k| <= v_max`` and
        ``|vy_k| <= v_max`` at every node. No velocity bound by default.
    solver_opts : dict, optional
        IPOPT/CasADi options passed to the solver. Defaults to
        `SOLVER_OPTS`.
    integrator : {"rk4", "euler"}, default "rk4"
        Discretization of the dynamics. ``"rk4"`` gives a nondegenerate
        problem, ``"euler"`` a structurally degenerate one (see `euler`).
    init_scale : float, default 1.0
        The initial condition is imposed as
        ``init_scale * X[0:5, 0] = 0``. Any nonzero value describes the
        same problem and only rescales these constraint rows (and their
        multipliers).
    goal_as_param : bool, default False
        If True, the parameter is ``theta = [m, p_x(T), p_y(T)]``, so the
        terminal position is a parameter as well. If False, ``theta = m``
        and the terminal position is fixed to `goal`.
    goal : tuple of float, default (0.5, 0.25)
        Terminal position ``(p_x(T), p_y(T))``. Ignored when
        `goal_as_param` is True.
 
    Returns
    -------
    nlp : DifferentiableNLP
        The parametric NLP, ``min f(x) s.t. g(x, theta) <= 0,
        h(x, theta) = 0``.
    dim_dict : dict of str to int
        Layout of the primal vector, with keys
 
        - ``"N"`` : number of shooting intervals,
        - ``"nx"``, ``"nu"`` : state and input dimensions,
        - ``"nx_tot"`` : number of state entries, ``nx * (N + 1)``,
        - ``"nu_tot"`` : number of input entries, ``nu * N``,
        - ``"n_tot"`` : total primal dimension.
    """

    # state and input dimensions
    nx, nu = 6, 2

    # full state and input trajectories
    x = MX_sym("x", nx, N+1)
    u = MX_sym("u", nu, N)

    # set parameters
    if goal_as_param:
        # theta = [ steering bound, p_x(T), p_y(T) ]
        theta = MX_sym("theta", 3)
        m, x_ref = theta[0], theta[1:3]
    else:
        # theta = steering bound
        theta = MX_sym("m", 1)
        x_ref = ca.DM(list(goal))
        m = theta

    # implement continuous dynamics, first define state and input
    xs, us, ms = MX_sym("xs", nx), MX_sym("us", nu), MX_sym("ms", 1)
    vel_x, vel_y, heading, time_dilation = xs[2], xs[3], xs[4], xs[5]
    acceleration, steering = us[0], us[1]

    # then implement dynamics
    rhs = ca.vertcat(
        vel_x, 
        vel_y,
        acceleration * ca.cos(heading), 
        acceleration * ca.sin(heading),
        steering * (ca.cos(heading) * vel_x + ca.sin(heading) * vel_y)
    )
    # multiply by time-dilation to have free-final time dynamics.
    # time-dilation is constant, hence the trailing zero.
    fc = ca.Function("fc", [xs, us, ms], [ca.vertcat(time_dilation * rhs, 0)])

    # time is normalized to (almost) 1.
    dt = 1.0 / (N + 1)

    # ---- equalities h(x, m) = 0 ----
    # start at rest at origin
    h = [init_scale * x[0:5, 0]]

    # loop through time-steps and enforce dynamics
    for k in range(N):
        if integrator == "rk4":
            x_next = rk4(fc, x[:, k], u[:, k], m, ca.DM(dt))
        elif integrator == "euler":
            x_next = euler(fc, x[:, k], u[:, k], m, ca.DM(dt))
        else:
            raise ValueError(f"unknown integrator '{integrator}'")
        h.append(x[:, k + 1] - x_next)

    # enforce final position
    h.append(x[0:2, N] - x_ref)
    
    # zero terminal velocity
    h.append(x[2:4, N])

    # stack together
    h = cast(ca.MX, ca.vertcat(*h))

    # ---- inequalities g(x, m) <= 0 ----
    g = []
    for k in range(N):

        # acceleration bounds
        g.append(u[0, k] - a_max)
        g.append(-u[0, k] - a_max)

        # steering bounds (with parameters)
        g.append(u[1, k] - s_max * m)
        g.append(-u[1, k] - s_max * m)

    # velocity bound if provided    
    if v_max is not None:
        for k in range(N + 1):
            g.append(x[2:4, k] - v_max)
            g.append(-x[2:4, k] - v_max)
    
    # time-dilation is non-negative
    g.append(-x[5, 0])

    # stack together
    g = cast(ca.MX, ca.vertcat(*g))


    # ---- cost f(x) ----
    # select time-dilation (at any time-step)
    f = x[5,N] 

    # build NLP
    x_opt = cast(ca.MX, ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)))
    nlp = DifferentiableNLP(
        x=x_opt, theta=theta, 
        f=f, g=g, h=h,
        solver_opts=dict(solver_opts or SOLVER_OPTS)
    )

    # build dimension dictionary
    dim_dict = dict(
        N = N, nx = nx, nu = nu,
        nx_tot = nx * (N + 1),
        nu_tot = nu * N,
        n_tot = nx * (N + 1) + nu * N
    )

    return nlp, dim_dict

def unpack(z: ca.DM | np.ndarray, L: dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a flat primal vector into state and input trajectories.
 
    Inverse of the column-major flattening done in `build_dubins_nlp`.
    Works for the primal solution itself or for any vector with the same
    layout, e.g. a sensitivity ``dx/dm``.
 
    Parameters
    ----------
    z : ca.DM or array_like
        Flat vector of length ``L["n_tot"]``, ordered as ``[vec(X); vec(U)]``.
    L : dict of str to int
        Layout dictionary returned by `build_dubins_nlp`. Uses the keys
        ``"N"``, ``"nx"``, ``"nu"`` and ``"nx_tot"``.
 
    Returns
    -------
    x : numpy.ndarray
        State trajectory, shape ``(nx, N + 1)``. Column ``k`` is node ``k``.
    u : numpy.ndarray
        Input trajectory, shape ``(nu, N)``. Column ``k`` is interval ``k``.
    """
    # convert to a flat numpy vector (works for ca.DM and numpy input alike)
    z = np.asarray(z, dtype=float).flatten()

    # column-major reshape, matching ca.reshape in build_dubins_nlp
    x = z[:L["nx_tot"]].reshape((L["nx"], L["N"] + 1), order="F")
    u = z[L["nx_tot"]:].reshape((L["nu"], L["N"]), order="F")
    return x, u

# =====================================================================
# ERROR COMPUTATION
# =====================================================================

def compute_error(a:ca.DM | np.ndarray, b: ca.DM | np.ndarray) -> Tuple[float, float]:
    """
    Compare a vector or matrix against a reference.

    Both inputs are flattened first, so for matrices the relative error is
    the Frobenius-norm error and the cosine similarity uses the Frobenius
    inner product.

    Parameters
    ----------
    a : ca.DM or numpy.ndarray
        Quantity to evaluate, e.g. a computed sensitivity ``dx/dtheta``.
    b : ca.DM or numpy.ndarray
        Reference with the same number of entries as `a`. Must be nonzero.

    Returns
    -------
    cos_sim : float
        Cosine similarity ``<a, b> / (|a| |b|)``, in ``[-1, 1]``.
    relative_error : float
        Relative error ``|a - b| / |b|``.
    """
    # convert to numpy and flatten
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    # relative error
    relative_error = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    
    # cosine similarity
    cos_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    return cos_sim, relative_error


# =====================================================================
# CLOSEST POINT SOLVER (PROXIMAL)
# =====================================================================

def closest_point_solver(nlp: DifferentiableNLP, eps=1e-8) -> DifferentiableNLP:
    """
    Build the closest-point (proximal) variant of an NLP.

    The returned problem is ``min f(x) + eps/2 |x - x_bar|^2`` subject to the
    constraints of `nlp`, with ``x_bar`` appended to the parameters. It has
    the same feasible set and cost as `nlp`, but when the original solution
    set is degenerate (flat directions), it picks the point closest to
    ``x_bar``. This gives a well-posed finite-difference reference for
    ``dx/dtheta``.

    Parameters
    ----------
    nlp : DifferentiableNLP
        Original problem. Its symbolic primal variable must be ``ca.MX``.
    eps : float, default 1e-8
        Weight of the proximal term. It should be small enough not to
        noticeably change the solution along nondegenerate directions.

    Returns
    -------
    DifferentiableNLP
        Proximal problem with parameter ``[theta; x_bar]``, of length
        ``n_theta + n_x``. It is solved with the module-level `SOLVER_OPTS`.
    """
    # extract nlp symbols
    sym = nlp.sym

    # we need a nominal solution as parameter
    xb = MX_sym("x_bar", nlp.n_x)

    # form new differentiable NLP and return
    return DifferentiableNLP(
        x=sym.primal, theta=cast(ca.MX, ca.vertcat(sym.param, xb)),
        f=sym.cost + eps / 2 * ca.sumsqr(sym.primal - xb),
        g=sym.ineq, h=sym.eq, solver_opts=dict(SOLVER_OPTS)
    )



# =====================================================================
# FINITE-DIFFERENCE JACOBIANS
# =====================================================================

def fd_jacobians(
    nlp: DifferentiableNLP,
    prox: DifferentiableNLP,
    theta0: np.ndarray,
    x_bar: np.ndarray,
    delta: float,
    with_warm: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute central finite-difference Jacobians ``dx/dtheta``.

    Each parameter component is perturbed by ``+-delta`` in turn, which
    costs two solves per component for each Jacobian.

    Parameters
    ----------
    nlp : DifferentiableNLP
        Original problem, with parameter `theta`.
    prox : DifferentiableNLP
        Closest-point variant of `nlp` (see `closest_point_solver`), with
        parameter ``[theta; x_bar]``.
    theta0 : numpy.ndarray
        Nominal parameter, shape ``(n_theta,)``.
    x_bar : numpy.ndarray
        Anchor of the proximal term, usually the nominal solution. Also
        used to warm-start the proximal solves.
    delta : float
        Finite-difference step.
    with_warm : bool, default True
        If False, skip the warm-started solves of `nlp` (e.g. for timing
        the proximal reference alone).

    Returns
    -------
    J_prox : numpy.ndarray
        Finite differences of the proximal problem, shape ``(n_x, n_theta)``.
    J_warm : numpy.ndarray or None
        Finite differences of the original problem, warm-started at the
        proximal solutions ``x^+`` and ``x^-``, shape ``(n_x, n_theta)``.
        None if `with_warm` is False.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    n_theta = theta0.size

    # allocate Jacobians
    J_prox = np.zeros((nlp.n_x, n_theta))
    J_warm = np.zeros((nlp.n_x, n_theta)) if with_warm else None

    for j in range(n_theta):

        # perturbation of the j-th parameter component
        e = np.zeros(n_theta)
        e[j] = delta

        # proximal problem, warm-started at x_bar
        xp = np.array(prox.solve(np.r_[theta0 + e, x_bar], x_warmstart=x_bar).x).flatten()
        xm = np.array(prox.solve(np.r_[theta0 - e, x_bar], x_warmstart=x_bar).x).flatten()
        J_prox[:, j] = (xp - xm) / (2 * delta)

        # original problem, warm-started at the proximal solutions
        if J_warm is not None:
            wp = np.array(nlp.solve(ca.DM(theta0 + e), x_warmstart=xp).x).flatten()
            wm = np.array(nlp.solve(ca.DM(theta0 - e), x_warmstart=xm).x).flatten()
            J_warm[:, j] = (wp - wm) / (2 * delta)

    return J_prox, J_warm


# =====================================================================
# TIMING
# =====================================================================

def time_it(fn: Callable[[], object], reps: int) -> np.ndarray:
    """
    Measure the wall-clock time of repeated calls.

    One untimed warm-up call is made first, so that one-off costs (e.g.
    CasADi function construction) are not included.

    Parameters
    ----------
    fn : callable
        Function to time, called without arguments.
    reps : int
        Number of timed calls.

    Returns
    -------
    numpy.ndarray
        Time of each call in seconds, shape ``(reps,)``.
    """
    # warm-up
    fn()

    # timed calls
    t = np.empty(reps)
    for i in range(reps):
        t0 = perf_counter()
        fn()
        t[i] = perf_counter() - t0
    return t


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    # ensure data directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # build NLP solver
    nlp, dim = build_dubins_nlp(N=N, integrator="euler")
    print(f"Dubins NLP (Euler): n_x={nlp.n_x}  n_ineq={nlp.n_ineq}  n_eq={nlp.n_eq}")

    # ---------------- 1. solve ----------------
    sol = nlp.solve(ca.DM(M0), x_warmstart=DM_ones(dim["n_tot"],1)*0.075)
    x0 = np.array(sol.x).flatten()
    Xs, Us = unpack(sol.x, dim)
    print(f"\n--- 1. Solution: status={nlp.solver.stats()['return_status']}, sigma={float(Xs[5, -1]):.6f}")

    # ---------------- 2. degeneracy ----------------
    # compute degeneracy information
    sub = nlp.degenerate_subspaces(np.array([M0]), sol.x, sol.lam, sol.nu, active_tol=TOL)
    w, V = np.linalg.eigh(sub["Pi_P"])
    print(
        f"\n--- 2. Degeneracy: n_active={len(sub['active_idx'])}, dim P={sub['dim_P']}, "
        f"dim U={sub['dim_U']}, weakly active={len(sub['weak_idx'])}, "
        f"min eig of reduced Hessian={sub['min_reduced_eig']:.1e}"
    )

    # print flat directions
    for c in np.where(w > 0.5)[0]:
        top = np.argsort(-np.abs(V[:, c]))[:2]
        print("    flat direction along index " + ", ".join(f"{i} ({V[i, c]:+.2f})" for i in top))

    # see if nominal KKT system can be solved
    try:
        nlp.differentiate(
            ca.DM([M0]), sol.x, sol.lam, sol.nu,
            mode="active_set",
            options = {"tol":TOL, "active_rule":"ratio"}
        )["x"]
        print("    unregularized KKT system: solved")
    except np.linalg.LinAlgError:
        print("    unregularized KKT system: singular")

    # ---------------- 3. references ----------------
    # `ref`: central finite difference of the closest-point re-solve (well-posed
    # even on the degenerate flat directions, since eps/2|x-x_bar|^2 picks a
    # unique minimizer). `warm`: central finite difference of the original
    # (degenerate) NLP re-solved from a warm start, for comparison.
    prox = closest_point_solver(nlp)
    xcp = lambda m: prox.solve(np.r_[m, x0], x_warmstart=x0).x

    def resolve(m: float, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-solve at parameter m around a (predicted) primal point z.

        Returns the solution of the original NLP(m) warm-started at z, and of
        the proximal NLP(m) anchored and warm-started at z. The original problem
        is degenerate, so the first is whichever point of the solution set IPOPT
        lands on; the second is the point of that set closest to z (up to O(eps)).
        """
        x_nlp = np.array(nlp.solve(ca.DM([m]), x_warmstart=z).x).flatten()
        x_prox = np.array(prox.solve(np.r_[m, z], x_warmstart=z).x).flatten()
        return x_nlp, x_prox

    # cost of the original problem, f(x, theta)
    f_fun = ca.Function("f_cost", [nlp.sym.primal, nlp.sym.param], [nlp.sym.cost])
    f_cost = lambda z, m: float(cast(ca.DM, f_fun(z, m)))

    ref_plus = xcp(M0 + DELTA)
    ref_minus = xcp(M0 - DELTA)
    ref = (ref_plus - ref_minus) / (2 * DELTA)
    warm = (
        nlp.solve(ca.DM([M0 + DELTA]), x_warmstart=ref_plus).x 
        - nlp.solve(ca.DM([M0 - DELTA]), x_warmstart=ref_minus).x
    ) / (2 * DELTA)

    # ---------------- 4. comparison ----------------
    # Compare every differentiation mode/setting against both references, using
    # the relative error over the full vector: `ref` (closest-point, well-posed
    # even on the degenerate directions) and `warm` (plain warm-started FD on
    # the original problem, itself just one method among the ones being judged).
    print("\n--- 3. dx/dm against the closest-point reference")
    print(f"    |dx/dm| (FD): {np.linalg.norm(np.array(ref)):.3e} (proximal), {np.linalg.norm(np.array(warm)):.3e} (warm-started)")
    print(f"    {'method':<26} {'err vs ref':>10} {'err vs warm':>12}")

    err_rows = []                  # (method, rho, err vs proximal FD, err vs warm-started FD)

    def row(label, v, method, rho=np.nan):
        """Print one comparison row (relative error of v against ref and warm) and store it."""
        err_rows.append((method, rho, compute_error(v, ref)[1], compute_error(v, warm)[1]))
        print(f"    {label:<26} {compute_error(v, ref)[1]:>10.1e} {compute_error(v, warm)[1]:>12.1e}")

    # references
    row("closest point (reference)", ref, "proxFD")
    row("warm-started IPOPT (FD)", warm, "warmFD")

    # projected method
    for r in RHOS:
        projected_result_rho = nlp.differentiate(
            ca.DM(M0), sol.x, sol.lam, sol.nu, 
            mode="projected", 
            options={"rho_1":r, "rho_2":r, "tol":TOL}
        )
        row(f"projected rho={r:.0e}", projected_result_rho["x"], "projected", r)

    # least squares
    least_squares_result = nlp.differentiate(
        ca.DM(M0), sol.x, sol.lam, sol.nu, 
        mode="active_set",
        options={"tol":TOL, "lstsq":True, "active_rule":"ratio"}
    )
    row("least squares", least_squares_result["x"], "lstsq")

    # original method
    for r in RHOS:
        original_result_rho = nlp.differentiate(
            ca.DM(M0), sol.x, sol.lam, sol.nu,
            mode="regularized",
            options={"rho_1":r, "rho_2":r}
        )
        row(f"previous (rho I) rho={r:.0e}", original_result_rho["x"], "uniform", r)

    # tab:dubins_errors. `nan` marks entries that are "--" in the table (no rho, or a
    # reference compared against itself). Readable with \pgfplotstableread.
    with open(os.path.join(RESULTS_DIR, "dubins_errors.dat"), "w") as fh:
        fh.write("method rho err_prox err_warm\n")
        for meth, r, ep, ew in err_rows:
            fh.write(f"{meth} {r:.1e} {ep:.3e} {ew:.3e}\n")
    print(f"    (saved to {RESULTS_DIR}/dubins_errors.dat)")

    # ---------------- 5. invariance to constraint scaling ----------------
    # Rebuild the NLP with the initial-condition constraint scaled by c (a
    # mathematically equivalent problem, see `init_scale`) and check that the
    # projected-regularization derivative of s_0 (the flat, standstill steering
    # direction) stays constant across c, while the least-squares one does not.
    # s_0 is input 1 at interval 0: inputs come after the nx_tot state entries,
    # so a_0 sits at nx_tot and s_0 right after it.
    i_s0 = dim["nx_tot"] + 1
    print("\n--- 4. Same problem, initial condition imposed as c * X[0:5,0] = 0")
    print(
        f"    {'c':>6} {'projected ds_0/dm':>18} {'err ref':>8} {'err warm':>9} "
        f"{'LS ds_0/dm':>12} {'err ref':>8} {'err warm':>9}"
    )

    scaling = []
    for c in C_LIST:

        # build and solve the scaled problem
        nlp_c, _ = build_dubins_nlp(N=N, integrator="euler", init_scale=c)
        sol_c = nlp_c.solve(ca.DM(M0), x_warmstart=x0)

        # projected method
        projected_c = np.array(nlp_c.differentiate(
            ca.DM(M0), sol_c.x, sol_c.lam, sol_c.nu,
            mode="projected",
            options={"rho_1":RHO, "rho_2":RHO, "tol":TOL}
        )["x"]).flatten()

        # least squares
        least_squares_c = np.array(nlp_c.differentiate(
            ca.DM(M0), sol_c.x, sol_c.lam, sol_c.nu,
            mode="active_set",
            options={"tol":TOL, "lstsq":True, "active_rule":"ratio"}
        )["x"]).flatten()

        # errors against both references
        p_ref, p_warm = compute_error(projected_c, ref)[1], compute_error(projected_c, warm)[1]
        l_ref, l_warm = compute_error(least_squares_c, ref)[1], compute_error(least_squares_c, warm)[1]

        scaling.append([c, projected_c[i_s0], p_ref, p_warm, least_squares_c[i_s0], l_ref, l_warm])
        print(
            f"    {c:>6.0e} {projected_c[i_s0]:>18.2e} {p_ref:>8.1e} {p_warm:>9.1e} "
            f"{least_squares_c[i_s0]:>12.4f} {l_ref:>8.1e} {l_warm:>9.1e}"
        )

    # tab:scaling (the table uses the *_prox columns; *_warm are for reference)
    np.savetxt(
        os.path.join(RESULTS_DIR, "dubins_scaling.dat"), np.array(scaling),
        header="c proj_ds0 proj_err_prox proj_err_warm ls_ds0 ls_err_prox ls_err_warm",
        comments="", fmt="%.6e"
    )
    print(f"    (saved to {RESULTS_DIR}/dubins_scaling.dat)")

    # ---------------- 6. Taylor test ----------------
    # For shrinking steps dm, check that x0 + dm * dx/dm approximates the
    # perturbed solution to first order: for a correct derivative, the
    # prediction error shrinks faster than the distance of x0 itself.
    print("\n--- 5. First-order prediction error of x0 + dm * dx/dm")

    # least squares is evaluated on the problem scaled with c = 0.1
    nlp_ls, _ = build_dubins_nlp(N=N, integrator="euler", init_scale=0.1)
    sol_ls = nlp_ls.solve(ca.DM(M0), x_warmstart=x0)

    # derivatives to test (flat numpy vectors)
    methods = {
        "projected": np.array(nlp.differentiate(
            ca.DM(M0), sol.x, sol.lam, sol.nu,
            mode="projected",
            options={"rho_1":RHO, "rho_2":RHO, "tol":TOL}
        )["x"]).flatten(),
        "LS c0.1": np.array(nlp_ls.differentiate(
            ca.DM(M0), sol_ls.x, sol_ls.lam, sol_ls.nu,
            mode="active_set",
            options={"tol":TOL, "lstsq":True, "active_rule":"ratio"}
        )["x"]).flatten(),
        "uniform 1e-5": np.array(nlp.differentiate(
            ca.DM(M0), sol.x, sol.lam, sol.nu,
            mode="regularized",
            options={"rho_1":1e-5, "rho_2":1e-5}
        )["x"]).flatten(),
        # FD of the original (degenerate) problem, no proximal term
        "FD unregularized": np.array(warm).flatten(),
    }

    # Each prediction z = x0 + dm * v is checked against the two re-solves at
    # m + dm returned by `resolve` (see section 3): the original NLP warm-started
    # at z, and the proximal NLP anchored at z. The row "x0" is the zeroth-order
    # prediction z = x0, for which the proximal target is x_cp(m + dm).
    nx_tot = dim["nx_tot"]
    taylor = []
    for dm in TAYLOR_STEPS:

        # errors of x0 (nominal) and of each prediction, full vector and states only
        # (for large dm the full error is dominated by the few steering inputs that
        # saturate, which a first-order prediction cannot capture)
        # together with the suboptimality f(x_prox) - f(x_nlp) of the proximal
        # re-solve in the original problem, which certifies that it lands on the
        # solution set (the constraints are the same, so it is feasible)
        errs, errs_state, gaps = [], [], []
        for v in [np.zeros_like(x0)] + list(methods.values()):
            pred = x0 + dm * v
            x_nlp, x_prox = resolve(M0 + dm, pred)
            for target in (x_nlp, x_prox):
                errs.append(np.linalg.norm(pred - target))
                errs_state.append(np.linalg.norm(pred[:nx_tot] - target[:nx_tot]))
            gaps.append(f_cost(x_prox, M0 + dm) - f_cost(x_nlp, M0 + dm))

        taylor.append([dm] + errs + errs_state + gaps)

    # print one table per re-solve (columns: x0 first, then the methods)
    for t, target in enumerate([
        "original NLP(m+dm), warm-started at z",
        "proximal NLP(m+dm), anchored and warm-started at z"
    ]):
        print(f"    |z - x| with x from the {target}")
        print(f"    {'dm':>8} {'x0':>10} " + " ".join(f"{k:>16}" for k in methods))
        for r in taylor:
            print(
                f"    {r[0]:>8.0e} {r[1 + t]:>10.3e} "
                + " ".join(f"{e:>16.3e}" for e in r[3 + t : 3 + 2 * len(methods) : 2])
            )
    print("    suboptimality f(x_prox) - f(x_nlp) of the proximal re-solve")
    print(f"    {'dm':>8} {'x0':>10} " + " ".join(f"{k:>16}" for k in methods))
    for r in taylor:
        gaps = r[-(len(methods) + 1):]
        print(f"    {r[0]:>8.0e} {gaps[0]:>10.3e} " + " ".join(f"{e:>16.3e}" for e in gaps[1:]))

    # fig:dubins_taylor
    names = ["nominal"] + [k.replace(" ", "_") for k in methods]
    header = (
        "dm "
        + " ".join(f"{k}_vs_nlp {k}_vs_prox" for k in names) + " "
        + " ".join(f"{k}_vs_nlp_states {k}_vs_prox_states" for k in names) + " "
        + " ".join(f"{k}_gap" for k in names)
    )
    np.savetxt(
        os.path.join(RESULTS_DIR, "dubins_taylor.dat"), np.array(taylor),
        header=header, comments=""
    )
    print(f"    (saved to {RESULTS_DIR}/dubins_taylor.dat for pgfplots)")

    # ---------------- 7. data for the trajectory figure (position.tex) ----------------
    # Three trajectories, each written as two-column pgfplots tables (t, value) for
    # the x-position, y-position, acceleration a and steering s:
    #   suffix ""           : nominal solution x(m)
    #   suffix "_m2"        : original NLP re-solved at m + dm, warm-started at the prediction
    #   suffix "_m2_approx" : first-order prediction x(m) + dm * dx/dm (projected)
    # The time axis is physical time t_k = k * sigma / (N + 1), with sigma the time
    # dilation of that trajectory (for the prediction, sigma is itself predicted to
    # first order, so its time axis stretches accordingly).
    os.makedirs(PLOT_DIR, exist_ok=True)

    def save_trajectory(z: np.ndarray, suffix: str) -> Tuple[float, dict[str, np.ndarray]]:
        """Write x_pos, y_pos, a, s tables for primal vector z; return final time and series."""
        x_traj, u_traj = unpack(z, dim)
        t = np.arange(dim["N"] + 1) * x_traj[5, -1] / (dim["N"] + 1)

        # inputs are held constant on [t_k, t_{k+1}): repeat the last one at t_N so
        # that every curve spans [0, t_N] (use `const plot` in pgfplots)
        series = {
            "x_pos": x_traj[0],
            "y_pos": x_traj[1],
            "a": np.r_[u_traj[0], u_traj[0, -1]],
            "s": np.r_[u_traj[1], u_traj[1, -1]],
        }
        for name, v in series.items():
            np.savetxt(
                os.path.join(PLOT_DIR, f"{name}{suffix}.tex"), np.column_stack([t, v]),
                header="t v", comments="", fmt="%.10e"
            )
        return float(t[-1]), series

    # first-order prediction
    x_app = x0 + PLOT_DM * np.array(nlp.differentiate(
        ca.DM(M0), sol.x, sol.lam, sol.nu,
        mode="projected",
        options={"rho_1":PLOT_RHO, "rho_2":PLOT_RHO, "tol":TOL}
    )["x"]).flatten()

    # re-solves at m + dm around the prediction (see `resolve`); the figure shows
    # the original NLP, the proximal one is only used for the error printout
    x_p, x_p_prox = resolve(M0 + PLOT_DM, x_app)
    status_p = nlp.solver.stats()["return_status"]
    status_p_prox = prox.solver.stats()["return_status"]

    # write tables
    tf, data = {}, {}
    for label, z, suffix in [
        ("nominal", x0, ""),
        ("perturbed", x_p, "_m2"),
        ("approx", x_app, "_m2_approx"),
    ]:
        tf[label], data[label] = save_trajectory(z, suffix)

    # final time for the x-axis limit (xmax=\dubinsTf in position.tex)
    with open(os.path.join(PLOT_DIR, "position_params.tex"), "w") as fh:
        fh.write(f"\\def\\dubinsTf{{{max(tf.values()):.6f}}}%\n")

    print(f"\n--- 6. Figure data (dm = {PLOT_DM:g}, rho = {PLOT_RHO:g}), saved to {PLOT_DIR}/")
    print(f"    re-solve at m + dm: status original={status_p}, proximal={status_p_prox}")
    print("    final time        " + "  ".join(f"{k}={v:.4f}" for k, v in tf.items()))
    print(f"    |x(m+dm) - x(m)|              = {np.linalg.norm(x_p - x0):.3e} (original)")
    print(
        f"    |x(m+dm) - x(m) - dm * dx/dm| = {np.linalg.norm(x_p - x_app):.3e} (original), "
        f"{np.linalg.norm(x_p_prox - x_app):.3e} (proximal)"
    )
    print("    data ranges (for the y-axis limits):")
    for name in ("x_pos", "y_pos", "a", "s"):
        v = np.concatenate([d[name] for d in data.values()])
        print(f"      {name:<6} [{v.min():+.3f}, {v.max():+.3f}]")

    # ---------------- 8. three-dimensional parameter ----------------
    # theta = [m, p_x(T), p_y(T)]: the steering-bound scaling plus the terminal
    # position. The reference is again the proximal central FD (one pair of solves
    # per component), anchored at x0. Errors are relative Frobenius-norm errors of
    # the full Jacobian dx/dtheta (n_x x 3), cosine similarity uses the Frobenius
    # inner product. Least squares depends on the constraint scaling, so both
    # methods are evaluated for every c in C_LIST (the paper uses c = 1).
    print("\n--- 7. Three-dimensional parameter theta = [m, p_x(T), p_y(T)]")

    # build and solve
    nlp3, _ = build_dubins_nlp(N=N, integrator="euler", goal_as_param=True)
    sol3 = nlp3.solve(ca.DM(THETA3), x_warmstart=x0)
    print(
        f"    nominal re-solve: status={nlp3.solver.stats()['return_status']}, "
        f"|x - x0| = {np.linalg.norm(np.array(sol3.x).flatten() - x0):.1e}"
    )

    # finite-difference references
    prox3 = closest_point_solver(nlp3)
    J_prox3, J_warm3 = fd_jacobians(nlp3, prox3, THETA3, x0, DELTA)
    J_warm3 = cast(np.ndarray, J_warm3)
    print(f"    consistency: column m of the FD Jacobian vs 1-D reference: "
          f"{compute_error(J_prox3[:, 0], ref)[1]:.1e}")

    # rows: (method, c, err prox, cos prox, err warm, cos warm)
    cos_wp, err_wp = compute_error(J_warm3, J_prox3)
    th3_rows = [("warmFD", np.nan, err_wp, cos_wp, 0.0, 1.0)]

    for c in C_LIST:

        # reuse the nominal problem for c = 1, rebuild otherwise
        if c == 1.0:
            nlp3_c, sol3_c = nlp3, sol3
        else:
            nlp3_c, _ = build_dubins_nlp(N=N, integrator="euler", init_scale=c, goal_as_param=True)
            sol3_c = nlp3_c.solve(ca.DM(THETA3), x_warmstart=x0)

        # projected method
        projected_3 = np.array(nlp3_c.differentiate(
            ca.DM(THETA3), sol3_c.x, sol3_c.lam, sol3_c.nu,
            mode="projected",
            options={"rho_1":RHO, "rho_2":RHO, "tol":TOL}
        )["x"]).reshape(nlp3_c.n_x, nlp3_c.n_theta)

        # least squares
        least_squares_3 = np.array(nlp3_c.differentiate(
            ca.DM(THETA3), sol3_c.x, sol3_c.lam, sol3_c.nu,
            mode="active_set",
            options={"tol":TOL, "lstsq":True, "active_rule":"ratio"}
        )["x"]).reshape(nlp3_c.n_x, nlp3_c.n_theta)

        for meth, J in [("projected", projected_3), ("lstsq", least_squares_3)]:
            cos_p, err_p = compute_error(J, J_prox3)
            cos_w, err_w = compute_error(J, J_warm3)
            th3_rows.append((meth, c, err_p, cos_p, err_w, cos_w))

    print(f"    {'method':<10} {'c':>6} {'err prox':>9} {'cos prox':>12} {'err warm':>9} {'cos warm':>12}")
    for meth, c, ep, cp, ew, cw in th3_rows:
        print(f"    {meth:<10} {c:>6.0e} {ep:>9.2e} {cp:>12.8f} {ew:>9.2e} {cw:>12.8f}")

    # 3-parameter test
    with open(os.path.join(RESULTS_DIR, "dubins_theta3.dat"), "w") as fh:
        fh.write("method c err_prox cos_prox err_warm cos_warm\n")
        for meth, c, ep, cp, ew, cw in th3_rows:
            fh.write(f"{meth} {c:.1e} {ep:.6e} {cp:.10f} {ew:.6e} {cw:.10f}\n")
    print(f"    (saved to {RESULTS_DIR}/dubins_theta3.dat)")

    # ---------------- 9. computation times ----------------
    # Both methods start from the same nominal primal-dual solution, whose
    # computation is not timed. "ours" = one call to the projected-regularization
    # derivative (Hessian and Jacobian evaluation, subspace computation and linear
    # solve). "fd" = central proximal finite differences as used for the reference:
    # 2 * n_theta IPOPT solves, warm-started at the nominal solution. Each is
    # repeated N_TIMING times after one untimed warm-up.
    print(f"\n--- 8. Computation times ({N_TIMING} evaluations each, seconds)")

    time_rows = []
    for nlp_t, sol_t, th0, prox_t in [
        (nlp, sol, np.array([M0]), prox),
        (nlp3, sol3, THETA3, prox3),
    ]:
        # projected method
        t_ours = time_it(
            lambda: nlp_t.differentiate(
                ca.DM(th0), sol_t.x, sol_t.lam, sol_t.nu,
                mode="projected",
                options={"rho_1":RHO, "rho_2":RHO, "tol":TOL}
            ),
            N_TIMING
        )

        # proximal finite differences (no warm-started companion solves)
        t_fd = time_it(
            lambda: fd_jacobians(nlp_t, prox_t, th0, x0, DELTA, with_warm=False),
            N_TIMING
        )

        for meth, t in [("ours", t_ours), ("fd", t_fd)]:
            time_rows.append((meth, th0.size, t.mean(), t.var(ddof=1), t.std(ddof=1), np.median(t)))

    print(f"    {'method':<6} {'n_theta':>7} {'mean':>10} {'variance':>10} {'std':>10} {'median':>10}")
    for meth, nth, mu, var, sd, med in time_rows:
        print(f"    {meth:<6} {nth:>7d} {mu:>10.3e} {var:>10.3e} {sd:>10.3e} {med:>10.3e}")

    # tab:comp_times
    with open(os.path.join(RESULTS_DIR, "dubins_times.dat"), "w") as fh:
        fh.write("method n_theta mean var std median\n")
        for meth, nth, mu, var, sd, med in time_rows:
            fh.write(f"{meth} {nth:d} {mu:.6e} {var:.6e} {sd:.6e} {med:.6e}\n")
    print(f"    (saved to {RESULTS_DIR}/dubins_times.dat)")