# =====================================================================
# IMPORTS
# =====================================================================

import os
import numpy as np
import casadi as ca
from time import perf_counter
from typing import Union, Literal, Optional, Tuple, cast

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
M0 = 1.0                    # nominal mass
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
N_TIMING = 100              # number of evaluations to compute timings
RESULTS_DIR = ".results"    # location of dat files


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
    """
    return x + T * fc(x, u, m)

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

def unpack(z: ca.DM, L:dict[str, int]) -> Tuple[ca.DM, ca.DM]:
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
    x = z[:L["nx_tot"]].reshape((L["nx"], L["N"] + 1))
    u = z[L["nx_tot"]:].reshape((L["nu"], L["N"]))
    return x, u

def dx_dm(nlp, m, sol, mode, **opts) -> ca.DM:
    """
    Compute the sensitivity ``dx/dm`` of the primal solution.
 
    Evaluates the derivative of the optimal primal vector with respect to
    the scalar parameter `m` at a given primal-dual solution.
 
    Parameters
    ----------
    nlp : DifferentiableNLP
        Problem with a scalar parameter, as built by `build_dubins_nlp`
        with ``goal_as_param=False``.
    m : float
        Parameter value at which the solution was computed.
    sol : object
        Solution returned by ``nlp.solve``, with attributes ``x``
        (primal), ``lam`` (inequality multipliers) and ``nu`` (equality
        multipliers).
    mode : str
        Differentiation method, e.g. ``"active_set"``, ``"projected"`` or
        ``"regularized"``. See `DifferentiableNLP.differentiate`.
    **opts
        Method-specific options forwarded to ``nlp.differentiate``, e.g.
        ``rho_1``, ``rho_2``, ``tol``, ``lstsq``, ``active_rule``.
 
    Returns
    -------
    ca.DM
        Sensitivity ``dx/dm``, same length and ordering as the primal
        vector.
 
    Raises
    ------
    numpy.linalg.LinAlgError
        If the chosen method solves an unregularized KKT system and that
        system is singular (e.g. ``mode="active_set"`` on the degenerate
        Euler problem).
    """
    return nlp.differentiate(
        ca.DM([m]), sol.x, sol.lam, sol.nu,
        mode=mode, options=opts
    )["x"]

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

    # print flat firections
    for c in np.where(w > 0.5)[0]:
        top = np.argsort(-np.abs(V[:, c]))[:2]
        print("    flat direction along index " + ", ".join(f"{i} ({V[i, c]:+.2f})" for i in top))

    # see if nominal KKT system can be solved
    try:
        dx_dm(nlp, M0, sol, "active_set", tol=TOL, active_rule="ratio")
        print("    unregularized KKT system: solved")
    except np.linalg.LinAlgError:
        print("    unregularized KKT system: singular")

    