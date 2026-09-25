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
  4. compares them to finite differences and to the m = 1 reference.

Results are saved to mpc_overactuated_results.npz and a preview figure, and exported
as pgfplots tables to Data/*.dat (read by Figures/fig3.tex).
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
import time
import numpy as np
import casadi as ca
from typing import Any, Optional, Sequence, Tuple, cast

from dbdnlp import DifferentiableNLP


# =====================================================================
# CONSTANT PARAMETERS
# =====================================================================

# IPOPT options
SOLVER_OPTS = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "yes",
    "ipopt.tol": 1e-12,
    "ipopt.constr_viol_tol": 1e-12,
    "ipopt.dual_inf_tol": 1e-10,
    "ipopt.max_iter": 3000,
}

ACTIVE_TOL = 1e-7               # active-set tolerance
M = 2                           # number of redundant actuators
THETA = 1.0                     # nominal parameter
T = 200                         # closed-loop horizon
N = 20                          # MPC horizon
U_MAX = 0.95                    # total input bound (the paper uses 2, which never saturates from x0 = (3, 0))
DELTA = 1e-6                    # finite-difference step
RHO = 1e-4                      # fixed rho in the sweeps (also used for the fig3 curve)
X0 = np.array([3.0, 0.0])       # initial state of the closed loop
RHO_GRID = np.logspace(
    -8, 0, 9
)                               # rho sweep (1e-8, 1e-7, ..., 1e0)
OUT_NAME = "mpc_overactuated"   # file prefix of the preview figure and the .npz results
DAT_DIR = "Data"                # .dat files for pgfplots (copy to Figures/Data/ in the paper)


# helper to suppress casadi's type warnings
def SX_sym(*args) -> ca.SX:
    return ca.SX.sym(*args)


# =====================================================================
# NLP CREATION
# =====================================================================

def plant_function(m: int) -> Tuple[ca.Function, ca.Function]:
    """
    Build the discrete-time plant with `m` redundant actuators.

    The plant is

        x1+ = x1 + 0.4 x2
        x2+ = 0.56 x2 + 0.1 x1 x2 + 0.4 * sum_i u_i + theta * x1 * exp(-x1)

    so the actuators only enter the dynamics through their sum.

    Parameters
    ----------
    m : int
        Number of actuators, i.e. the input dimension.

    Returns
    -------
    f : ca.Function
        Plant map ``f(x, u, theta) -> x+``, with ``x`` of shape ``(2, 1)``,
        ``u`` of shape ``(m, 1)`` and scalar ``theta``.
    jac : ca.Function
        Plant Jacobians ``jac(x, u, theta) -> (df/dx, df/du, df/dtheta)``,
        of shapes ``(2, 2)``, ``(2, m)`` and ``(2, 1)``.
    """
    # state, input and parameter
    x = SX_sym("x", 2)
    u = SX_sym("u", m)
    th = SX_sym("th", 1)

    # next state
    xn = ca.vertcat(
        x[0] + 0.4 * x[1],
        0.56 * x[1] + 0.1 * x[0] * x[1] + 0.4 * ca.sum1(u) + th * x[0] * ca.exp(-x[0]),
    )

    # plant map and its Jacobians w.r.t. state, input and parameter
    f = ca.Function("f", [x, u, th], [xn])
    jac = ca.Function(
        "f_jac", [x, u, th],
        [ca.jacobian(xn, x), ca.jacobian(xn, u), ca.jacobian(xn, th)]
    )
    return f, jac

def build_mpc_nlp(
    m : int = 2,
    N : int = 20,
    u_max : float = 2.0,
    x2_max : float = 2.0,
    Q : Tuple[float, float] = (1e-2, 1.0),
    redundant : bool = True,
    solver_opts : Optional[dict] = None
) -> Tuple[DifferentiableNLP, dict[str, Any]]:
    """
    Build the over-actuated MPC problem as a `DifferentiableNLP`.

    Over a horizon of `N` stages, the MPC minimizes a quadratic state cost
    subject to the plant dynamics (see `plant_function`), the per-actuator
    input bounds, the (optional) redundant bound on the total input, and a
    bound on the second state. The initial state and the plant parameter
    are both NLP parameters, so that the derivative of the first input
    w.r.t. ``(x_init, theta)`` is available for the closed-loop sensitivity.

    Parameters
    ----------
    m : int, default 2
        Number of redundant actuators. ``m = 1`` gives the original MPC,
        which serves as the reference.
    N : int, default 20
        MPC horizon. The state trajectory has ``N + 1`` nodes and the input
        trajectory has ``N`` stages.
    u_max : float, default 2.0
        Bound on the total input. Each actuator is bounded by
        ``|u_i| <= u_max / m``.
    x2_max : float, default 2.0
        Bound on the second state, ``|x2_k| <= x2_max`` for ``k = 1, ..., N``.
        Node ``k = 0`` is fixed by the initial condition and is left out, to
        avoid a trivial loss of LICQ.
    Q : tuple of float, default (1e-2, 1.0)
        Diagonal of the state weight. The cost is
        ``sum_{k=0}^{N} x_k^T diag(Q) x_k``, with no input penalty.
    redundant : bool, default True
        If True and ``m > 1``, also impose the redundant bound
        ``|sum_i u_i| <= u_max`` at every stage. Whenever the total input
        saturates, this bound and all `m` individual bounds are active
        simultaneously.
    solver_opts : dict, optional
        IPOPT/CasADi options passed to the solver. Defaults to
        `SOLVER_OPTS`.

    Returns
    -------
    nlp : DifferentiableNLP
        The parametric NLP, ``min f(x) s.t. g(x, p) <= 0, h(x, p) = 0``,
        with parameter ``p = [x_init; theta]`` of length 3.
    layout : dict of str to Any
        Layout of the primal vector ``[vec(X); vec(U)]`` (column-major),
        with keys

        - ``"N"`` : MPC horizon,
        - ``"m"`` : number of actuators,
        - ``"nXtot"`` : number of state entries, ``2 * (N + 1)``,
        - ``"u_max"`` : total input bound.
    """

    # plant dynamics
    f, _ = plant_function(m)

    # full state and input trajectories
    X = SX_sym("X", 2, N + 1)
    U = SX_sym("U", m, N)

    # parameters p = [ initial state, plant parameter ]
    p = SX_sym("p", 3)
    x_init, th = p[0:2], p[2]

    # ---- equalities h(x, p) = 0 ----
    # start at the measured state
    h = [X[:, 0] - x_init]

    # loop through time-steps and enforce dynamics
    for k in range(N):
        h.append(X[:, k + 1] - cast(ca.SX, f(X[:, k], U[:, k], th)))

    # stack together
    h = cast(ca.SX, ca.vertcat(*h))

    # ---- inequalities g(x, p) <= 0 ----
    g = []
    for k in range(N):

        # individual actuator bounds
        for i in range(m):
            g.append(U[i, k] - u_max / m)
            g.append(-U[i, k] - u_max / m)

        # redundant bound on the total input
        if m > 1 and redundant:
            g.append(ca.sum1(U[:, k]) - u_max)
            g.append(-ca.sum1(U[:, k]) - u_max)

    # state bounds (k = 0 is fixed by x_init: omit to avoid trivial LICQ loss)
    for k in range(1, N + 1):
        g.append(X[1, k] - x2_max)
        g.append(-X[1, k] - x2_max)

    # stack together
    g = cast(ca.SX, ca.vertcat(*g))

    # ---- cost f(x) ----
    # quadratic state cost over all nodes (no input penalty)
    Qm = ca.diag(ca.DM(Q))
    cost = sum(ca.bilin(Qm, X[:, k], X[:, k]) for k in range(N + 1))

    # build NLP
    x = cast(ca.SX, ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)))
    nlp = DifferentiableNLP(
        x=x, theta=p,
        f=cost, g=g, h=h,
        solver_opts=dict(solver_opts or SOLVER_OPTS)
    )

    # build layout dictionary
    layout = dict(
        N = N, m = m,
        nXtot = 2 * (N + 1),
        u_max = u_max
    )

    return nlp, layout

def unpack(z: ca.DM | np.ndarray, L: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a flat primal vector into state and input trajectories.

    Inverse of the column-major flattening done in `build_mpc_nlp`.
    Works for the primal solution itself or for any vector with the same
    layout.

    Parameters
    ----------
    z : ca.DM or array_like
        Flat vector of length ``L["nXtot"] + m * N``, ordered as
        ``[vec(X); vec(U)]``.
    L : dict of str to Any
        Layout dictionary returned by `build_mpc_nlp`. Uses the keys
        ``"N"``, ``"m"`` and ``"nXtot"``.

    Returns
    -------
    X : numpy.ndarray
        State trajectory, shape ``(2, N + 1)``. Column ``k`` is node ``k``.
    U : numpy.ndarray
        Input trajectory, shape ``(m, N)``. Column ``k`` is stage ``k``.
    """
    # convert to a flat numpy vector (works for ca.DM and numpy input alike)
    z = np.asarray(z, float).flatten()

    # column-major reshape, matching ca.reshape in build_mpc_nlp
    X = z[:L["nXtot"]].reshape(2, L["N"] + 1, order="F")
    U = z[L["nXtot"]:].reshape(L["m"], L["N"], order="F")
    return X, U

def u0_rows(L: dict[str, Any]) -> np.ndarray:
    """
    Locate the first-stage inputs in the primal vector.

    Parameters
    ----------
    L : dict of str to Any
        Layout dictionary returned by `build_mpc_nlp`. Uses the keys
        ``"m"`` and ``"nXtot"``.

    Returns
    -------
    numpy.ndarray
        Indices of ``u_{0|t}`` (all `m` actuators) inside the primal
        vector, shape ``(m,)``.
    """
    # inputs come after the nXtot state entries, stage 0 first
    return L["nXtot"] + np.arange(L["m"])

def shift_warmstart(z: np.ndarray, L: dict[str, Any]) -> np.ndarray:
    """
    Shift a previous MPC solution by one stage, to warm-start the next one.

    The first node/stage is dropped and the last one is repeated, so the
    result has the same layout as `z`.

    Parameters
    ----------
    z : numpy.ndarray
        Previous primal solution, flat, ordered as ``[vec(X); vec(U)]``.
    L : dict of str to Any
        Layout dictionary returned by `build_mpc_nlp`.

    Returns
    -------
    numpy.ndarray
        Shifted primal vector, same shape and layout as `z`.
    """
    # previous trajectories
    X, U = unpack(z, L)

    # drop the first node/stage, repeat the last one
    Xs = np.hstack([X[:, 1:], X[:, -1:]])
    Us = np.hstack([U[:, 1:], U[:, -1:]])

    # flatten again (column-major, as in build_mpc_nlp)
    return np.concatenate([Xs.flatten(order="F"), Us.flatten(order="F")])


# =====================================================================
# CLOSED LOOP
# =====================================================================

def simulate(
    nlp: DifferentiableNLP,
    L: dict[str, Any],
    theta: float,
    x0: np.ndarray,
    T: int,
    warm: Optional[Sequence[Any]] = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Simulate the MPC closed loop.

    At each step ``t``, the MPC is solved at the current state, its first
    input ``u_{0|t}`` (all `m` actuators) is applied, and the plant is
    propagated to the next state.

    Parameters
    ----------
    nlp : DifferentiableNLP
        MPC problem, from `build_mpc_nlp`.
    L : dict of str to Any
        Layout dictionary returned by `build_mpc_nlp`.
    theta : float
        Plant parameter, used both in the MPC and in the simulated plant.
    x0 : numpy.ndarray
        Initial state of the closed loop, shape ``(2,)``.
    T : int
        Number of closed-loop steps.
    warm : sequence of MPC solutions, optional
        If given, step ``t`` is warm-started with ``warm[t].x`` (used for
        finite differences, as in the paper). Otherwise, step ``t`` is
        warm-started with the shifted previous solution (see
        `shift_warmstart`), and the first step with zeros.

    Returns
    -------
    xs : numpy.ndarray
        Closed-loop states, shape ``(T + 1, 2)``. Row ``t`` is ``x_t``.
    us : numpy.ndarray
        Applied inputs, shape ``(T, m)``. Row ``t`` is ``u_{0|t}``.
    sols : list
        MPC solutions returned by `nlp.solve`, one per step (length `T`).
    """
    # plant used to simulate the closed loop
    f, _ = plant_function(L["m"])

    # allocate trajectories
    xs = np.zeros((T + 1, 2))
    xs[0] = x0
    us = np.zeros((T, L["m"]))
    sols = []

    # warm start of the first step (shifted solutions afterwards)
    z_ws = np.zeros(nlp.n_x)

    for t in range(T):

        # solve the MPC at the current state
        ws = warm[t].x if warm is not None else z_ws
        sol = nlp.solve(np.r_[xs[t], theta], x_warmstart=ws)
        sols.append(sol)

        # apply the first input and propagate the plant
        z = np.array(sol.x).flatten()
        us[t] = z[u0_rows(L)]
        xs[t + 1] = np.array(f(xs[t], us[t], theta)).flatten()

        # shifted warm start for the next step
        z_ws = shift_warmstart(z, L)

    return xs, us, sols

def propagate(
    nlp: DifferentiableNLP,
    L: dict[str, Any],
    theta: float,
    xs: np.ndarray,
    us: np.ndarray,
    sols: Sequence[Any],
    mode: str,
    options: Optional[dict] = None
) -> np.ndarray:
    """
    Propagate the closed-loop sensitivity ``dx_t / dtheta`` along a closed loop.

    The initial state does not depend on `theta`, so ``dx_0 / dtheta = 0``,
    and the sensitivity is propagated with

        dx_{t+1} = f_x dx_t + f_th + f_u (dMPC/dx_t dx_t + dMPC/dtheta),

    where ``f_x``, ``f_u`` and ``f_th`` are the plant Jacobians at
    ``(x_t, u_t, theta)``, and ``dMPC/d(x_t, theta)`` is the derivative of
    the first input ``u_{0|t}`` computed by `nlp.differentiate`.

    Parameters
    ----------
    nlp : DifferentiableNLP
        MPC problem, from `build_mpc_nlp`.
    L : dict of str to Any
        Layout dictionary returned by `build_mpc_nlp`.
    theta : float
        Plant parameter.
    xs : numpy.ndarray
        Closed-loop states, shape ``(T + 1, 2)``, from `simulate`.
    us : numpy.ndarray
        Applied inputs, shape ``(T, m)``, from `simulate`.
    sols : sequence of MPC solutions
        MPC solutions along the closed loop, from `simulate`. Its length
        sets the number of steps ``T``.
    mode : str
        Differentiation mode passed to `nlp.differentiate`, e.g.
        ``"projected"``, ``"regularized"`` or ``"active_set"``.
    options : dict, optional
        Options passed to `nlp.differentiate` along with `mode`.

    Returns
    -------
    numpy.ndarray
        Closed-loop sensitivity, shape ``(T + 1, 2)``. Row ``t`` is
        ``dx_t / dtheta``.
    """
    # plant Jacobians
    _, fjac = plant_function(L["m"])

    # allocate sensitivity (row 0 stays zero)
    T = len(sols)
    dxs = np.zeros((T + 1, 2))

    # position of the first-stage inputs in the primal vector
    rows = u0_rows(L)

    for t, sol in enumerate(sols):

        # derivative of the MPC solution w.r.t. p = (x_init, theta)
        d = nlp.differentiate(
            sol.theta, sol.x, sol.lam, sol.nu,
            mode=mode,
            options=options
        )
        dU = np.array(ca.DM(d["x"]).full())[rows, :]          # (m, 3): d u0 / d(x_init, theta)

        # plant Jacobians at the current state and input
        fx, fu, fth = (np.array(J.full()) for J in fjac(xs[t], us[t], theta))

        # chain rule through the plant and the MPC
        dxs[t + 1] = (
            fx @ dxs[t] + fth.flatten()
            + fu @ (dU[:, :2] @ dxs[t] + dU[:, 2])
        )

    return dxs

def finite_differences(
    nlp: DifferentiableNLP,
    L: dict[str, Any],
    theta: float,
    x0: np.ndarray,
    T: int,
    sols_nom: Sequence[Any],
    delta: float
) -> np.ndarray:
    """
    Compute the closed-loop sensitivity ``dx_t / dtheta`` by central finite differences.

    The closed loop is simulated at ``theta + delta`` and ``theta - delta``,
    with every MPC instance warm-started at the corresponding nominal
    solution. This costs ``2 T`` MPC solves.

    Parameters
    ----------
    nlp : DifferentiableNLP
        MPC problem, from `build_mpc_nlp`.
    L : dict of str to Any
        Layout dictionary returned by `build_mpc_nlp`.
    theta : float
        Nominal plant parameter.
    x0 : numpy.ndarray
        Initial state of the closed loop, shape ``(2,)``.
    T : int
        Number of closed-loop steps.
    sols_nom : sequence of MPC solutions
        Nominal MPC solutions along the closed loop (from `simulate`), used
        as warm starts. Must have at least `T` entries.
    delta : float
        Finite-difference step.

    Returns
    -------
    numpy.ndarray
        Closed-loop sensitivity, shape ``(T + 1, 2)``. Row ``t`` is
        ``dx_t / dtheta``.
    """
    # perturbed closed loops, warm-started at the nominal MPC solutions
    xp, _, _ = simulate(nlp, L, theta + delta, x0, T, warm=sols_nom)
    xm, _, _ = simulate(nlp, L, theta - delta, x0, T, warm=sols_nom)

    # central difference
    return (xp - xm) / (2 * delta)


# =====================================================================
# DEGENERACY
# =====================================================================

def degeneracy_map(nlp: DifferentiableNLP, sols: Sequence[Any]) -> dict[str, np.ndarray]:
    """
    Compute the degeneracy structure of every MPC instance along a closed loop.

    Parameters
    ----------
    nlp : DifferentiableNLP
        MPC problem, from `build_mpc_nlp`.
    sols : sequence of MPC solutions
        MPC solutions along the closed loop, from `simulate`.

    Returns
    -------
    dict of str to numpy.ndarray
        Arrays indexed by time step (length ``len(sols)``), with keys

        - ``"dim_P"`` : dimension of the flat-direction subspace P,
        - ``"dim_U"`` : dimension of the redundant-active subspace U,
        - ``"n_weak"`` : number of weakly active constraints,
        - ``"n_active"`` : number of active constraints,
        - ``"min_eig"`` : smallest eigenvalue of the reduced Hessian,
        - ``"gap"`` : smallest non-flat eigenvalue of the reduced Hessian.
    """
    # one list per quantity, filled step by step
    out = {k: [] for k in ("dim_P", "dim_U", "n_weak", "n_active", "min_eig", "gap")}

    for sol in sols:

        # degeneracy information at this MPC solution
        s = nlp.degenerate_subspaces(sol.theta, sol.x, sol.lam, sol.nu, active_tol=ACTIVE_TOL)

        # store
        out["dim_P"].append(s["dim_P"])
        out["dim_U"].append(s["dim_U"])
        out["n_weak"].append(len(s["weak_idx"]))
        out["n_active"].append(len(s["active_idx"]))
        out["min_eig"].append(s["min_reduced_eig"])
        out["gap"].append(s["smallest_nonflat_eig"])

    return {k: np.array(v) for k, v in out.items()}


# =====================================================================
# ERROR COMPUTATION
# =====================================================================

def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the relative error of `a` against a reference `b`.

    Parameters
    ----------
    a : numpy.ndarray
        Quantity to evaluate, e.g. a closed-loop sensitivity.
    b : numpy.ndarray
        Reference with the same shape as `a`. Must be nonzero.

    Returns
    -------
    float
        Relative error ``|a - b| / |b|`` (Frobenius norm for 2-D arrays).
    """
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between `a` and `b`.

    Both inputs are flattened first, so for matrices this uses the
    Frobenius inner product.

    Parameters
    ----------
    a : numpy.ndarray
        First array.
    b : numpy.ndarray
        Second array, with the same number of entries as `a`.

    Returns
    -------
    float
        Cosine similarity ``<a, b> / (|a| |b|)``, in ``[-1, 1]``.
    """
    # flatten
    a, b = a.flatten(), b.flatten()

    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# =====================================================================
# TIMING
# =====================================================================

LAST_TIME = [0.0]               # mutable box holding the per-call timing from the last safe_propagate call

def safe_propagate(*args, **kwargs) -> Optional[np.ndarray]:
    """
    Call `propagate`, time it, and return None instead of raising on failure.

    Solver and linear-algebra failures (which happen for some rho values in
    the sweeps) are caught and printed. On success, the average time per
    MPC instance is stored in ``LAST_TIME[0]``. On failure, `LAST_TIME` is
    left unchanged.

    Parameters
    ----------
    *args, **kwargs
        Passed to `propagate` unchanged. The MPC solutions `sols` must be
        passed positionally (6th argument), since their number is used to
        normalize the timing.

    Returns
    -------
    numpy.ndarray or None
        Output of `propagate`, shape ``(T + 1, 2)``, or None if a
        ``numpy.linalg.LinAlgError`` or ``ValueError`` was raised.
    """
    try:
        t0 = time.perf_counter()
        out = propagate(*args, **kwargs)
        LAST_TIME[0] = (time.perf_counter() - t0) / len(args[5])   # per MPC instance
        return out
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"      failed: {type(e).__name__}: {e}")
        return None


# =====================================================================
# FULL COMPARISON
# =====================================================================

def run(
    m: int,
    theta: float,
    x0: np.ndarray,
    T: int,
    N: int,
    delta: float,
    rho_grid: np.ndarray,
    rho_fixed: float,
    u_max: float = 0.95,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run the full comparison for a given number of actuators.

    Simulates the ``m = 1`` and over-actuated closed loops, characterizes
    their degeneracy, computes finite-difference and reference
    sensitivities, then sweeps the "projected" and "regularized" (old)
    sensitivity methods over `rho_grid` and evaluates the unregularized
    least-squares and plain solves.

    Parameters
    ----------
    m : int
        Number of redundant actuators of the over-actuated MPC.
    theta : float
        Nominal plant parameter.
    x0 : numpy.ndarray
        Initial state of the closed loop, shape ``(2,)``.
    T : int
        Number of closed-loop steps.
    N : int
        MPC horizon.
    delta : float
        Finite-difference step.
    rho_grid : numpy.ndarray
        Regularization values of the sweeps. In the projected sweeps, one of
        ``rho_1``, ``rho_2`` takes these values while the other is fixed to
        `rho_fixed`. In the old sweep, ``rho_1 = rho_2`` take these values.
    rho_fixed : float
        Fixed regularization, used in the projected sweeps, for the
        ``m = 1`` reference and for the stored projected sensitivity.
    u_max : float, default 0.95
        Bound on the total input (see `build_mpc_nlp`).
    verbose : bool, default True
        If True, print the intermediate results.

    Returns
    -------
    dict of str to Any
        Results, consumed by `plot` and `export_dat`, with keys

        - ``"m"`` : number of actuators,
        - ``"xs"``, ``"us"`` : over-actuated closed loop, shapes ``(T + 1, 2)``
          and ``(T, m)``,
        - ``"xs_ref"`` : ``m = 1`` closed-loop states, shape ``(T + 1, 2)``,
        - ``"n_x"`` : primal dimension of the over-actuated MPC,
        - ``"deg"`` : degeneracy structure of the over-actuated MPC (see
          `degeneracy_map`),
        - ``"dfd"``, ``"dfd_ref"`` : finite-difference sensitivities for
          `m` and ``m = 1`` actuators, shape ``(T + 1, 2)``,
        - ``"dref"`` : ``m = 1`` reference sensitivity (projected method),
          shape ``(T + 1, 2)``,
        - ``"proj_rho1"``, ``"proj_rho2"``, ``"old"`` : sweep results, shape
          ``(len(rho_grid), 4)``, columns (err vs FD, err vs reference,
          cosine similarity vs FD, ms per MPC instance), NaN where the solve
          failed,
        - ``"lstsq"``, ``"plain"`` : the same four quantities for the
          unregularized least-squares and plain solves, as tuples,
        - ``"d_proj"`` : over-actuated sensitivity with the projected method
          at `rho_fixed`, shape ``(T + 1, 2)``.
    """
    res : dict = {"m": m}

    # ---------------- 1. reference (m = 1) and over-actuated closed loops ----------------
    t0 = time.time()

    # reference
    nlp1, L1 = build_mpc_nlp(m=1, N=N, u_max=u_max)
    xs1, us1, sols1 = simulate(nlp1, L1, theta, x0, T)

    # over-actuated
    nlpm, Lm = build_mpc_nlp(m=m, N=N, u_max=u_max)
    xsm, usm, solsm = simulate(nlpm, Lm, theta, x0, T)

    if verbose:
        print(
            f"\n=== m = {m}: n_x = {nlpm.n_x}, n_ineq = {nlpm.n_ineq}, n_eq = {nlpm.n_eq} "
            f"({time.time() - t0:.1f}s for both closed loops)"
        )

        # both closed loops should coincide
        print(f"  max |x_cl(m) - x_cl(1)|           = {np.max(np.abs(xsm - xs1)):.2e}")
        print(f"  max |sum u(m) - u(1)|             = {np.max(np.abs(usm.sum(1) - us1[:, 0])):.2e}")

        # steps at which the total input saturates
        sat = np.where(np.abs(np.abs(us1[:, 0]) - u_max) < 1e-6)[0]
        print(
            f"  saturated steps                   = {sat.size}"
            + (f" (t = {sat[0]}..{sat[-1]})" if sat.size else "")
        )

    res.update(xs=xsm, us=usm, xs_ref=xs1, n_x=nlpm.n_x)

    # ---------------- 2. degeneracy structure ----------------
    deg1 = degeneracy_map(nlp1, sols1)
    degm = degeneracy_map(nlpm, solsm)
    res["deg"] = degm

    if verbose:
        print(f"  degeneracy along the closed loop       m = 1        m = {m:d}")

        # number of degenerate steps
        for k, lbl in [
            ("dim_P", "steps with dim P > 0"),
            ("dim_U", "steps with dim U > 0"),
            ("n_weak", "steps with weakly active"),
        ]:
            print(f"    {lbl:<32} {np.sum(deg1[k] > 0):>6}/{T}   {np.sum(degm[k] > 0):>6}/{T}")

        # average subspace dimensions and eigenvalue stats
        print(
            f"    {'mean dim P / mean dim U':<32} {deg1['dim_P'].mean():5.1f}/{deg1['dim_U'].mean():<4.1f}"
            f"  {degm['dim_P'].mean():5.1f}/{degm['dim_U'].mean():<4.1f}"
        )
        print(f"    min eigenvalue of reduced Hessian  {deg1['min_eig'].min():+.1e}   {degm['min_eig'].min():+.1e}")
        print(f"    smallest non-flat eigenvalue       {deg1['gap'].min():.1e}    {degm['gap'].min():.1e}")

    # ---------------- 3. finite differences and reference sensitivity ----------------
    # finite differences for both problems
    dfd = finite_differences(nlpm, Lm, theta, x0, T, solsm, delta)
    dfd1 = finite_differences(nlp1, L1, theta, x0, T, sols1, delta)

    # Reference: the m = 1 problem with the projected method (exact where nondegenerate).
    dref = propagate(
        nlp1, L1, theta, xs1, us1, sols1,
        "projected",
        {"rho_1":rho_fixed, "rho_2":rho_fixed, "tol":ACTIVE_TOL}
    )
    res.update(dfd=dfd, dfd_ref=dfd1, dref=dref)

    if verbose:
        print(f"  FD(m) vs FD(1)          rel. err = {rel_err(dfd, dfd1):.2e}")
        print(f"  reference vs FD(1)      rel. err = {rel_err(dref, dfd1):.2e}")

    # ---------------- 4. sensitivities of the over-actuated MPC ----------------
    def report(name, d) -> tuple:
        """Print and return (err vs FD, err vs m=1 reference, cos-sim vs FD, ms/instance) for d, or NaNs if d is None."""
        if d is None:
            return (np.nan, np.nan, np.nan, np.nan)
        e = (rel_err(d, dfd), rel_err(d, dref), cos_sim(d, dfd), 1e3 * LAST_TIME[0])
        if verbose:
            print(f"    {name:<28} {e[0]:>11.3e} {e[1]:>11.3e} {e[2]:>9.5f} {e[3]:>9.2f}")
        return e

    if verbose:
        print(f"\n  {'method':<30} {'err vs FD':>11} {'err vs ref':>11} {'cos(FD)':>9} {'ms/inst':>9}")

    sweeps = {"proj_rho1": [], "proj_rho2": [], "old": []}

    # projected method, sweep rho_1 (rho_2 fixed)
    for r in rho_grid:
        d = safe_propagate(
            nlpm, Lm, theta, xsm, usm, solsm,
            "projected",
            {"rho_1":r, "rho_2":rho_fixed, "tol":ACTIVE_TOL}
        )
        sweeps["proj_rho1"].append(report(f"projected rho1={r:.0e}", d))

    # projected method, sweep rho_2 (rho_1 fixed)
    for r in rho_grid:
        d = safe_propagate(
            nlpm, Lm, theta, xsm, usm, solsm,
            "projected",
            {"rho_1":rho_fixed, "rho_2":r, "tol":ACTIVE_TOL}
        )
        sweeps["proj_rho2"].append(report(f"projected rho2={r:.0e}", d))

    # previous method (rho I)
    for r in rho_grid:
        d = safe_propagate(
            nlpm, Lm, theta, xsm, usm, solsm,
            "regularized",
            {"rho_1":r, "rho_2":r}
        )
        sweeps["old"].append(report(f"old rho*I  rho={r:.0e}", d))

    # least squares
    d_ls = safe_propagate(
        nlpm, Lm, theta, xsm, usm, solsm,
        "active_set",
        {"tol":ACTIVE_TOL, "lstsq":True}
    )
    res["lstsq"] = report("unregularized, lstsq", d_ls)

    # plain (unregularized) solve
    d_as = safe_propagate(
        nlpm, Lm, theta, xsm, usm, solsm,
        "active_set",
        {"tol":ACTIVE_TOL}
    )
    res["plain"] = report("unregularized, solve", d_as)

    res.update({k: np.array(v) for k, v in sweeps.items()})

    # projected sensitivity at the fixed rho (shown in fig3)
    res["d_proj"] = propagate(
        nlpm, Lm, theta, xsm, usm, solsm,
        "projected",
        {"rho_1":rho_fixed, "rho_2":rho_fixed, "tol":ACTIVE_TOL}
    )
    return res


# =====================================================================
# PLOTTING
# =====================================================================

def plot(res: dict[str, Any], rho_grid: np.ndarray, T: int, fname: str) -> None:
    """
    Save a 3-panel preview figure.

    The panels show

    1. the relative error against the ``m = 1`` reference of each
       sensitivity method, over `rho_grid`,
    2. the degeneracy structure (dim P, dim U, number of weakly active
       constraints) along the closed loop,
    3. the finite-difference and projected-method state sensitivities.

    Parameters
    ----------
    res : dict of str to Any
        Results returned by `run`.
    rho_grid : numpy.ndarray
        Regularization values used in `run`.
    T : int
        Number of closed-loop steps.
    fname : str
        Output file name (the format follows the extension).
    """
    # imported here, so that the Agg backend is selected before pyplot is loaded
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # ---- panel 1: error vs rho ----
    a = ax[0]
    # error against the m = 1 reference (exact); FD noise would floor all curves at ~1e-7
    a.loglog(rho_grid, res["proj_rho1"][:, 1], "o-", label=r"projected, sweep $\rho_1$")
    a.loglog(rho_grid, res["proj_rho2"][:, 1], "s--", label=r"projected, sweep $\rho_2$")
    a.loglog(rho_grid, res["old"][:, 1], "^-", label=r"previous ($\rho I$)")
    if np.isfinite(res["lstsq"][1]):
        a.axhline(res["lstsq"][1], color="k", ls=":", label="least squares")
    a.axhline(rel_err(res["dfd"], res["dref"]), color="gray", ls="-.", lw=1, label="finite differences")
    a.set_xlabel(r"$\rho$")
    a.set_ylabel("rel. error vs m = 1 reference")
    a.set_title(f"closed-loop sensitivity, m = {res['m']}")
    a.legend(fontsize=8)
    a.grid(True, which="both", alpha=.3)

    # ---- panel 2: degeneracy along the closed loop ----
    a = ax[1]
    t = np.arange(T)
    a.step(t, res["deg"]["dim_P"], where="post", label=r"dim $\mathcal{P}$")
    a.step(t, res["deg"]["dim_U"], where="post", label=r"dim $\mathcal{U}$")
    a.step(t, res["deg"]["n_weak"], where="post", label="# weakly active")
    a.set_xlabel("closed-loop step t")
    a.set_title("degeneracy of each MPC instance")
    a.legend(fontsize=8)

    # ---- panel 3: state sensitivities ----
    a = ax[2]
    tt = np.arange(T + 1)
    for i in range(2):
        a.plot(tt, res["dfd"][:, i], "-", color=f"C{i}", label=rf"FD $\partial x^{i+1}/\partial\theta$")
        a.plot(
            tt, res["d_proj"][:, i], "--", color="k" if i == 0 else "gray",
            label=rf"projected $\partial x^{i+1}/\partial\theta$"
        )
    a.set_xlabel("t")
    a.set_title("closed-loop state sensitivity")
    a.legend(fontsize=8)

    # save
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"\nfigure saved to {fname}")


# =====================================================================
# EXPORT FOR PGFPLOTS
# =====================================================================

def write_dat(fname: str, columns: dict[str, Any], comment: str = "") -> None:
    """
    Write a whitespace-separated table readable by pgfplots / pgfplotstable.

    The first non-comment line holds the column names, so the columns can
    be selected in LaTeX with ``\\addplot table[x=<name>, y=<name>] {file.dat};``.
    Lines starting with ``%`` are comments (ignored by pgfplots). NaNs are
    written as ``nan``, which pgfplots skips (``unbounded coords=discard``,
    the default).

    Parameters
    ----------
    fname : str
        Output file name.
    columns : dict of str to array_like
        Ordered mapping ``{name: 1-D array}``. All arrays must have the
        same length.
    comment : str, default ""
        Optional (multi-line) comment written above the column names, each
        line prefixed with ``%``.
    """
    # stack the columns in order
    names = list(columns)
    data = np.column_stack([np.asarray(columns[k], float).ravel() for k in names])

    # comment lines, then column names
    header = ""
    if comment:
        header += "\n".join("% " + line for line in comment.strip().splitlines()) + "\n"
    header += " ".join(names)

    # write
    np.savetxt(fname, data, fmt="%.10e", header=header, comments="")
    print(f"  wrote {fname}  ({data.shape[0]} rows x {data.shape[1]} cols)")

def export_dat(
    res: dict[str, Any],
    rho_grid: np.ndarray,
    T: int,
    rho_fixed: float,
    delta: float,
    out_dir: str
) -> None:
    """
    Export the results of `run` as .dat files for the pgfplots figures.

    Three files are written to `out_dir`:

    - ``fig3_sensitivity.dat`` : closed-loop state and state sensitivity
      (fig3.tex), columns t, x1, x2, dx1_fd, dx2_fd, dx1_proj, dx2_proj,
      dx1_ref, dx2_ref (``T + 1`` rows),
    - ``mpc_instances.dat`` : per-MPC-instance quantities, i.e. applied
      inputs and degeneracy structure (``T`` rows),
    - ``rho_sweep.dat`` : errors / cosine similarity of the sensitivity
      methods over `rho_grid` (one row per rho).

    Parameters
    ----------
    res : dict of str to Any
        Results returned by `run`.
    rho_grid : numpy.ndarray
        Regularization values used in `run`.
    T : int
        Number of closed-loop steps.
    rho_fixed : float
        Fixed regularization used in `run` (only written to the header).
    delta : float
        Finite-difference step used in `run` (only written to the header).
    out_dir : str
        Output directory, created if needed.
    """
    # ensure data directory exists
    os.makedirs(out_dir, exist_ok=True)

    # header shared by all files
    m = res["m"]
    info = (
        f"Over-actuated MPC example, m = {m}, T = {T}, "
        f"rho_1 = rho_2 = {rho_fixed:g}, FD step = {delta:g}\n"
        f"Generated by {os.path.basename(__file__)}"
    )

    # ---- fig3: closed-loop state sensitivity (FD vs our method vs m = 1 reference) ----
    cols = {
        "t": np.arange(T + 1),
        "x1": res["xs"][:, 0], "x2": res["xs"][:, 1],
        "dx1_fd": res["dfd"][:, 0], "dx2_fd": res["dfd"][:, 1],
        "dx1_proj": res["d_proj"][:, 0], "dx2_proj": res["d_proj"][:, 1],
        "dx1_ref": res["dref"][:, 0], "dx2_ref": res["dref"][:, 1],
    }
    write_dat(
        os.path.join(out_dir, "fig3_sensitivity.dat"), cols,
        info + "\nfd = finite differences (m actuators), proj = projected method, "
               "ref = m = 1 reference"
    )

    # ---- per MPC instance: inputs and degeneracy ----
    cols = {"t": np.arange(T), "u": res["us"].sum(axis=1)}
    cols.update({f"u{i + 1}": res["us"][:, i] for i in range(m)})
    cols.update({k: res["deg"][k] for k in ("dim_P", "dim_U", "n_weak", "n_active")})
    write_dat(
        os.path.join(out_dir, "mpc_instances.dat"), cols,
        info + "\nu = sum of the applied inputs u1..um"
    )

    # ---- rho sweep: columns <method>_err_fd, <method>_err_ref, <method>_cos ----
    cols = {"rho": rho_grid}
    for key in ("proj_rho1", "proj_rho2", "old"):
        s = res[key]   # (len(rho_grid), 4): err vs FD, err vs ref, cos vs FD, ms/instance
        cols.update({
            f"{key}_err_fd": s[:, 0], f"{key}_err_ref": s[:, 1],
            f"{key}_cos": s[:, 2], f"{key}_ms": s[:, 3],
        })

    # rho-independent baselines, repeated on every row so they can be drawn as horizontal lines
    ones = np.ones_like(rho_grid, dtype=float)
    cols.update({
        "lstsq_err_fd": res["lstsq"][0] * ones, "lstsq_err_ref": res["lstsq"][1] * ones,
        "lstsq_cos": res["lstsq"][2] * ones,
        "fd_err_ref": rel_err(res["dfd"], res["dref"]) * ones,
    })
    write_dat(
        os.path.join(out_dir, "rho_sweep.dat"), cols,
        info + "\nproj_rho1: sweep rho_1 (rho_2 fixed), proj_rho2: sweep rho_2 (rho_1 fixed), "
               "old: rho*I regularization\nnan = solve failed"
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    # ---------------- 1. full comparison ----------------
    res = run(M, THETA, X0, T, N, DELTA, RHO_GRID, RHO, U_MAX)

    # ---------------- 2. preview figure ----------------
    plot(res, RHO_GRID, T, OUT_NAME + ".png")

    # ---------------- 3. raw results ----------------
    # all arrays in res, plus the degeneracy structure and the rho grid
    save = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    save.update({f"deg_{k}": v for k, v in res["deg"].items()})
    save["rho_grid"] = RHO_GRID

    np.savez(OUT_NAME + "_results.npz", **save)
    print(f"results saved to {OUT_NAME}_results.npz")

    # ---------------- 4. pgfplots tables ----------------
    export_dat(res, RHO_GRID, T, RHO, DELTA, DAT_DIR)