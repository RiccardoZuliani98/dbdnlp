# Copyright (c) 2026 ETH Zurich, Riccardo Zuliani
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the project root.

import casadi as ca
import numpy as np
from typing import Dict, Optional, Tuple, cast, Any
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve, svd, qr, null_space
from time import perf_counter

from dbdnlp._impl.types import (
    NLPFunctions, NLPSolution, NLPSymbolic, Symbolic,
    CostFuncType, IneqFuncType, EqFuncType, CstFuncType,
    LagrFuncType, KKTFuncType, IFTFuncType, ArrayLike
)

class _SparseKernelFailure(RuntimeError):
    """Raised when the sparse null-space computation cannot certify its result."""


class DifferentiableNLP:
    """
    A class to solve and differentiate non-linear programs (NLPs) in CasADi.
    
    This solver manages a single NLP instance. Regularized versions of the problem 
    should be constructed externally via a factory function that returns a new 
    DifferentiableNLP instance.
    """

    def __init__(
        self,
        x: Symbolic,
        theta: Symbolic,
        f: Symbolic,
        g: Optional[Symbolic] = None,
        h: Optional[Symbolic] = None,
        solver_opts: Optional[Dict] = None
    ) -> None:
        """
        Constructs the NLP solver, its symbolic representation, and eagerly 
        compiles all numerical functions required for solving and differentiation.

        Parameters
        ----------
        x : Symbolic
            Decision variables vector of shape (n_x, 1).
        theta : Symbolic
            Parameters vector of shape (n_theta, 1).
        f : Symbolic
            Scalar objective function expression f(x, theta).
        g : Optional[Symbolic], optional
            Inequality constraint expressions g(x, theta) <= 0.
        h : Optional[Symbolic], optional
            Equality constraint expressions h(x, theta) == 0.
        solver_opts : Optional[Dict], optional
            Dictionary of options passed to `casadi.nlpsol`.
        """

        # identify symbolic type
        if isinstance(x, ca.SX):
            sym_type = ca.SX
        else:
            sym_type = ca.MX

        # 1. Normalize Constraints and Dimensions
        g = g if g is not None else sym_type(0, 1)
        h = h if h is not None else sym_type(0, 1)

        self.n_x     = cast(int, x.size1())
        self.n_theta = cast(int, theta.size1())
        self.n_ineq  = cast(int, g.size1())
        self.n_eq    = cast(int, h.size1())
        self.n_c     = self.n_ineq + self.n_eq

        # 2. Initialize Symbolic Multipliers
        lam = cast(Symbolic, sym_type.sym('lam', self.n_ineq))  #type:ignore
        nu  = cast(Symbolic, sym_type.sym('nu',  self.n_eq))    #type:ignore

        # 3. Build NLPSymbolic Dataclass
        cst = cast(Symbolic, ca.vertcat(g, h))
        dual_cst = cast(Symbolic, ca.vertcat(lam, nu))

        self.sym = NLPSymbolic(
            cost=f, ineq=g, eq=h, cst=cst, 
            primal=x, param=theta,
            dual_ineq=lam, dual_eq=nu, 
            dual_cst=dual_cst
        )

        # 4. Build Eagerly Compiled Functions
        cost_func = ca.Function('cost', [x, theta], [f])
        ineq_func = ca.Function('ineq', [x, theta], [g])
        eq_func   = ca.Function('eq',   [x, theta], [h])
        cst_func  = ca.Function('cst',  [x, theta], [cst])

        # Lagrangian: L = f + lam^T g + nu^T h
        lagr = f
        if self.n_ineq > 0: lagr += ca.dot(lam, g)
        if self.n_eq > 0:   lagr += ca.dot(nu, h)
        lagr = cast(Symbolic, lagr)
        lagr_func = ca.Function('lagr', [x, lam, nu, theta], [lagr])

        # KKT Conditions
        stationarity     = cast(Symbolic, ca.jacobian(lagr, x).T)
        inequality       = cast(Symbolic, ca.fmax(g, 0.0))
        equality         = h
        slackness        = cast(Symbolic, lam * g)
        dual_feasibility = cast(Symbolic, ca.fmin(lam, 0.0))
        
        kkt_func = ca.Function(
            'kkt', 
            [x, lam, nu, theta],
            [stationarity, inequality, equality, slackness, dual_feasibility]
        )

        # Implicit Function Theorem (IFT) Derivatives
        H_xx     = cast(Symbolic, ca.hessian(lagr, x)[0])
        H_xtheta = cast(Symbolic, ca.jacobian(stationarity, theta))
        
        # Safely handle empty constraints for Jacobians
        J_gx     = cast(Symbolic, ca.jacobian(g, x)) if self.n_ineq > 0 else sym_type(0, self.n_x)
        J_gtheta = cast(Symbolic, ca.jacobian(g, theta)) if self.n_ineq > 0 else sym_type(0, self.n_theta)
        J_hx     = cast(Symbolic, ca.jacobian(h, x)) if self.n_eq > 0 else sym_type(0, self.n_x)
        J_htheta = cast(Symbolic, ca.jacobian(h, theta)) if self.n_eq > 0 else sym_type(0, self.n_theta)

        ift_func = ca.Function(
            'ift', 
            [x, lam, nu, theta], 
            [H_xx, H_xtheta, J_gx, J_gtheta, J_hx, J_htheta]
        )

        # Define Constraint Bounds
        # Inequalities are g(x) <= 0, Equalities are h(x) == 0
        lbg = ca.DM(np.concatenate([-np.inf * np.ones(self.n_ineq), np.zeros(self.n_eq)]))
        ubg = ca.DM(np.concatenate([np.zeros(self.n_ineq), np.zeros(self.n_eq)]))

        self.funcs = NLPFunctions(
            cost = cast(CostFuncType, cost_func),
            ineq = cast(IneqFuncType, ineq_func),
            eq   = cast(EqFuncType, eq_func),
            cst  = cast(CstFuncType, cst_func),
            lagr = cast(LagrFuncType, lagr_func),
            kkt  = cast(KKTFuncType, kkt_func),
            ift  = cast(IFTFuncType, ift_func),
            lbg  = lbg, ubg = ubg
        )

        # 5. Initialize Solver Configuration
        self.opts = dict(solver_opts) if solver_opts else {}
        self.plugin = self.opts.pop("plugin", "ipopt")

        nlp_dict = {
            'x': self.sym.primal,
            'p': self.sym.param,
            'f': self.sym.cost,
            'g': self.sym.cst
        }
        
        self.solver = ca.nlpsol('solver', self.plugin, nlp_dict, self.opts)

        # null-space dimensions of the last `degenerate_subspaces` call, see `_null_spaces`
        self._null_space_hint: Optional[Tuple[int, int]] = None

    def solve(
        self,
        theta: ArrayLike,
        x_warmstart: Optional[ArrayLike] = None,
        lam_warmstart: Optional[ArrayLike] = None,
        nu_warmstart: Optional[ArrayLike] = None
    ) -> NLPSolution:
        """
        Solves the NLP for a given parameter vector theta and optional warmstart values.

        Parameters
        ----------
        theta : ArrayLike
            Numerical value for parameter theta of shape (n_theta, 1).
        x_warmstart : Optional[ArrayLike], optional
            Warmstart guess for primal decision variable x (n_x, 1).
        lam_warmstart : Optional[ArrayLike], optional
            Warmstart guess for inequality multipliers lambda (n_ineq, 1).
        nu_warmstart : Optional[ArrayLike], optional
            Warmstart guess for equality multipliers nu (n_eq, 1).

        Returns
        -------
        NLPSolution
            Object containing the numerical solution, evaluated parameters,
            dual multipliers, and exact KKT residual analysis.
        """

        theta_DM = ca.DM(theta)

        args: Dict[str, Any] = {
            'p': theta_DM,
            'lbg': self.funcs.lbg,
            'ubg': self.funcs.ubg
        }

        # Handle warmstarts
        if x_warmstart is not None:
            args['x0'] = x_warmstart

        if lam_warmstart is not None or nu_warmstart is not None:
            # Handle partial warmstarts by filling missing multiplier components with zeros
            lam_0 = lam_warmstart if lam_warmstart is not None else ca.DM(self.n_ineq,1)
            nu_0  = nu_warmstart  if nu_warmstart  is not None else ca.DM(self.n_eq,1)
            args['lam_g0'] = ca.vertcat(lam_0, nu_0)

        # Solve the compiled NLP
        res = self.solver(**args)

        # Extract solution components
        x_opt   = ca.DM(res['x'])
        lam_c   = ca.DM(res['lam_g'])
        lam_opt = ca.DM(lam_c[:self.n_ineq])
        nu_opt  = ca.DM(lam_c[self.n_ineq:])

        # Package the point and evaluate its KKT residuals
        return self.evaluate(theta_DM, x_opt, lam_opt, nu_opt)

    def evaluate(
        self,
        theta: ArrayLike,
        x: ArrayLike,
        lam: Optional[ArrayLike] = None,
        nu: Optional[ArrayLike] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> NLPSolution:
        """
        Packages an arbitrary primal-dual point as an NLPSolution, evaluating its
        KKT residuals. No solve is performed.

        Useful for scoring a warmstart, wrapping a point obtained elsewhere
        (an analytic solution, another solver), or re-checking a stored solution
        at a new theta. `solve` uses it to build its return value.

        Parameters
        ----------
        theta : ArrayLike
            Parameter vector (n_theta, 1).
        x : ArrayLike
            Primal point (n_x, 1).
        lam, nu : Optional[ArrayLike]
            Multipliers (n_ineq, 1) and (n_eq, 1). None means zeros.
        properties : Optional[Dict[str, Any]]
            Initial contents of `NLPSolution.properties`.

        Returns
        -------
        NLPSolution
        """
        theta_DM = ca.DM(theta)
        x_DM     = self._col(x, self.n_x)
        lam_DM   = self._col(lam, self.n_ineq)
        nu_DM    = self._col(nu, self.n_eq)

        stat, ineq, eq, slack, dual_feas = self.funcs.kkt(x_DM, lam_DM, nu_DM, theta_DM)

        return NLPSolution(
            x=x_DM,
            theta=theta_DM,
            lam=lam_DM,
            nu=nu_DM,
            kkt_residuals={
                'stationarity':     ca.DM(stat),
                'inequality':       ca.DM(ineq),
                'equality':         ca.DM(eq),
                'slackness':        ca.DM(slack),
                'dual_feasibility': ca.DM(dual_feas)
            },
            properties=dict(properties) if properties else {},
        )

    @staticmethod
    def _active_set(g_np: np.ndarray, lam_np: np.ndarray, tol: float, rule: str) -> np.ndarray:
        """Indices of active inequalities (g <= 0 convention), see `degenerate_subspaces`."""
        if g_np.size == 0:
            return np.array([], dtype=int)
        mask = g_np >= -tol
        if rule == "ratio":
            mask |= lam_np > -g_np
        elif rule != "threshold":
            raise ValueError(f"Unknown active_rule '{rule}'. Choose 'threshold' or 'ratio'.")
        return np.where(mask)[0]

    @staticmethod
    def _col(v: Optional[ArrayLike], n: int) -> ca.DM:
        """Length-n column DM; None or an empty input gives the (n, 1) zero vector."""
        if v is None:
            return ca.DM(n, 1)
        d = ca.DM(v)
        if d.numel() == 0:
            return ca.DM(n, 1)
        if d.numel() != n:
            raise ValueError(f"expected {n} entries, got {d.numel()}")
        return ca.reshape(d, n, 1)

    # ------------------------------------------------------------------
    # Fast conversion and evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sp(v: ArrayLike) -> sp.csc_matrix:
        """CasADi matrix -> scipy CSC matrix.

        `DM.sparse()` is two orders of magnitude faster than `DM.full()` for
        large matrices (for a 1206 x 1206 Hessian ~0.5 ms vs ~80 ms), and
        `.toarray()` on the result is cheap, so dense arrays are built this way too.
        """
        return ca.DM(v).sparse()

    @classmethod
    def _np(cls, v: ArrayLike) -> np.ndarray:
        """CasADi matrix -> dense 2-D numpy array (fast path, see `_sp`)."""
        return cls._sp(v).toarray()

    def _eval_point(
        self, theta: ArrayLike, x: ArrayLike, lam: ArrayLike, nu: ArrayLike
    ) -> Dict[str, Any]:
        """
        Evaluates the IFT function and g once and returns sparse matrices.

        Keys: H (n_x, n_x), H_xtheta, J_g, J_gtheta, J_h, J_htheta (all scipy
        CSC), g (n_ineq,) and lam (n_ineq,) as flat numpy arrays.
        """
        H, H_xth, J_g, J_gth, J_h, J_hth = self.funcs.ift(x, lam, nu, theta)
        g = self._np(self.funcs.ineq(x, theta)).ravel() if self.n_ineq > 0 else np.zeros(0)
        return {
            "H": self._sp(H),
            "H_xtheta": self._sp(H_xth),
            "J_g": self._sp(J_g),
            "J_gtheta": self._sp(J_gth),
            "J_h": self._sp(J_h),
            "J_htheta": self._sp(J_hth),
            "g": g,
            "lam": self._np(self._col(lam, self.n_ineq)).ravel(),
        }

    # ------------------------------------------------------------------
    # Null spaces of J (right and left) for `degenerate_subspaces`
    # ------------------------------------------------------------------

    # below this size of [0 J^T; J 0] the dense SVD is fast enough and is used
    _SPARSE_MIN_SIZE = 300

    @staticmethod
    def _null_spaces_dense(J: Any, rcond: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orthonormal bases of null(J) (n, d_Z) and null(J^T) (m, d_U) from a
        single full SVD, with the same rank rule as `scipy.linalg.null_space`
        (singular values <= rcond * s_max count as zero).
        """
        Jd = J.toarray() if sp.issparse(J) else np.asarray(J)
        U, s, Vh = svd(Jd, full_matrices=True)
        tol = float(np.max(s, initial=0.0)) * rcond
        r = int(np.sum(s > tol))
        return Vh[r:, :].T.copy(), U[:, r:].copy()

    @staticmethod
    def _norm2_estimate(
        J: sp.spmatrix, rng: np.random.Generator, max_iter: int = 100, rtol: float = 1e-3
    ) -> float:
        """
        Largest singular value of J, by power iteration on J^T J.

        It only scales the zero threshold of the sparse null-space method, so a
        relative accuracy of `rtol` is plenty. This is ~10x cheaper than `svds`,
        whose ARPACK driver needs hundreds of matrix-vector products.
        """
        v = rng.standard_normal(J.shape[1])
        s2, s2_old = 0.0, 0.0
        for _ in range(max_iter):
            v /= np.linalg.norm(v)
            v = J.T @ (J @ v)
            s2 = float(np.linalg.norm(v))
            if s2 == 0.0 or abs(s2 - s2_old) <= rtol * s2:
                break
            s2_old = s2
        return float(np.sqrt(s2))

    @staticmethod
    def _ritz_singular_values(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """All b singular values (b,) of the m x b matrix B, also when m < b, and the
        right singular vectors (b, b)."""
        if B.shape[0] < B.shape[1]:
            B = np.vstack([B, np.zeros((B.shape[1] - B.shape[0], B.shape[1]))])
        _, s, Vh = np.linalg.svd(B, full_matrices=False)
        return s, Vh.T

    @classmethod
    def _null_spaces_sparse(
        cls,
        J: sp.spmatrix,
        rcond: float,
        hint: Optional[Tuple[int, int]] = None,
        seed: int = 0,
        oversample: int = 8,
        max_iter: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same output and rank rule as `_null_spaces_dense`, exploiting sparsity.

        Uses one sparse LU of the quasi-definite matrix A = [delta I, J^T; J, -delta I].
        Since A^2 = blkdiag(J^T J + delta^2 I, J J^T + delta^2 I), a solve with a block
        [Q_Z; 0] returns delta (J^T J + delta^2 I)^{-1} Q_Z in its first block, and a
        solve with [0; Q_U] returns -delta (J J^T + delta^2 I)^{-1} Q_U in its second
        block. Each solve is thus a step of block inverse iteration on J^T J and on
        J J^T (without forming them), which damps the component along a singular
        value s by (delta / s)^2 relative to the null-space components. After
        orthonormalizing the block Q, the singular values of J Q (resp. J^T Q) are
        exactly |J z| for the Ritz vectors z, so those <= rcond * s_max are null
        vectors under the same rule as the dense SVD.

        Each block starts at the null-space dimension of `hint` (the result of the
        previous call) or at the one forced by rank-nullity, |n - m|, plus
        `oversample`. A block whose Ritz values are all near zero may be too small:
        its null vectors are kept and a block twice as large, orthogonal to them,
        is tried next (deflation).

        Raises `_SparseKernelFailure` when the result cannot be certified; the
        caller then falls back to the dense SVD.
        """
        m, n = J.shape
        rng = np.random.default_rng(seed)
        s_max = cls._norm2_estimate(J, rng)
        if s_max == 0.0:
            return np.eye(n), np.eye(m)

        tau = rcond * s_max          # zero threshold, as in scipy.linalg.null_space
        band = 1e3 * tau             # Ritz values in (tau_acc, band] may still be converging,
        tau_acc = min(tau, 1e2 * np.finfo(float).eps * s_max)   # those below are at rounding level
        delta = 1e-10 * s_max        # shift: far below the nonzero singular values, above rounding

        A = sp.bmat([[delta * sp.eye(n), J.T], [J, -delta * sp.eye(m)]], format="csc")
        lu = spla.splu(cast(sp.csc_matrix, A))

        # side 0 is null(J) in R^n, side 1 is null(J^T) in R^m
        dims = (n, m)
        rows = (slice(0, n), slice(n, n + m))
        ops = (J, J.T.tocsr())
        found = [np.zeros((n, 0)), np.zeros((m, 0))]
        start = hint if hint is not None else (max(n - m, 0), max(m - n, 0))
        b = [start[0] + oversample, start[1] + oversample]
        active = [True, True]

        def deflate(Y: np.ndarray, F: np.ndarray) -> np.ndarray:
            return Y - F @ (F.T @ Y) if F.shape[1] else Y

        while active[0] or active[1]:
            sides = [i for i in (0, 1) if active[i]]
            for i in sides:
                if found[i].shape[1] > dims[i] // 2:
                    raise _SparseKernelFailure("null space too large for the sparse method")
                b[i] = min(b[i], dims[i] - found[i].shape[1])

            # both sides share each solve: right-hand side blkdiag(Q_Z, Q_U)
            Q = {i: deflate(rng.standard_normal((dims[i], b[i])), found[i]) for i in sides}
            cols = np.cumsum([0] + [b[i] for i in sides])
            ritz = {}
            for _ in range(max_iter):
                R = np.zeros((n + m, cols[-1]))
                for j, i in enumerate(sides):
                    R[rows[i], cols[j]:cols[j + 1]] = Q[i]
                X = lu.solve(R)
                for j, i in enumerate(sides):
                    Q[i] = qr(deflate(X[rows[i], cols[j]:cols[j + 1]], found[i]), mode="economic")[0]
                    ritz[i] = cls._ritz_singular_values(ops[i] @ Q[i])
                if not any(np.any((s > tau_acc) & (s <= band)) for s, _ in ritz.values()):
                    break

            for i in sides:
                s, V = ritz[i]
                found[i] = np.hstack([found[i], Q[i] @ V[:, s <= tau]])
                if np.sum(s <= band) < b[i]:
                    active[i] = False   # a Ritz value beyond the band: the block was large enough
                else:
                    b[i] *= 2

        Z, N_U = found
        if Z.shape[1] - N_U.shape[1] != n - m:   # rank-nullity check
            raise _SparseKernelFailure("inconsistent null-space dimensions")
        return Z, N_U

    def _null_spaces(
        self, J: sp.spmatrix, rank_tol: Optional[float], method: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Dispatches to the sparse or dense null-space routine, see `degenerate_subspaces`."""
        m, n = J.shape
        rcond = rank_tol if rank_tol else np.finfo(float).eps * max(m, n)
        if method not in ("auto", "sparse", "dense"):
            raise ValueError(f"Unknown null-space method '{method}'. Choose 'auto', 'sparse' or 'dense'.")
        used = "dense"
        if method == "sparse" or (method == "auto" and m + n >= self._SPARSE_MIN_SIZE):
            try:
                Z, N_U = self._null_spaces_sparse(J, rcond, hint=self._null_space_hint)
                used = "sparse"
            except (_SparseKernelFailure, RuntimeError, ValueError):
                pass  # fall back to the dense SVD
        if used == "dense":
            Z, N_U = self._null_spaces_dense(J, rcond)
        # repeated calls (e.g. timing, or a sweep over theta) have similar null
        # spaces: their dimensions size the first block of the next sparse call
        self._null_space_hint = (Z.shape[1], N_U.shape[1])
        return Z, N_U, used

    def _subspace_bases(
        self,
        ev: Dict[str, Any],
        active_tol: float,
        rank_tol: Optional[float],
        flat_tol: float,
        weak_as_active: bool,
        active_rule: str,
        method: str,
    ) -> Dict[str, Any]:
        """Core of `degenerate_subspaces`, working on the output of `_eval_point`.

        Returns orthonormal bases B_P (n_x, dim_P) and N_U (n_m, dim_U) instead of
        the dense projectors.
        """
        t0 = perf_counter()
        H = 0.5 * (ev["H"] + ev["H"].T)
        g_np, lam_np = ev["g"], ev["lam"]

        active = self._active_set(g_np, lam_np, active_tol, active_rule)
        strong = active[lam_np[active] > active_tol] if active.size else active
        weak = np.setdiff1d(active, strong)
        sys_idx = active if weak_as_active else strong

        J_sys = cast(sp.csc_matrix, sp.vstack([ev["J_g"][sys_idx, :], ev["J_h"]], format="csc"))
        n_m = J_sys.shape[0]
        t1 = perf_counter()

        # --- U and null(J_sys) ---------------------------------------------
        if n_m > 0:
            Z, N_U, used = self._null_spaces(J_sys, rank_tol, method)
        else:
            Z, N_U, used = np.eye(self.n_x), np.zeros((0, 0)), "none"
        t2 = perf_counter()

        # --- P: flat directions of H on null(J_sys) --------------------------
        min_eig, gap = np.inf, np.inf
        B_P = np.zeros((self.n_x, 0))
        if Z.shape[1] > 0:
            H_red = Z.T @ (H @ Z)
            H_red = 0.5 * (H_red + H_red.T)
            w, W = np.linalg.eigh(H_red)
            scale = max(1.0, float(np.max(np.abs(w))))
            flat = np.abs(w) <= flat_tol * scale
            min_eig = float(w[0])
            if np.any(~flat):
                gap = float(np.min(np.abs(w[~flat])))
            if np.any(flat):
                B_P = Z @ W[:, flat]
        t3 = perf_counter()

        return {
            "B_P": B_P,
            "N_U": N_U if N_U.size else np.zeros((n_m, 0)),
            "dim_P": int(B_P.shape[1]),
            "dim_U": int(N_U.shape[1]) if N_U.size else 0,
            "sys_idx": sys_idx,
            "active_idx": active,
            "strong_idx": strong,
            "weak_idx": weak,
            "min_reduced_eig": min_eig,
            "smallest_nonflat_eig": gap,
            "null_space_method": used,
            "timing": {
                "active_set": t1 - t0,
                "null_spaces": t2 - t1,
                "reduced_hessian": t3 - t2,
            },
        }

    def differentiate(
        self,
        theta: ArrayLike,
        x: ArrayLike,
        lam: ArrayLike,
        nu: ArrayLike,
        mode: str = "full",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ca.DM]:
        """
        Computes the parametric sensitivities (dx/dtheta, dlam/dtheta, dnu/dtheta)
        by evaluating the Implicit Function Theorem (IFT) on the KKT system.

        Parameters
        ----------
        theta : ArrayLike
            Numerical value for parameter theta (n_theta, 1).
        x : ArrayLike
            Primal decision variable vector at solution (n_x, 1).
        lam : ArrayLike
            Inequality dual multipliers at solution (n_ineq, 1).
        nu : ArrayLike
            Equality dual multipliers at solution (n_eq, 1).
        mode : str, default="full"
            Sensitivity computation mode:
            - "full": Differentiates the full KKT system directly (non-symmetric).
            - "active_set": Identifies active inequalities and solves a reduced, 
                            symmetric KKT system. options['lstsq']=True returns
                            the minimum-norm least-squares solution instead.
                            options['prox']=eps adds eps * I to the Hessian, i.e.
                            differentiates the problem with the proximal term
                            (eps/2) ||x - x_bar||^2 added to the cost.
            - "regularized": Previous regularization (rho_1 * I, rho_2 * I).
            - "projected": Current regularization (rho_1 * Pi_P, rho_2 * Pi_U),
                           see `degenerate_subspaces`. Options: rho_1, rho_2,
                           active_tol, rank_tol, flat_tol, weak_as_active,
                           null_space_method ('auto', 'sparse' or 'dense').
                           The system is solved with a sparse LU, using the
                           low-rank structure of Pi_P and Pi_U.
        options : Optional[Dict[str, Any]], optional
            Dictionary of options (e.g., {'tol': 1e-5} for active set thresholding).

        Returns
        -------
        Dict[str, ca.DM]
            Sensitivity matrices containing:
            - 'x'   : dx/dtheta matrix of shape (n_x, n_theta)
            - 'lam' : dlam/dtheta matrix of shape (n_ineq, n_theta)
            - 'nu'  : dnu/dtheta matrix of shape (n_eq, n_theta)
        """
        opts = options or {}
        tol: float = opts.get("tol", 1e-5)

        # 1. Evaluate IFT Jacobians & Hessians once, as sparse matrices
        #    (DM.full() is very slow on large matrices, DM.sparse() is not)
        ev = self._eval_point(theta, x, lam, nu)
        g_np, lam_np = ev["g"], ev["lam"]

        n_x, n_theta = self.n_x, self.n_theta
        n_ineq, n_eq = self.n_ineq, self.n_eq

        # 2. Dense copies for the dense modes (the projected mode stays sparse)
        if mode in ("full", "active_set", "regularized"):
            H_xx_np     = ev["H"].toarray()
            H_xtheta_np = ev["H_xtheta"].toarray()
            J_gx_np     = ev["J_g"].toarray()
            J_gtheta_np = ev["J_gtheta"].toarray()
            J_hx_np     = ev["J_h"].toarray()
            J_htheta_np = ev["J_htheta"].toarray()

        # 3. Assemble and Solve Linear System M * d = -N_rhs
        if mode == "full":
            N_full = n_x + n_ineq + n_eq
            M = np.zeros((N_full, N_full))
            N_rhs = np.zeros((N_full, n_theta))

            # Block 1: Stationarity
            M[:n_x, :n_x] = H_xx_np
            N_rhs[:n_x, :] = H_xtheta_np
            if n_ineq > 0:
                M[:n_x, n_x:n_x + n_ineq] = J_gx_np.T
            if n_eq > 0:
                M[:n_x, n_x + n_ineq:] = J_hx_np.T

            # Block 2: Complementary Slackness (lam * g = 0)
            if n_ineq > 0:
                M[n_x:n_x + n_ineq, :n_x] = np.diag(lam_np) @ J_gx_np
                M[n_x:n_x + n_ineq, n_x:n_x + n_ineq] = np.diag(g_np)
                N_rhs[n_x:n_x + n_ineq, :] = np.diag(lam_np) @ J_gtheta_np

            # Block 3: Equality Constraints
            if n_eq > 0:
                M[n_x + n_ineq:, :n_x] = J_hx_np
                N_rhs[n_x + n_ineq:, :] = J_htheta_np

            # Solve linear system
            # TODO: switch to scipy sparse?
            d_full = np.linalg.solve(M, -N_rhs)

            dx = d_full[:n_x, :]
            dlam = d_full[n_x:n_x + n_ineq, :] if n_ineq > 0 else np.zeros((0, n_theta))
            dnu = d_full[n_x + n_ineq:, :] if n_eq > 0 else np.zeros((0, n_theta))

        elif mode == "active_set":
            # Identify active inequalities (g <= 0, so g >= -tol means active)
            active_idx = self._active_set(g_np, lam_np, tol, opts.get("active_rule", "threshold"))
            n_act = len(active_idx)

            # Assemble reduced symmetric KKT system
            N_act = n_x + n_act + n_eq
            M = np.zeros((N_act, N_act))
            N_rhs = np.zeros((N_act, n_theta))

            # Block 1: Stationarity (with optional proximal term eps * I)
            M[:n_x, :n_x] = H_xx_np + opts.get("prox", 0.0) * np.eye(n_x)
            N_rhs[:n_x, :] = H_xtheta_np

            # Block 2: Active Inequalities
            if n_act > 0:
                J_gx_act = J_gx_np[active_idx, :]
                M[:n_x, n_x:n_x + n_act] = J_gx_act.T
                M[n_x:n_x + n_act, :n_x] = J_gx_act
                N_rhs[n_x:n_x + n_act, :] = J_gtheta_np[active_idx, :]

            # Block 3: Equality Constraints
            if n_eq > 0:
                M[:n_x, n_x + n_act:] = J_hx_np.T
                M[n_x + n_act:, :n_x] = J_hx_np
                N_rhs[n_x + n_act:, :] = J_htheta_np

            # Solve symmetric linear system. With options['lstsq']=True the
            # minimum-norm least-squares solution is returned instead, which is
            # the naive baseline when the KKT matrix is singular.
            #TODO: will this be fast, or should I convert M to a sparse scipy type?
            if opts.get("lstsq", False):
                d_act = np.linalg.lstsq(M, -N_rhs, rcond=opts.get("rcond", None))[0]
            else:
                d_act = solve(M, -N_rhs, assume_a='sym')

            dx = d_act[:n_x, :]
            dlam = np.zeros((n_ineq, n_theta))
            if n_act > 0:
                dlam[active_idx, :] = d_act[n_x:n_x + n_act, :]
            dnu = d_act[n_x + n_act:, :] if n_eq > 0 else np.zeros((0, n_theta))

        elif mode == "regularized":

            # get regularization parameters
            rho_1 = opts.get("rho_1")
            rho_2 = opts.get("rho_2")

            if rho_1 is None or rho_2 is None:
                raise ValueError("Options 'rho_1' and 'rho_2' must be provided for 'regularized' mode.")

            N_full = n_x + n_ineq + n_eq
            M = np.zeros((N_full, N_full))
            N_rhs = np.zeros((N_full, n_theta))

            # Block 1: Regularized Stationarity
            M[:n_x, :n_x] = H_xx_np + rho_1 * np.eye(n_x)
            N_rhs[:n_x, :] = H_xtheta_np
            if n_ineq > 0: 
                M[:n_x, n_x:n_x + n_ineq] = J_gx_np.T
            if n_eq > 0: 
                M[:n_x, n_x + n_ineq:] = J_hx_np.T
            
            # Block 2: Regularized Inequality Constraints
            if n_ineq > 0:
                M[n_x:n_x + n_ineq, :n_x] = J_gx_np
                
                # Compute denominator (gamma - rho * z) with safety clamping
                eps = 1e-10
                denom = lam_np - rho_1 * g_np
                sgn = np.where(denom < 0.0, -1.0, 1.0)
                denom = sgn * np.maximum(np.abs(denom), eps)
                
                # Compute diagonal modifier term
                diag_val = g_np / denom - rho_2
                M[n_x:n_x + n_ineq, n_x:n_x + n_ineq] = np.diag(diag_val)
                
                # Compute RHS multiplier
                rhs_mult = lam_np / denom
                N_rhs[n_x:n_x + n_ineq, :] = np.diag(rhs_mult) @ J_gtheta_np
                
            # Block 3: Regularized Equality Constraints
            if n_eq > 0:
                M[n_x + n_ineq:, :n_x] = J_hx_np
                M[n_x + n_ineq:, n_x + n_ineq:] = -rho_2 * np.eye(n_eq)
                N_rhs[n_x + n_ineq:, :] = J_htheta_np
                
            # Solve symmetric linear system using scipy
            #TODO: will this be fast, or should I convert M to a sparse scipy type?
            d_full = solve(M, -N_rhs, assume_a='sym')
            
            dx = d_full[:n_x, :]
            dlam = d_full[n_x:n_x + n_ineq, :] if n_ineq > 0 else np.zeros((0, n_theta))
            dnu = d_full[n_x + n_ineq:, :] if n_eq > 0 else np.zeros((0, n_theta))

        elif mode == "projected":
            # Projected regularization (Prop. 4.1):
            #   [ H + rho_1 Pi_P      J_sys^T       ] [dx  ]     [H_xtheta    ]
            #   [ J_sys          -rho_2 Pi_U        ] [dpsi] = - [J_sys,theta ]
            # J_sys stacks the active inequalities and the equalities. Pi_P and
            # Pi_U vanish when the problem is nondegenerate, in which case this
            # is exactly the "active_set" system, independently of rho.
            rho_1 = opts.get("rho_1")
            rho_2 = opts.get("rho_2")
            if rho_1 is None or rho_2 is None:
                raise ValueError("Options 'rho_1' and 'rho_2' must be provided for 'projected' mode.")

            sub = self._subspace_bases(
                ev,
                active_tol=opts.get("active_tol", tol),
                rank_tol=opts.get("rank_tol", None),
                flat_tol=opts.get("flat_tol", 1e-9),
                weak_as_active=opts.get("weak_as_active", True),
                active_rule=opts.get("active_rule", "ratio"),
                method=opts.get("null_space_method", "auto"),
            )

            sys_idx = sub["sys_idx"]
            n_sys = len(sys_idx)

            J_sys = sp.vstack([ev["J_g"][sys_idx, :], ev["J_h"]], format="csc")
            J_sys_theta = sp.vstack([ev["J_gtheta"][sys_idx, :], ev["J_htheta"]]).toarray()
            n_m = J_sys.shape[0]

            # Pi_P = B_P B_P^T and Pi_U = N_U N_U^T are dense, so M is not formed.
            # With K_P = [H + rho_1 Pi_P, J_sys^T; J_sys, 0]:
            #  - (0, N_U) is an invariant subspace of M with eigenvalue -rho_2
            #    (J_sys^T N_U = 0), so the U-part of the solution is explicit,
            #    dpsi_U = -Pi_U rhs_psi / rho_2, and on the orthogonal complement M
            #    coincides with K_P.
            #  - The kernel of K_P is (0, N_U). Diagonal entries -gamma on rows S of
            #    the multiplier block, with N_U[S] nonsingular, make K_P nonsingular
            #    without changing the solution for a right-hand side orthogonal to
            #    (0, N_U), up to a kernel component, which is projected out.
            #  - rho_1 Pi_P enters as a border of dim(P) dense columns,
            #      [ A      W_P ]          W_P = [sqrt(rho_1) B_P; 0],
            #      [ W_P^T  -I  ],         eliminating the border adds W_P W_P^T to A.
            # This keeps the LU sparse: only the few columns of B_P are dense,
            # while bordering with N_U as well causes a lot of fill-in.
            B_P, N_U = sub["B_P"], sub["N_U"]
            n_p, n_u = B_P.shape[1], N_U.shape[1]
            if (n_p > 0 and rho_1 == 0) or (n_u > 0 and rho_2 == 0):
                raise np.linalg.LinAlgError("projected KKT matrix is singular: rho = 0 on a degenerate subspace")

            rhs = np.vstack([-ev["H_xtheta"].toarray(), -J_sys_theta])
            rhs_U = N_U @ (N_U.T @ rhs[n_x:]) if n_u > 0 else np.zeros((n_m, n_theta))
            rhs[n_x:] -= rhs_U

            if n_u > 0:
                S = qr(N_U.T, mode="r", pivoting=True)[1][:n_u]
                gamma = float(abs(J_sys).max()) or 1.0   # any nonzero value works, this one matches J
                D_S = sp.csc_matrix((-gamma * np.ones(n_u), (S, S)), shape=(n_m, n_m))
            else:
                D_S = sp.csc_matrix((n_m, n_m))
            A = sp.bmat([[ev["H"], J_sys.T], [J_sys, D_S]], format="csc")
            if n_p > 0:
                W_P = sp.csc_matrix(np.vstack([np.sqrt(rho_1) * B_P, np.zeros((n_m, n_p))]))
                A = sp.bmat([[A, W_P], [W_P.T, -sp.eye(n_p)]], format="csc")
                rhs = np.vstack([rhs, np.zeros((n_p, n_theta))])

            try:
                d_p = spla.splu(cast(sp.csc_matrix, A)).solve(rhs)[:n_x + n_m]
            except RuntimeError as err:  # SuperLU: "Factor is exactly singular"
                raise np.linalg.LinAlgError(str(err)) from err

            if n_u > 0:
                d_p[n_x:] -= N_U @ (N_U.T @ d_p[n_x:]) + rhs_U / rho_2

            dx = d_p[:n_x, :]
            dlam = np.zeros((n_ineq, n_theta))
            if n_sys > 0:
                dlam[sys_idx, :] = d_p[n_x:n_x + n_sys, :]
            dnu = d_p[n_x + n_sys:, :] if n_eq > 0 else np.zeros((0, n_theta))

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'full', 'active_set', 'regularized', or 'projected'."
            )

        return {
            "x": ca.DM(dx),
            "lam": ca.DM(dlam),
            "nu": ca.DM(dnu)
        }

    def check_properties(
        self,
        theta: "ArrayLike",
        x: "ArrayLike",
        lam: "ArrayLike",
        nu: "ArrayLike",
        active_tol: float = 1e-6,
        rank_tol: Optional[float] = None,
        pd_tol: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        Evaluates LICQ, SCS, SOSC and SSOSC at the primal-dual point (x, lam, nu).

        Works unmodified for both the base NLP and the augmented problem returned by
        `build_regularized_nlp`, since the latter is itself a DifferentiableNLP whose
        primal is xi, parameter is p_reg, inequality is z <= 0 and equality is h_reg.

        Sign convention follows the rest of the class: L = f + lam^T g + nu^T h with
        g(x) <= 0 and lam >= 0.

        Parameters
        ----------
        active_tol : float
            Threshold on g for declaring an inequality active, and on lam for
            declaring it strongly active.
        rank_tol : Optional[float]
            Passed to `np.linalg.matrix_rank`. None uses NumPy's scale-aware default
            (max(M, N) * eps * sigma_max), which is what you want unless you have a
            specific absolute threshold in mind.
        pd_tol : float
            Relative threshold on the smallest eigenvalue of the reduced Hessian.

        Returns
        -------
        Dict[str, Any]
            licq, scs, sosc, ssosc (bool), plus diagnostics: dim_kernel,
            n_active, n_weakly_active, sosc_exact, and kkt_residual.
        """
        # --- 1. Reuse the eagerly compiled ift function -------------------------
        ev = self._eval_point(theta, x, lam, nu)
        H_L = ev["H"].toarray()
        J_g = ev["J_g"].toarray()
        J_h = ev["J_h"].toarray()
        g_np, lam_np = ev["g"], ev["lam"]

        # --- 2. Guard: these conditions are only meaningful at a KKT point ------
        stat, _, eq, _, dual_feas = self.funcs.kkt(x, lam, nu, theta)
        kkt_res = float(max(
            np.max(np.abs(np.array(ca.DM(stat).full()))) if self.n_x else 0.0,
            np.max(np.abs(np.array(ca.DM(eq).full()))) if self.n_eq else 0.0,
            np.max(np.abs(np.array(ca.DM(dual_feas).full()))) if self.n_ineq else 0.0,
        ))

        # --- 3. Active / strongly active / weakly active sets -------------------
        active = np.where(g_np >= -active_tol)[0]
        strict = active[lam_np[active] > active_tol] if active.size else active
        weak = np.setdiff1d(active, strict)

        def stacked(idx: np.ndarray) -> np.ndarray:
            return np.vstack([J_h, J_g[idx, :]])  # both are (k, n_x), empty is fine

        J_act = stacked(active)
        J_str = stacked(strict)

        # --- 4. LICQ: active constraint gradients linearly independent ----------
        licq = True
        if J_act.shape[0] > 0:
            licq = bool(np.linalg.matrix_rank(J_act, tol=rank_tol) == J_act.shape[0])

        # --- 5. SCS: no weakly active inequality --------------------------------
        scs = bool(weak.size == 0)

        # --- 6. Second order conditions -----------------------------------------
        def pd_on_nullspace(H: np.ndarray, J: np.ndarray) -> bool:
            Z = np.eye(H.shape[0]) if J.shape[0] == 0 else null_space(J, rcond=rank_tol)
            if Z.shape[1] == 0:
                return True  # vacuously: the critical subspace is trivial
            H_red = Z.T @ H @ Z
            H_red = 0.5 * (H_red + H_red.T)
            scale = max(1.0, float(np.max(np.abs(H_red))))
            return bool(np.min(np.linalg.eigvalsh(H_red)) > pd_tol * scale)

        # SSOSC drops the weakly active rows, so null(J_str) contains null(J_act):
        # it is the strictly stronger test, and SSOSC => SOSC.
        sosc_subspace = pd_on_nullspace(H_L, J_act)
        ssosc = pd_on_nullspace(H_L, J_str)

        # The true SOSC critical cone is
        #   C = {d : J_h d = 0, J_g[i] d = 0 for strongly active i,
        #                       J_g[i] d <= 0 for weakly active i}
        # which is a cone, not a subspace. null(J_act) is a strict subset of C
        # whenever a weakly active constraint exists, so the subspace test is
        # optimistic there. It is exact iff SCS holds; SSOSC is a rigorous
        # sufficient certificate in all cases.
        sosc = bool(ssosc or sosc_subspace)
        sosc_exact = bool(scs or ssosc)

        # --- 7. Dimension of ker K, K = [[H_L, J_act^T], [J_act, 0]] ------------
        # This is zero exactly when the IFT applies at this point, and its value
        # counts the independent degeneracy directions (LICQ-type and SOSC-type
        # together), which licq/sosc alone do not distinguish.
        K = np.block([[H_L, J_act.T], [J_act, np.zeros((J_act.shape[0], J_act.shape[0]))]])
        dim_kernel = int(K.shape[0] - np.linalg.matrix_rank(K, tol=rank_tol))

        return {
            "licq": licq,
            "scs": scs,
            "sosc": sosc,
            "ssosc": ssosc,
            "sosc_exact": sosc_exact,
            "dim_kernel": dim_kernel,
            "n_active": int(active.size),
            "n_weakly_active": int(weak.size),
            "kkt_residual": kkt_res,
        }

    def degenerate_subspaces(
        self,
        theta: ArrayLike,
        x: ArrayLike,
        lam: ArrayLike,
        nu: ArrayLike,
        active_tol: float = 1e-6,
        rank_tol: Optional[float] = None,
        flat_tol: float = 1e-9,
        weak_as_active: bool = True,
        active_rule: str = "ratio",
        null_space_method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Computes the orthogonal projectors onto the degenerate subspaces U and P
        of Section 3 at the primal-dual point (x, lam, nu).

        U is the left null space of J_sys = [J_g[sys_idx]; J_h], expressed in the
        coordinates of the multipliers (psi_in[sys_idx], psi_eq). It is trivial
        iff LICQ holds on the constraints entering the linear system.

        P is spanned by the directions of null(J_sys) along which the Hessian of
        the Lagrangian is flat, i.e. the eigenvectors of Z^T H Z (Z an orthonormal
        basis of null(J_sys)) with eigenvalue below flat_tol * scale.

        Active set: with active_rule="threshold" an inequality is active iff
        g >= -active_tol. With "ratio" (default) it is also active if lam > -g,
        which is the natural test for interior-point output, where a constraint
        with a small multiplier sits at distance ~ mu / lam from its bound.

        Weakly active inequalities: with weak_as_active=True they enter J_sys
        (they are assumed to remain active, giving one of the one-sided
        derivatives). With False they are dropped and P is computed on K_+, as in
        Corollary 3.2. Under SCS both choices coincide.

        Null spaces: both null(J_sys) and null(J_sys^T) use the rank rule of
        `scipy.linalg.null_space` (singular values <= rank_tol * s_max are zero,
        rank_tol=None meaning eps * max(shape)). With null_space_method="dense"
        they come from one full SVD of J_sys. With "sparse" they come from block
        inverse iteration on J^T J and J J^T, through one sparse LU of
        [delta I, J^T; J, -delta I] (see `_null_spaces_sparse`); this is much
        faster when the null spaces are small compared to J (the usual case), and
        falls back to the dense SVD when it cannot certify its result. "auto"
        (default) uses "sparse" except for small matrices.

        Returns
        -------
        Dict with keys: Pi_P (n_x, n_x), Pi_U (n_m, n_m), B_P (n_x, dim_P) and
        N_U (n_m, dim_U) orthonormal bases with Pi_P = B_P B_P^T and
        Pi_U = N_U N_U^T, dim_P, dim_U, sys_idx, active_idx, strong_idx,
        weak_idx, min_reduced_eig (negative values of significant size indicate
        that Assumption 3.1 fails), smallest_nonflat_eig (gap to the flat
        eigenvalues, useful to sanity check flat_tol), null_space_method (the
        method actually used) and timing.
        """
        t0 = perf_counter()
        ev = self._eval_point(theta, x, lam, nu)
        t1 = perf_counter()

        sub = self._subspace_bases(
            ev,
            active_tol=active_tol,
            rank_tol=rank_tol,
            flat_tol=flat_tol,
            weak_as_active=weak_as_active,
            active_rule=active_rule,
            method=null_space_method,
        )
        sub["Pi_P"] = sub["B_P"] @ sub["B_P"].T
        sub["Pi_U"] = sub["N_U"] @ sub["N_U"].T
        t2 = perf_counter()

        sub["timing"] = {"convert": t1 - t0, **sub["timing"], "total": t2 - t0}
        return sub

    def annotate_solution(self, sol: "NLPSolution", **kwargs) -> Dict[str, Any]:
        """Thin wrapper: evaluate properties at `sol` and store them on it."""
        props = self.check_properties(sol.theta, sol.x, sol.lam, sol.nu, **kwargs)
        if sol.properties is None:
            sol.properties = {}
        sol.properties.update(props)
        return props