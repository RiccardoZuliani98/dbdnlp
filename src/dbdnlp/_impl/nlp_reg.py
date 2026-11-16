# Copyright (c) 2026 Automatic Control Laboratory, ETH Zurich, 8092 Zurich, Switzerland, Riccardo Zuliani
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the project root.

from typing import Dict, Optional, Any, cast
import casadi as ca

from dbdnlp._impl.types import NLPSymbolic, Symbolic
from dbdnlp._impl.nlp import DifferentiableNLP

def build_regularized_nlp(
    base_sym: NLPSymbolic, 
    solver_opts: Optional[Dict[str, Any]] = None
) -> DifferentiableNLP:
    """
    Constructs a regularized `DifferentiableNLP` instance from a base symbolic NLP definition.

    Transforms the original primal NLP $P_1(\\theta)$ into an augmented primal-dual 
    regularized formulation $P_2(p_{\\text{reg}})$ to guarantee differentiability 
    and strong local convexity, even when strict complementarity or Linear 
    Independence Constraint Qualifications (LICQ) fail.

    Mathematical Formulation
    ------------------------
    **Augmented Decision Variables:**
        $$\\xi = \\begin{bmatrix} x \\\\ \\lambda_{\\text{reg}} \\\\ \\nu_{\\text{reg}} \\\\ z_{\\text{reg}} \\end{bmatrix} \\in \\mathbb{R}^{n_x + n_g + n_h + n_g}$$

    **Augmented Parameters:**
        $$p_{\\text{reg}} = \\begin{bmatrix} \\theta \\\\ \\rho_1 \\\\ \\rho_2 \\\\ \\bar{x} \\\\ \\bar{\\lambda} \\\\ \\bar{\\nu} \\end{bmatrix} \\in \\mathbb{R}^{n_\\theta + 1 + 1 + n_x + n_g + n_h}$$

    **Regularized Objective Function ($f_{\\text{reg}}$):**
        $$f_{\\text{reg}}(\\xi, p_{\\text{reg}}) = f(x, \\theta) + \\frac{\\rho_1}{2} \\left( \\|x - \\bar{x}\\|^2 + \\|z_{\\text{reg}} - g(\\bar{x}, \\theta)\\|^2 \\right) + \\frac{\\rho_2}{2} \\left( \\|\\lambda_{\\text{reg}}\\|^2 + \\|\\nu_{\\text{reg}}\\|^2 \\right)$$

    **Regularized Equality Constraints ($h_{\\text{reg}}$):**
        $$h_{\\text{reg}}(\\xi, p_{\\text{reg}}) = \\begin{bmatrix} g(x, \\theta) - z_{\\text{reg}} + \\rho_2(\\bar{\\lambda} - \\lambda_{\\text{reg}}) \\\\ h(x, \\theta) + \\rho_2(\\bar{\\nu} - \\nu_{\\text{reg}}) \\end{bmatrix} = 0$$

    **Regularized Inequality Constraints ($g_{\\text{reg}}$):**
        $$g_{\\text{reg}}(\\xi, p_{\\text{reg}}) = z_{\\text{reg}} \\le 0$$

    Parameters
    ----------
    base_sym : NLPSymbolic
        The symbolic representation of the base NLP containing primal/parameter
        variables, objective, and constraint expressions.
    solver_opts : Optional[Dict[str, Any]], default=None
        Dictionary of solver options passed directly to the underlying `casadi.nlpsol`
        plugin (e.g., IPOPT settings).

    Returns
    -------
    DifferentiableNLP
        A fully initialized solver instance representing the augmented, regularized NLP.
    """
    # 1. Inspect graph type (ca.MX vs ca.SX) to preserve compatibility
    sym_type = type(base_sym.primal)
    
    def typed_sym(name: str, nrow: int = 1, ncol: int = 1) -> Symbolic:
        return cast(Symbolic, sym_type.sym(name, nrow, ncol)) #type: ignore

    # 2. Extract dimensions and base symbolic expressions
    n_x: int = base_sym.primal.size1()
    n_theta: int = base_sym.param.size1()
    n_g: int = base_sym.ineq.size1()
    n_h: int = base_sym.eq.size1()
    
    x: Symbolic = base_sym.primal
    theta: Symbolic = base_sym.param

    # 3. Create regularized parameters p_reg = [theta; rho_1; rho_2; x_bar; lam_bar; nu_bar]
    rho_1: Symbolic = typed_sym('rho_1')
    rho_2: Symbolic = typed_sym('rho_2')
    x_bar: Symbolic = typed_sym('x_bar', n_x)
    lam_bar: Symbolic = typed_sym('lam_bar', n_g)
    nu_bar: Symbolic = typed_sym('nu_bar', n_h)

    p_reg: Symbolic = cast(
        Symbolic, 
        ca.vertcat(theta, rho_1, rho_2, x_bar, lam_bar, nu_bar)
    )

    # 4. Create augmented decision variables xi = [x; lam_reg; nu_reg; z_reg]
    lam_reg: Symbolic = typed_sym('lam_reg', n_g)
    nu_reg: Symbolic = typed_sym('nu_reg', n_h)
    z_reg: Symbolic = typed_sym('z_reg', n_g)

    xi: Symbolic = cast(
        Symbolic, 
        ca.vertcat(x, lam_reg, nu_reg, z_reg)
    )

    # 5. Formulate Regularized Objective Function f_reg
    g_x_bar: Symbolic = (
        cast(Symbolic, ca.substitute(base_sym.ineq, x, x_bar)) 
        if n_g > 0 
        else typed_sym('empty_g_x_bar', 0, 1)
    )
    
    penalty_primal: Symbolic = (
        (rho_1 / 2.0) * (ca.sumsqr(x - x_bar) + ca.sumsqr(z_reg - g_x_bar))
    )
    penalty_dual: Symbolic = (
        (rho_2 / 2.0) * (ca.sumsqr(lam_reg) + ca.sumsqr(nu_reg))
    )
    
    f_reg: Symbolic = cast(Symbolic, base_sym.cost + penalty_primal + penalty_dual)

    # 6. Formulate Regularized Equality Constraints h_reg
    h_reg_list = []
    if n_g > 0:
        h_reg_list.append(base_sym.ineq - z_reg + rho_2 * (lam_bar - lam_reg))
    if n_h > 0:
        h_reg_list.append(base_sym.eq + rho_2 * (nu_bar - nu_reg))
        
    h_reg: Symbolic = (
        cast(Symbolic, ca.vertcat(*h_reg_list)) 
        if h_reg_list 
        else typed_sym('empty_h_reg', 0, 1)
    )

    # 7. Formulate Regularized Inequality Constraints g_reg (z_reg <= 0)
    g_reg: Symbolic = z_reg if n_g > 0 else typed_sym('empty_g_reg', 0, 1)

    # 8. Return a fresh DifferentiableNLP instance
    return DifferentiableNLP(
        x=xi,
        theta=p_reg,
        f=f_reg,
        g=g_reg,
        h=h_reg,
        solver_opts=solver_opts
    )