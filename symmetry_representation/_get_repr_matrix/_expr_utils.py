"""
Utilities for handling algebraic expressions, such as turning them to vector or matrix form.
"""

import random

import numpy as np
import numpy.linalg as nl
# import scipy.linalg as la
import sympy as sp

VEC = sp.symbols('x, y, z')


def _get_substitution(rotation_matrix_cartesian):
    return list(zip(VEC, rotation_matrix_cartesian.T @ sp.Matrix(VEC)))


def _expr_to_vector(
    expr, basis, *, random_fct=lambda: random.randint(-100, 100), numeric
):
    """
    Converts an algebraic (sympy) expression into vector form.

    :param expr: Algebraic expression
    :type expr: sympy.Expr

    :param expr: Basis of the vector space, w.r.t. which the vector will be expressed.
    :type expr: list[sympy.Expr]

    :param random_fct: Function creating random numbers on which the expression will be evaluated.
    """
    dim = len(basis)
    # create random values for the coordinates and evaluate
    # both the basis functions and the expression to generate
    # the linear equation to be solved
    A = []  # pylint: disable=invalid-name
    b = []  # pylint: disable=invalid-name
    for _ in range(2 * dim):
        if not numeric:
            if sp.Matrix(A).rank() >= len(basis):
                break
        vals = [(k, random_fct()) for k in VEC]
        A.append([b.subs(vals) for b in basis])
        b.append(expr.subs(vals))
    else:
        # this could happen if the random_fct is bad, or the 'basis' is not
        # linearly independent
        if not numeric:
            raise ValueError(
                'Could not find a sufficient number of linearly independent vectors'
            )

    if numeric:
        vec = nl.lstsq(
            np.array(A).astype(complex),
            np.array(b).astype(complex),
        )[0]
    else:
        res = sp.linsolve((sp.Matrix(A), sp.Matrix(b)), sp.symbols('a b c'))
        if len(res) != 1:
            raise ValueError(
                'Invalid result {res} when trying to match expression {expr} to basis {basis}.'
                .format(res=res, expr=expr, basis=basis)
            )
        vec = next(iter(res))
        vec = tuple(v.nsimplify() for v in vec)
    return vec
