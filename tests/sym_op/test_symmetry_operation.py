# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the SymmetryOperation class.
"""

import pytest

import numpy as np
import sympy as sp

import symmetry_representation as sr


@pytest.mark.parametrize(['left', 'right', 'result'],
                         [(
                             sr.SymmetryOperation(
                                 rotation_matrix=np.eye(3),
                                 repr_matrix=[[0, 1], [1, 0]],
                                 repr_has_cc=True
                             ),
                             sr.SymmetryOperation(
                                 rotation_matrix=np.eye(3),
                                 repr_matrix=[[0, 1], [1, 0]],
                                 repr_has_cc=True
                             ),
                             sr.SymmetryOperation(
                                 rotation_matrix=np.eye(3),
                                 repr_matrix=np.eye(2),
                                 repr_has_cc=False
                             )
                         ),
                          (
                              sr.SymmetryOperation(
                                  rotation_matrix=sp.eye(3, 3),
                                  repr_matrix=[[0, 1], [1, 0]],
                                  repr_has_cc=True,
                                  numeric=False
                              ),
                              sr.SymmetryOperation(
                                  rotation_matrix=sp.eye(3, 3),
                                  repr_matrix=[[0, 1], [1, 0]],
                                  repr_has_cc=True,
                                  numeric=False
                              ),
                              sr.SymmetryOperation(
                                  rotation_matrix=np.eye(3),
                                  repr_matrix=np.eye(2),
                                  repr_has_cc=False,
                                  numeric=False
                              )
                          ),
                          (
                              sr.SymmetryOperation(
                                  rotation_matrix=[[0, 1], [1, 0]],
                                  repr_matrix=[[0, 1j], [1j, 0]],
                                  translation_vector=[0.5, 0.5],
                                  repr_has_cc=True,
                                  numeric=True
                              ),
                              sr.SymmetryOperation(
                                  rotation_matrix=np.eye(2),
                                  repr_matrix=[[0, -1j], [-1j, 0]],
                                  repr_has_cc=False,
                                  numeric=True
                              ),
                              sr.SymmetryOperation(
                                  rotation_matrix=[[0, 1], [1, 0]],
                                  translation_vector=[0.5, 0.5],
                                  repr_matrix=-np.eye(2),
                                  repr_has_cc=True,
                                  numeric=True
                              )
                          ),
                          (
                              sr.SymmetryOperation(
                                  rotation_matrix=[[0, 1], [1, 0]],
                                  repr_matrix=[[0, sp.I], [sp.I, 0]],
                                  translation_vector=[0.5, 0.5],
                                  repr_has_cc=True,
                                  numeric=False
                              ),
                              sr.SymmetryOperation(
                                  rotation_matrix=np.eye(2),
                                  repr_matrix=[[0, -sp.I], [-sp.I, 0]],
                                  repr_has_cc=False,
                                  numeric=False
                              ),
                              sr.SymmetryOperation(
                                  rotation_matrix=[[0, 1], [1, 0]],
                                  translation_vector=[sp.Rational(1, 2)] * 2,
                                  repr_matrix=-np.eye(2),
                                  repr_has_cc=True,
                                  numeric=False
                              )
                          )])
def test_matmul(left, right, result):
    """
    Check the matrix multiplication operator against the expected result.
    """
    assert (left @ right) == result


@pytest.mark.parametrize([
    'rotation_matrix', 'translation_vector', 'repr_matrix', 'repr_has_cc',
    'result'
], [
    ([[0, 1], [1, 0]], [0.5, 0.5], -sp.eye(2, 2), True, 2),
    (sp.eye(2, 2), None, sp.eye(2, 2), False, 1),
    (sp.eye(2, 2), None, sp.I * sp.eye(2, 2), False, 4),
])
def test_get_order(
    rotation_matrix, translation_vector, repr_matrix, repr_has_cc, result,
    numeric
):
    """
    Check that the ``get_order`` method matches the expected result.
    """
    sym_op = sr.SymmetryOperation(
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        repr_matrix=repr_matrix,
        repr_has_cc=repr_has_cc,
        numeric=numeric
    )
    assert sym_op.get_order() == result


def test_get_order_invalid(numeric):
    """
    Check that the ``get_order`` method raises an error when the symmetry
    operations are invalid (have no power that is the identity).
    """
    sym_op = sr.SymmetryOperation(
        rotation_matrix=[[1, 1], [0, 1]],
        translation_vector=None,
        repr_matrix=sp.eye(2, 2),
        repr_has_cc=True,
        numeric=numeric
    )
    with pytest.raises(ValueError):
        sym_op.get_order()
