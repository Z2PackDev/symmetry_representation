# © 2017-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the RealSpaceOperator class.
"""

import pytest

import numpy as np

import symmetry_representation as sr


@pytest.mark.parametrize(['real_space_op', 'vec', 'res'], [
    (
        sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]), [0.2, 0.7, 0.1],
        [0.3, 0.9, 0.4]
    ),
    (
        sr.RealSpaceOperator([[0, 1], [1, 1]], [0.1, 0.2]), [0.3, 0.6],
        [0.7, 1.1]
    ),
])
def test_apply(real_space_op, vec, res):
    """
    Tests applying a symmetry operation to a real-space vector.
    """
    assert np.allclose(real_space_op.apply(vec), res)


@pytest.mark.parametrize(
    ['real_space_op', 'is_pure_translation', 'is_lattice_translation'], [
        (sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]), True, False),
        (sr.RealSpaceOperator([[0, 1], [1, 0]]), False, False),
    ]
)
def test_translation_tests(
    real_space_op, is_pure_translation, is_lattice_translation
):
    """
    Tests the Boolean properties which describe if an operation is a pure
    translation, and a lattice translation.
    """
    assert real_space_op.is_pure_translation == is_pure_translation
    assert real_space_op.is_lattice_translation == is_lattice_translation


@pytest.mark.parametrize(
    ['real_space_op_1', 'real_space_op_2', 'result'],
    [(
        sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]),
        sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]), True
    ),
     (
         sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]),
         sr.RealSpaceOperator(np.eye(3), [0.9, 0.2, 0.3]), False
     ),
     (
         sr.RealSpaceOperator([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                              [0.1, 0.2, 0.3]),
         sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]), False
     )]
)
def test_equal(real_space_op_1, real_space_op_2, result):
    """
    Tests the equality operator.
    """
    assert (real_space_op_1 == real_space_op_2) == result


@pytest.mark.parametrize(['left', 'right', 'result'], [
    (
        sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]),
        sr.RealSpaceOperator(np.eye(3), [0.1, 0.2, 0.3]),
        sr.RealSpaceOperator(np.eye(3), [0.2, 0.4, 0.6]),
    ),
    (
        sr.RealSpaceOperator([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                             [0.1, 0.2, 0.3]),
        sr.RealSpaceOperator([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                             [0.5, 0.4, 0.9]),
        sr.RealSpaceOperator(np.eye(3), [0.5, 0.7, 1.2]),
    ),
    (
        sr.RealSpaceOperator([[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                             [0.1, 0.2, 0.3]),
        sr.RealSpaceOperator([[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                             [0.5, 0.4, 0.9]),
        sr.RealSpaceOperator([[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                             [0.5, 1.1, 0.8]),
    )
])
def test_matmul(left, right, result):
    """
    Tests the matrix multiplication operator.
    """
    assert (left @ right) == result


def test_invalid_not_square():
    """
    Tests that an error is raised when constructing an operator with a matrix
    which is not square.
    """
    with pytest.raises(ValueError):
        sr.RealSpaceOperator([[0, 1]])


def test_translation_dimension():
    """
    Tests that an error is raised when the dimension of the translation vector
    doesn't match the dimension of the rotation matrix.
    """
    with pytest.raises(ValueError):
        sr.RealSpaceOperator([[0, 1], [1, 0]], [1, 2, 3])
