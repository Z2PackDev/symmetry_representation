# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the Representation class.
"""

import pytest

import numpy as np
import sympy as sp

import symmetry_representation as sr


@pytest.mark.parametrize(
    ['mat_1', 'has_cc_1', 'mat_2', 'has_cc_2', 'mat_res', 'has_cc_res'],
    [
        ([[0, 1], [1, 0]], False, [[0, 1], [1, 0]], True, [[1, 0], [0, 1]
                                                           ], True),
        ([[1, 0], [0, 1]], True, [[0, sp.I], [-sp.I, 0]
                                  ], False, [[0, -sp.I], [sp.I, 0]], True),
    ]  # pylint: disable=too-many-arguments
)
def test_representation_mul(
    mat_1, has_cc_1, mat_2, has_cc_2, mat_res, has_cc_res, numeric
):
    """
    Test the multiplication of representations.
    """
    repr_1 = sr.Representation(mat_1, has_cc=has_cc_1, numeric=numeric)
    repr_2 = sr.Representation(mat_2, has_cc=has_cc_2, numeric=numeric)
    repr_res = repr_1 @ repr_2
    if numeric:
        assert np.allclose(repr_res.matrix, np.array(mat_res).astype(complex))
    else:
        assert repr_res.matrix.equals(sp.Matrix(mat_res))
    assert repr_res.has_cc == has_cc_res


def test_repr_mul_invalid():
    """
    Test that trying to multiply incompatible representation raises an error.
    """
    with pytest.raises(ValueError):
        (  # pylint: disable=expression-not-assigned
            sr.Representation([[1]], numeric=True)
            @ sr.Representation([[1]], numeric=False)
        )


@pytest.mark.parametrize(['mat', 'has_cc', 'result'],
                         [(sp.eye(3, 3), False, True),
                          (sp.eye(3, 3), True, False),
                          ([[sp.I, 0], [0, -sp.I]], False, False),
                          ([[0, 1], [1, 0]], False, False)])
def test_is_identity(mat, has_cc, result, numeric):
    """
    Check that the property describing if the representation is identity matches
    the expected result.
    """
    representation = sr.Representation(mat, has_cc=has_cc, numeric=numeric)
    assert representation.is_identity == result


def test_not_unitary(numeric):
    """
    An error should be raised when the representation matrix is not unitary.
    """
    with pytest.raises(ValueError):
        sr.Representation([[1, 1], [0, 1]], numeric=numeric)


@pytest.mark.parametrize(['val1', 'val2', 'result'], [
    (
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=True),
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=True), True
    ),
    (
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=False),
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=False), True
    ),
    (
        sr.Representation([[1, 0], [0, 1]], has_cc=False, numeric=True),
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=True), False
    ),
    (
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=False),
        sr.Representation([[1, 0], [0, 1]], has_cc=True, numeric=True), False
    ),
])
def test_equal(val1, val2, result):
    """
    Test the equality operator.
    """
    assert (val1 == val2) == result
