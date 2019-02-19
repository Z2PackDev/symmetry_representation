#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for saving and loading ``symmetry-representation`` objects.
"""

import tempfile

import pytest
import numpy as np
import sympy as sp

import symmetry_representation as sr

SYM_OP = sr.SymmetryOperation(
    rotation_matrix=np.array([[1, 2, 3], [4, 5, 9], [10, 11, 12]]),
    repr_matrix=np.array([[0, 1], [1, 0]]),
    repr_has_cc=True
)
SYM_OP_ANALYTIC = sr.SymmetryOperation(
    rotation_matrix=np.array([[1, 2, 3], [4, 5, 9], [10, 11, 12]]),
    repr_matrix=sp.Matrix([[0, sp.I], [-sp.I, 0]]),
    repr_has_cc=True
)
REPR_MATRIX = sr.Representation(matrix=np.array([[1j, 0], [0, -1j]]))
REPR_MATRIX_ANALYTIC = sr.Representation(
    matrix=sp.Matrix([[sp.I, 0], [0, sp.I]])
)
SYM_GROUP = sr.SymmetryGroup(symmetries=[SYM_OP, SYM_OP], full_group=True)


@pytest.mark.parametrize(
    'data', [
        SYM_OP, [SYM_OP], REPR_MATRIX, [REPR_MATRIX],
        [SYM_OP, [SYM_OP], REPR_MATRIX], SYM_GROUP,
        [SYM_GROUP, SYM_OP, REPR_MATRIX], REPR_MATRIX_ANALYTIC,
        [REPR_MATRIX_ANALYTIC], [SYM_GROUP, SYM_OP, REPR_MATRIX_ANALYTIC],
        SYM_OP_ANALYTIC
    ]
)
def test_save_load(data):
    """
    Test that objects are the same after saving and loading.
    """
    with tempfile.NamedTemporaryFile() as f:
        print(data)
        sr.io.save(data, f.name)
        result = sr.io.load(f.name)
        np.testing.assert_equal(result, data)


@pytest.mark.parametrize(
    'sample_name', ['symmetries.hdf5', 'symmetries_old.hdf5']
)
def test_load_samples(sample, sample_name):
    """
    Test loading of given sample files.
    """
    res = sr.io.load(sample(sample_name))
    assert isinstance(res, list)
    assert isinstance(res[0], sr.SymmetryOperation)
    assert isinstance(res[1], sr.SymmetryGroup)
