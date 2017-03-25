#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import tempfile

import pytest
import numpy as np

import symmetry_representation as sr

SYM_OP = sr.SymmetryOperation(
    rotation_matrix=np.array([[1, 2, 3], [4, 5j, 9]]),
    repr_matrix=np.array([[0, 1], [3, 5]]),
    repr_has_cc=True
)
REPR_MATRIX = sr.Representation(matrix=np.array([[1j, 0], [-2j, 3j]]))
SYM_GROUP = sr.SymmetryGroup(symmetries=[SYM_OP, SYM_OP], full_group=True)

@pytest.mark.parametrize('data',[
    SYM_OP,
    [SYM_OP],
    REPR_MATRIX,
    [REPR_MATRIX],
    [SYM_OP, [SYM_OP], REPR_MATRIX],
    SYM_GROUP,
    [SYM_GROUP, SYM_OP, REPR_MATRIX]
])
def test_save_load(data):
    with tempfile.NamedTemporaryFile() as f:
        print(data)
        sr.io.save(data, f.name)
        result = sr.io.load(f.name)
        np.testing.assert_equal(result, data)
