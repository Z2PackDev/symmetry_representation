#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import types

import numpy as np
from fsc.export import export

@export
class SymmetryGroup(types.SimpleNamespace):
    """
    Describes a symmetry group.

    :param symmetries: Elements of the symmetry group.
    :type symmetries: list(SymmetryOperation)

    :param full_group: Flag which determines whether the symmetry elements describe the full group or just a generating subset.
    :type full_group: bool

    :ivar symmetries: Elements of the symmetry group.
    :vartype symmetries: list(SymmetryOperation)

    :ivar full_group: Flag which determines whether the symmetry elements describe the full group or just a generating subset.
    :vartype full_group: bool
    """
    def __init__(self, symmetries, full_group=False):
        self.symmetries = list(symmetries)
        self.full_group = full_group

@export
class SymmetryOperation(types.SimpleNamespace):
    """
    Describes a symmetry operation.

    :param rotation_matrix: Real-space rotation matrix of the symmetry.
    :type rotation_matrix: array

    :param repr_matrix: Matrix of the representation corresponding to the symmetry operation.
    :type repr_matrix: array

    :param repr_has_cc: Specifies whether the representation contains a complex conjugation.
    :type repr_has_cc: bool

    :ivar rotation_matrix: Real-space rotation matrix.
    :vartype rotation_matrix: array

    :ivar repr: Symmetry representation.
    :vartype repr: :class:`.Representation`

    .. note :: Currently, only point-group symmetries are implemented.
    """
    def __init__(self, *, rotation_matrix, repr_matrix, repr_has_cc=False):
        self.rotation_matrix = rotation_matrix
        self.repr = Representation(matrix=repr_matrix, has_cc=repr_has_cc)

    def __eq__(self, val):
        return np.all(self.rotation_matrix == val.rotation_matrix) and self.repr == val.repr

@export
class Representation(types.SimpleNamespace):
    """
    Describes an (anti-)unitary representation of a symmetry operation.
    """
    def __init__(self, matrix, has_cc=False):
        self.matrix = matrix
        self.has_cc = has_cc

    def __eq__(self, val):
        return np.all(self.matrix == val.matrix) and self.has_cc == val.has_cc
