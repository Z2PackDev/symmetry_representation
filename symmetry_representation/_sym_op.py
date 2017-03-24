#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

class SymmetryOperation:
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

class Representation:
    """
    Describes an (anti-)unitary representation of a symmetry operation.
    """
    def __init__(self, matrix, has_cc=False):
        self.matrix = matrix
        self.has_cc = has_cc
