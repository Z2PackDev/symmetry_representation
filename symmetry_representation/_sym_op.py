#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

from collections import namedtuple

from fsc.export import export

@export
class SymmetryGroup(namedtuple('SymmetryGroupBase', ['symmetries', 'full_group'])):
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
    pass

@export
class SymmetryOperation(namedtuple('SymmetryOperationBase', ['rotation_matrix', 'repr'])):
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
    def __new__(cls, *, rotation_matrix, repr_matrix, repr_has_cc=False):
        return super().__new__(
            cls,
            rotation_matrix=rotation_matrix,
            repr=Representation(matrix=repr_matrix, has_cc=repr_has_cc)
        )

@export
class Representation(namedtuple('RepresentationBase', ['matrix', 'has_cc'])):
    """
    Describes an (anti-)unitary representation of a symmetry operation.
    """
    def __new__(cls, matrix, has_cc=False):
        return super().__new__(
            cls,
            matrix=matrix,
            has_cc=has_cc
        )
