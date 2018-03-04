#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy as np
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, HDF5Enabled, to_hdf5, from_hdf5


@export
@subscribe_hdf5('symmetry_representation.symmetry_group')
class SymmetryGroup(HDF5Enabled, types.SimpleNamespace):
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

    def to_hdf5(self, hdf5_handle):
        to_hdf5(self.symmetries, hdf5_handle.create_group('symmetries'))
        hdf5_handle['full_group'] = self.full_group

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            symmetries=from_hdf5(hdf5_handle['symmetries']),
            full_group=hdf5_handle['full_group'].value
        )


@export
@subscribe_hdf5('symmetry_representation.symmetry_operation')
class SymmetryOperation(HDF5Enabled, types.SimpleNamespace):
    """
    Describes a symmetry operation.

    :param rotation_matrix: Real-space rotation matrix of the symmetry (in reduced coordinates).
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
        return np.all(
            self.rotation_matrix == val.rotation_matrix
        ) and self.repr == val.repr

    def to_hdf5(self, hdf5_handle):
        hdf5_handle['rotation_matrix'] = np.array(self.rotation_matrix)
        repr_hf = hdf5_handle.create_group('repr')
        to_hdf5(self.repr, repr_hf)

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        representation = Representation.from_hdf5(hdf5_handle['repr'])
        return cls(
            rotation_matrix=np.array(hdf5_handle['rotation_matrix']),
            repr_matrix=representation.matrix,
            repr_has_cc=representation.has_cc
        )


@export
@subscribe_hdf5('symmetry_representation.representation')
class Representation(HDF5Enabled, types.SimpleNamespace):
    """
    Describes an (anti-)unitary representation of a symmetry operation.
    """

    def __init__(self, matrix, has_cc=False):
        self.matrix = matrix
        self.has_cc = has_cc

    def __eq__(self, val):
        return np.all(self.matrix == val.matrix) and self.has_cc == val.has_cc

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            matrix=np.array(hdf5_handle['matrix']),
            has_cc=hdf5_handle['has_cc'].value
        )

    def to_hdf5(self, hdf5_handle):
        hdf5_handle['has_cc'] = self.has_cc
        hdf5_handle['matrix'] = np.array(self.matrix)
