#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy as np
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('symmetry_representation.symmetry_group')
class SymmetryGroup(SimpleHDF5Mapping, types.SimpleNamespace):
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
    HDF5_ATTRIBUTES = ['symmetries', 'full_group']

    def __init__(self, symmetries, full_group=False):
        self.symmetries = list(symmetries)
        self.full_group = full_group


@export
@subscribe_hdf5('symmetry_representation.symmetry_operation')
class SymmetryOperation(SimpleHDF5Mapping, types.SimpleNamespace):
    """
    Describes a symmetry operation.

    :param rotation_matrix: Real-space rotation matrix of the symmetry (in reduced coordinates).
    :type rotation_matrix: array

    :param repr_matrix: Matrix of the representation corresponding to the symmetry operation.
    :type repr_matrix: array

    :param translation_vector: Real-space displacement vector of the symmetry (in reduced coordinates).
    :type translation_vector: array

    :param repr_has_cc: Specifies whether the representation contains a complex conjugation.
    :type repr_has_cc: bool

    :ivar real_space_operator: Real-space operator of the symmetry.
    :vartype real_space_operator: :class:`.RealSpaceOperator`

    :ivar repr: Symmetry representation.
    :vartype repr: :class:`.Representation`

    .. note :: Currently, only point-group symmetries are implemented.
    """

    HDF5_ATTRIBUTES = ['real_space_operator', 'repr']

    def __init__(
        self,
        *,
        rotation_matrix,
        repr_matrix,
        translation_vector=None,
        repr_has_cc=False
    ):
        self.real_space_operator = RealSpaceOperator(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector
        )
        self.repr = Representation(matrix=repr_matrix, has_cc=repr_has_cc)

    @classmethod
    def from_real_space_operator(cls, *, real_space_operator, **kwargs):
        return cls(
            rotation_matrix=real_space_operator.rotation_matrix,
            translation_vector=real_space_operator.translation_vector,
            **kwargs
        )

    @classmethod
    def from_orbitals(
        cls,
        *,
        orbitals,
        real_space_operator,
        rotation_matrix_cartesian,
        numeric,
        **kwargs
    ):
        from . import _get_repr_matrix
        repr_matrix = _get_repr_matrix.get_repr_matrix(
            orbitals=orbitals,
            real_space_operator=real_space_operator,
            rotation_matrix_cartesian=rotation_matrix_cartesian,
            numeric=numeric
        )
        return cls.from_real_space_operator(
            real_space_operator=real_space_operator,
            repr_matrix=repr_matrix,
            **kwargs
        )

    @property
    def rotation_matrix(self):
        return self.real_space_operator.rotation_matrix

    @property
    def translation_vector(self):
        return self.real_space_operator.translation_vector

    def __eq__(self, other):
        return (
            self.real_space_operator == other.real_space_operator
            and self.repr == other.repr
        )

    def __matmul__(self, other):
        if not isinstance(other, SymmetryOperation):
            raise TypeError(
                'Cannot matrix-multiply objects of type {} and {}'.format(
                    type(self), type(other)
                )
            )
        new_real_space_op = self.real_space_operator @ other.real_space_operator
        new_repr = self.repr @ other.repr
        return SymmetryOperation(
            rotation_matrix=new_real_space_op.rotation_matrix,
            translation_vector=new_real_space_op.translation_vector,
            repr_matrix=new_repr.matrix,
            repr_has_cc=new_repr.has_cc
        )

    def get_order(self, max_order=20):
        curr_val = self
        for i in range(1, max_order + 1):
            if curr_val.repr.is_identity and curr_val.real_space_operator.is_lattice_translation:
                return i
            curr_val @= self
        else:  # pylint: disable=useless-else-on-loop
            raise ValueError(
                'Order of the symmetry operation could not be determined.'
            )

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        if 'real_space_operator' in hdf5_handle:
            real_space_operator = RealSpaceOperator.from_hdf5(
                hdf5_handle['real_space_operator']
            )
            rotation_matrix = real_space_operator.rotation_matrix
            translation_vector = real_space_operator.translation_vector
        # handle old version without RealSpaceOperator
        else:
            rotation_matrix = np.array(hdf5_handle['rotation_matrix'])
            translation_vector = None

        representation = Representation.from_hdf5(hdf5_handle['repr'])
        return cls(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            repr_matrix=representation.matrix,
            repr_has_cc=representation.has_cc
        )


@export
@subscribe_hdf5('symmetry_representation.real_space_operator')
class RealSpaceOperator(SimpleHDF5Mapping, types.SimpleNamespace):
    """
    Describes the real-space operator of a symmetry operation.
    """

    HDF5_ATTRIBUTES = ['rotation_matrix', 'translation_vector']

    def __init__(self, rotation_matrix, translation_vector=None):
        self.rotation_matrix = rotation_matrix
        if translation_vector is None:
            translation_vector = np.zeros(len(self.rotation_matrix))
        self.translation_vector = translation_vector

    @classmethod
    def from_pymatgen(cls, pymatgen_op):
        return cls(
            rotation_matrix=pymatgen_op.rotation_matrix,
            translation_vector=pymatgen_op.translation_vector
        )

    def __matmul__(self, other):
        if not isinstance(other, RealSpaceOperator):
            raise TypeError(
                'Cannot matrix-multiply objects of type {} and {}'.format(
                    type(self), type(other)
                )
            )
        return RealSpaceOperator(
            rotation_matrix=self.rotation_matrix @ other.rotation_matrix,
            translation_vector=self.translation_vector +
            self.rotation_matrix @ other.translation_vector
        )

    def apply(self, r):
        """
        Apply symmetry operation to a vector in reduced real-space coordinates.
        """
        return self.rotation_matrix @ r + self.translation_vector

    @property
    def is_pure_translation(self):
        n, m = self.rotation_matrix.shape
        if n != m:
            return False
        return np.allclose(self.rotation_matrix, np.eye(n))

    @property
    def is_lattice_translation(self):
        return self.is_pure_translation and np.allclose(
            self.translation_vector, np.round(self.translation_vector)
        )

    def __eq__(self, other):
        return np.all(
            self.rotation_matrix == other.rotation_matrix
        ) and np.all(self.translation_vector == other.translation_vector)


@export
@subscribe_hdf5('symmetry_representation.representation')
class Representation(SimpleHDF5Mapping, types.SimpleNamespace):
    """
    Describes an (anti-)unitary representation of a symmetry operation.
    """
    HDF5_ATTRIBUTES = ['matrix', 'has_cc']

    def __init__(self, matrix, has_cc=False):
        self.matrix = matrix
        self.has_cc = has_cc

    def __matmul__(self, other):
        if not isinstance(other, Representation):
            raise TypeError(
                'Cannot matrix-multiply objects of type {} and {}'.format(
                    type(self), type(other)
                )
            )
        if self.has_cc:
            new_mat = self.matrix @ other.matrix.conjugate()
        else:
            new_mat = self.matrix @ other.matrix
        new_has_cc = self.has_cc != other.has_cc
        return Representation(matrix=new_mat, has_cc=new_has_cc)

    @property
    def is_identity(self):
        n, m = self.matrix.shape
        if n != m:
            return False
        return (not self.has_cc) and np.allclose(self.matrix, np.eye(n))

    def __eq__(self, val):
        return np.all(self.matrix == val.matrix) and self.has_cc == val.has_cc
