#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines symmetry operations and groups.
"""

import types

import numpy as np
import sympy as sp
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('symmetry_representation.symmetry_group')
class SymmetryGroup(SimpleHDF5Mapping, types.SimpleNamespace):
    """
    Describes a symmetry group.

    Arguments
    ---------
    symmetries : List[SymmetryOperation]
        Elements of the symmetry group.
    full_group : bool
        Flag which determines whether the symmetry elements describe the full group or just a generating subset.

    Attributes
    ----------
    symmetries : List[SymmetryOperation]
        Elements of the symmetry group.
    full_group : bool
        Flag which determines whether the symmetry elements describe the full group or just a generating subset.
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

    Arguments
    ---------
    rotation_matrix : array
        Real-space rotation matrix of the symmetry (in reduced coordinates).
    translation_vector : array
        Real-space displacement vector of the symmetry (in reduced coordinates).
    repr_matrix : array
        Matrix of the representation corresponding to the symmetry operation.
    repr_has_cc : bool
        Specifies whether the representation contains a complex conjugation.
    numeric : bool
        Specifies whether the symmetry operation contains a numeric or analytic
        values. By default, this is determined by the type of the rotation
        matrix.

    Attributes
    ----------
    real_space_operator : RealSpaceOperator
        Real-space operator of the symmetry.
    repr : Representation
        Symmetry representation.
    """

    HDF5_ATTRIBUTES = ['real_space_operator', 'repr']
    HDF5_OPTIONAL = ['numeric']

    def __init__(
        self,
        *,
        rotation_matrix,
        translation_vector=None,
        repr_matrix,
        repr_has_cc=False,
        numeric=None
    ):

        self.real_space_operator = RealSpaceOperator(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            numeric=numeric
        )
        self.repr = Representation(
            matrix=repr_matrix, has_cc=repr_has_cc, numeric=self.numeric
        )

    @property
    def numeric(self):
        return self.real_space_operator.numeric

    @classmethod
    def from_real_space_operator(cls, *, real_space_operator, **kwargs):
        return cls(
            rotation_matrix=real_space_operator.rotation_matrix,
            translation_vector=real_space_operator.translation_vector,
            **kwargs
        )

    @classmethod
    def from_orbitals(
        cls, *, orbitals, real_space_operator, rotation_matrix_cartesian,
        numeric, **kwargs
    ):
        """
        Construct a (unitary) symmetry operation from the basis orbitals, real
        space operator and cartesian rotation matrix. The automatic construction
        of the representation matrix is used.

        Arguments
        ---------
        orbitals : Iterable[Orbital]
            The basis of orbitals with respect to which the represenation matrix
            is constructed.
        real_space_operator : RealSpaceOperator
            The real space operator of the matrix.
        rotation_matrix_cartesian : array
            The rotation matrix of the symmetry, in cartesian coordinates.
        numeric : bool
            Determines whether a numeric (numpy) or analytic (sympy)
            representation matrix is constructed.
        """
        from . import _get_repr_matrix
        if kwargs.get('repr_has_cc', False):
            raise NotImplementedError
        repr_matrix = _get_repr_matrix.get_repr_matrix(
            orbitals=orbitals,
            real_space_operator=real_space_operator,
            rotation_matrix_cartesian=rotation_matrix_cartesian,
            numeric=numeric
        )
        return cls.from_real_space_operator(
            real_space_operator=real_space_operator,
            repr_matrix=repr_matrix,
            numeric=numeric,
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
        """
        Defines the product of two symmetry operations.
        """
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
        """
        Get the order of a symmetry, i.e. the lowest power to which the symmetry
        is identity.
        """
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

    Arguments
    ---------
    rotation_matrix : array
        Describes the rotation matrix of the symmetry, in reduced coordinates
    translation_vector : array
        The translation vector of the symmetry.
    numeric : bool
        Specifies whether the symmetry operation contains a numeric or analytic
        values. By default, this is determined by the type of the rotation
        matrix.
    """

    HDF5_ATTRIBUTES = ['rotation_matrix', 'translation_vector']
    HDF5_OPTIONAL = ['numeric']

    def __init__(self, rotation_matrix, translation_vector=None, numeric=None):
        if numeric is None:
            numeric = not isinstance(rotation_matrix, sp.Matrix)
        self.numeric = numeric
        if numeric:
            rotation_matrix = np.array(rotation_matrix).astype(float)
        else:
            rotation_matrix = sp.Matrix(rotation_matrix)
        n, m = rotation_matrix.shape
        if n != m:
            raise ValueError('The rotation matrix must be square.')
        self.rotation_matrix = rotation_matrix

        if translation_vector is None:
            if self.numeric:
                translation_vector = np.zeros(n)
            else:
                translation_vector = sp.zeros(n, 1)
        else:
            if self.numeric:
                translation_vector = np.array(translation_vector).astype(float)
            else:
                translation_vector = sp.Matrix(translation_vector)
        if len(translation_vector) != n:
            raise ValueError(
                'The length of the translation vector must match the matrix dimension.'
            )
        self.translation_vector = translation_vector

    @classmethod
    def from_pymatgen(cls, pymatgen_op):
        return cls(
            rotation_matrix=pymatgen_op.rotation_matrix,
            translation_vector=pymatgen_op.translation_vector
        )

    def __matmul__(self, other):
        """
        Defines the product of real-space operations.
        """
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
        if self.numeric:
            r = np.array(r).astype(float)
        else:
            r = sp.Matrix(r)
        return self.rotation_matrix @ r + self.translation_vector

    @property
    def is_pure_translation(self):
        """
        Checks whether the operation is a pure translation, without rotation or
        reflection part.
        """
        n, m = self.rotation_matrix.shape
        assert n == m
        if self.numeric:
            return np.allclose(self.rotation_matrix, np.eye(n))
        else:
            return sp.eye(n, n).equals(self.rotation_matrix)

    @property
    def is_lattice_translation(self):
        """
        Checks if the operation is a lattice translation, i.e. a pure translation
        where the translation vector is a lattice vector.
        """
        if not self.is_pure_translation:
            return False
        if self.numeric:
            return np.allclose(
                self.translation_vector, np.round(self.translation_vector)
            )
        else:
            return self.translation_vector.equals(
                sp.Matrix([x.round() for x in self.translation_vector])
            )

    def __eq__(self, other):
        if self.numeric:
            return np.all(
                self.rotation_matrix == other.rotation_matrix
            ) and np.all(self.translation_vector == other.translation_vector)
        else:
            return self.rotation_matrix.equals(
                other.rotation_matrix
            ) and self.translation_vector.equals(other.translation_vector)


@export
@subscribe_hdf5('symmetry_representation.representation')
class Representation(SimpleHDF5Mapping, types.SimpleNamespace):
    r"""
    Describes an (anti-)unitary representation of a symmetry operation. For
    unitary symmetry, the representation is given as a unitary matrix :math:`U_g`. For
    anti-unitary symmetries, it is given as :math:`U_g \hat{K}`, where :math:`\hat{K}` is
    the complex conjugation operator.

    Arguments
    ---------
    matrix : array
        The unitary matrix of the representation.
    has_cc : bool
        Determines whether the representation contains complex conjugation
        (that is, whether it is anti-unitary).
    numeric : bool
        Determines if the representation matrix is numeric or analytic. By default
        this is determined from the type of the passed matrix.
    """
    HDF5_ATTRIBUTES = ['matrix', 'has_cc']
    HDF5_OPTIONAL = ['numeric']

    def __init__(self, matrix, has_cc=False, numeric=None):
        if numeric is None:
            numeric = not isinstance(matrix, sp.Matrix)
        if numeric:
            matrix = np.array(matrix).astype(complex)
            if not np.allclose(
                matrix @ matrix.T.conjugate(), np.eye(matrix.shape[0])
            ):
                raise ValueError(
                    'Input matrix is not unitary: {}'.format(matrix)
                )
        else:
            matrix = sp.Matrix(matrix)
            if not sp.eye(*matrix.shape).equals(matrix @ matrix.H):  # pylint: disable=not-an-iterable
                raise ValueError(
                    'Input matrix is not unitary: {}'.format(matrix)
                )
        self.matrix = matrix
        self.has_cc = has_cc
        self.numeric = numeric

    def __matmul__(self, other):
        """
        Defines the product of representations.
        """
        if self.numeric != other.numeric:
            raise ValueError(
                'Cannot multiply numeric and analytic representations.'
            )
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
        """
        Checks if a representation is the identity.
        """
        n, m = self.matrix.shape
        assert n == m
        if self.has_cc:
            return False
        if self.numeric:
            return np.allclose(self.matrix, np.eye(n))
        else:
            return sp.eye(n, n).equals(self.matrix)

    def __eq__(self, other):
        if self.numeric != other.numeric:
            return False
        if self.has_cc != other.has_cc:
            return False
        if self.numeric:
            return np.all(self.matrix == other.matrix)
        else:
            return self.matrix == other.matrix
