"""
Defines the functionality for automatically creating representation matrices.
"""

from fractions import Fraction

import numpy as np
import sympy as sp
import scipy.linalg as la

from fsc.export import export

from .._sym_op import RealSpaceOperator, SymmetryOperation

from ._orbitals import Spin
from ._orbital_constants import SPIN_UP, SPIN_DOWN
from ._spin_reps import _spin_reps
from ._expr_utils import _get_substitution, _expr_to_vector


@export
def get_time_reversal(*, orbitals, numeric):
    """
    Create the symmetry operation for time-reversal.

    Arguments
    ---------
    orbitals : List(Orbital)
        Basis orbitals with respect to which the representation should be created.
    numeric : bool
        Flag to determine whether numeric (numpy) or symbolic (sympy) computation
        should be used.
    """
    dim = len(orbitals[0].position)
    if numeric:
        unity = np.eye(dim)
    else:
        unity = sp.eye(dim)
    real_space_operator = RealSpaceOperator(rotation_matrix=unity)
    repr_matrix = _get_repr_matrix_impl(
        orbitals=orbitals,
        real_space_operator=real_space_operator,
        rotation_matrix_cartesian=unity,
        spin_rot_function=_apply_spin_time_reversal,
        numeric=numeric,
    )
    return SymmetryOperation.from_real_space_operator(
        real_space_operator=real_space_operator,
        repr_matrix=repr_matrix,
        repr_has_cc=True
    )


@export
def get_repr_matrix(
    *, orbitals, real_space_operator, rotation_matrix_cartesian, numeric
):
    """
    Create the representation matrix for a unitary operator.

    Arguments
    ---------
    orbitals : List(Orbital)
        Basis orbitals with respect to which the representation should be created.
    real_space_operator : .RealSpaceOperator
        Real-space operator of the symmetry operation.
    rotation_matrix_cartesian : np.array or sp.Matrix
        Rotation matrix of the symmetry operation in cartesian coordinates.
    numeric : bool
        Flag to determine whether numeric (numpy) or symbolic (sympy) computation
        should be used.
    """
    return _get_repr_matrix_impl(
        orbitals=orbitals,
        real_space_operator=real_space_operator,
        rotation_matrix_cartesian=rotation_matrix_cartesian,
        spin_rot_function=_apply_spin_rotation,
        numeric=numeric
    )


def _get_repr_matrix_impl(  # pylint: disable=too-many-locals
    *, orbitals, real_space_operator, rotation_matrix_cartesian,
    spin_rot_function, numeric
):
    """
    Implements the functionality for getting the representation matrix. The
    function to get the spin rotation matrix can be defined, to allow use in the
    time-reversal case.

    Arguments
    ---------
    orbitals : List(Orbital)
        Basis orbitals with respect to which the representation should be created.
    real_space_operator : .RealSpaceOperator
        Real-space operator of the symmetry operation.
    rotation_matrix_cartesian : np.array or sp.Matrix
        Rotation matrix of the symmetry operation in cartesian coordinates.
    spin_rot_function : Callable
        A function which applies the spin rotation, given the initial spin and
        cartesian rotation matrix.
    numeric : bool
        Flag to determine whether numeric (numpy) or symbolic (sympy) computation
        should be used.
    """

    orbitals = list(orbitals)

    positions_mapping = _get_positions_mapping(
        orbitals=orbitals, real_space_operator=real_space_operator
    )
    repr_matrix = sp.zeros(len(orbitals))

    expr_substitution = _get_substitution(rotation_matrix_cartesian)
    for i, orb in enumerate(orbitals):
        res_pos_idx = positions_mapping[i]
        spin_res = spin_rot_function(
            rotation_matrix_cartesian=rotation_matrix_cartesian, spin=orb.spin
        )

        new_func = orb.function.subs(expr_substitution, simultaneous=True)
        for new_spin, spin_value in spin_res.items():
            res_pos_idx_reduced = [
                idx for idx in res_pos_idx if orbitals[idx].spin == new_spin
            ]
            func_basis_reduced = [
                orbitals[idx].function for idx in res_pos_idx_reduced
            ]
            func_vec = _expr_to_vector(
                new_func, basis=func_basis_reduced, numeric=numeric
            )
            func_vec_norm = la.norm(np.array(func_vec).astype(complex))
            if not np.isclose(func_vec_norm, 1):
                raise ValueError(
                    'Norm {} of vector {} for expression {} created from orbital {} is not one.\nCartesian rotation matrix: {}'
                    .format(
                        func_vec_norm, func_vec, new_func, orb,
                        rotation_matrix_cartesian
                    )
                )
            for idx, func_value in zip(res_pos_idx_reduced, func_vec):
                repr_matrix[idx, i] += func_value * spin_value
    # check that the matrix is unitary
    repr_matrix_numeric = np.array(repr_matrix).astype(complex)
    if not np.allclose(
        repr_matrix_numeric @ repr_matrix_numeric.conj().T,
        np.eye(*repr_matrix_numeric.shape)
    ):
        max_mismatch = np.max(
            np.abs(
                repr_matrix_numeric @ repr_matrix_numeric.conj().T -
                np.eye(*repr_matrix_numeric.shape)
            )
        )
        raise ValueError(
            'Representation matrix is not unitary. Maximum mismatch to unity: {}'
            .format(max_mismatch)
        )
    if numeric:
        return repr_matrix_numeric
    else:
        return repr_matrix


def _get_positions_mapping(orbitals, real_space_operator):
    """
    Calculates the mapping from initial to final positions, given the orbital
    basis and real space operator.
    """
    positions = [orbital.position for orbital in orbitals]
    res = {}
    for i, pos1 in enumerate(positions):
        new_pos = real_space_operator.apply(pos1)
        res[i] = [
            j for j, pos2 in enumerate(positions)
            if _is_same_position(new_pos, pos2)
        ]
    return res


def _is_same_position(pos1, pos2):
    """
    Checks if two positions are the same, up to a lattice vector.
    """
    return np.isclose(_pos_distance(pos1, pos2), 0, atol=1e-6)


def _pos_distance(pos1, pos2):
    """
    Returns the periodic distance between two positions.
    """
    delta = np.array(pos1) - np.array(pos2)
    delta %= 1
    return la.norm(np.array(np.minimum(delta, 1 - delta)).astype(float))


def _apply_spin_time_reversal(rotation_matrix_cartesian, spin):
    """
    Applies the effect of time-reversal on a spin.
    """
    dim = rotation_matrix_cartesian.shape[0]
    assert np.all(rotation_matrix_cartesian == np.eye(dim)
                  ) or rotation_matrix_cartesian == sp.eye(dim)
    if spin.total == 0:
        return {spin: 1}

    # time-reversal is represented by sigma_y * complex conjugation
    elif spin.total == Fraction(1, 2):
        if spin == SPIN_UP:
            return {SPIN_DOWN: 1j}
        else:
            assert spin == SPIN_DOWN
            return {SPIN_UP: -1j}
    else:
        raise NotImplementedError('Spins larger than 1/2 are not implemented.')


def _apply_spin_rotation(rotation_matrix_cartesian, spin):
    """
    Applies the effect of a given rotation on spin.
    """
    if spin.total == 0:
        return {spin: 1}
    elif spin.total == Fraction(1, 2):
        spin_vec = _spin_to_vector(spin)
        spin_vec_res = _spin_reps(rotation_matrix_cartesian).dot(spin_vec)
        return _vec_to_spins(spin_vec_res)
    else:
        raise NotImplementedError('Spins larger than 1/2 are not implemented.')


def _spin_to_vector(spin):
    """
    Helper function to convert a spin to vector representation.
    """
    size = int(2 * spin.total + 1)
    idx = int(spin.total - spin.z_component)
    res = np.zeros(size)
    res[idx] = 1
    return res


def _vec_to_spins(vec):
    """
    Helper function to convert a vector back to spins. The result is a dictionary,
    where the key is the spin, and the value is its vector component.
    """
    total = Fraction(vec.size - 1, 2)
    res = {}
    for i, val in enumerate(vec):
        if val != 0:
            res[Spin(total=total, z_component=total - i)] = val
    return res
