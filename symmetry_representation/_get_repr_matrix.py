import numpy as np
import sympy as sp
import scipy.linalg as la

from fsc.export import export

X, Y, Z = sp.symbols('x, y, z')


@export
def get_repr_matrix(*, system, real_space_operator, rotation_matrix_cartesian):
    print(
        _get_positions_mapping(
            system=system, real_space_operator=real_space_operator
        )
    )


def _get_positions_mapping(system, real_space_operator):
    positions = [orbital.position for orbital in system.orbitals]
    res = {}
    for i, pos1 in enumerate(positions):
        new_pos = real_space_operator.apply(pos1)
        res[i] = [
            j for j, pos2 in enumerate(positions)
            if _is_same_position(new_pos, pos2)
        ]
    return res


def _is_same_position(pos1, pos2):
    return np.isclose(_pos_distance(pos1, pos2), 0, atol=1e-6)


def _pos_distance(pos1, pos2):
    delta = np.array(pos1) - np.array(pos2)
    delta %= 1
    return la.norm(np.minimum(delta, 1 - delta))
