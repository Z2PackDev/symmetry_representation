# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test that the automatically generated symmetry representations match a reference.
"""

import pytest

import numpy as np
from numpy.testing import assert_allclose

import sympy as sp

import pymatgen as mg
import pymatgen.symmetry.analyzer  # pylint: disable=unused-import

import symmetry_representation as sr


def test_auto_repr(sample):
    """
    Test that the symmetry group created with the automatic representation matrix
    is the matches a reference.
    """
    pos_In = (0, 0, 0)  # pylint: disable=invalid-name
    pos_As = (0.25, 0.25, 0.25)  # pylint: disable=invalid-name

    orbitals = []
    for spin in (sr.SPIN_UP, sr.SPIN_DOWN):
        orbitals.extend([
            sr.Orbital(position=pos_In, function_string=fct, spin=spin)
            for fct in sr.WANNIER_ORBITALS['s'] + sr.WANNIER_ORBITALS['p']
        ])
        orbitals.extend([
            sr.Orbital(position=pos_As, function_string=fct, spin=spin)
            for fct in sr.WANNIER_ORBITALS['p']
        ])

    symops, symops_cart = mg.loadfn(sample('InAs_symops.json'))

    symmetry_group = sr.SymmetryGroup(
        symmetries=[
            sr.SymmetryOperation.from_orbitals(
                orbitals=orbitals,
                real_space_operator=sr.RealSpaceOperator.
                from_pymatgen(sym_reduced),
                rotation_matrix_cartesian=sym_cart.rotation_matrix,
                numeric=True
            ) for sym_reduced, sym_cart in zip(symops, symops_cart)
        ],
        full_group=True
    )
    reference = sr.io.load(sample('symmetries_InAs.hdf5'))
    assert symmetry_group.full_group == reference.full_group
    for sym1, sym2 in zip(symmetry_group.symmetries, reference.symmetries):
        assert_allclose(
            sym1.real_space_operator.rotation_matrix,
            sym2.real_space_operator.rotation_matrix,
            atol=1e-12
        )
        assert_allclose(
            sym1.real_space_operator.translation_vector,
            sym2.real_space_operator.translation_vector,
            atol=1e-12
        )
        assert sym1.repr.has_cc == sym2.repr.has_cc
        assert_allclose(sym1.repr.matrix, sym2.repr.matrix, atol=1e-12)


@pytest.mark.parametrize(
    ('orbitals', 'result_repr_matrix'),
    [
        ([
            sr.Orbital(
                position=(0.1, 0.2, 0.3), function_string='1', spin=None
            )
        ], sp.Matrix([[1]])),
        ([
            sr.Orbital(
                position=(0.4, 0.2, 0.8), function_string='1', spin=sr.SPIN_UP
            ),
            sr.Orbital(
                position=(0.4, 0.2, 0.8),
                function_string='1',
                spin=sr.SPIN_DOWN
            )
        ], sp.Matrix([[0, -sp.I], [sp.I, 0]])),
        ([
            sr.Orbital(
                position=(0.1, 0.2, 0.3), function_string='x', spin=None
            )
        ], sp.Matrix([[1]])),
    ],
)
def test_time_reversal(orbitals, result_repr_matrix, numeric):
    """
    Test the generation of a representation matrix for time-reversal.
    """
    result = sr.get_time_reversal(orbitals=orbitals, numeric=numeric)
    if numeric:
        assert isinstance(result.repr.matrix, np.ndarray)
        assert np.allclose(
            result.repr.matrix,
            np.array(result_repr_matrix).astype(complex)
        )

        assert isinstance(
            result.real_space_operator.rotation_matrix, np.ndarray
        )
        assert np.allclose(
            result.real_space_operator.rotation_matrix, np.eye(3)
        )
        assert isinstance(
            result.real_space_operator.translation_vector, np.ndarray
        )
        assert np.allclose(
            result.real_space_operator.translation_vector, np.zeros(3)
        )
    else:
        assert isinstance(result.repr.matrix, sp.Matrix)
        assert result.repr.matrix == result_repr_matrix

        assert isinstance(
            result.real_space_operator.rotation_matrix, sp.Matrix
        )
        assert result.real_space_operator.rotation_matrix.equals(sp.eye(3, 3))
        assert isinstance(
            result.real_space_operator.translation_vector, sp.Matrix
        )
        assert result.real_space_operator.translation_vector.equals(
            sp.zeros(3, 1)
        )
    assert result.repr.has_cc


@pytest.mark.parametrize(['orbitals', 'rotation_matrix', 'reference'], [
    ([sr.Orbital(position=(0, 0, 0), function_string='1', spin=None)
      ], np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), sp.eye(1, 1)),
    ([
        sr.Orbital(position=(0, 0, 0), function_string='x', spin=None),
        sr.Orbital(position=(0, 0, 0), function_string='y', spin=None)
    ], np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), sp.Matrix([[0, 1], [1, 0]]
                                                              )),
    ([
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_UP),
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_DOWN)
    ], np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
     (sp.sqrt(2) / 2) * sp.Matrix([[0, -1 + sp.I], [1 + sp.I, 0]])),
    ([
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_UP),
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_DOWN)
    ], np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
     (sp.sqrt(2) / 2) * sp.Matrix([[1 - sp.I, 0], [0, 1 + sp.I]])),
    ([
        sr.Orbital(
            position=(0.1, 0.2, 0.3), function_string='x', spin=sr.SPIN_UP
        ),
        sr.Orbital(
            position=(0.4, 0.9, 0.1), function_string='y', spin=sr.SPIN_DOWN
        )
    ], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), sp.eye(2, 2)),
    ([
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_UP),
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_DOWN)
    ], np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), sp.eye(2, 2)),
    ([
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_UP),
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_DOWN)
    ], np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
                ), sp.Matrix([[-sp.I, 0], [0, sp.I]])),
    ([
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_UP),
        sr.Orbital(position=(0, 0, 0), function_string='1', spin=sr.SPIN_DOWN)
    ],
     np.array([[0.5, np.sin(2 * np.pi / 3), 0],
               [-np.sin(2 * np.pi / 3), 0.5, 0], [0, 0, -1]]),
     sp.Matrix([[-sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2, 0],
                [0, -sp.Rational(1, 2) - sp.sqrt(3) * sp.I / 2]])),
])
def test_simple_repr(orbitals, rotation_matrix, reference, numeric):
    """
    Test some representation matrices for simple rotations / rotoreflections,
    where the cartesian and fractional coordinates are the same.
    """
    result = sr.get_repr_matrix(
        orbitals=orbitals,
        real_space_operator=sr.RealSpaceOperator(
            rotation_matrix=rotation_matrix
        ),
        rotation_matrix_cartesian=rotation_matrix,
        numeric=numeric
    )
    if numeric:
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.array(reference).astype(complex))
    else:
        assert isinstance(result, sp.Matrix)
        assert result.equals(reference)
