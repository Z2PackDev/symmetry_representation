"""
Test that the automatically generated symmetry representations match a reference.
"""

from numpy.testing import assert_allclose
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
