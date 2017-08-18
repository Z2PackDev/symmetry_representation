import pytest
import numpy as np
import pymatgen as mg

import symmetry_representation as sr

@pytest.fixture
def all_symmetries(sample):
    symmetry, group = sr.io.load(sample('symmetries.hdf5'))
    return [symmetry] + group.symmetries

def test_is_compatible(sample, all_symmetries):
    structure = mg.Structure.from_file(sample('POSCAR'))
    for sym in all_symmetries:
        assert sr.is_compatible(structure=structure, symmetry=sym)

def test_not_all_compatible(sample, all_symmetries):
    structure = mg.Structure.from_file(sample('POSCAR_110_bi_0.04'))
    compatible_rotations = [
        np.eye(3),
        np.array([[0, 1, 0], [1, 0, 0], [-1, -1, -1]]),
        np.array([[1, 0, 0], [0, 1, 0], [-1, -1, -1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    ]
    def check(sym):
        for c in compatible_rotations:
            if np.allclose(c, sym.rotation_matrix):
                return True
        return False

    for sym in all_symmetries:
        assert sr.is_compatible(structure=structure, symmetry=sym) == check(sym)
