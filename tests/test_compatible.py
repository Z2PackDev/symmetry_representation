# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the symmetry compatibility checks.
"""

import pytest
import numpy as np
import pymatgen as mg

import symmetry_representation as sr


@pytest.fixture
def unstrained_structure(unstrained_poscar):
    return mg.Structure.from_file(unstrained_poscar)


@pytest.fixture
def strained_structure(strained_poscar):
    return mg.Structure.from_file(strained_poscar)


@pytest.fixture(params=[False, True])
def structure(request, unstrained_structure, strained_structure):  # pylint: disable=redefined-outer-name
    """
    Parametrized fixture for getting both the strained and unstrained structure.
    """
    if request.param:
        return strained_structure
    else:
        return unstrained_structure


@pytest.fixture
def all_symmetries(symmetries_file_content):
    symmetry, group = symmetries_file_content
    return [symmetry] + group.symmetries


def test_is_compatible(unstrained_structure, all_symmetries):  # pylint: disable=redefined-outer-name
    for sym in all_symmetries:
        assert sr.is_compatible(structure=unstrained_structure, symmetry=sym)


def test_not_all_compatible(strained_structure, all_symmetries):  # pylint: disable=redefined-outer-name
    """
    Test is_compatible against hard-coded result.
    """
    compatible_rotations = [
        np.eye(3),
        np.array([[0, 1, 0], [1, 0, 0], [-1, -1, -1]]),
        np.array([[1, 0, 0], [0, 1, 0], [-1, -1, -1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    ]

    def check(sym):
        for rot in compatible_rotations:
            if np.allclose(rot, sym.rotation_matrix):
                return True
        return False

    for sym in all_symmetries:
        assert sr.is_compatible(
            structure=strained_structure, symmetry=sym
        ) == check(sym)


def test_filter_compatible(unstrained_structure, all_symmetries):  # pylint: disable=redefined-outer-name
    assert (
        len(
            sr.filter_compatible(
                all_symmetries, structure=unstrained_structure
            )
        ) == len(all_symmetries)
    )


def test_filter_compatible_strained(strained_structure, all_symmetries):  # pylint: disable=redefined-outer-name
    assert (
        len(
            sr.filter_compatible(all_symmetries, structure=strained_structure)
        ) == 5
    )


def test_nested_filter(structure, symmetries_file_content):  # pylint: disable=redefined-outer-name
    print(structure)
    sr.filter_compatible(symmetries_file_content, structure=structure)
