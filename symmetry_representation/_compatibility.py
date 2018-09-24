#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines functions to determine if symmetries are compatible with a given structure.
"""

from collections.abc import Iterable
from functools import singledispatch

import numpy as np
import pymatgen as mg
from fsc.export import export

from . import SymmetryGroup, SymmetryOperation


@export
def is_compatible(*, structure, symmetry):
    """
    Checks whether a given symmetry's real space action (rotation + translation vector) is consistent with a given structure.

    Arguments
    ---------
    structure : pymatgen.Structure
        The crystal structure.
    symmetry : SymmetryOperation
        The symmetry operation that is checked for compatibility.
    """
    analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
    valid_sym_ops = analyzer.get_symmetry_operations(cartesian=False)
    for sym_op in valid_sym_ops:
        if (
            np.
            allclose(sym_op.translation_vector, symmetry.translation_vector)
            and np.allclose(sym_op.rotation_matrix, symmetry.rotation_matrix)
        ):
            return True
    return False


@export
@singledispatch
def filter_compatible(symmetries, *, structure):  # pylint: disable=unused-argument
    """
    Returns the symmetries which are compatible with the given structure.

    Arguments
    ---------
    symmetries : SymmetryGroup, Iterable
        The symmetries which should be checked for compatibility. If a :class:`.SymmetryGroup`
        is given, the result is also given as a :class:`.SymmetryGroup`.
    structure : pymatgen.Structure
        The crystal structure.
    """
    raise ValueError(
        "Unrecognized type '{}' for 'symmetries'".format(type(symmetries))
    )


@filter_compatible.register(Iterable)
def _(symmetries, *, structure):  # pylint: disable=missing-docstring
    filtered_syms = [
        filter_compatible(s, structure=structure) for s in symmetries
    ]
    return [s for s in filtered_syms if s is not None]


@filter_compatible.register(SymmetryOperation)
def _(symmetry, *, structure):  # pylint: disable=missing-docstring
    if is_compatible(symmetry=symmetry, structure=structure):
        return symmetry
    else:
        return None


@filter_compatible.register(SymmetryGroup)
def _(symmetry_group, *, structure):  # pylint: disable=missing-docstring
    filtered_syms = filter_compatible(  # pylint: disable=assignment-from-no-return
        symmetry_group.symmetries, structure=structure
    )
    if filtered_syms:
        return SymmetryGroup(
            symmetries=filtered_syms, full_group=symmetry_group.full_group
        )
    else:
        return None
