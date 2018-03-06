#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    :param structure: Structure
    :type structure: pymatgen.Structure

    :type symmetry: SymmetryOperation
    """
    analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
    valid_sym_ops = analyzer.get_symmetry_operations(cartesian=False)
    for sym_op in valid_sym_ops:
        if (
            np.allclose(sym_op.translation_vector, [0.] * 3)
            and np.allclose(sym_op.rotation_matrix, symmetry.rotation_matrix)
        ):
            return True
    return False


@export
@singledispatch
def filter_compatible(symmetries, *, structure):
    raise ValueError(
        "Unrecognized type '{}' for 'symmetries'".format(type(symmetries))
    )


@filter_compatible.register(Iterable)
def _(symmetries, *, structure):
    filtered_syms = [
        filter_compatible(s, structure=structure) for s in symmetries
    ]
    return [s for s in filtered_syms if s is not None]


@filter_compatible.register(SymmetryOperation)
def _(symmetry, *, structure):
    if is_compatible(symmetry=symmetry, structure=structure):
        return symmetry
    else:
        return None


@filter_compatible.register(SymmetryGroup)
def _(symmetry_group, *, structure):
    filtered_syms = filter_compatible(
        symmetry_group.symmetries, structure=structure
    )
    if filtered_syms:
        return SymmetryGroup(
            symmetries=filtered_syms, full_group=symmetry_group.full_group
        )
    else:
        return None
