#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pymatgen as mg
from fsc.export import export

from . import SymmetryGroup

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
            np.allclose(sym_op.translation_vector, [0.] * 3) and
            np.allclose(sym_op.rotation_matrix, symmetry.rotation_matrix)
        ):
            return True
    return False

@export
def filter_compatible(*, structure, symmetry_group):
    return SymmetryGroup(
        symmetries=[
            s for s in symmetry_group.symmetries if
            is_compatible(structure=structure, symmetry=s)
        ],
        full_group=symmetry_group.full_group
    )
