#!/usr/bin/env python

import numpy as np
import pymatgen as mg
import pymatgen.symmetry.analyzer
import symmetry_representation as sr

POS_In = (0, 0, 0)
POS_As = (0.25, 0.25, 0.25)
SPIN_UP = sr.Spin(total=0.5, z_component=0.5)
SPIN_DOWN = sr.Spin(total=0.5, z_component=-0.5)

system = sr.System(
    sr.Orbital(position=POS_In, function_string='1', spin=SPIN_UP),
    sr.Orbital(position=POS_In, function_string='x', spin=SPIN_UP),
    sr.Orbital(position=POS_In, function_string='y', spin=SPIN_UP),
    sr.Orbital(position=POS_In, function_string='z', spin=SPIN_UP),
    sr.Orbital(position=POS_As, function_string='x', spin=SPIN_UP),
    sr.Orbital(position=POS_As, function_string='y', spin=SPIN_UP),
    sr.Orbital(position=POS_As, function_string='z', spin=SPIN_UP),
    sr.Orbital(position=POS_In, function_string='1', spin=SPIN_DOWN),
    sr.Orbital(position=POS_In, function_string='x', spin=SPIN_DOWN),
    sr.Orbital(position=POS_In, function_string='y', spin=SPIN_DOWN),
    sr.Orbital(position=POS_In, function_string='z', spin=SPIN_DOWN),
    sr.Orbital(position=POS_As, function_string='x', spin=SPIN_DOWN),
    sr.Orbital(position=POS_As, function_string='y', spin=SPIN_DOWN),
    sr.Orbital(position=POS_As, function_string='z', spin=SPIN_DOWN),
)

structure = mg.Structure(
    lattice=[[0., 3.029, 3.029], [3.029, 0., 3.029], [3.029, 3.029, 0.]],
    species=['In', 'As'],
    coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
)

analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
symops = analyzer.get_symmetry_operations(cartesian=False)
symops_cart = analyzer.get_symmetry_operations(cartesian=True)

for sym_reduced, sym_cart in zip(symops, symops_cart):
    sr.get_repr_matrix(
        system=system,
        real_space_operator=sr.RealSpaceOperator.from_pymatgen(sym_reduced),
        rotation_matrix_cartesian=sym_cart.rotation_matrix
    )
