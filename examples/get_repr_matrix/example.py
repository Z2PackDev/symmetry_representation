#!/usr/bin/env python

# © 2017-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
import pymatgen as mg
import pymatgen.symmetry.analyzer
import symmetry_representation as sr

POS_In = (0, 0, 0)
POS_As = (0.25, 0.25, 0.25)

orbitals = []
for spin in (sr.SPIN_UP, sr.SPIN_DOWN):
    orbitals.extend([
        sr.Orbital(position=POS_In, function_string=fct, spin=spin)
        for fct in sr.WANNIER_ORBITALS['s'] + sr.WANNIER_ORBITALS['p']
    ])
    orbitals.extend([
        sr.Orbital(position=POS_As, function_string=fct, spin=spin)
        for fct in sr.WANNIER_ORBITALS['p']
    ])

structure = mg.Structure(
    lattice=[[0., 3.029, 3.029], [3.029, 0., 3.029], [3.029, 3.029, 0.]],
    species=['In', 'As'],
    coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
)

analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
symops = analyzer.get_symmetry_operations(cartesian=False)
symops_cart = analyzer.get_symmetry_operations(cartesian=True)

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

sr.io.save(symmetry_group, 'symmetries.hdf5')
