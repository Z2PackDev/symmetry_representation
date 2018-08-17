import types
from fractions import Fraction

import sympy as sp
import numpy as np
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


# TODO: Maybe remove System class
@export
@subscribe_hdf5('symmetry_representation.system')
class System(SimpleHDF5Mapping, types.SimpleNamespace):
    HDF5_ATTRIBUTES = ['orbitals']

    def __init__(self, *orbitals):
        self.orbitals = orbitals


@export
@subscribe_hdf5('symmetry_representation.orbital')
class Orbital(SimpleHDF5Mapping, types.SimpleNamespace):
    HDF5_ATTRIBUTES = ['position', 'function_string', 'has_spin']

    def __init__(self, *, position, function_string, spin=None):
        self.position = np.array(position) % 1
        self.function_string = function_string
        self.function = sp.sympify(self.function_string)

        if spin is None:
            spin = Spin(total=0, z_component=0)
        self.spin = spin


@export
@subscribe_hdf5('symmetry_representation.spin')
class Spin(SimpleHDF5Mapping, types.SimpleNamespace):
    HDF5_ATTRIBUTES = ['total', 'z_component']

    def __init__(self, *, total=Fraction(1, 2), z_component):
        self.total = Fraction(total)
        self.z_component = Fraction(z_component)

        for value in [self.total, self.z_component]:
            if value.denominator not in [1, 2]:
                raise ValueError('Invalid value for spin: {}'.format(value))

        if self.total > Fraction(1, 2):
            raise NotImplementedError(
                'Spin values larger than 1/2 are not implemented.'
            )

        if ((abs(self.z_component) > self.total)
            or ((self.total - self.z_component) % 1 != 0)):
            raise ValueError(
                'Spin z component {} is incompatible with total spin {}'.
                format(self.z_component, self.total)
            )
