"""
Defines orbitals and their spin, which are used as basis for constructing
representation matrices.
"""

import types
from fractions import Fraction
from collections import namedtuple

import sympy as sp
import numpy as np
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('symmetry_representation.orbital')
class Orbital(SimpleHDF5Mapping, types.SimpleNamespace):
    """
    Defines a basis orbital.

    Attributes
    ----------
    position : array
        Position of the orbital, in reduced coordinates.
    function_string : str
        String describing the function of the orbital.
    spin : Spin
        Spin component of the orbital.
    """

    HDF5_ATTRIBUTES = ['position', 'function_string', 'spin']

    def __init__(self, *, position, function_string, spin=None):
        self.position = np.array(position) % 1
        self.function_string = function_string
        self.function = sp.sympify(self.function_string)

        if spin is None:
            spin = Spin(total=0, z_component=0)
        self.spin = spin


_SpinBase = namedtuple('_SpinBase', ['total', 'z_component'])


@export
@subscribe_hdf5('symmetry_representation.spin')
class Spin(SimpleHDF5Mapping, _SpinBase):
    """
    Describes the spin component of an orbital.

    Attributes
    ----------
    total : Fraction
        The total spin.
    z_component : Fraction
        The component of spin in z direction.
    """
    HDF5_ATTRIBUTES = ['total', 'z_component']

    def __new__(cls, total=Fraction(0), z_component=Fraction(0)):
        total = Fraction(total)
        z_component = Fraction(z_component)

        for value in [total, z_component]:
            if value.denominator not in [1, 2]:
                raise ValueError('Invalid value for spin: {}'.format(value))

        if total > Fraction(1, 2):
            raise NotImplementedError(
                'Spin values larger than 1/2 are not implemented.'
            )

        if ((abs(z_component) > total) or ((total - z_component) % 1 != 0)):
            raise ValueError(
                'Spin z component {} is incompatible with total spin {}'.
                format(z_component, total)
            )

        return super().__new__(cls, total=total, z_component=z_component)
