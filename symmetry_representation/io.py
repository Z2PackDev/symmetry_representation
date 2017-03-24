#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

from collections.abc import Iterable
from singledispatch import singledispatch

import h5py
from fsc.export import export

from ._sym_op import SymmetryOperation, Representation

@export
def save(obj, file_path):
    """
    Saves symmetry_representation objects (or nested lists thereof) to a file, using the HDF5 format.
    """
    with h5py.File(filename, 'w') as hf:
        _encode(obj, hf)

@singledispatch
def _encode(obj, hf):
    raise ValueError('Cannot encode object of type {}'.format(type(obj)))

@_encode.register(SymmetryOperation)
def _(obj, hf):
    hf['rotation_matrix'] = np.array(obj.rotation_matrix)
    repr_hf = hf.create_group('repr')
    _encode(obj.repr, repr_hf)

@_encode.register(Representation)
def _(obj, hf):
    hf['has_cc'] = obj.has_cc
    hf['matrix'] = np.array(obj.matrix)

# @export
# def load(filename):
