#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

from collections.abc import Iterable
from functools import singledispatch

import h5py
import numpy as np
from fsc.export import export

from ._sym_op import SymmetryGroup, SymmetryOperation, Representation

@export
def save(obj, file_path):
    """
    Saves symmetry_representation objects (or nested lists thereof) to a file, using the HDF5 format.
    """
    with h5py.File(file_path, 'w') as hf:
        _encode(obj, hf)

@singledispatch
def _encode(obj, hf):
    raise ValueError('Cannot encode object of type {}'.format(type(obj)))

@_encode.register(Iterable)
def _(obj, hf):
    for i, part in enumerate(obj):
        sub_group = hf.create_group(str(i))
        _encode(part, sub_group)

@_encode.register(SymmetryGroup)
def _(obj, hf):
    _encode(obj.symmetries, hf.create_group('symmetries'))
    hf['full_group'] = obj.full_group

@_encode.register(SymmetryOperation)
def _(obj, hf):
    hf['rotation_matrix'] = np.array(obj.rotation_matrix)
    repr_hf = hf.create_group('repr')
    _encode(obj.repr, repr_hf)

@_encode.register(Representation)
def _(obj, hf):
    hf['has_cc'] = obj.has_cc
    hf['matrix'] = np.array(obj.matrix)

@export
def load(file_path):
    with h5py.File(file_path, 'r') as hf:
        return _decode(hf)

def _decode(hf):
    if 'symmetries' in hf:
        return _decode_symgroup(hf)
    elif 'rotation_matrix' in hf:
        return _decode_symop(hf)
    elif 'matrix' in hf:
        return _decode_repr(hf)
    elif '0' in hf:
        return _decode_iterable(hf)
    else:
        raise ValueError('File structure not understood.')

def _decode_iterable(hf):
    return [_decode(hf[key]) for key in hf]

def _decode_symgroup(hf):
    return SymmetryGroup(
        symmetries=_decode_iterable(hf['symmetries']),
        full_group=hf['full_group'].value
    )

def _decode_symop(hf):
    representation = _decode_repr(hf['repr'])
    return SymmetryOperation(
        rotation_matrix=np.array(hf['rotation_matrix']),
        repr_matrix=representation.matrix,
        repr_has_cc=representation.has_cc
    )

def _decode_repr(hf):
    return Representation(
        matrix=np.array(hf['matrix']),
        has_cc=hf['has_cc'].value
    )
