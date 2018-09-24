#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines a decode function to read a legacy HDF5 format.
"""

import h5py
import numpy as np

from ._sym_op import SymmetryGroup, SymmetryOperation, Representation


def load(file_path):
    with h5py.File(file_path, 'r') as hdf5_handle:
        return _decode(hdf5_handle)


def _decode(hdf5_handle):
    """
    Construct the object stored at the given HDF5 location.
    """
    if 'symmetries' in hdf5_handle:
        return _decode_symgroup(hdf5_handle)
    elif 'rotation_matrix' in hdf5_handle:
        return _decode_symop(hdf5_handle)
    elif 'matrix' in hdf5_handle:
        return _decode_repr(hdf5_handle)
    elif '0' in hdf5_handle:
        return _decode_iterable(hdf5_handle)
    else:
        raise ValueError('File structure not understood.')


def _decode_iterable(hdf5_handle):
    return [_decode(hdf5_handle[key]) for key in sorted(hdf5_handle, key=int)]


def _decode_symgroup(hdf5_handle):
    return SymmetryGroup(
        symmetries=_decode_iterable(hdf5_handle['symmetries']),
        full_group=hdf5_handle['full_group'].value
    )


def _decode_symop(hdf5_handle):
    representation = _decode_repr(hdf5_handle['repr'])
    return SymmetryOperation(
        rotation_matrix=np.array(hdf5_handle['rotation_matrix']),
        repr_matrix=representation.matrix,
        repr_has_cc=representation.has_cc
    )


def _decode_repr(hdf5_handle):
    return Representation(
        matrix=np.array(hdf5_handle['matrix']),
        has_cc=hdf5_handle['has_cc'].value
    )
