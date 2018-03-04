#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from functools import singledispatch

import h5py
import numpy as np

from ._sym_op import SymmetryGroup, SymmetryOperation, Representation


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
    return [_decode(hf[key]) for key in sorted(hf, key=int)]


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
        matrix=np.array(hf['matrix']), has_cc=hf['has_cc'].value
    )
