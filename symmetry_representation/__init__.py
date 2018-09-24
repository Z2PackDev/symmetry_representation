#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A tool for describing symmetry operations and their representations.
"""

__version__ = '0.2.0'

from . import io
from ._sym_op import *
from ._compatibility import *
from ._get_repr_matrix import *

__all__ = [
    'io'
] + _sym_op.__all__ + _compatibility.__all__ + _get_repr_matrix.__all__  # pylint: disable=undefined-variable
