#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
# pylint: disable=cyclic-import
"""
A tool for describing symmetry operations and their representations.
"""

__version__ = '0.3.3'

from . import io
from ._sym_op import *
from ._compatibility import *
from ._get_repr_matrix import *

__all__ = [
    'io'
] + _sym_op.__all__ + _compatibility.__all__ + _get_repr_matrix.__all__  # pylint: disable=undefined-variable
