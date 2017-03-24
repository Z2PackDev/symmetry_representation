#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import numpy as np
import symmetry_representation as sr

test_1 = sr.SymmetryOperation(rotation_matrix=np.eye(3), repr_matrix=np.eye(2), repr_has_cc=True)

sr.io.save([test_1, test_1, test_1.repr], 'test.hdf5')
res = sr.io.load('test.hdf5')
print(res)
