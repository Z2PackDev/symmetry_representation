#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>
"""
Defines the function for getting the spin representation of rotations.
"""

# pylint: disable=invalid-name

import numpy as np
import scipy.linalg as la


def _spin_reps(rotation_matrix_cartesian):
    """
    Calculates the spin rotation matrices. The formulas to determine the rotation axes and angles
    are taken from `here <http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf>`_.

    :param prep:   List that contains 3d rotation matrices.
    :type prep:    list(array)
    """
    # general representation of the D1/2 rotation about the axis (l,m,n) around the
    # angle phi
    rot = rotation_matrix_cartesian
    D12 = lambda l, m, n, phi: np.array([[np.cos(phi / 2.) - 1j * n * np.sin(phi / 2.), (-1j * l - m) * np.sin(phi / 2.)],
                                         [(-1j * l + m) * np.sin(phi / 2.), np.cos(phi / 2.) + 1j * n * np.sin(phi / 2.)]])

    n = np.zeros(3)
    tr = np.trace(rot)
    det = np.round(np.linalg.det(rot), 5)
    if det == 1.:  # rotations
        theta = np.arccos(0.5 * (tr - 1.))
        if theta != 0:
            n[0] = rot[2, 1] - rot[1, 2]
            n[1] = rot[0, 2] - rot[2, 0]
            n[2] = rot[1, 0] - rot[0, 1]
            if np.round(np.linalg.norm(n),
                        5) == 0.:  # theta = pi, that is C2 rotations
                e, v = la.eig(rot)
                n = v[:, list(np.round(e, 10)).index(1.)]
                spin = np.round(D12(n[0], n[1], n[2], np.pi), 15)
            else:
                n /= np.linalg.norm(n)
                spin = np.round(D12(n[0], n[1], n[2], theta), 15)
        else:  # case of unity
            spin = D12(0, 0, 0, 0)
    elif det == -1.:  # improper rotations and reflections
        theta = np.arccos(0.5 * (tr + 1.))
        if np.round(theta, 5) != np.round(np.pi, 5):
            n[0] = rot[2, 1] - rot[1, 2]
            n[1] = rot[0, 2] - rot[2, 0]
            n[2] = rot[1, 0] - rot[0, 1]
            if np.round(np.linalg.norm(n), 5) == 0.:  # theta = 0 (reflection)
                e, v = la.eig(rot)
                # normal vector is eigenvector to eigenvalue -1
                n = v[:, list(np.round(e, 10)).index(-1.)]
                # spin is a pseudovector!
                spin = np.round(D12(n[0], n[1], n[2], np.pi), 15)
            else:
                n /= np.linalg.norm(n)
                # rotation followed by reflection:
                spin = np.round(
                    np.dot(
                        D12(n[0], n[1], n[2], np.pi),
                        D12(n[0], n[1], n[2], theta)
                    ), 15
                )
        else:  # case of inversion (does not do anything to spin)
            spin = D12(0, 0, 0, 0)
    return np.array(spin)
