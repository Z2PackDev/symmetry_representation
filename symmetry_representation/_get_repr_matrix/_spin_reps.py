#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2018, ETH Zurich, Institut für Theoretische Physik
# Authors: Georg Winkler, Dominik Gresch <greschd@gmx.ch>
"""
Defines the function for getting the spin representation of rotations.
"""

# pylint: disable=invalid-name

import numpy as np
import sympy as sp
import scipy.linalg as la


def _spin_reps(rotation_matrix_cartesian, numeric):
    """
    Calculates the spin rotation matrices. The formulas to determine the rotation axes and angles
    are taken from `here <http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf>`_.

    :param prep:   List that contains 3d rotation matrices.
    :type prep:    list(array)
    """
    if numeric:
        return _spin_reps_numeric(rotation_matrix_cartesian)
    else:
        return _spin_reps_analytic(rotation_matrix_cartesian)


def _spin_reps_numeric(rotation_matrix_cartesian):
    """
    Generate the spin representation matrices for the case of numeric (numpy array)
    output.
    """

    # general representation of the D1/2 rotation about the axis (l,m,n) around the
    # angle phi
    rot = rotation_matrix_cartesian

    def D12(l, m, n, phi):
        return np.array([[
            np.cos(phi / 2.) - 1j * n * np.sin(phi / 2.),
            (-1j * l - m) * np.sin(phi / 2.)
        ],
                         [(-1j * l + m) * np.sin(phi / 2.),
                          np.cos(phi / 2.) + 1j * n * np.sin(phi / 2.)]])

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


def _spin_reps_analytic(rotation_matrix_cartesian):  # pylint: disable=too-many-branches
    """
    Generate the spin representation matrices for the case of analytic (sympy)
    output.
    """
    # general representation of the D1/2 rotation about the axis (l,m,n) around the
    # angle phi
    rot = rotation_matrix_cartesian

    def D12(l, m, n, phi):
        phi_half = phi / sp.Integer(2)
        return sp.Matrix([[
            sp.cos(phi_half) - sp.I * n * sp.sin(phi_half),
            (-sp.I * l - m) * sp.sin(phi_half)
        ],
                          [(-sp.I * l + m) * sp.sin(phi_half),
                           sp.cos(phi_half) + sp.I * n * sp.sin(phi_half)]])

    n = sp.zeros(3, 1)
    tr = rot.trace()
    det = rot.det()
    if det == 1.:  # rotations
        theta = sp.acos(sp.Rational(1, 2) * (tr - 1))
        if theta != 0:
            n[0] = rot[2, 1] - rot[1, 2]
            n[1] = rot[0, 2] - rot[2, 0]
            n[2] = rot[1, 0] - rot[0, 1]
            if n.norm() == 0:  # theta = pi, that is C2 rotations
                for val, _, basis in rot.eigenvects():
                    if val == 1:
                        n = basis[0].normalized()
                        break
                else:
                    raise ValueError(
                        'No eigenvectors with eigenvalue 1 found.'
                    )
                spin = D12(n[0], n[1], n[2], sp.pi)
            else:
                n = n.normalized()
                spin = D12(n[0], n[1], n[2], theta)
        else:  # case of unity
            spin = D12(0, 0, 0, 0)
    elif det == -1:  # improper rotations and reflections
        theta = sp.acos(sp.Rational(1, 2) * (tr + 1))
        if theta != sp.pi:
            n[0] = rot[2, 1] - rot[1, 2]
            n[1] = rot[0, 2] - rot[2, 0]
            n[2] = rot[1, 0] - rot[0, 1]
            if n.norm() == 0:  # theta = 0 (reflection)
                for val, _, basis in rot.eigenvects():
                    if val == -1:
                        n = basis[0].normalized()
                        break
                else:
                    raise ValueError(
                        'No eigenvectors with eigenvalue -1 found.'
                    )
                # spin is a pseudovector!
                spin = D12(*n, sp.pi)
            else:
                n = n.normalized()
                # rotation followed by reflection:
                spin = D12(n[0], n[1], n[2],
                           sp.pi) @ D12(n[0], n[1], n[2], theta)
        else:  # case of inversion (does not do anything to spin)
            spin = D12(0, 0, 0, 0)
    return spin
