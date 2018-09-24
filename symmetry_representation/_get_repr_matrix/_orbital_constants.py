"""
Defines constants which are useful for creating orbitals.
"""

from fractions import Fraction

from ._orbitals import Spin

__all__ = ['WANNIER_ORBITALS', 'SPIN_UP', 'SPIN_DOWN', 'NO_SPIN']

WANNIER_ORBITALS = {
    's': ['1'],
    'p': ['z', 'x', 'y'],
    'd': ['z**2', 'x * z', 'y * z', 'x**2 - y**2', 'x * y'],
    'f': [
        'z**3', 'x * z**2', 'y * z**2', 'z * (x**2 - y**2)', 'x * y * z',
        'x * (x**2 - 3 * y**2)', 'y * (3 * x**2 - y**2)'
    ],
    'sp': ['1 + x', '1 - x'],
    'sp2': [
        '1 / sqrt(3) - x / sqrt(6) + y / sqrt(2)',
        '1 / sqrt(3) - x / sqrt(6) - y / sqrt(2)',
        '1 / sqrt(3) + 2 * x / sqrt(6)'
    ],
    'sp3':
    ['1 + x + y + z', '1 + x - y  - z', '1 - x + y - z', '1 - x - y + z'],
    'sp3d': [
        '1 / sqrt(3) - x / sqrt(6) + y / sqrt(2)',
        '1 / sqrt(3) - x / sqrt(6) - y / sqrt(2)',
        '1 / sqrt(3) + 2 * x / sqrt(6)', 'z / sqrt(2) + z**2 / sqrt(2)',
        '-z / sqrt(2) + z**2 / sqrt(2)'
    ],
    'sp3d2': [
        '1 / sqrt(6) - x / sqrt(2) - z**2 / sqrt(12) + (x**2 - y**2) / 2',
        '1 / sqrt(6) + x / sqrt(2) - z**2 / sqrt(12) + (x**2 - y**2) / 2',
        '1 / sqrt(6) - y / sqrt(2) - z**2 / sqrt(12) - (x**2 - y**2) / 2',
        '1 / sqrt(6) + y / sqrt(2) - z**2 / sqrt(12) - (x**2 - y**2) / 2',
        '1 / sqrt(6) - z / sqrt(2) + z**2 / sqrt(3)',
        '1 / sqrt(6) + z / sqrt(2) + z**2 / sqrt(3)',
    ]
}

NO_SPIN = Spin(total=0, z_component=0)
SPIN_UP = Spin(total=Fraction(1, 2), z_component=Fraction(1, 2))
SPIN_DOWN = Spin(total=Fraction(1, 2), z_component=Fraction(-1, 2))
