#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
try:
    from setuptools import setup
except:
    from distutils.core import setup

import sys
if sys.version_info < (3, 4):
    raise 'must use Python version 3.4 or higher'

readme = """A tool to describe symmetry operations and their representation."""

with open('./symmetry_representation/_version.py', 'r') as f:
    match_expr = "__version__[^'" + '"]+([' + "'" + r'"])([^\1]+)\1'
    version = re.search(match_expr, f.read()).group(2)

setup(
    name='symmetry-representation',
    version=version,
    url='http://z2pack.ethz.ch/symmetry-representation',
    author='Dominik Gresch',
    author_email='greschd@gmx.ch',
    description='Provides an interface to describe symmetry representations.',
    install_requires=[
        'numpy', 'sympy', 'fsc.export', 'fsc.hdf5-io>=0.3', 'h5py', 'pymatgen',
        'click'
    ],
    extras_require={
        'dev': [
            'pytest', 'pytest-cov', 'yapf', 'pre-commit', 'sphinx',
            'sphinx-rtd-theme', 'sphinx-click'
        ]
    },
    long_description=readme,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English', 'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Development Status :: 4 - Beta'
    ],
    license='GPL',
    packages=['symmetry_representation'],
    entry_points='''
        [console_scripts]
        symmetry-repr=symmetry_representation._cli:cli
    ''',
)
