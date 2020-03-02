#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import re
from setuptools import setup, find_packages

import sys
if sys.version_info < (3, 6):
    raise 'must use Python version 3.6 or higher'

README = """A tool to describe symmetry operations and their representation."""

with open('./symmetry_representation/__init__.py', 'r') as f:
    MATCH_EXPR = "__version__[^'\"]+(['\"])([^'\"]+)"
    VERSION = re.search(MATCH_EXPR, f.read()).group(2).strip()

EXTRAS_REQUIRE = {
    'doc': ['sphinx', 'sphinx-rtd-theme', 'sphinx-click', 'ipython>=6.2'],
    'dev': [
        'pytest', 'pytest-cov', 'yapf==0.29', 'pre-commit',
        'prospector==1.2.0', 'pylint==2.4.4'
    ]
}

EXTRAS_REQUIRE['dev'] += EXTRAS_REQUIRE['doc']

setup(
    name='symmetry-representation',
    version=VERSION,
    url='https://symmetry-representation.greschd.ch',
    author='Dominik Gresch',
    author_email='greschd@gmx.ch',
    license='Apache 2.0',
    description='Provides an interface to describe symmetry representations.',
    install_requires=[
        'numpy', 'sympy', 'fsc.export', 'fsc.hdf5-io>=0.6', 'h5py', 'pymatgen',
        'click>=7.0'
    ],
    python_requires=">=3.6",
    extras_require=EXTRAS_REQUIRE,
    long_description=README,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English', 'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Development Status :: 4 - Beta'
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts':
        ['symmetry-repr = symmetry_representation._cli:cli'],
        'fsc.hdf5_io.load':
        ['symmetry_representation = symmetry_representation']
    },
)
