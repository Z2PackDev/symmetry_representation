#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Configuration file for pytest tests.
"""

import os
import json

import pytest

import symmetry_representation as sr


@pytest.fixture
def sample():
    """
    Returns the path to the sample of the given name.
    """
    def inner(name):
        return os.path.join(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'samples'
            ), name
        )

    return inner


@pytest.fixture
def test_name(request):
    """Returns module_name.function_name for a given test"""
    return request.module.__name__ + '/' + request._parent_request._pyfuncitem.name  # pylint: disable=protected-access


@pytest.fixture
def compare_data(request, test_name, scope="session"):  # pylint: disable=unused-argument,redefined-outer-name
    """Returns a function which either saves some data to a file or (if that file exists already) compares it to pre-existing data using a given comparison function."""
    def inner(compare_fct, data, tag=None):
        full_name = test_name + (tag or '')

        # get rid of json-specific quirks
        # store as string because I cannot add the decoder to the pytest cache
        data_str = json.dumps(data)
        data = json.loads(data_str)
        val = json.loads(request.config.cache.get(full_name, 'null'))

        if val is None:
            request.config.cache.set(full_name, data_str)
            raise ValueError('Reference data does not exist.')
        assert compare_fct(val, data)

    return inner


@pytest.fixture
def compare_equal(compare_data):  # pylint: disable=redefined-outer-name
    return lambda data, tag=None: compare_data(lambda x, y: x == y, data, tag)


@pytest.fixture
def symmetries_file(sample):  # pylint: disable=redefined-outer-name
    return sample('symmetries.hdf5')


@pytest.fixture
def symmetries_file_content(symmetries_file):  # pylint: disable=redefined-outer-name
    return sr.io.load(symmetries_file)


@pytest.fixture
def unstrained_poscar(sample):  # pylint: disable=redefined-outer-name
    return sample('POSCAR')


@pytest.fixture
def strained_poscar(sample):  # pylint: disable=redefined-outer-name
    return sample('POSCAR_110_bi_0.04')


@pytest.fixture(params=[True, False])
def numeric(request):
    return request.param
