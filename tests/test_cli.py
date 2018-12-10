# © 2017-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the command-line interface ``symmetry-repr``.
"""

import tempfile

from click.testing import CliRunner

import symmetry_representation as sr
from symmetry_representation._cli import cli


def test_filter_symmetries_noop(unstrained_poscar, symmetries_file):
    """
    Test that filtering symmetries does nothing when the unstrained structure
    is given.
    """
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        runner.invoke(
            cli, [
                'filter-symmetries', '-s', symmetries_file, '-l',
                unstrained_poscar, '-o', out_file.name
            ],
            catch_exceptions=False
        )
        result = sr.io.load(out_file.name)
    reference = sr.io.load(symmetries_file)
    assert len(result) == len(reference)
    assert len(result[1].symmetries) == len(reference[1].symmetries)


def test_filter_symmetries_strained(strained_poscar, symmetries_file):
    """
    Test that filtering symmetries works when a strained structure is given.
    """
    runner = CliRunner()
    with tempfile.NamedTemporaryFile() as out_file:
        runner.invoke(
            cli, [
                'filter-symmetries', '-s', symmetries_file, '-l',
                strained_poscar, '-o', out_file.name
            ],
            catch_exceptions=False
        )
        result = sr.io.load(out_file.name)
    reference = sr.io.load(symmetries_file)
    assert len(result) == len(reference)
    assert len(result[1].symmetries) == 4
