"""
Defines the command-line tool ``symmetry-repr``.
"""

import click
import pymatgen as mg

from . import io
from . import filter_compatible


@click.group()
def cli():
    pass


@cli.command(
    short_help='Filter symmetries that are compatible with a given structure.'
)
@click.option(
    '--symmetries',
    '-s',
    type=click.Path(exists=True, dir_okay=False),
    default='symmetries.hdf5',
    help='File containing the symmetries (in HDF5 format).'
)
@click.option(
    '--lattice',
    '-l',
    type=click.Path(exists=True, dir_okay=False),
    default='lattice.cif',
    help='File containing the lattice structure.'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(dir_okay=False),
    default='symmetries_out.hdf5',
    help='File where the filtered symmetries are written (in HDF5 format).'
)
def filter_symmetries(symmetries, lattice, output):
    """
    Selects symmetries which are compatible with the given lattice.
    """
    click.echo(
        "Loading initial symmetries from file '{}'...".format(symmetries)
    )
    symmetries = io.load(symmetries)
    click.echo("Loading structure from file '{}'...".format(lattice))
    structure = mg.Structure.from_file(lattice)
    click.echo("Filtering symmetries...")
    filtered_symmetries = filter_compatible(symmetries, structure=structure)  # pylint: disable=assignment-from-no-return
    click.echo("Saving filtered symmetries to file '{}'...".format(output))
    io.save(filtered_symmetries, output)
    click.echo("Done!")
