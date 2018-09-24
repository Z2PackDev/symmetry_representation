"""
Defines the functions to save and load objects to HDF5 files.
"""

import fsc.hdf5_io
from fsc.export import export

from . import _legacy_io

__all__ = ['save']

save = fsc.hdf5_io.save  # pylint: disable=invalid-name


@export
def load(hdf5_file):
    """
    Load an object from the given HDF5 file.

    Arguments
    ---------
    hdf5_file : str
        Path of the HDF5 file to load.
    """
    try:
        return fsc.hdf5_io.load(hdf5_file)
    except ValueError:
        return _legacy_io.load(hdf5_file)
