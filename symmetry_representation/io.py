import fsc.hdf5_io
from fsc.export import export

from . import _legacy_io

__all__ = ['save']

save = fsc.hdf5_io.save


@export
def load(hdf5_file):
    try:
        return fsc.hdf5_io.load(hdf5_file)
    except ValueError:
        return _legacy_io.load(hdf5_file)
