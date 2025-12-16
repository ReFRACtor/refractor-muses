from pathlib import Path
import os
import netCDF4
import h5py  # type: ignore
from typing import Self

class InputFileMonitor:
    '''The retrieval opens a large number of files across the OSP directory and
    other places. It can be useful to have a central place where we know what
    has been accessed (e.g., to put metadata in an output file about the inputs
    used to generate it).

    This class provides a simple observer that we can do thing with. This class
    doesn't do anything with information supplied, but it provides a place where
    we can attach other objects that can make use of this information.
    '''
    def notify_file_input(self, fname: Path) -> None:
        pass

    @classmethod
    def open_ncdf(cls, fname: str | os.PathLike[str],
                  ifile_mon: Self | None) -> netCDF4.Dataset:
        '''Small wrapper to call notify_file_input if InputFileMonitor is not None'''
        if ifile_mon is not None:
            ifile_mon.notify_file_input(Path(fname))
        return netCDF4.Dataset(fname, "r")

    @classmethod
    def open_h5(cls, fname: str | os.PathLike[str],
                ifile_mon: Self | None) -> h5py.File:
        '''Small wrapper to call notify_file_input if InputFileMonitor is not None'''
        if ifile_mon is not None:
            ifile_mon.notify_file_input(Path(fname))
        return h5py.File(fname,"r")
    

