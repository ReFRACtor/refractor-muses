from pathlib import Path
import os
import netCDF4
import h5py  # type: ignore
from typing import Self


class InputFileHelper:
    """The retrieval opens a large number of files across the OSP
    directory and other places. It can be useful to have a central
    place where we know what has been accessed (e.g., to put metadata
    in an output file about the inputs used to generate it).

    This class provides a simple observer that we can do thing
    with. This class doesn't do anything with information supplied,
    but it provides a place where we can attach other objects that can
    make use of this information.

    We also have handling for opening TesFile and a few other types,
    mapping file names relate to the OSP or other directory to the
    final location. This is a little convenient, but the main reason
    for this is to support changing how we find these files.  For
    example, we may want to use cloud storage. Not really sure here
    what we want, but by having a simple interface here we have a
    place to make these changes (by introducing a new
    InputFileCloudHelper or something like that).

    Note that in general having these central global like classes is a
    bad idea, it can create coupling between all the different parts
    of the system. However, there are a few things that are naturally
    common across the system - the canonical example being the logger
    where we have the same logger object throughout the system.

    I think this class falls into the same category. We purposely have
    a very simple interface - the only things we do are
    notify_file_input and open various files. Any other file logic is
    in other places. This hopefully makes these easily replaceable.

    We can evaluate this over time. If it turns out this class is too
    centralizing/coupling we can rethink this. But the old py-retrieve
    had logic for handling files spread throughout the entire code
    base which wasn't a good solution. It was difficult to know what
    all was even being read in a retrieval. We'll try this design for
    now.
    """

    def notify_file_input(self, fname: Path) -> None:
        pass

    @classmethod
    def open_ncdf(
        cls, fname: str | os.PathLike[str], ifile_mon: Self | None
    ) -> netCDF4.Dataset:
        """Small wrapper to call notify_file_input if InputFileMonitor is not None"""
        if ifile_mon is not None:
            ifile_mon.notify_file_input(Path(fname))
        return netCDF4.Dataset(fname, "r")

    @classmethod
    def open_h5(
        cls, fname: str | os.PathLike[str], ifile_mon: Self | None
    ) -> h5py.File:
        """Small wrapper to call notify_file_input if InputFileMonitor is not None"""
        if ifile_mon is not None:
            ifile_mon.notify_file_input(Path(fname))
        return h5py.File(fname, "r")
