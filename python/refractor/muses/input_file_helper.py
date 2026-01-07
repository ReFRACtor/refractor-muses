from __future__ import annotations
from pathlib import Path
from .tes_file import TesFile
from loguru import logger
import os
import netCDF4
import re
import h5py  # type: ignore
from typing import Any, Iterator
import abc


class InputFileLogging:
    """Simple observer to add logging when a file is read"""

    def notify_file_input(
        self, ifile_hlp: InputFileHelper, fname: str | os.PathLike[str] | InputFilePath
    ) -> None:
        logger.opt(colors=True).debug(f"input file <red>{fname}</>")


class InputFilePath(object, metaclass=abc.ABCMeta):
    """This is just like a pathlib.Path for the OSP, GMAO or other base pathes. But
    we pull this out, because we may end up wanting to support other
    path like objects such as cloudpathlib
    (https://github.com/drivendataorg/cloudpathlib), or zipfile.Path.
    These are very much like the pathlib interface, but have differences
    because not all pathlib.Path functions are supported (see for example
    https://cloudpathlib.drivendata.org/stable/#supported-methods-and-properties).

    We only need a small number of functions supported. We pull these
    out here. We can then know exactly what functionality we need for
    supporting other path like interfaces.

    If you use InputFilePathImp, this is just a thin wrapper around pathlib.Path
    """

    # Want to be hashable. Note this is important, it gets used in OmiFmObjectCreator
    # ILS to eliminate duplicates
    @abc.abstractmethod
    def __eq__(self, x: object) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def __truediv__(self, rel_path: str | os.PathLike[str]) -> InputFilePath:
        '''Used to have a path like combination, so "path / 'filename'"'''
        raise NotImplementedError()

    @abc.abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError()

    @abc.abstractproperty
    def parent(self) -> InputFilePath:
        raise NotImplementedError()

    @abc.abstractmethod
    def absolute(self) -> InputFilePath:
        raise NotImplementedError()

    @abc.abstractmethod
    def resolve(self) -> InputFilePath:
        raise NotImplementedError()

    @abc.abstractmethod
    def as_posix(self) -> str:
        """Not positive on this one, we may need to remove this. Certainly a
        Path has this, but not sure about all pathlike classes we want to support"""
        raise NotImplementedError()

    @abc.abstractmethod
    def glob(self, pattern: str, **kwargs: Any) -> Iterator[InputFilePath]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abc.abstractproperty
    def name(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def sub_fname(self, pattern: str, repl: str) -> InputFilePath:
        """Replace a pattern with a replacement value for the name part of the
        path. So this is like self.parent / re.sub(pattern, repl, self.name), but
        might get implemented differently."""
        raise NotImplementedError()

    @classmethod
    def create_input_file_path(
        cls, fname: str | os.PathLike[str] | InputFilePath
    ) -> InputFilePath:
        """We frequently have arguments passed that are either pathlike or already
        an InputFilePath. This creation function handles this, either returning the
        InputFilePath if fname is already that, or creating a InputFilePathImp if it is
        a string or pathlike."""
        if isinstance(fname, InputFilePath):
            return fname
        return InputFilePathImp(fname)


class InputFilePathImp(InputFilePath):
    """InputFilePath that just uses a standard pathlib.Path to implement."""

    def __init__(
        self,
        base_path: str | os.PathLike[str] | InputFilePathImp,
        rel_path: str | os.PathLike[str] = ".",
    ) -> None:
        if isinstance(base_path, InputFilePathImp):
            self._base_path: Path = Path(base_path._base_path)
            self._rel_path: Path = Path(base_path._rel_path)
        else:
            self._base_path = Path(base_path).absolute()
            self._rel_path = Path(rel_path)

    def __eq__(self, x: object) -> bool:
        return isinstance(x, InputFilePath) and str(self) == str(x)

    def __hash__(self) -> int:
        return hash(str(self))

    def __truediv__(self, rel_path: str | os.PathLike[str]) -> InputFilePath:
        return InputFilePathImp(self._base_path, self._rel_path / rel_path)

    def exists(self) -> bool:
        return (self._base_path / self._rel_path).exists()

    @property
    def parent(self) -> InputFilePath:
        return InputFilePathImp(self._base_path, self._rel_path.parent)

    def absolute(self) -> InputFilePath:
        # File is already absolute, so just return self.
        # Note, only called in muses-py, but supply so we don't need to change that code.
        return self

    def resolve(self) -> InputFilePath:
        # Similar for resolve
        # Note, only called in muses-py, but supply so we don't need to change that code.
        return self

    def as_posix(self) -> str:
        return str(self)

    def glob(self, pattern: str, **kwargs: Any) -> Iterator[InputFilePath]:
        for x in (self._base_path / self._rel_path).glob(pattern, **kwargs):
            yield InputFilePathImp(self._base_path, x.relative_to(self._base_path))

    def __str__(self) -> str:
        return str(self._base_path / self._rel_path)

    @property
    def name(self) -> str:
        return (self._base_path / self._rel_path).name

    def sub_fname(self, pattern: str, repl: str) -> InputFilePath:
        return self.parent / re.sub(pattern, repl, self.name)


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

    def __init__(
        self,
        osp_dir: str | os.PathLike[str] | None = None,
        gmao_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.osp_dir = InputFilePathImp(
            osp_dir
            if osp_dir is not None
            else os.environ.get("MUSES_OSP_PATH", "../OSP")
        )
        self.gmao_dir = InputFilePathImp(
            gmao_dir
            if gmao_dir is not None
            else os.environ.get("MUSES_GMAO_PATH", "../GMAO")
        )
        self._observers: set[Any] = set()

    def add_observer(self, obs: Any) -> None:
        # Often we want weakref, so we don't prevent objects from
        # being deleted just because they are observing this. But in
        # this particular case, we actually do want to maintain the
        # lifetime. These observers will do things like write out
        # output, but have no real life outside of being attached to
        # this class.  It is easy enough to change this to weakref if
        # that proves useful
        self._observers.add(obs)
        if hasattr(obs, "notify_add"):
            obs.notify_add(self)

    def remove_observer(self, obs: Any) -> None:
        self._observers.discard(obs)
        if hasattr(obs, "notify_remove"):
            obs.notify_remove(self)

    def clear_observers(self) -> None:
        # We change self._observers, in our loop so grab a copy of the
        # list before we start
        lobs = list(self._observers)
        for obs in lobs:
            self.remove_observer(obs)

    def notify_file_input(self, fname: str | os.PathLike[str] | InputFilePath) -> None:
        if False:
            logger.debug(f"input file {fname}")
        for obs in self._observers:
            obs.notify_file_input(self, fname)

    def open_ncdf(
        self, fname: str | os.PathLike[str] | InputFilePath
    ) -> netCDF4.Dataset:
        """Small wrapper to call notify_file_input if InputFileMonitor is not None"""
        self.notify_file_input(fname)
        return netCDF4.Dataset(
            str(fname) if isinstance(fname, InputFilePath) else fname, "r"
        )

    def open_h5(self, fname: str | os.PathLike[str] | InputFilePath) -> h5py.File:
        """Small wrapper to call notify_file_input if InputFileMonitor is not None"""
        self.notify_file_input(fname)
        return h5py.File(str(fname) if isinstance(fname, InputFilePath) else fname, "r")

    def open_tes(self, fname: str | os.PathLike[str] | InputFilePath) -> TesFile:
        self.notify_file_input(fname)
        return TesFile(str(fname) if isinstance(fname, InputFilePath) else fname)


__all__ = ["InputFilePath", "InputFileHelper", "InputFileLogging"]
