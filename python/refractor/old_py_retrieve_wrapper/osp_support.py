from __future__ import annotations
from contextlib import contextmanager
import typing
from typing import Generator
import os
import tempfile

if typing.TYPE_CHECKING:
    from refractor.muses import InputFileHelper


@contextmanager
def osp_setup(
    ifile_hlp: InputFileHelper | None = None,
) -> Generator[None, None, None]:
    """Some of the readers assume the OSP is available as "../OSP". We
    are trying to get away from assuming we are in a run directory
    whenever we do things, it limits using the code in various
    contexts.  So this handles things by taking the osp_dir and
    setting up a temporary directory so things look like muses_py
    assumes.

    We can perhaps just move the muses-py code over at some point and
    handle this more cleanly, but for now we do this.
    """
    if ifile_hlp is None:
        from refractor.muses import InputFileHelper

        ifile_hlp = InputFileHelper()
    curdir = os.path.abspath(os.path.curdir)
    try:
        with tempfile.TemporaryDirectory() as tname:
            os.chdir(tname)
            os.symlink(str(ifile_hlp.osp_dir.path_for_muses_py), "OSP")
            os.mkdir("subdir")
            os.chdir("./subdir")
            yield
    finally:
        os.chdir(curdir)


__all__ = [
    "osp_setup",
]
