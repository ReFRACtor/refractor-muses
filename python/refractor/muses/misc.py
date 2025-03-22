from __future__ import annotations
from contextlib import contextmanager
import tempfile
import os
from typing import Generator


@contextmanager
def osp_setup(
    osp_dir: str | os.PathLike[str] | None = None,
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
    if osp_dir is None:
        dname = os.path.abspath("../OSP")
    else:
        dname = os.path.abspath(str(osp_dir))
    curdir = os.path.abspath(os.path.curdir)
    try:
        with tempfile.TemporaryDirectory() as tname:
            os.chdir(tname)
            os.symlink(dname, "OSP")
            os.mkdir("subdir")
            os.chdir("./subdir")
            yield
    finally:
        os.chdir(curdir)
