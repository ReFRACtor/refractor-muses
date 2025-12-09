from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import shutil
from functools import cache
from .mpy import have_muses_py, mpy_cli_options


@cache
def vlidort_cli_from_path() -> Path:
    """We use to need to point to vlidort_cli. There might still be special cases
    where we want to do this, so muses_py_call will allow this to get passed in. But
    most of the time, you want to just grab the one in the path. This figure this out
    """
    t = shutil.which("vlidort_cli")
    if t is None:
        raise RuntimeError("Can't find vlidort_cli in the path")
    return Path(t).parent


@cache
def ring_cli_from_path() -> Path:
    """We use to need to point to ring_cli. There might still be special cases
    where we want to do this, so muses_py_call will allow this to get passed in. But
    most of the time, you want to just grab the one in the path. This figure this out
    """
    t = shutil.which("ring_cli")
    if t is None:
        raise RuntimeError("Can't find ring_cli in the path")
    return Path(t).parent


@contextmanager
def muses_py_call(
    rundir: str | os.PathLike[str],
    vlidort_cli: str | os.PathLike[str] | None = None,
    ring_cli: str | os.PathLike[str] | None = None,
    debug: bool = False,
    vlidort_nstokes: int = 2,
    vlidort_nstreams: int = 4,
    # For MusesForwardModel, we don't need to be in the rundir. But for a number
    # of old_py_retrieve tests, we do since we are using older code
    change_to_rundir: bool = False,
) -> Generator[None, None, None]:
    """There is some cookie cutter code needed to call a py_retrieve function.
    We collect that here as a context manager, so you can just do something
    like:

    with muses_py_call(rundir):
       mpy.run_retrieval(...)

    without all the extra stuff. Note that we handle changing to the rundir
    before calling, so you don't need to do that before hand (or just
    pass "." if for whatever reason you have already done that).

    The "debug" flag turns on the per iteration directory diagnostics in omi and
    tropomi. I don't think it actually changes anything else, as far as I can tell
    only this get changed by the flags. The per iteration stuff is needed for some
    of the diagnostics (e.g., RefractorTropOmiFmMusesPy.surface_albedo)."""
    curdir = os.getcwd()
    old_run_dir = os.environ.get("MUSES_DEFAULT_RUN_DIR")
    if vlidort_cli is None and have_muses_py:
        vlidort_cli = vlidort_cli_from_path()
    if ring_cli is None and have_muses_py:
        ring_cli = ring_cli_from_path()
    if have_muses_py:
        assert mpy_cli_options is not None
        old_vlidort_cli = mpy_cli_options.get("vlidort_cli")
        old_ring_cli = mpy_cli_options.get("ring_cli")
        old_debug = mpy_cli_options.get("debug")
        old_vlidort_nstokes = mpy_cli_options.vlidort.get("nstokes")
        old_vlidort_nstreams = mpy_cli_options.vlidort.get("nstreams")
    try:
        # This gets uses on a couple of top level py-retrieve functions.
        # (script_retrieval_setup_ms.py, get_emis_uwis.py. I don't think
        # we call either of these anywhere anymore, but go ahead and set
        # this anyways since it doesn't hurt.
        os.environ["MUSES_DEFAULT_RUN_DIR"] = os.path.abspath(str(rundir))
        if change_to_rundir:
            os.chdir(rundir)
        if have_muses_py:
            assert mpy_cli_options is not None
            # These variables control running vlidort and ring (the raman scattering
            # executable).
            if vlidort_cli is not None:
                mpy_cli_options.vlidort_cli = str(vlidort_cli)
            if ring_cli is not None:
                mpy_cli_options.ring_cli = str(ring_cli)
            mpy_cli_options.debug = debug
            mpy_cli_options.vlidort.nstokes = vlidort_nstokes
            mpy_cli_options.vlidort.nstreams = vlidort_nstreams
        yield
    finally:
        os.chdir(curdir)
        if have_muses_py:
            assert mpy_cli_options is not None
            mpy_cli_options.vlidort_cli = old_vlidort_cli
            mpy_cli_options.ring_cli = old_ring_cli
            mpy_cli_options.debug = old_debug
            mpy_cli_options.vlidort.nstokes = old_vlidort_nstokes
            mpy_cli_options.vlidort.nstreams = old_vlidort_nstreams
        if old_run_dir:
            os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
        else:
            del os.environ["MUSES_DEFAULT_RUN_DIR"]
