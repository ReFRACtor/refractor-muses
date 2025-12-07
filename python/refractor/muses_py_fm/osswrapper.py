from __future__ import annotations
from .mpy import (
    have_muses_py,
    mpy_register_observer_function,
    mpy_pyoss_dir,
    mpy_fm_oss_init,
    mpy_fm_oss_windows,
    mpy_fm_oss_delete,
)
import itertools
from refractor.muses import suppress_replacement
import os
from contextlib import contextmanager
import sys
import numpy as np
from types import TracebackType
from typing import Any, Self, Generator


@contextmanager
def suppress_stdout() -> Generator[None, None, None]:
    """A context manager to temporarily redirect stdout to /dev/null"""
    oldstdchannel = None
    dest_file = None
    try:
        oldstdchannel = os.dup(sys.stdout.fileno())
        dest_file = open(os.devnull, "w")
        os.dup2(dest_file.fileno(), sys.stdout.fileno())
        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, sys.stdout.fileno())
        if dest_file is not None:
            dest_file.close()


# This implements mpy.ObserveFunctionObject, but we don't actually derive from
# that so we don't depend on mpy being available.
# class WatchOssInit(mpy.ObserveFunctionObject):
class WatchOssInit:
    """Helper object to update osswrapper.have_oss when py-retrieve calls
    fm_oss_init."""

    def should_replace_function(self, func_name: str, parms: dict[str, Any]) -> bool:
        self.notify_function_call(func_name, parms)
        return False

    def notify_function_call(self, func_name: str, parms: dict[str, Any]) -> None:
        osswrapper.have_oss = True
        osswrapper.first_oss_initialize = False


# This implements mpy.ObserveFunctionObject, but we don't actually derive from
# that so we don't depend on mpy being available.
# class WatchOssDelete(mpy.ObserveFunctionObject):
class WatchOssDelete:
    """Helper object to update osswrapper.have_oss when py-retrieve calls
    fm_oss_delete."""

    def should_replace_function(self, func_name: str, parms: dict[str, Any]) -> bool:
        self.notify_function_call(func_name, parms)
        return False

    def notify_function_call(self, func_name: str, parms: dict[str, Any]) -> None:
        osswrapper.have_oss = False


class osswrapper:
    """The OSS library needs to be initialized, have windows set up,
    and freed when done. But it is a global function, e.g., you can't
    have two window sets available (standard global variables in
    fortran code). Depending on how a function that needs the OSS is
    called this may or may not have already been set up.

    This simple class provides a context manager than ensure that we only
    do the initialization once, and clean up wherever that occurs. This
    wrapper can then be nested - so a function in an osswrapper can call
    another function that uses the osswrapper and the OSS initialization
    will only happen once.

    Note that if we do the initialization, the uip passed in is
    modified to add oss_jacobianList, oss_dir_lut, oss_frequencyList
    and oss_frequencyListFull.  This duplicates what muses-py does,
    and this is really like an internal way to pass these variables
    around, so this isn't really a problem.

    We also interact with muse py to catch calls to fm_oss_init and
    fm_oss_delete done outside of ReFRACtor (e.g., in run_retrieval).

    We unfortunately can't do anything to ensure that we don't try
    creating two oss_wrapper with different uip. This doesn't work
    and will fail.

    Note, another probably bug with OSS is that *first* call to it
    returns different results then future calls. Not clear what is going
    on here, but it makes it hard to have repeatable code. To work around
    this the very first time we initialize code we do it twice -
    initialize + delete followed by a second initialization. This should
    probably get sorted out at some point, but for now we just work around
    this.

    """

    have_oss = False
    first_oss_initialize = True

    def __init__(self, uip: dict[str, Any]) -> None:
        if hasattr(uip, "as_dict"):
            self.uip = uip.as_dict(uip)
        else:
            self.uip = uip
        self.need_cleanup = False

    @classmethod
    def register_with_muses_py(self) -> None:
        mpy_register_observer_function("fm_oss_init", WatchOssInit())
        mpy_register_observer_function("fm_oss_delete", WatchOssDelete())

    def __enter__(self) -> Self:
        from .refractor_uip import AttrDictAdapter

        uip_all = None
        if not osswrapper.have_oss:
            for inst in ("CRIS", "AIRS", "TES"):
                # I don't think the logic here is correct if we have multiple
                # instruments. But I don't think we error have more than one,
                # so we can just assume that here. Revisit if needed.
                if f"uip_{inst}" in self.uip:
                    # Suppress warning message print out, it clutters output
                    # if True:
                    with suppress_stdout():
                        # Used by fm_oss_load to point to the pyoss library.
                        # This is a wrapper, that is created in py-retrieve
                        # (in setup.py, as a C extension). We determine the
                        # location in muses_py __init__, and just need to set
                        # it here. Kind of a round about way, but this is deep
                        # in py-retrieve and we don't want to change how it works.
                        os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy_pyoss_dir
                        # Delete frequencyList if found. I don't think we
                        # run into that in actual muses-py runs, but we do
                        # with some of our test data based on where we are
                        # in the processing.
                        self.uip.pop("frequencyList", None)
                        uip_all = self.struct_combine(self.uip, self.uip[f"uip_{inst}"])
                        # Special handling for the first time through, working
                        # around what is a bug or "feature" of the OSS code
                        if osswrapper.first_oss_initialize:
                            with suppress_replacement("fm_oss_init"):
                                (_, frequencyListFullOSS, jacobianList) = (
                                    mpy_fm_oss_init(AttrDictAdapter(uip_all), inst)
                                )
                            # This can potentially change oss_frequencyList.
                            # Neither py-retrieve or refractor is set up to handle
                            # that.
                            mpy_fm_oss_windows(AttrDictAdapter(uip_all))
                            flen = len(uip_all["frequencyList"])
                            flen2 = len(uip_all["oss_frequencyList"])
                            if flen != flen2:
                                raise RuntimeError(
                                    "fm_oss_window changed the size of oss_frequencyList. Neither py-retrieve or refractor is set up to handle this"
                                )
                            with suppress_replacement("fm_oss_delete"):
                                mpy_fm_oss_delete()
                            osswrapper.first_oss_initialize = False
                        with suppress_replacement("fm_oss_init"):
                            (_, frequencyListFullOSS, jacobianList) = mpy_fm_oss_init(
                                AttrDictAdapter(uip_all), inst
                            )
                            self.uip["oss_jacobianList"] = jacobianList
                        # This can potentially change oss_frequencyList.
                        # Neither py-retrieve or refractor is set up to handle
                        # that.
                        mpy_fm_oss_windows(AttrDictAdapter(uip_all))
                        flen = len(uip_all["frequencyList"])
                        flen2 = len(uip_all["oss_frequencyList"])
                        if flen != flen2:
                            raise RuntimeError(
                                "fm_oss_window changed the size of oss_frequencyList. Neither py-retrieve or refractor is set up to handle this"
                            )
                        self.need_cleanup = True
                        osswrapper.have_oss = True
        if uip_all is not None:
            self.oss_dir_lut = uip_all["oss_dir_lut"]
            self.oss_jacobianList = uip_all["oss_jacobianList"]
            self.oss_frequencyList = uip_all["oss_frequencyList"]
            self.oss_frequencyListFull = uip_all["oss_frequencyListFull"]
            self.uip["oss_dir_lut"] = self.oss_dir_lut
            self.uip["oss_jacobianList"] = self.oss_jacobianList
            self.uip["oss_frequencyList"] = self.oss_frequencyList
            self.uip["oss_frequencyListFull"] = self.oss_frequencyListFull
        else:
            self.oss_dir_lut = None
            self.oss_jacobianList = None
            self.oss_frequencyList = None
            self.oss_frequencyListFull = None
        return self

    def struct_combine(self, d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
        """This is similar to just d1 | d2, but the logic is a little different. In particular,
        the left side of duplicates is chosen rather than the right side, and some
        items are copied depending on the type."""
        res = {}
        for k, v in itertools.chain(d1.items(), d2.items()):
            if k not in res:
                res[k] = v.copy() if type(v) in (np.ndarray, list, dict) else v
        return res

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.need_cleanup:
            with suppress_replacement("fm_oss_delete"):
                mpy_fm_oss_delete()
            self.need_cleanup = False
            osswrapper.have_oss = False


if have_muses_py:
    osswrapper.register_with_muses_py()

__all__ = ["osswrapper"]
