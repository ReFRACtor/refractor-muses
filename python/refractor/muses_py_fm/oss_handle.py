from __future__ import annotations
from .mpy import (
    have_muses_py,
    mpy_register_observer_function,
    mpy_register_replacement_function,
    mpy_pyoss_dir,
    mpy_fm_oss_init,
    mpy_fm_oss_windows,
    mpy_fm_oss_delete,
    mpy_fm_oss_stack,
    mpy_redo_fm_oss_init,
    
)
from contextlib import contextmanager

import numpy as np
import os
import sys
from typing import Self, Iterator, Any
import typing

if typing.TYPE_CHECKING:
    from .refractor_uip import RefractorUip
    from refractor.muses import InputFileHelper, InstrumentIdentifier


@contextmanager
def suppress_stdout() -> Iterator[None]:
    """A context manager to temporarily redirect stdout to /dev/null. We do that because
    the OSS init is noisy, complaining about things we can' change."""
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


@contextmanager
def suppress_replacement(func_name: str) -> Iterator[None]:
    old_f = mpy_register_replacement_function(func_name, None)
    try:
        yield
    finally:
        mpy_register_replacement_function(func_name, old_f)


# Because the OSS wrapper in muses_oss is global, we need to know if py-retrieve
# either initializes or destroys this. This doesn't happen during a normal refractor
# run, but if we are testing with old py-retrieve code we may run into that.
class WatchOss:
    """Helper object to update oss_handle.have_oss when py-retrieve calls
    fm_oss_init or fm_oss_delete."""

    def should_replace_function(self, func_name: str, parms: dict[str, Any]) -> bool:
        self.notify_function_call(func_name, parms)
        return False

    def notify_function_call(self, func_name: str, parms: dict[str, Any]) -> None:
        if func_name == "fm_oss_init":
            oss_handle.have_oss_py_retrieve = True
            oss_handle.first_oss_initialize = False
            breakpoint()
        elif func_name == "fm_oss_delete":
            oss_handle.have_oss_py_retrieve = False
            oss_handle.have_oss = False


if have_muses_py:
    t = WatchOss()
    mpy_register_observer_function("fm_oss_init", t)
    mpy_register_observer_function("fm_oss_delete", t)


class OssHandle:
    """We use to do mpy_fm_oss_init and mpy_fm_oss_delete each time we
    did a OSS retrieval. It turns out this is relatively expense,
    there is a lot of work done in the initialization. Most of the
    time, we are calling the same exact initialization.  So we have
    pulled this out into a handler class. On initialization, we skip
    and use our previous initialization unless something has changed.

    The OSS library needs to be initialized, have windows set up,
    and freed when done. But it is a global function, e.g., you can't
    have two window sets available (standard global variables in
    fortran code). Depending on how a function that needs the OSS is
    called this may or may not have already been set up.

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

    def __init__(self) -> None:
        # Because OSSWrapper in muses_oss is global, we need to know if one has been
        # initialized or not. Also, we need special handling for the first time
        # fm_oss_init is called.
        self.have_oss = False
        self.first_oss_initialize = True
        # We also want to stay out of the way if we are calling thing through
        # py-retrieve, which handles is for us
        self.have_oss_py_retrieve = False
        # Used by fm_oss_load to point to the pyoss library.
        # This is a wrapper, that is created in py-retrieve
        # (in setup.py, as a C extension). We determine the
        # location in muses_py __init__, and just need to set
        # it here. Kind of a round about way, but this is deep
        # in py-retrieve and we don't want to change how it works.
        os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy_pyoss_dir

    def initialize_oss(self) -> None:
        """Do the initialization for oss. This deletes any existing oss, so the logic
        of if we do this or not gets handled at a higher level - initialize_oss
        *always* cleans up and recreates the OSSWrapper data."""
        from refractor.muses import AttrDictAdapter, InstrumentIdentifier

        cris_l1b_type = ''
        if InstrumentIdentifier("CRIS") in self.rf_uip.instrument:
            jacobians = self.rf_uip.uip["uip_CRIS"]["jacobians"]
            frequency_list = self.rf_uip.uip["uip_CRIS"]["frequencyList"]
            cris_l1b_type = self.rf_uip.uip["crisPars"]["l1bType"]
            inst = InstrumentIdentifier("CRIS")
        elif InstrumentIdentifier("AIRS") in self.rf_uip.instrument:
            jacobians = self.rf_uip.uip["uip_AIRS"]["jacobians"]
            frequency_list = self.rf_uip.uip["uip_AIRS"]["frequencyList"]
            inst = InstrumentIdentifier("AIRS")
        elif InstrumentIdentifier("TES") in self.rf_uip.instrument:
            jacobians = self.rf_uip.uip["uip_TES"]["jacobians"]
            frequency_list = self.rf_uip.uip["uip_TES"]["frequencyList"]
            inst = InstrumentIdentifier("TES")
        else:
            # Don't recognize the instrument, so we can't initialize OSS
            return
        atmosphere_params = self.rf_uip.uip["atmosphere_params"]
        nlevels = self.rf_uip.uip["atmosphere"].shape[1]
        nfreq = self.rf_uip.uip["emissivity"]["frequency"].shape[0]
        self.oss_init(inst, jacobians, atmosphere_params, nlevels, nfreq, cris_l1b_type)
        self.oss_window(frequency_list, self.oss_frequency_list_full)
        self.rf_uip.uip["oss_dir_lut"] = self.oss_dir_lut
        self.rf_uip.uip["oss_jacobianList"] = self.oss_jacobian_list
        self.rf_uip.uip["oss_frequencyListFull"] = self.oss_frequency_list_full
        self.rf_uip.uip["oss_frequencyList"] = self.oss_frequency_list
        self.have_oss = True

    def oss_init(self, inst : InstrumentIdentifier, jacobians : list[str], atmosphere_params: list[str], nlevels: int,
                 nfreq: int, cris_l1b_type : str) -> None:
        from refractor.muses import AttrDictAdapter
        
        if self.have_oss:
            with suppress_replacement("fm_oss_delete"):
                mpy_fm_oss_delete()
            self.have_oss = False
        t = {'jacobians': jacobians,
             'atmosphere_params' : atmosphere_params,
             'atmosphere' : np.empty((1, nlevels)),
             'emissivity' : {'frequency' : np.empty((nfreq,))},
              'crisPars' : {'l1bType' : cris_l1b_type}
             }
        with suppress_replacement("fm_oss_init"), suppress_stdout():
            (_, self.oss_frequency_list_full, self.oss_jacobian_list) = mpy_fm_oss_init(
                AttrDictAdapter(t),
                str(inst),
                i_osp_dir=self.ifile_hlp.osp_dir.path_for_muses_py,
            )
        self.oss_dir_lut = t["oss_dir_lut"]
        # List of files used by fm_oss_init
        for t2 in ("sel_file", "od_file", "sol_file", "fix_file", "chsel_file"):
            fname = t[t2]
            if fname != "NULL":
                self.ifile_hlp.notify_file_input(fname)
        # Special handling for the first time through, working
        # around what is a bug or "feature" of the OSS code
        if self.first_oss_initialize:
            with suppress_replacement("fm_oss_init"), suppress_stdout():
                mpy_redo_fm_oss_init(AttrDictAdapter(t))
                self.first_oss_initialize = False

    def oss_window(self, frequency_list: list[float], oss_frequency_list_full : np.ndarray) -> None:
        from refractor.muses import AttrDictAdapter
        
        t = {"frequencyList" : frequency_list,
             "oss_frequencyListFull" : oss_frequency_list_full}
        mpy_fm_oss_windows(AttrDictAdapter(t))
        self.frequency_list = frequency_list
        self.oss_frequency_list = t["oss_frequencyList"]
        if len(self.frequency_list) != len(self.oss_frequency_list):
            raise RuntimeError(
                "fm_oss_window changed the size of oss_frequencyList. Neither py-retrieve or refractor is set up to handle this"
            )
        

    @contextmanager
    def handle(
        self, rf_uip: RefractorUip, ifile_hlp: InputFileHelper | None
    ) -> Iterator[Self]:
        if ifile_hlp is None:
            from refractor.muses import InputFileHelper

            self.ifile_hlp = InputFileHelper()
        else:
            self.ifile_hlp = ifile_hlp
        self.rf_uip = rf_uip
        assert ifile_hlp is not None
        if not self.have_oss_py_retrieve:
            self.initialize_oss()
        yield self

    def radiance_and_jacobian(
        self, instrument_name: InstrumentIdentifier
    ) -> tuple[np.ndarray, np.ndarray]:
        """Call OSS to get the radiance and jacobian"""
        return mpy_fm_oss_stack(self.rf_uip.uip_all(instrument_name))


oss_handle = OssHandle()

__all__ = [
    "oss_handle",
]
