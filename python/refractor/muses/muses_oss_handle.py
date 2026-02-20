from __future__ import annotations
from .identifier import InstrumentIdentifier, StateElementIdentifier
import os
import ctypes
from ctypes import c_int, POINTER, c_float, c_char_p
from pathlib import Path
import numpy as np
from contextlib import contextmanager
import sys
import typing
from typing import Any, Iterator

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFileHelper, InputFilePath

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

class MusesOssHandle:
    """This handles the OSS interface. Because it is a global, we need a central
    place to handle this.  This class should be a singleton, use oss_handle
    rather than creating directly from this class."""

    def __init__(self) -> None:
        # Note on the implementation here. ctypes is nice, but a bit cumbersome to
        # use. We could have swig wrappers instead, but we only call a handful of
        # functions in liboss so the simplicity of not depending on a swig library
        # outweighs the simpler swig interface. The complexity (which isn't *that*
        # much) is hidden in OssHandle so this isn't a big deal.

        # We need special handling for the first time oss_init is called
        self.first_oss_initialize = True

        # We can do a more complicated discovery of the path to
        # liboss.so if needed, but this is probably sufficient for the
        # way we use this.
        #
        # Note that py-retrieve used a pyoss library it builds. We
        # aren't using that here, the pyoss library really had no
        # point, it just forwarded everything to liboss. I think this
        # was because it was converted from IDL, there were some IDL
        # wrappers that have a interface more appropriate to IDL. But
        # liboss already has a c interface that ctypes can directly
        # interact with.

        # We use the logic here that we open up the library if found, but we don't
        # want refractor-muses to depend on having this library. If it isn't found,
        # we just silently set self.liboss to None. We then fail later when we try
        # to use the library
        self.library_error_message = None
        self.liboss = None
        if "CONDA_PREFIX" not in os.environ:
            self.library_error_message = "CONDA_PREFIX not found in environment. Currently OssHandle only work with a conda environment."
        else:
            libpath = Path(os.environ["CONDA_PREFIX"]) / "lib" / "liboss.so"
            if not libpath.exists():
                self.library_error_message = f"Did not find oss library at {libpath}"
            else:
                self.liboss = ctypes.CDLL(libpath)

        # Set up the argtypes for the functions we will call. I don't think this
        # is strictly required with ctypes, but it is recommended and there doesn't
        # seem to be any reason not to do this.
        #
        # We get this by looking at CPPwrapper.f90 in the muses-oss code
        # (see git@github.jpl.nasa.gov:MUSES-Processing/muses-oss.git)
        #
        # Note that the ISO c bindings take everything as pointers, so you
        # will see a c_int_p where you might expect just a c_int. This is just
        # the way fortran works, you just need to be careful when passing thing
        # to send pointers. Note in general even though we are passing by reference,
        # things don't get updated. Look in the fortran to see "INTENT". If it is
        # "INOUT" then the value may get updated.
        #
        # Ctypes doesn't have a "const pointer" type, so you can't tell
        # just from the argtypes below what gets updated. But fortran won't update
        # something passed in with a type "INTENT(IN)".
        if self.liboss is not None:
            # fmt: off
            c_int_p = POINTER(c_int)
            self.liboss.cppinitwrapper.argtypes = [
                c_int_p, c_int_p, c_char_p,  # name gas
                c_int_p, c_int_p, c_char_p,  # name jac
                c_char_p, c_int_p,           # sel file
                c_char_p, c_int_p,           # od file
                c_char_p, c_int_p,           # sol file
                c_char_p, c_int_p,           # fix file
                c_char_p, c_int_p,           # channel select file
                c_int_p, c_int_p,            # nlevu, nsf
                POINTER(c_float),            # minExtCld
                c_int_p,                     # nchanOSS, output
                POINTER(c_float),            # WN OSS, output
                c_int_p,                     # max channel
            ]
            self.liboss.cppinitwrapper.restype = None
            self.liboss.cppdestrwrapper.argtypes = []
            self.liboss.cppdestrwrapper.restype = None
            # fmt: off
            self.liboss.cppupdatejacobwrapper.argtypes = [
                c_int_p, c_int_p, c_char_p,  # name gas
                c_int_p, c_int_p, c_char_p,  # name jac
            ]
            self.liboss.cppupdatejacobwrapper.restypes = None
        self.have_oss = False
        # Values we used initializing, we check if this has changed to see if we
        # need to update the initialization
        self.sel_file: str | os.PathLike[str] | InputFilePath = ""
        self.od_file: str | os.PathLike[str] | InputFilePath  = ""
        self.sol_file: str | os.PathLike[str] | InputFilePath  = ""
        self.fix_file: str | os.PathLike[str] | InputFilePath  = ""
        self.nlevels = -1
        self.nfreq = -1
        self.atm_spec : list[StateElementIdentifier] = []
        self.jac_spec : list[StateElementIdentifier] = []
        # We handle the channel selection outside of the OSS code. This selects
        # the subsets of the full range of frequencies that we actually run the
        # forward model on
        self.chsel_file = "NULL"

    def check_have_library(self) -> None:
        """Check if the library is available, and if not throw an exception"""
        if self.liboss is None:
            if self.library_error_message is not None:
                raise RuntimeError(self.library_error_message)
            else:
                raise RuntimeError("Trouble setting up the liboss library")

    def to_c_str(
        self, s: str | os.PathLike[str] | InputFilePath
    ) -> tuple[c_char_p, Any]:
        """Convert a string like s to the types needed to pass to liboss"""
        sb = str(s).encode("utf-8")
        return c_char_p(sb), ctypes.byref(c_int(len(sb)))

    def to_c_str_arr(
        self, slist: list[StateElementIdentifier], slen: int = 6
    ) -> tuple[Any, Any, c_char_p]:
        """Convert a list of StateElementIdentifier to the types
        needed to pass to liboss.

        The slen value here corresponds to a hard coded number in
        "ConvertModule.f90" in muses-oss, this is "lenMol". These jacobian and
        gas names get looked up in that table, with left spaces trimmed. So we
        just need a value large enough to allow matching all the molecule names.
        Fitting to lenMol works with that.
        """
        # Truncate and/or add spaces to make exactly slen
        t = [str(s).encode("utf-8")[:slen].ljust(slen) for s in slist]
        # Join into on array, and return
        return (
            ctypes.byref(c_int(len(slist))),
            ctypes.byref(c_int(slen)),
            c_char_p(b"".join(t)),
        )

    def oss_cleanup(self):
        '''We don't normally need to call this, but when coordinating with the old
        py-retrieve code we need to since we are sharing a global resource. Clean
        up any allocation, and mark us as not having OSS.'''
        self.liboss.cppdestrwrapper()
        self.have_oss = False
        
    def oss_init(
        self,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        nlevels: int,
        nfreq: int, # This seems to be the size of the emissivity. Perhaps verify,
                    # And if so change it name. This has nothing to do with the
                    # size of freq_oss that gets filled in
        sel_file : str | os.PathLike[str] | InputFilePath,
        od_file : str | os.PathLike[str] | InputFilePath,
        sol_file : str | os.PathLike[str] | InputFilePath,
        fix_file : str | os.PathLike[str] | InputFilePath,
    ) -> None:
        self.check_have_library()
        assert self.liboss is not None
        # Skip initialization if the only thing that changes in the jac_spec list,
        # (we can separately update that part). The initialization is expensive relative
        # to other OSS functions, so we skip if we can.
        if (self.have_oss and
            self.sel_file == sel_file and
            self.od_file == od_file and
            self.sol_file == sol_file and
            self.fix_file == fix_file and
            self.nlevels == nlevels and
            self.nfreq == nfreq and
            self.species_list == species_list):
            do_jac_only = True
        else:
            do_jac_only = False
        self.current_jac_spec = self.jac_spec
        self.sel_file = sel_file
        self.od_file = od_file
        self.sol_file = sol_file
        self.fix_file = fix_file
        self.nlevels = nlevels
        self.nfreq = nfreq
        self.species_list = species_list
        self.nlevels = nlevels
        self.nfreq = nfreq
        self.retrieval_state_element_id = retrieval_state_element_id
        self.species_list = species_list

        # Strip out items that aren't atmospheric, and also TATM (which gets
        # marked as atmospheric but isn't a gas species)
        self.atm_spec = [
            s
            for s in self.species_list
            if s.is_atmospheric_species and s != StateElementIdentifier("TATM")
        ]
        self.jac_spec = [
            s
            for s in self.retrieval_state_element_id
            if s.is_atmospheric_species and s != StateElementIdentifier("TATM")
        ]
        # Some of the species names have a different name in OSS. Not sure of
        # the history of this, but map names to OSS name. This comes from py-retrieve
        spec_rename = [
            (StateElementIdentifier("CFC11"), StateElementIdentifier("F11")),
            (StateElementIdentifier("CFC12"), StateElementIdentifier("F12")),
            (StateElementIdentifier("ISOP"), StateElementIdentifier("C5H8")),
            (StateElementIdentifier("CFC22"), StateElementIdentifier("CHCLF2")),
        ]
        for nm, rname in spec_rename:
            if nm in self.atm_spec:
                self.atm_spec[self.atm_spec.index(nm)] = rname
            if nm in self.jac_spec:
                self.jac_spec[self.jac_spec.index(nm)] = rname

        # Special case, turns out OSS doesn't work with no jacobians. So if our list
        # is empty, just add H2O so that there is something there
        if len(self.jac_spec) == 0:
            self.jac_spec = [StateElementIdentifier("H2O")]

        if do_jac_only:
            # Nothing to do if Jacobian matches
            if self.jac_spec == self.current_jac_spec:
                return
            # Otherwise, update just the jacobian part
            self.liboss.cppupdatejacobwrapper(
                *self.to_c_str_arr(self.atm_spec),
                *self.to_c_str_arr(self.jac_spec),
            )
            return
            

        # Should perhaps move this logic out of here and into the forward models.
        # We could just pass in sel_file, od_file, sol_file and fix_file
        if False:
            instrument = "blah"
            if instrument == InstrumentIdentifier("TES"):
                if self.dir_lut is None:
                    self.dir_lut = ifile_hlp.osp_dir / "OSS_FM" / "TES " / "2018-03-14"
                self.sel_file = (
                    self.dir_lut
                    / "aqua-tes-B2B11B22A11A1-unapod-loc-clear-23V-M12.4-v1.2.train.sel"
                )
                self.od_file = (
                    self.dir_lut
                    / "aqua-tes-B2B11B22A11A1-unapod-loc-clear-23V-M12.4-v1.2.train.lut"
                )
                self.sol_file = self.dir_lut / "newkur.dat"
                self.fix_file = self.dir_lut / "default.dat"
            elif instrument == InstrumentIdentifier("AIRS"):
                if self.dir_lut is None:
                    self.dir_lut = ifile_hlp.osp_dir / "OSS_FM" / "AIRS" / "2017-07"
                self.sel_file = (
                    self.dir_lut
                    / "aqua-airs-B1B2B3-unapod-loc-clear-23V-M12.4-v1.0.train.sel"
                )
                self.od_file = (
                    self.dir_lut
                    / "aqua-airs-B1B2B3-unapod-loc-clear-23V-M12.4-v1.0.train.lut"
                )
                self.sol_file = self.dir_lut / "newkur.dat"
                self.fix_file = self.dir_lut / "default.dat"
            else:
                raise RuntimeError("Instrument does not match possible list")

        # Notify that we read these files, since this is done in fortran we can't
        # notify when we actually open the file like we do most places
        ifile_hlp.notify_file_input(self.sel_file)
        ifile_hlp.notify_file_input(self.od_file)
        ifile_hlp.notify_file_input(self.sol_file)
        ifile_hlp.notify_file_input(self.fix_file)
        # chsel_file isn't used
        # ifile_hlp.notify_file_input(self.chsel_file)

        # Hardcoded value from py-retrieve. Not sure exactly what this corresponds to
        min_ext_cld = c_float(0.0000001)

        # Maximum number of wavenumbers that can be returned. This just
        # needs to be large enough so there is enough space in freq_oss. We
        # get the actual number back from our call
        mx_nfreq = 20000
        freq_oss = np.zeros(shape=(mx_nfreq), dtype=c_float)
        # Will get filled in
        n_freq = c_int(0)

        # Special handling for the first time through, working around
        # what is a bug or "feature" of the OSS code (see MUSESCI-1240
        # in jira). We need to run through the initialization again
        # the first time
        do_init = True
        
        while do_init:
            # Clean up any old initialization. Note this is safe to call even
            # if no initialization has occurred yet.
            self.liboss.cppdestrwrapper()

            # n_freq and freq_oss get updated by this call, everything else is
            # just a input.
            # Suppress error messages we can't do anything about. *Note* if you
            # are debugging a problem here make sure to turn the suppression off,
            # otherwise you won't be able to see the problem.
            with suppress_stdout():
                self.liboss.cppinitwrapper(
                    *self.to_c_str_arr(self.atm_spec),
                    *self.to_c_str_arr(self.jac_spec),
                    *self.to_c_str(self.sel_file),
                    *self.to_c_str(self.od_file),
                    *self.to_c_str(self.sol_file),
                    *self.to_c_str(self.fix_file),
                    *self.to_c_str(self.chsel_file),
                    ctypes.byref(c_int(self.nlevels)),
                    ctypes.byref(c_int(self.nfreq)),
                    ctypes.byref(min_ext_cld),
                    ctypes.byref(n_freq),
                    freq_oss.ctypes.data_as(POINTER(c_float)),
                    ctypes.byref(c_int(mx_nfreq)),
                )
            self.freq_oss = freq_oss[: n_freq.value]
            # Check that self.freq_oss is sorted in ascending
            # order. We assume that when looking up data. We could
            # change our code to not assume this, but it seems like a
            # good idea to take advantage of this. Note numpy doesn't
            # have a "is_sorted" test, but this takes the difference between
            # adjacent elements and makes sure this is >= 0. I think this
            # is probably strictly ascending (>0), but we don't actually require that.
            # All we need is for searchsorted to work.
            if not np.all(np.diff(self.freq_oss) >= 0):
                raise RuntimeError("freq_oss is not sorted in ascending order")
            if self.first_oss_initialize:
                # First time through, repeat the initialization
                self.first_oss_initialize = False
                do_init = True
            else:
                self.have_oss = True
                do_init = False

muses_oss_handle = MusesOssHandle()

__all__ = [
    "muses_oss_handle",
]
