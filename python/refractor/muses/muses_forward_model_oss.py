from __future__ import annotations
from .identifier import InstrumentIdentifier, StateElementIdentifier
import os
import ctypes
from ctypes import c_int, POINTER, c_float, c_char_p, c_int_p
from pathlib import Path
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFileHelper, InputFilePath

# This is a work in progress. We would like to move over and simplify the vlidort
# forward model, and hopefully remove using the UIP etc. But for right now, we
# leverage off of muses-py
#
# Note that this has direct copied of stuff from muses_py_fm/muses_forward_model.py,
# since we want to independent update stuff. This is obviously not desirable long
# term.


class OssHandle:
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

    def check_have_library(self):
        """Check if the library is available, and if not throw an exception"""
        if self.liboss is None:
            if self.library_error_message is not None:
                raise RuntimeError(self.library_error_message)
            else:
                raise RuntimeError("Trouble setting up the liboss library")

    def to_c_str(
        self, s: str | os.PathLike[str] | InputFilePath
    ) -> tuple[c_char_p, c_int_p]:
        """Convert a string like s to the types needed to pass to liboss"""
        sb = str(s).encode("utf-8")
        return c_char_p(sb), ctypes.byref(c_int(len(sb)))

    def to_c_str_arr(
        self, slist: list[StateElementIdentifier], slen: int = 6
    ) -> tuple[c_int_p, c_int_p, c_char_p]:
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

    def oss_init(
        self,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        nlevels: int,
        nfreq: int,
        instrument: InstrumentIdentifier,
        cris_l1b_type: str = "",
        dir_lut: InputFilePath | None = None,
    ) -> None:
        # TODO Can we move the cris_l1b_type into the InstrumentIdentifier. It seems like
        # cris actually has two instrument types, and it might make more sense to have it
        # handled like that rather than a separate l1b_type carried around.
        self.check_have_library()
        self.retrieval_state_element_id = retrieval_state_element_id
        self.species_list = species_list
        self.instrument = instrument
        self.dir_lut = dir_lut

        # Strip out items that aren't atmospheric, and also TATM (which gets
        # marked as atmospheric but isn't a gas species)
        atm_spec = [
            s
            for s in self.species_list
            if s.is_atmospheric_species and s != StateElementIdentifier("TATM")
        ]
        jac_spec = [
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
            if nm in atm_spec:
                atm_spec[atm_spec.index(nm)] = rname
            if nm in jac_spec:
                jac_spec[jac_spec.index(nm)] = rname

        # Special case, turns out OSS doesn't work with no jacobians. So if our list
        # is empty, just add H2O so that there is something there
        if len(jac_spec) == 0:
            jac_spec = [StateElementIdentifier("H2O")]

        # We handle the channel selection outside of the OSS code. This selects
        # the subsets of the full range of frequencies that we actually run the
        # forward model on
        self.chsel_file = "NULL"

        # Should perhaps move this logic out of here and into the forward models.
        # We could just pass in sel_file, od_file, sol_file and fix_file
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
        elif instrument == InstrumentIdentifier("CRIS") and "nsr" in cris_l1b_type:
            # updated 1/2023
            if self.dir_lut is None:
                self.dir_lut = ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2023-01-nsr"
            self.sel_file = (
                self.dir_lut
                + "suomi-cris-B1B2B3-unapod-loc-clear-19V-M12.4-v1.0.train.sel"
            )
            self.od_file = (
                self.dir_lut
                + "suomi-cris-B1B2B3-unapod-loc-clear-19V-M12.4-v1.0.train.lut"
            )
            self.sol_file = self.dir_lut / "newkur.dat"
            self.fix_file = self.dir_lut / "default.dat"
        elif instrument == InstrumentIdentifier("CRIS"):
            if self.dir_lut is None:
                self.dir_lut = ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2017-08"
            self.sel_file = (
                self.dir_lut
                / "suomi-cris-fsr-B1B2B3-unapod-loc-cloudy-23V-M12.4-v1.0.train.sel"
            )

            self.od_file = (
                self.dir_lut
                / "suomi-cris-fsr-B1B2B3-unapod-loc-cloudy-23V-M12.4-v1.0.train.lut"
            )

            self.sol_file = self.dir_lut / "newkur.dat"
            self.fix_file = self.dir_lut / "default.dat"
        else:
            raise RuntimeError("Instrument does not match possible list")

        # Hardcoded value from py-retrieve. Not sure exactly what this corresponds to
        min_ext_cld = c_float(0.0000001)

        # Maximum number of wavenumbers that can be returned. This just
        # needs to be large enough so there is enough space in freq_oss. We
        # get the actual number back from our call
        mx_nfreq = 20000
        freq_oss = np.zeros(shape=(mx_nfreq), dtype=c_float)
        # Will get filled in
        n_freq = c_int(0)

        # Clean up any old initialization. Note this is safe to call even
        # if no initialization has occurred yet.
        self.liboss.cppdestrwrapper()

        # nfreq and freq_oss get updated by this call, everything else is
        # just a input.
        self.liboss.cppinitwrapper(
            *self.to_c_str_arr(atm_spec),
            *self.to_c_str_arr(jac_spec),
            *self.to_c_str(self.sel_file),
            *self.to_c_str(self.od_file),
            *self.to_c_str(self.sol_file),
            *self.to_c_str(self.fix_file),
            *self.to_c_str(self.chsel_file),
            ctypes.byref(c_int(nlevels)),
            ctypes.byref(c_int(nfreq)),
            ctypes.byref(min_ext_cld),
            ctypes.byref(n_freq),
            freq_oss.ctypes.data_as(POINTER(c_float)),
            ctypes.byref(c_int(mx_nfreq)),
        )
        self.freq_oss = freq_oss[: n_freq.value]


oss_handle = OssHandle()

__all__ = [
    "oss_handle",
]
