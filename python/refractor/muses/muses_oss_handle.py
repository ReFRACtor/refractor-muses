from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .identifier import StateElementIdentifier
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
            c_float_p = POINTER(c_float)
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
            self.liboss.cppupdatejacobwrapper.restype = None
            self.liboss.cpploadchanselect.argtypes=[
                c_int_p, c_int_p,            # Channel index
                c_int_p                      # index select, output
            ]
            self.liboss.cpploadchanselect.restype= None
            self.liboss.cppsetchanselect.argtypes=[
                c_int_p                      # index select
            ]
            self.liboss.cppsetchanselect.restype=None
            self.liboss.cppreloadchanselect.argtypes=[
                c_int_p, c_int_p,            # Channel index
                c_int_p                      # index select
            ]
            self.liboss.cppreloadchanselect.restype= None
            self.liboss.cppfwdwrapper.argtypes = [
                c_int_p,              # Number levels
                c_int_p,              # Number gases in vmr
                c_float_p,            # Pressure
                c_float_p,            # TATM
                c_float_p,            # TSUR
                c_float_p,            # VMR
                c_int_p,              # Number EMIS
                c_float_p,            # EMIS
                c_float_p,            # Reflectance
                c_float_p,            # Scale Cloud
                c_float_p,            # Pressure Cloud
                c_int_p,              # Number cloudext
                c_float_p,            # CLOUDEXT
                c_float_p,            # emis_freq
                c_float_p,            # cloud_freq
                c_float_p,            # pointing angle
                c_float_p,            # sun angle
                c_float_p,            # latitude
                c_float_p,            # surface altitude
                c_int_p,              # Lambertian
                c_int_p,              # Number jacobian
                c_int_p,              # Number channels
                c_float_p,            # y (out)
                c_float_p,            # dy_dtemp (out)
                c_float_p,            # dy_dtsur (out)
                c_float_p,            # xkOutGas (out)
                c_float_p,            # xkEm (out)
                c_float_p,            # xkRf (out)
                c_float_p,            # xkCldlnPres (out)
                c_float_p             # xkCldlnExt (out)
            ]
            self.liboss.cppfwdwrapper.restype = None
        self.have_oss = False
        # Values we used initializing, we check if this has changed to see if we
        # need to update the initialization
        self.sel_file: str | os.PathLike[str] | InputFilePath = ""
        self.od_file: str | os.PathLike[str] | InputFilePath = ""
        self.sol_file: str | os.PathLike[str] | InputFilePath = ""
        self.fix_file: str | os.PathLike[str] | InputFilePath = ""
        self.species_list: list[StateElementIdentifier] = []
        self.nlevels = -1
        self.nfreq = -1
        self._atm_spec: list[StateElementIdentifier] = []
        self._atm_jac_spec: list[StateElementIdentifier] = []
        # We handle the channel selection outside of the OSS code. This selects
        # the subsets of the full range of frequencies that we actually run the
        # forward model on
        self.chsel_file = "NULL"
        self.channel_indx = np.array(
            [
                -1,
            ],
            dtype=c_int,
        )

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

    def to_int_arr(self, d: np.ndarray) -> tuple[Any, Any]:
        """Arguments needed to pass a c_int array."""
        return ctypes.byref(c_int(d.shape[0])), d.ctypes.data_as(POINTER(c_int))

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
        t = [s.oss_species_name.encode("utf-8")[:slen].ljust(slen) for s in slist]
        # Join into on array, and return
        return (
            ctypes.byref(c_int(len(slist))),
            ctypes.byref(c_int(slen)),
            c_char_p(b"".join(t)),
        )

    def oss_cleanup(self) -> None:
        """We don't normally need to call this, but when coordinating with the old
        py-retrieve code we need to since we are sharing a global resource. Clean
        up any allocation, and mark us as not having OSS."""
        if self.liboss is not None:
            self.liboss.cppdestrwrapper()
        self.have_oss = False

    def oss_init(
        self,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        nlevels: int,
        nfreq: int,  # This seems to be the size of the emissivity. Perhaps verify,
        # And if so change it name. This has nothing to do with the
        # size of freq_oss that gets filled in
        sel_file: str | os.PathLike[str] | InputFilePath,
        od_file: str | os.PathLike[str] | InputFilePath,
        sol_file: str | os.PathLike[str] | InputFilePath,
        fix_file: str | os.PathLike[str] | InputFilePath,
    ) -> None:
        self.check_have_library()
        assert self.liboss is not None
        # Skip initialization if the only thing that changes in the jac_spec list,
        # (we can separately update that part). The initialization is expensive relative
        # to other OSS functions, so we skip if we can.
        if (
            self.have_oss
            and self.sel_file == sel_file
            and self.od_file == od_file
            and self.sol_file == sol_file
            and self.fix_file == fix_file
            and self.nlevels == nlevels
            and self.nfreq == nfreq
            and self.species_list == species_list
        ):
            do_jac_only = True
        else:
            do_jac_only = False
        self.current_jac_spec = self._atm_jac_spec
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
        self._atm_spec = [
            s
            for s in self.species_list
            if s.is_atmospheric_species and s != StateElementIdentifier("TATM")
        ]
        self._atm_jac_spec = [
            s
            for s in self.retrieval_state_element_id
            if s.is_atmospheric_species and s != StateElementIdentifier("TATM")
        ]

        # Special case, turns out OSS doesn't work with no jacobians. So if our list
        # is empty, just add H2O so that there is something there. self._atm_jac_spec2
        # either self.atm_jac_spec or just H2O
        if len(self.atm_jac_spec) == 0:
            self._atm_jac_spec2 = [StateElementIdentifier("H2O")]
        else:
            self._atm_jac_spec2 = self.atm_jac_spec

        if do_jac_only:
            # Nothing to do if Jacobian matches
            if self.atm_jac_spec == self.current_jac_spec:
                return
            # Otherwise, update just the jacobian part
            self.liboss.cppupdatejacobwrapper(
                *self.to_c_str_arr(self.atm_spec),
                *self.to_c_str_arr(self._atm_jac_spec2),
            )
            return

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
                    *self.to_c_str_arr(self._atm_jac_spec2),
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
            # Turns out freq_oss isn't always sorted. Not sure of the history of that,
            # but it is quicker for us to look stuff up in sorted order. So create
            # a sorted version, but also with a map for going back to the unsorted
            # data since that is what the channel select works on.
            self.freq_oss_argsort = np.argsort(self.freq_oss)
            self.freq_oss_sorted = self.freq_oss[self.freq_oss_argsort]
            # Initialize channel select interface. We update this later, but we
            # need to do an initial just to get stuff bootstrapped
            self.channel_indx = np.array(
                [
                    1,
                ],
                dtype=c_int,
            )
            self.channel_id_set = c_int(-1)
            self.liboss.cpploadchanselect(
                *self.to_int_arr(self.channel_indx), ctypes.byref(self.channel_id_set)
            )
            self.liboss.cppsetchanselect(ctypes.byref(self.channel_id_set))
            if self.first_oss_initialize:
                # First time through, repeat the initialization
                self.first_oss_initialize = False
                do_init = True
            else:
                self.have_oss = True
                do_init = False

    @property
    def atm_spec(self) -> list[StateElementIdentifier]:
        """The list of gases we use in the OSS calculation. Note that
        the MusesRadiativeTransferOss select only a subset of these for
        a particular OSS run, but it keeps the full list of gases
        setting the vmr of gases not in the desired subset to very
        small values (1e-20), effectively removing them from the calculation."""
        return self._atm_spec

    @property
    def atm_jac_spec(self) -> list[StateElementIdentifier]:
        """The subset of atm_spec that we calculate jacobians for."""
        return self._atm_jac_spec

    def oss_channel_select(self, sd_desired: rf.SpectralDomain) -> None:
        """Set the channels selected in OSS (the terminology of the OSS code - pick the
        indices of freq_oss to calculate)."""
        self.check_have_library()
        assert self.liboss is not None
        freq_desired = sd_desired.convert_wave("cm^-1")
        # Make sure the freq_desired actually matches the freq_oss values. We allow
        # a small amount of slop for round off, but need to be pretty close
        tolerance = 0.001
        # searchsorted returns index of first value <= freq_desired. Due to round off,
        # freq_desired might have point slightly larger. So subtract our tolerance
        channel_indx = np.searchsorted(
            muses_oss_handle.freq_oss_sorted, freq_desired - tolerance
        )
        # Map back to the unsorted data
        channel_indx = self.freq_oss_argsort[channel_indx]
        if not np.all(
            np.abs(muses_oss_handle.freq_oss[channel_indx] - freq_desired) <= tolerance
        ):
            raise RuntimeError(
                "Desired frequency doesn't match the available frequencies in muses_oss_handle.freq_oss"
            )
        # Fortran is 1 based, so add that to our zero based indices
        channel_indx += 1
        channel_indx = channel_indx.astype(c_int)
        # if channel_indx matches our last update, we can skip this step
        if list(self.channel_indx) == list(channel_indx):
            return
        self.channel_indx = channel_indx
        self.liboss.cppreloadchanselect(
            *self.to_int_arr(self.channel_indx), ctypes.byref(self.channel_id_set)
        )

    def oss_forward_model(
        self,
        tsur: float,
        scale_cloud: float,
        pcloud: float,
        pointing_angle: float,
        sun_angle: float,
        latitude: float,
        surface_altitude: float,
        lambertian_flag: int,
        pressure: np.ndarray,
        tatm: np.ndarray,
        atmosphere: np.ndarray,
        emis_freq: np.ndarray,
        emis: np.ndarray,
        cloud_freq: np.ndarray,
        cloudext: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        This information comes from the developers guide in muses-oss

        tsuf - surface temperature, in kelvin
        pointing_angle - in degrees
        sun_angle - in degrees
        latitude - in degrees
        pressure - from high to low, in mbar
        cloud_ext - cloud peak extinction in km^-1
        emis - emissivity (unitless)
        reflectance - not passed in, we have this as 1 - emis
        emis_freq - wavenumbers in cm^-1
        cloud_freq - wavenumbers in cm^-1
        lambertian_flag - if set to 0, specular refection, otherwise, a
            diffusion approximation is used to calculate downwelling ir
            reflection from the surface

        returns radiance in W / (m^2 sr cm^-1)
        """

        self.check_have_library()
        assert self.liboss is not None
        if pressure.shape[0] != tatm.shape[0]:
            raise RuntimeError(
                "Pressure and tatm need to have the same number of levels"
            )
        if pressure.shape[0] != atmosphere.shape[0]:
            raise RuntimeError(
                "Pressure and atmosphere need to have the same number of levels"
            )
        if emis.shape[0] != emis_freq.shape[0]:
            raise RuntimeError("emis and emis_freq need to be the same size")
        if cloudext.shape[0] != cloud_freq.shape[0]:
            raise RuntimeError("cloudext and cloud_freq need to be the same size")

        # Setup return stuff
        nlevels = atmosphere.shape[0]
        nrad = self.channel_indx.shape[0]
        nemis = emis.shape[0]
        ncloud = cloudext.shape[0]
        njac = len(self._atm_jac_spec2)
        rad = np.zeros((nrad,), dtype=c_float, order="F")
        drad_dtemp = np.zeros((nlevels, nrad), dtype=c_float, order="F")
        drad_dtsur = np.zeros((nrad,), dtype=c_float, order="F")
        xkem = np.zeros((nemis, nrad), dtype=c_float, order="F")
        xkrf = np.zeros((nemis, nrad), dtype=c_float, order="F")
        xkcldlnpres = np.zeros((nrad,), dtype=c_float, order="F")
        xkcldlnext = np.zeros((ncloud, nrad), dtype=c_float, order="F")
        drad_datm_jac_spec = np.zeros((nlevels, nrad, njac), dtype=c_float, order="F")

        self.liboss.cppfwdwrapper(
            ctypes.byref(c_int(atmosphere.shape[0])),
            ctypes.byref(c_int(atmosphere.shape[1])),
            np.asfortranarray(pressure, dtype=c_float).ctypes.data_as(POINTER(c_float)),
            np.asfortranarray(tatm, dtype=c_float).ctypes.data_as(POINTER(c_float)),
            ctypes.byref(c_float(tsur)),
            np.asfortranarray(atmosphere, dtype=c_float).ctypes.data_as(
                POINTER(c_float)
            ),
            ctypes.byref(c_int(emis.shape[0])),
            np.asfortranarray(emis, dtype=c_float).ctypes.data_as(POINTER(c_float)),
            (1.0 - np.asfortranarray(emis, dtype=c_float)).ctypes.data_as(
                POINTER(c_float)
            ),
            ctypes.byref(c_float(scale_cloud)),
            ctypes.byref(c_float(pcloud)),
            ctypes.byref(c_int(cloudext.shape[0])),
            np.asfortranarray(cloudext, dtype=c_float).ctypes.data_as(POINTER(c_float)),
            np.asfortranarray(emis_freq, dtype=c_float).ctypes.data_as(
                POINTER(c_float)
            ),
            np.asfortranarray(cloud_freq, dtype=c_float).ctypes.data_as(
                POINTER(c_float)
            ),
            ctypes.byref(c_float(pointing_angle)),
            ctypes.byref(c_float(sun_angle)),
            ctypes.byref(c_float(latitude)),
            ctypes.byref(c_float(surface_altitude)),
            ctypes.byref(c_int(lambertian_flag)),
            ctypes.byref(c_int(drad_datm_jac_spec.shape[2])),
            ctypes.byref(c_int(rad.shape[0])),
            rad.ctypes.data_as(POINTER(c_float)),
            drad_dtemp.ctypes.data_as(POINTER(c_float)),
            drad_dtsur.ctypes.data_as(POINTER(c_float)),
            drad_datm_jac_spec.ctypes.data_as(POINTER(c_float)),
            xkem.ctypes.data_as(POINTER(c_float)),
            xkrf.ctypes.data_as(POINTER(c_float)),
            xkcldlnpres.ctypes.data_as(POINTER(c_float)),
            xkcldlnext.ctypes.data_as(POINTER(c_float)),
        )
        return (
            rad,
            drad_dtemp,
            drad_dtsur,
            drad_datm_jac_spec,
            xkem,
            xkrf,
            xkcldlnpres,
            xkcldlnext,
        )


muses_oss_handle = MusesOssHandle()

__all__ = [
    "muses_oss_handle",
]
