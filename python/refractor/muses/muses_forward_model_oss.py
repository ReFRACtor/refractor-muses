from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier, StateElementIdentifier
from .muses_tes_observation import MusesTesObservation
from .misc import ResultIrk
import os
import ctypes
from ctypes import c_int, POINTER, c_float, c_char_p
from pathlib import Path
import numpy as np
from functools import cached_property
from contextlib import contextmanager
import copy
import sys
import typing
from typing import Any, Iterator

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFileHelper, InputFilePath
    from .current_state import CurrentState
    from .muses_observation import MusesObservation

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
            return
        
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

        # Should perhaps move this logic out of here and into the forward models.
        # We could just pass in sel_file, od_file, sol_file and fix_file
        if False:
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
            if self.first_oss_initialize:
                # First time through, repeat the initialization
                self.first_oss_initialize = False
                do_init = True
            else:
                self.have_oss = True
                do_init = False

oss_handle = OssHandle()

class MusesForwardModelOssBase(rf.ForwardModel):
    '''Base class for our OSS forward models.

    A note on the design here. We usually have a rf.RadiativeTransfer class
    for our forward model (e.g, VLIDORT, LIDORT), and then something like
    a rf.StandardForwardModel that wraps around this. The ForwardModel class
    handles things like spectrum effects (e.g., radiance scaling, raman scattering),
    the ILS if needed, and determines what rf.SpectralDomain the RadiativeTransfer
    should run on.

    Our OSS code is a little different. It can *only* be uses on a predetermined
    frequency grid. We can toss points out, but not add them. So rather than
    having something tell the OSS what rf.SpectralDomain to run on, it tells us
    what rf.SpectralDomain it can run on. We could still shove this into a
    rf.RadiativeTransfer class by just doing something like checking that the
    requested rf.SpectralDomain fits the allowed frequency grid and throwing an
    error if doesn't. But there doesn't seem to be a strong reason do that here.
    Instead, we calculate this at the rf.ForwardModel level and skip a
    rf.RadiativeTransfer object.

    We can revisit this if we determine that there is a strong reason to use the
    traditional structure instead.
    '''
    def __init__(self,
                 obs: MusesObservation,
                 retrieval_config: RetrievalConfiguration,
                 dir_lut: Path | InputFilePath | None = None,
                 **kwargs: Any) -> None:
        super().__init__()
        self.obs = obs
        self.retrieval_config = retrieval_config
        self.ifile_hlp = retrieval_config.input_file_helper
        self.dir_lut = dir_lut

    def setup_grid(self) -> None:
        # Nothing that we need to do for this
        pass

    def _v_num_channels(self) -> int:
        return 1

    def _init_oss(self) -> None():
        raise NotImplementedError()

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        if sensor_index > 0:
            raise RuntimeError("sensor_index out of range")
        sd = np.concatenate(
            [self.obs.spectral_domain(i).data for i in range(self.obs.num_channels)]
        )
        return rf.SpectralDomain(sd, rf.Unit("nm"))

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        if sensor_index > 0:
            raise RuntimeError("sensor_index out of range")
        self._init_oss()
        raise NotImplementedError()


class MusesForwardModelOssIrk(MusesForwardModelOssBase):
    """This add the irk function to calculate the IRK. It seems like this
    calculation similar for different instruments, although classes that support
    this just need to supply a irk function, not necessarily inherit from this
    one."""

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return []

    @property
    def flux_freq_range(self) -> tuple[float, float]:
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers
        return (980.0, 1080.0)

    @property
    def seg_freq_range(self) -> tuple[float, float]:
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers
        return (970.0, 1120.0)

    @property
    def irk_average_freq_range(self) -> tuple[float, float]:
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers
        return (979.99, 1078.999)

    @property
    def irk_weight(self) -> list[float]:
        # The weights to use when creating radianceWeighted and jacWeighted.
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers.
        # Note that this needs to have the same length as the
        # irk_angle
        return [0, 0.096782, 0.167175, 0.146387, 0.073909, 0.015748]

    def irk_radiance(
        self,
        cstate: CurrentState,
        pointing_angle: rf.DoubleWithUnit,
    ) -> tuple[rf.Spectrum, None | np.ndarray]:
        """Calculate radiance/jacobian for the IRK calculation, for the
        given angle. If pointing_angle is 0, we also return dEdOD."""
        # TODO, remove uip here
        from refractor.muses_py_fm import RefractorUip
        rf_uip_pointing = RefractorUip.create_uip_from_refractor_objects(
            [
                self.obs,
            ],
            cstate,
            self.rconf,
            pointing_angle=pointing_angle,
        )
        rf_uip_original = self.rf_uip
        try:
            self.rf_uip = rf_uip_pointing
            with self.obs.modify_spectral_window(include_bad_sample=True):
                r = self.radiance_all(False)
            if pointing_angle.value == 0.0:
                ray_info = rf_uip_pointing.ray_info(
                    self.obs.instrument_name, set_cloud_extinction_one=True
                )
                dEdOD = 1.0 / ray_info["cloud"]["tau_total"]
            else:
                dEdOD = None
        finally:
            self.rf_uip = rf_uip_original
        return r, dEdOD

    def irk(self, current_state: CurrentState) -> ResultIrk:
        """This was originally the run_irk.py code from py-retrieve. We
        have our own copy of this so we can clean this code up a bit.
        """
        t = self.obs.radiance_all_extended(include_bad_sample=True)
        frq_l1b = np.array(t.spectral_domain.data)
        rad_l1b = np.array(t.spectral_range.data)
        radiance = []
        jacobian = []
        for gi_angle in self.irk_angle():
            if gi_angle == 0.0:
                r, dEdOD = self.irk_radiance(
                    current_state, rf.DoubleWithUnit(0.0, "deg")
                )
                frequency = r.spectral_domain.data
            else:
                r, _ = self.irk_radiance(
                    current_state, rf.DoubleWithUnit(gi_angle, "deg")
                )
            radiance.append(r.spectral_range.data)
            jacobian.append(r.spectral_range.data_ad.jacobian.transpose())

        freq_step = frequency[1:] - frequency[:-1]
        freq_step = np.array([freq_step[0], *freq_step])
        n_l1b = len(frq_l1b)

        # need remove missing data in L1b radiance
        ifrq_missing = np.where(rad_l1b == 0.0)
        valid_indices = np.where(rad_l1b != 0.0)[0]  # Ensure 1-D array
        rad_l1b[ifrq_missing] = np.interp(
            ifrq_missing, valid_indices, rad_l1b[valid_indices]
        )

        freq_step_l1b_temp = (frq_l1b[2:] - frq_l1b[0 : n_l1b - 2]) / 2.0
        freq_step_l1b = np.concatenate(
            (
                np.asarray([frq_l1b[1] - frq_l1b[0]]),
                freq_step_l1b_temp,
                np.asarray([frq_l1b[n_l1b - 1] - frq_l1b[n_l1b - 2]]),
            ),
            axis=0,
        )

        radianceWeighted = 2.0 * sum(r * w for r, w in zip(radiance, self.irk_weight))

        radratio = radiance[0] / radianceWeighted
        ifrq = self._find_bin(frequency, frq_l1b)
        radratio = radratio[ifrq]
        ifreq = np.where(
            (frequency >= self.flux_freq_range[0])
            & (frequency <= self.flux_freq_range[1])
        )[0]
        flux = 1e4 * np.pi * np.sum(freq_step[ifreq] * radianceWeighted[ifreq])
        ifreq_l1b = np.where(
            (frq_l1b >= self.flux_freq_range[0]) & (frq_l1b <= self.flux_freq_range[1])
        )[0]
        flux_l1b = (
            1e4
            * np.pi
            * np.sum(
                freq_step_l1b[ifreq_l1b] * rad_l1b[ifreq_l1b] / radratio[ifreq_l1b]
            )
        )
        minn = np.amin(frequency)
        maxx = np.amax(frequency)
        minn, maxx = self.seg_freq_range
        nf = int((maxx - minn) / 3)
        freqSegments: np.ndarray = np.ndarray(shape=(nf), dtype=np.float32)
        freqSegments.fill(
            0
        )  # It is import to start with 0 because not all elements will be calculated.
        fluxSegments: np.ndarray = np.ndarray(shape=(nf), dtype=np.float32)
        fluxSegments.fill(
            0
        )  # It is import to start with 0 because not all elements will be calculated.
        fluxSegments_l1b: np.ndarray = np.ndarray(shape=(nf), dtype=np.float32)
        fluxSegments_l1b.fill(
            0
        )  # It is import to start with 0 because not all elements will be calculated.

        # get split into 3 cm-1 segments
        for ii in range(nf):
            ind = np.where(
                (frequency >= minn + ii * 3) & (frequency < minn + ii * 3 + 3)
            )[0]
            ind_l1b = np.where(
                (frq_l1b >= minn + ii * 3)
                & ((frq_l1b < minn + ii * 3 + 3) & (rad_l1b > 0.0))
            )[0]
            if len(ind_l1b) > 0:
                fluxSegments_l1b[ii] = (
                    1e4
                    * np.pi
                    * np.sum(
                        freq_step_l1b[ind_l1b] * rad_l1b[ind_l1b] / radratio[ind_l1b]
                    )
                )
            if (
                len(ind) > 0
            ):  # We only calculate fluxSegments, fluxSegments_l1b, and freqSegments if there is at least 1 value in ind vector.
                fluxSegments[ii] = (
                    1e4 * np.pi * np.sum(freq_step[ind] * radianceWeighted[ind])
                )
                freqSegments[ii] = np.mean(frequency[ind])

        jacWeighted = 2.0 * sum(jac * w for jac, w in zip(jacobian, self.irk_weight))

        # weight by freq_step
        jacWeighted *= freq_step[np.newaxis, :]

        o_results_irk = ResultIrk(
            {
                "flux": flux,
                "flux_l1b": flux_l1b,
                "fluxSegments": fluxSegments,
                "freqSegments": freqSegments,
                "fluxSegments_l1b": fluxSegments_l1b,
            }
        )

        # smaller range for irk average
        indf = np.where(
            (frequency >= self.irk_average_freq_range[0])
            & (frequency <= self.irk_average_freq_range[1])
        )[0]

        irk_array = 1e4 * np.pi * self.my_total(jacWeighted[:, indf], True)

        minn, maxx = self.flux_freq_range

        nf = int((maxx - minn) / 3)
        irk_segs = np.zeros(shape=(jacWeighted.shape[0], nf), dtype=np.float32)
        freq_segs = np.zeros(shape=(nf), dtype=np.float32)

        for ii in range(nf):
            ind = np.where(
                (frequency >= minn + ii * 3) & (frequency < minn + ii * 3 + 3)
            )[0]
            if (
                len(ind) > 1
            ):  # We only calculate irk_segs and freq_segs if there are more than 1 values in ind vector.
                irk_segs[:, ii] = 1e4 * np.pi * self.my_total(jacWeighted[:, ind], True)
                freq_segs[ii] = np.mean(frequency[ind])
        # end for ii in range(nf):

        # AT_LINE 333 src_ms-2018-12-10/run_irk.pro
        radarr_fm = np.concatenate(radiance, axis=0)
        radInfo = {
            "gi_angle": gi_angle,
            "radarr_fm": radarr_fm,
            "freq_fm": frequency,
            "rad_L1b": rad_l1b,
            "freq_L1b": frq_l1b,
        }
        o_results_irk["freqSegments_irk"] = freq_segs
        o_results_irk["radiances"] = radInfo

        # calculate irk for each type
        for selem_id in current_state.retrieval_state_vector_element_list:
            species_name = str(selem_id)
            pstart, plen = current_state.fm_sv_loc[selem_id]
            ii = pstart
            jj = pstart + plen
            vmr = current_state.initial_guess_full[ii:jj]
            vmr = (
                current_state.state_mapping(selem_id)
                .mapped_state(rf.ArrayAd_double_1(vmr))
                .value
            )
            pressure = current_state.pressure_list_fm(selem_id)

            myirfk = copy.deepcopy(irk_array[ii:jj])
            myirfk_segs = copy.deepcopy(irk_segs[ii:jj, :])

            # TODO This looks like the sort of thing that can be
            # replaced with our StateElement data, to get away from
            # having all this hard coded. But for now, leave this like
            # it was

            # convert cloudext to cloudod
            # dL/dod = dL/dext * dext/dod
            if species_name == "CLOUDEXT":
                if dEdOD is None:
                    raise RuntimeError("dEdOD should not be None")
                myirfk = np.multiply(myirfk, dEdOD)
                for pp in range(dEdOD.shape[0]):
                    myirfk_segs[pp, :] = myirfk_segs[pp, :] * dEdOD[pp]

                species_name = "CLOUDOD"
                vmr = np.divide(vmr, dEdOD)

            mm = jj - ii
            if species_name == "TATM" or species_name == "TSUR":
                mylirfk = np.multiply(myirfk, vmr)
                mylirfk_segs = copy.deepcopy(myirfk_segs)
                for kk in range(mm):
                    mylirfk_segs[kk, :] = mylirfk_segs[kk, :] * vmr[kk]
            else:
                mylirfk = copy.deepcopy(myirfk)
                myirfk = np.divide(myirfk, vmr)
                mylirfk_segs = copy.deepcopy(myirfk_segs)
                for kk in range(mm):
                    myirfk_segs[kk, :] = myirfk_segs[kk, :] / vmr[kk]

            if species_name == "O3":
                mult_factor = 1.0 / 1e9  # W/m2/ppb
                unit = "W/m2/ppb"
            elif species_name == "H2O":
                mult_factor = 1.0 / 1e6  # W/m2/ppm
                unit = "W/m2/ppm"
            elif species_name == "TATM":
                mult_factor = 1.0
                unit = "W/m2/K"
            elif species_name == "TSUR":
                mult_factor = 1.0
                unit = "W/m2/K"
            elif species_name == "EMIS":
                mult_factor = 1.0
                unit = "W/m2"
            elif species_name == "CLOUDDOD":
                mult_factor = 1.0
                unit = "W/m2"
            elif species_name == "PCLOUD":
                mult_factor = 1.0
                unit = "W/m2/hPa"
            else:
                # Fall back
                mult_factor = 1.0
                unit = " "

            myirfk = np.multiply(myirfk, mult_factor)
            myirfk_segs = np.multiply(myirfk_segs, mult_factor)

            # subset only freqs in range
            if species_name == "CLOUDOD":
                myirfk_segs = myirfk_segs[:, 0]
                myirfk_segs = np.reshape(myirfk_segs, (myirfk_segs.shape[0]))

                mylirfk_segs = mylirfk_segs[:, 0]
                mylirfk_segs = np.reshape(mylirfk_segs, (mylirfk_segs.shape[0]))

            vmr = np.divide(vmr, mult_factor)

            # Build a structure of result for each species_name.
            result_per_species = {
                "irfk": myirfk,
                "lirfk": mylirfk,
                "pressure": pressure,
                "unit": unit,
                "irfk_segs": myirfk_segs,
                "lirfk_segs": mylirfk_segs,
                "vmr": vmr,
            }

            # Add the result for each species_name to our structure to return.
            # Note that the name of the species is the key for the dictionary structure.

            o_results_irk[species_name] = copy.deepcopy(
                result_per_species
            )  # o_results_irk
        # end for ispecies in range(len(jacobian_speciesIn)):
        return o_results_irk

    def my_total(self, matrix_in: np.ndarray, ave_index: bool = False) -> np.ndarray:
        size_out = matrix_in.shape[0] if ave_index else matrix_in.shape[1]
        arrayOut = np.ndarray(shape=(size_out,), dtype=np.float64)
        for ii in range(size_out):
            my_vector = matrix_in[ii, :] if ave_index else matrix_in[:, ii]
            # Filter our -999 values
            val = np.sum(my_vector[np.abs(my_vector - (-999)) > 0.1])
            arrayOut[ii] = val
        return arrayOut

    def _find_bin(self, x: float, y: np.ndarray) -> np.ndarray:
        # IDL_LEGACY_NOTE: This function _find_bin is the same as findbin in run_irk.pro file.
        #
        # Returns the bin numbers for nearest value of x array to values of y
        #       returns nearest bin for values outside the range of x
        #
        ny = len(y)

        o_bin: np.ndarray = np.ndarray(shape=(ny), dtype=np.int32)
        for iy in range(0, ny):
            ix = np.argmin(abs(x - y[iy]))
            o_bin[iy] = ix

        if ny == 1:
            o_bin = np.asarray([o_bin[0]])

        return o_bin

class MusesCrisForwardModelOss(MusesForwardModelOssIrk):
    def _init_oss(self) -> None():
        # Different files depends on l1b_type
        if self.obs.instrument_name  == InstrumentIdentifier("CRIS", "suomi_nasa_nsr"):
            if self.dir_lut is None:
                self.dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2023-01-nsr"
            sel_file = (
                    self.dir_lut
                    / "suomi-cris-B1B2B3-unapod-loc-clear-19V-M12.4-v1.0.train.sel"
            )
            od_file = (
                self.dir_lut
                / "suomi-cris-B1B2B3-unapod-loc-clear-19V-M12.4-v1.0.train.lut"
            )
            sol_file = self.dir_lut / "newkur.dat"
            fix_file = self.dir_lut / "default.dat"
        else:
            if self.dir_lut is None:
                self.dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2017-08"
            sel_file = (
                self.dir_lut
                / "suomi-cris-fsr-B1B2B3-unapod-loc-cloudy-23V-M12.4-v1.0.train.sel"
            )

            od_file = (
                self.dir_lut
                / "suomi-cris-fsr-B1B2B3-unapod-loc-cloudy-23V-M12.4-v1.0.train.lut"
            )

            sol_file = self.dir_lut / "newkur.dat"
            fix_file = self.dir_lut / "default.dat"
        # The retrieval and species list seem to be hardcoded. I think this
        # corresponds to what is available in the various input files
        retrieval_state_element_id = [
            StateElementIdentifier(i)
            for i in ["H2O", "O3", "TSUR", "EMIS", "CLOUDEXT", "PCLOUD"]
        ]
        species_list = [
                StateElementIdentifier(i)
                for i in [
                    "PRESSURE",
                    "TATM",
                    "H2O",
                    "CO2",
                    "O3",
                    "N2O",
                    "CO",
                    "CH4",
                    "SO2",
                    "NH3",
                    "HNO3",
                    "OCS",
                    "N2",
                    "HCN",
                    "SF6",
                    "HCOOH",
                    "CCL4",
                    "CFC11",
                    "CFC12",
                    "CFC22",
                    "HDO",
                    "CH3OH",
                    "C2H4",
                    "PAN",
                ]
            ]
        # We need to come up with a way to get these values
        nlevels = 64
        nfreq = 121
        oss_handle.oss_init(self.ifile_hlp, retrieval_state_element_id,
                            species_list, nlevels, nfreq, sel_file,
                            od_file, sol_file, fix_file)

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.2906, 31.8588, 46.9590, 57.3154, 61.5613]


class MusesAirsForwardModelOss(MusesForwardModelOssIrk):
    def _init_oss(self) -> None():
        raise NotImplementedError()
    
    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]

    @cached_property
    def irk_obs(self) -> MusesObservation:
        """Observation to use in IRK calculation."""
        # Replace with a fake TES observation. This is done to get the
        # full TES frequency range.
        tes_frequency_fname = (
            f"{self.rconf['spectralWindowDirectory']}/../../tes_frequency.nc"
        )
        return MusesTesObservation.create_fake_for_irk(
            tes_frequency_fname, self.obs.spectral_window, self.rconf.input_file_helper
        )

    def irk_radiance(
        self,
        cstate: CurrentState,
        pointing_angle: rf.DoubleWithUnit,
    ) -> tuple[rf.Spectrum, None | np.ndarray]:
        """Calculate radiance/jacobian for the IRK calculation, for the
        given angle. We also return the UIP we used for the calculation"""
        # For AIRS, we use the TES forward model instead. Based on comments in
        # the code it looks like this was done use the more complete frequency
        # range of TES
        obs_original = self.obs
        try:
            self.obs = self.irk_obs
            self.instrument_name = InstrumentIdentifier("TES")
            return super().irk_radiance(cstate, pointing_angle)
        finally:
            self.obs = obs_original
            self.instrument_name = InstrumentIdentifier("AIRS")


class MusesTesForwardModelOss(MusesForwardModelOssIrk):
    def _init_oss(self) -> None():
        raise NotImplementedError()

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]
    
    
__all__ = [
    "oss_handle", "MusesForwardModelOssBase", "MusesForwardModelOssIrk",
    "MusesCrisForwardModelOss", "MusesAirsForwardModelOss", "MusesTesForwardModelOss",
]
