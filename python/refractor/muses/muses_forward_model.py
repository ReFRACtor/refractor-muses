from __future__ import annotations
from . import fake_muses_py as mpy  # type: ignore
from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
from .osswrapper import osswrapper
from .refractor_capture_directory import muses_py_call
from .muses_observation import MusesTesObservation
from .identifier import InstrumentIdentifier
import refractor.framework as rf  # type: ignore
from functools import cached_property
from loguru import logger
import numpy as np
import copy
from collections import UserDict
import typing
from typing import Callable, Any, TypeVar

if typing.TYPE_CHECKING:
    from .refractor_uip import RefractorUip
    from .muses_observation import MusesObservation, MeasurementId
    from .current_state import CurrentState

# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel

# There are a number of things in common with the different forward models,
# so we capture these in these base classes.


class MusesForwardModelBase(rf.ForwardModel):
    """Common behavior for the different MUSES forward models"""

    def __init__(
        self,
        rf_uip: RefractorUip,
        instrument_name: InstrumentIdentifier,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.instrument_name = instrument_name
        self.rf_uip = rf_uip
        self.obs = obs
        self.kwargs = kwargs
        self.measurement_id = measurement_id

    def bad_sample_mask(self, sensor_index: int) -> np.ndarray:
        bmask = self.obs.bad_sample_mask(sensor_index)
        if self.obs.spectral_window.include_bad_sample:
            bmask[:] = False
        # This is the full bad sample mask, for all the indices. But here we only
        # want the portion that fits in the spectral window
        with self.obs.modify_spectral_window(include_bad_sample=True):
            sd = self.obs.spectral_domain_full(sensor_index)
            gindex = self.obs.spectral_window.grid_indexes(sd, sensor_index)
        return bmask[list(gindex)]

    def setup_grid(self) -> None:
        # Nothing that we need to do for this
        pass

    def _v_num_channels(self) -> int:
        return 1

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        if sensor_index > 0:
            raise RuntimeError("sensor_index out of range")
        sd = np.concatenate(
            [self.obs.spectral_domain(i).data for i in range(self.obs.num_channels)]
        )
        return rf.SpectralDomain(sd, rf.Unit("nm"))


# Wrapper so we can get timing at a top level of ReFRACtor relative to the rest of the code
# using something like --profile-svg in pytest
class RefractorForwardModel(rf.ForwardModel):
    def __init__(self, fm: rf.ForwardModel) -> None:
        super().__init__()
        self.fm = fm

    def setup_grid(self) -> None:
        self.fm.setup_grid()

    def _v_num_channels(self) -> int:
        return self.fm.num_channels

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        return self.fm.spectral_domain(sensor_index)

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        return self.fm.radiance(sensor_index, skip_jacobian)


class MusesOssForwardModelBase(MusesForwardModelBase):
    """Common behavior for the OSS based forward models"""

    def __init__(
        self,
        rf_uip: RefractorUip,
        instrument_name: InstrumentIdentifier,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__(rf_uip, instrument_name, obs, measurement_id, **kwargs)

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        if sensor_index != 0:
            raise ValueError("sensor_index must be 0")
        with osswrapper(self.rf_uip.uip):
            rad, jac = mpy.fm_oss_stack(self.rf_uip.uip_all(self.instrument_name))
        # This is for the full set of frequences that fm_oss_stack works with
        # (which of course includes bad samples)
        gmask = self.bad_sample_mask(sensor_index) != True
        # This already has bad samples removed
        sd = self.spectral_domain(sensor_index)
        # We ran into this issue because the fm_oss_stack uses the UIP for the
        # frequency grid, and self.obs uses MusesSpectralWindow. These are
        # generally the same, but we might get odd situations where they aren't.
        # In particular, this led us to a special case for TES where we needed
        # TesSpectralWindow instead of MusesSpectralWindow.
        #
        # In case we run into this again, give a clearer error message. Still
        # need to fix the underlying cause, but at least have a clear description
        # of what went wrong.
        if gmask.shape[0] != rad.shape[0]:
            raise RuntimeError(
                f"gmask and rad don't match in size. gmask size is {gmask.shape[0]} and rad size if {rad.shape[0]}"
            )
        if skip_jacobian:
            sr = rf.SpectralRange(rad[gmask], rf.Unit("sr^-1"))
        else:
            # jacobian is 1) on the forward model grid and
            # 2) transposed from the ReFRACtor convention of the
            # column being the state vector variables. So
            # translate the oss jac to what we want from ReFRACtor

            sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(
                self.instrument_name
            )
            if jac is not None and jac.ndim > 0:
                if self.rf_uip.is_bt_retrieval:
                    # Only one column has data, although oss returns a larger
                    # jacobian. Note that fm_wrapper just "knows" this, it
                    # would be nice if this wasn't sort of magic knowledge.
                    jac = jac.transpose()[:, 0:1]
                else:
                    jac = np.matmul(sub_basis_matrix, jac).transpose()
                a = rf.ArrayAd_double_1(rad[gmask], jac[gmask, :])
            else:
                a = rf.ArrayAd_double_1(rad[gmask])
            sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)


class ResultIrk(UserDict):
    """This holds the results of IRK. This is basically just a dict,
    with a few extra functions. We can perhaps create a proper class
    at some point, but right now there isn't much of a need for this.
    The old py-retrieve code that uses ResultIrk is expecting a dict."""

    def __getattr__(self, nm: str) -> Any:
        if nm in self.data:
            return self[nm]
        raise AttributeError()

    def __setattr__(self, nm: str, value: Any) -> None:
        if nm in ("data",):
            super().__setattr__(nm, value)
        if nm in self.data:
            self[nm] = value
        else:
            super().__setattr__(nm, value)

    def get_state(self) -> dict[str, Any]:
        res = dict(copy.deepcopy(self.data))
        for k in res.keys():
            if k in (
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
            ):
                res[k] = res[k].tolist()
            if k == "radiances":
                for k2 in ("radarr_fm", "freq_fm", "rad_L1b", "freq_L1b"):
                    res[k][k2] = res[k][k2].tolist()
            if k not in (
                "flux",
                "flux_l1b",
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
                "radiances",
            ):
                for k2 in (
                    "irfk",
                    "lirfk",
                    "pressure",
                    "irfk_segs",
                    "lirfk_segs",
                    "vmr",
                ):
                    if res[k][k2] is None:
                        res[k][k2] = None
                    else:
                        res[k][k2] = res[k][k2].tolist()
        return res

    def set_state(self, d: dict[str, Any]) -> None:
        self.data = d
        for k in self.data.keys():
            if k in (
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
            ):
                self.data[k] = np.array(self.data[k])
            if k == "radiances":
                for k2 in ("radarr_fm", "freq_fm", "rad_L1b", "freq_L1b"):
                    self.data[k][k2] = np.array(self.data[k][k2])
            if k not in (
                "flux",
                "flux_l1b",
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
                "radiances",
            ):
                for k2 in (
                    "irfk",
                    "lirfk",
                    "pressure",
                    "irfk_segs",
                    "lirfk_segs",
                    "vmr",
                ):
                    self.data[k][k2] = np.array(self.data[k][k2])


class MusesForwardModelIrk(MusesOssForwardModelBase):
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
        pointing_angle: rf.DoubleWithUnit,
        rf_uip_func: Callable[[MusesObservation, rf.DoubleWithUnit], RefractorUip],
    ) -> tuple[rf.Spectrum, None | np.ndarray]:
        """Calculate radiance/jacobian for the IRK calculation, for the
        given angle. If pointing_angle is 0, we also return dEdOD."""
        rf_uip_pointing = rf_uip_func(self.obs, pointing_angle)
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

    def irk(
        self,
        current_state: CurrentState,
        rf_uip_func: Callable[[MusesObservation, rf.DoubleWithUnit], RefractorUip],
    ) -> ResultIrk:
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
                r, dEdOD = self.irk_radiance(rf.DoubleWithUnit(0.0, "deg"), rf_uip_func)
                frequency = r.spectral_domain.data
            else:
                r, _ = self.irk_radiance(
                    rf.DoubleWithUnit(gi_angle, "deg"), rf_uip_func
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

        irk_array = 1e4 * np.pi * mpy.my_total(jacWeighted[:, indf], 1)

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
                irk_segs[:, ii] = 1e4 * np.pi * mpy.my_total(jacWeighted[:, ind], 1)
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


class MusesTropomiOrOmiForwardModelBase(MusesForwardModelBase):
    """Common behavior for the omi/tropomi based forward models"""

    def __init__(
        self,
        rf_uip: RefractorUip,
        instrument_name: InstrumentIdentifier,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        vlidort_nstokes: int = 2,
        vlidort_nstreams: int = 4,
        **kwargs: Any,
    ) -> None:
        MusesForwardModelBase.__init__(
            self, rf_uip, instrument_name, obs, measurement_id, **kwargs
        )
        self.vlidort_nstreams = vlidort_nstreams
        self.vlidort_nstokes = vlidort_nstokes

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        if sensor_index != 0:
            raise ValueError("sensor_index must be 0")
        with muses_py_call(
            self.rf_uip.run_dir,
            vlidort_nstokes=self.vlidort_nstokes,
            vlidort_nstreams=self.vlidort_nstreams,
        ):
            if self.instrument_name == InstrumentIdentifier("TROPOMI"):
                jac, rad, _, success_flag = mpy.tropomi_fm(
                    self.rf_uip.uip_all(self.instrument_name)
                )
            elif self.instrument_name == InstrumentIdentifier("OMI"):
                jac, rad, _, success_flag = mpy.omi_fm(
                    self.rf_uip.uip_all(self.instrument_name)
                )
            else:
                raise RuntimeError(
                    f"Unrecognized instrument name {self.instrument_name}"
                )
        # Haven't filled everything in yet, but mark as cache full.
        # otherwise bad_sample_mask and spectral_domain will enter an
        # infinite loop
        self.cache_valid_flag = True
        gmask = np.concatenate(
            [self.bad_sample_mask(i) != True for i in range(self.obs.num_channels)]
        )
        sd = self.spectral_domain(0)
        # jacobian is 1) on the forward model grid and
        # 2) transposed from the ReFRACtor convention of the
        # column being the state vector variables. So
        # translate the oss jac to what we want from ReFRACtor
        # The logic in pack_omi_jacobian and pack_tropomi_jacobian
        # over counts the size of atmosphere jacobians by 1 for each
        # species. This is harmless,
        # it gives an extra row of zeros that then gets trimmed before leaving
        # fm_wrapper. But because we are calling the lower level function
        # ourselves we need to trim this.
        sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name)
        if jac is not None and jac.shape[0] > 0 and sub_basis_matrix.shape[1] > 0:
            jac = np.matmul(
                sub_basis_matrix, jac[: sub_basis_matrix.shape[1], :]
            ).transpose()
            a = rf.ArrayAd_double_1(rad[gmask], jac[gmask, :])
        else:
            a = rf.ArrayAd_double_1(rad[gmask])
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)


class MusesTropomiForwardModel(MusesTropomiOrOmiForwardModelBase):
    def __init__(
        self,
        rf_uip: RefractorUip,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rf_uip, InstrumentIdentifier("TROPOMI"), obs, measurement_id, **kwargs
        )


class MusesOmiForwardModel(MusesTropomiOrOmiForwardModelBase):
    def __init__(
        self,
        rf_uip: RefractorUip,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rf_uip, InstrumentIdentifier("OMI"), obs, measurement_id, **kwargs
        )


class MusesCrisForwardModel(MusesForwardModelIrk):
    """Wrapper around fm_oss_stack call for CRiS instrument"""

    def __init__(
        self,
        rf_uip: RefractorUip,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rf_uip, InstrumentIdentifier("CRIS"), obs, measurement_id, **kwargs
        )

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.2906, 31.8588, 46.9590, 57.3154, 61.5613]


class MusesAirsForwardModel(MusesForwardModelIrk):
    """Wrapper around fm_oss_stack call for Airs instrument"""

    def __init__(
        self,
        rf_uip: RefractorUip,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rf_uip, InstrumentIdentifier("AIRS"), obs, measurement_id, **kwargs
        )

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]

    @cached_property
    def irk_obs(self) -> MusesObservation:
        """Observation to use in IRK calculation."""
        # Replace with a fake TES observation. This is done to get the
        # full TES frequency range.
        tes_frequency_fname = (
            f"{self.measurement_id['spectralWindowDirectory']}/../../tes_frequency.nc"
        )
        return MusesTesObservation.create_fake_for_irk(
            tes_frequency_fname, self.obs.spectral_window
        )

    def irk_radiance(
        self,
        pointing_angle: rf.DoubleWithUnit,
        rf_uip_func: Callable[[MusesObservation, rf.DoubleWithUnit], RefractorUip],
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
            return super().irk_radiance(pointing_angle, rf_uip_func)
        finally:
            self.obs = obs_original
            self.instrument_name = InstrumentIdentifier("AIRS")


class MusesTesForwardModel(MusesForwardModelIrk):
    """Wrapper around fm_oss_stack call for TES instrument"""

    def __init__(
        self,
        rf_uip: RefractorUip,
        obs: MusesObservation,
        measurement_id: MeasurementId,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rf_uip, InstrumentIdentifier("TES"), obs, measurement_id, **kwargs
        )

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]


class StateVectorPlaceHolder(rf.StateVectorObserver):
    """Place holder for parts of the StateVector that ReFRACtor objects
    don't need. Just gives the right name in the state vector, and
    act as a placeholder for any future stuff."""

    def __init__(self, pstart: int, plen: int, species_name: str) -> None:
        super().__init__()
        self.pstart = pstart
        self.plen = plen
        self.species_name = species_name
        self.coeff = None

    def notify_update(self, sv: rf.StateVector) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.coeff = sv.state[self.pstart : (self.pstart + self.plen)]

    def state_vector_name(self, sv: rf.StateVector, sv_namev: list[str]) -> None:
        svnm = [
            "",
        ] * len(sv.state)
        for i in range(self.plen):
            svnm[i + self.pstart] = f"{self.species_name} coefficient {i + 1}"
        self.sv_name = svnm
        super().state_vector_name(sv, sv_namev)


C = TypeVar("C", bound=rf.ForwardModel)


class MusesForwardModelHandle(ForwardModelHandle):
    def __init__(
        self, instrument_name: InstrumentIdentifier, cls: type[C], **creator_kwargs: Any
    ) -> None:
        self.creator_kwargs = creator_kwargs
        self.instrument_name = instrument_name
        self.cls = cls
        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        rf_uip_func: Callable[[InstrumentIdentifier | None], RefractorUip] | None,
        **kwargs: Any,
    ) -> None | rf.ForwardModel:
        if instrument_name != self.instrument_name:
            return None
        if rf_uip_func is None:
            return None
        logger.debug(f"Creating forward model {self.cls.__name__}")
        # Note MeasurementId also has access to all the stuff in
        # RetrievalConfiguration
        return self.cls(
            rf_uip_func(instrument_name), obs, self.measurement_id, **kwargs
        )


# The Muses code is the fallback, so add with the lowest priority
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("CRIS"), MusesCrisForwardModel),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("AIRS"), MusesAirsForwardModel),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("TES"), MusesTesForwardModel),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("TROPOMI"), MusesTropomiForwardModel),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("OMI"), MusesOmiForwardModel),
    priority_order=-1,
)

__all__ = [
    "MusesCrisForwardModel",
    "MusesAirsForwardModel",
    "MusesTesForwardModel",
    "MusesTropomiForwardModel",
    "MusesOmiForwardModel",
    "ResultIrk",
]
