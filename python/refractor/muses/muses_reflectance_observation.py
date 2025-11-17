from __future__ import annotations
from .misc import osp_setup
from .observation_handle import ObservationHandleSet
from .muses_observation import (
    MusesObservationImp,
    MusesObservationHandle,
    MeasurementId,
    MusesObservation,
)
from .muses_spectral_window import MusesSpectralWindow
from .mpy import (
    mpy_read_tropomi,
    mpy_read_tropomi_surface_altitude,
    mpy_read_omi,
)
from .refractor_uip import AttrDictAdapter
import os
import numpy as np
import refractor.framework as rf  # type: ignore
from loguru import logger
import pickle
import subprocess
import re
from datetime import datetime
from typing import Any, Self
import typing
import math
from .identifier import InstrumentIdentifier, StateElementIdentifier, FilterIdentifier
import netCDF4

if typing.TYPE_CHECKING:
    from .current_state import CurrentState

# Logically you might expect MusesOmiObservation and MusesTropomiObservation to be in
# refractor.omi and refractor.tropomi. However they share a whole lot in common, both
# working in reflectance rather than radiance.
#
# Instead of artificially separating them, we just have both of these here in
# refractor.muses. We can revisit this if needed, but for now this makes the most sense.


def _new_from_init(cls, *args):  # type: ignore
    """For use with pickle, covers common case where we just store the
    arguments needed to create an object.
    """
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst


class MusesDispersion:
    """Helper class, just pull out the calculation of the wavelength
    at the pixel grid.  This is pretty similar to
    rf.DispersionPolynomial, but there are enough differences that it
    is worth pulling this out.  Note that for convenience we don't
    actually handle the StateVector here, instead we just handle the
    routing from the classes that use this.

    Also, we include all pixels, including bad samples. Filtering of
    bad sample happens outside of this class.

    """

    def __init__(
        self,
        original_wav: np.ndarray,
        bad_sample_mask: np.ndarray,
        parent_obj: MusesObservation,
        offset_index: int,
        slope_index: int,
        order: int,
    ) -> None:
        """For convenience, we take the offset and slope as a index
        into parent.mapped_state.  This allows us to directly use data
        from the Observation class without needing to worry about
        routing this.

        Note we had previously just passed a lambda function of offset
        and slope, but we want to be able to pickle this object and we
        can't pickle lambdas, at least without extra work (e.g., using
        dill or hand coding something)

        """
        self.orgwav = original_wav.copy()
        self.parent_obj = parent_obj
        self.offset_index = offset_index
        self.slope_index = slope_index
        self.order = order
        self.orgwav_mean = np.mean(original_wav[bad_sample_mask != True])

    def pixel_grid(self) -> list[rf.AutoDerivativeDouble]:
        """Return the pixel grid. This is in "nm", although for
        convenience we just return the data.

        """
        if self.order == 1:
            offset = self.parent_obj.mapped_state[self.offset_index]
            return [
                rf.AutoDerivativeDouble(float(self.orgwav[i])) - offset
                for i in range(self.orgwav.shape[0])
            ]
        elif self.order == 2:
            offset = self.parent_obj.mapped_state[self.offset_index]
            slope = self.parent_obj.mapped_state[self.slope_index]
            return [
                rf.AutoDerivativeDouble(float(self.orgwav[i]))
                - (
                    offset
                    + (
                        rf.AutoDerivativeDouble(float(self.orgwav[i]))
                        - self.orgwav_mean
                    )
                    * slope
                )
                for i in range(self.orgwav.shape[0])
            ]
        else:
            raise RuntimeError("order needs to be 1 or 2.")


class LinearInterpolate(rf.LinearInterpolateAutoDerivative):
    """The refractor LinearInterpolateAutoDerivative is what we want
    to use for our interpolation, but it is pretty low level and is
    also not something that can be pickled. We add a little higher
    level interface here. This might be generally useful, we can
    elevate this if it turns out to be useful. But right now, this
    just lives in our MusesObservation code.

    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.copy()
        self.y = y.copy()
        x_ad = rf.vector_auto_derivative()
        y_ad = rf.vector_auto_derivative()
        for xv in self.x:
            x_ad.append(rf.AutoDerivativeDouble(float(xv)))
        for yv in self.y:
            y_ad.append(rf.AutoDerivativeDouble(float(yv)))
        super().__init__(x_ad, y_ad)

    def __reduce__(self):  # type: ignore
        return (_new_from_init, (self.__class__, self.x, self.y))


class LinearInterpolate2(rf.LinearInterpolateAutoDerivative):
    def __init__(self, x: np.ndarray, y: list[rf.AutoDerivativeDouble]) -> None:
        self.x = x.copy()
        self.y = y.copy()
        x_ad = rf.vector_auto_derivative()
        y_ad = rf.vector_auto_derivative()
        for xv in self.x:
            x_ad.append(rf.AutoDerivativeDouble(float(xv)))
        for yv in self.y:
            y_ad.append(yv)
        super().__init__(x_ad, y_ad)


class MusesReflectanceObservation(MusesObservationImp):
    """Both omi and tropomi actually use reflectance rather than
    radiance. In addition, both the solar model and the radiance data
    have state elements that control the Dispersion for the data.

    This object captures the common behavior between the two.

    """

    def __init__(
        self,
        muses_py_dict: dict[str, Any],
        sdesc: dict[str, Any],
        filter_list: list[FilterIdentifier],
        existing_obs: Self | None = None,
        coeff: np.ndarray | None = None,
        mp: rf.StateMapping | None = None,
    ) -> None:
        self.filter_list = filter_list
        # Placeholder values if not passed in
        if coeff is None:
            coeff = np.zeros((len(self.filter_list) * 3))
            mp = rf.StateMappingLinear()
        super().__init__(
            muses_py_dict, sdesc, num_channels=len(self.filter_list), coeff=coeff, mp=mp
        )

        # Grab values from existing_obs if available
        if existing_obs is not None:
            self._freq_data: list[np.ndarray] = existing_obs._freq_data
            self._nesr_data: list[np.ndarray] = existing_obs._nesr_data
            self._bsamp: list[np.ndarray] = existing_obs._bsamp
            self._solar_interp: list[LinearInterpolate] = existing_obs._solar_interp
            self._earth_rad: list[np.ndarray] = existing_obs._earth_rad
            self._nesr: list[np.ndarray] = existing_obs._nesr
            self._solar_spectrum: list[rf.Spectrum] = existing_obs._solar_spectrum
        else:
            # Stash some values we use in later calculations. Note
            # that the radiance data is all smooshed together, so we
            # separate this.
            #
            # It isn't clear here if the best indexing is the full
            # instrument (so 8 bands) with only some of the bands
            # filled in, or instead the index number into the passed
            # in filter_list. For now, we are using the index into the
            # filter_list. We can possibly reevaluate this - it
            # wouldn't be huge change in the code we have here.
            self._freq_data = []
            self._nesr_data = []
            self._bsamp = []
            self._solar_interp = []
            self._earth_rad = []
            self._nesr = []
            self._solar_spectrum = []
            erad = muses_py_dict["Earth_Radiance"]
            srad = muses_py_dict["Solar_Radiance"]
            for i, flt in enumerate([str(i) for i in filter_list]):
                flt_sub = erad["EarthWavelength_Filter"] == str(flt)
                self._freq_data.append(erad["Wavelength"][flt_sub])
                self._nesr_data.append(erad["EarthRadianceNESR"][flt_sub])
                self._bsamp.append(
                    (erad["EarthRadianceNESR"][flt_sub] <= 0.0)
                    | (srad["AdjustedSolarRadiance"][flt_sub] <= 0.0)
                )
                self._earth_rad.append(erad["CalibratedEarthRadiance"][flt_sub])
                self._nesr.append(erad["EarthRadianceNESR"][flt_sub])

                # Note this looks wrong (why not use Solar_Radiance
                # Wavelength here?), but is actually correct. The
                # solar data has already been interpolated to the same
                # wavelengths as the Earth_Radiance, this happens in
                # daily_tropomi_irad for TROPOMI, and similarly for
                # OMI. Not sure why the original wavelengths are left
                # in rad_info['Solar_Radiance'], that is actually
                # misleading.

                sol_domain = rf.SpectralDomain(
                    erad["Wavelength"][flt_sub], rf.Unit("nm")
                )
                sol_range = rf.SpectralRange(
                    srad["AdjustedSolarRadiance"][flt_sub], rf.Unit("ph / nm / s")
                )
                self._solar_spectrum.append(rf.Spectrum(sol_domain, sol_range))

                # Create a interpolator for the solar model, only using good data.
                solar_data = srad["AdjustedSolarRadiance"][flt_sub]
                orgwav_good = self._freq_data[i][self.bad_sample_mask(i) != True]
                solar_good = solar_data[self.bad_sample_mask(i) != True]
                self._solar_interp.append(LinearInterpolate(orgwav_good, solar_good))

        # Always create a new _solar_wav and _norm_rad_wav because the
        # dispersion will have independent values
        self._solar_wav = []
        self._norm_rad_wav = []
        for i in range(len(filter_list)):
            self._solar_wav.append(
                MusesDispersion(
                    self._freq_data[i],
                    self.bad_sample_mask(i),
                    self,
                    0 * len(self.filter_list) + i,
                    -1,
                    order=1,
                )
            )
            self._norm_rad_wav.append(
                MusesDispersion(
                    self._freq_data[i],
                    self.bad_sample_mask(i),
                    self,
                    1 * len(self.filter_list) + i,
                    2 * len(self.filter_list) + i,
                    order=2,
                )
            )

    def desc(self) -> str:
        return "MusesReflectanceObservation"

    @property
    def filter_data(self) -> list[tuple[FilterIdentifier, int]]:
        self._filter_data_name = self.filter_list
        self._filter_data_swin = self._spectral_window
        return super().filter_data

    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._freq_data[sensor_index]

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._nesr_data[sensor_index]

    def bad_sample_mask(self, sensor_index: int) -> np.ndarray:
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._bsamp[sensor_index]

    def solar_spectrum(self, sensor_index: int) -> rf.Spectrum:
        """Not sure how much sense it makes, but the RamanSioris gets
        it solar model from the observation. I suppose this sort of
        makes sense, because we already need the solar model.  It
        seems like this should be a separate thing, and perhaps at
        some point we will pull this out. But for now make this
        available here.

        Note this is the original, unaltered solar model - so without
        the TROPOMISOLARSHIFT or OMINRADWAV.

        """
        return self.spectral_window.apply(
            self._solar_spectrum[sensor_index], sensor_index
        )

    def solar_radiance(self, sensor_index: int) -> list[rf.AutoDerivativeDouble]:
        """Use our interpolator to get the solar model at the shifted
        spectrum. This is for all data, so filtering out bad sample
        happens outside of this function.

        """
        pgrid = self._solar_wav[sensor_index].pixel_grid()
        return [self._solar_interp[sensor_index](wav) for wav in pgrid]

    def norm_radiance(self, sensor_index: int) -> list[rf.AutoDerivativeDouble]:
        """Calculate the normalized radiance. This is for all data, so
        filtering out bad sample happens outside of this function.

        """
        pgrid = self._norm_rad_wav[sensor_index].pixel_grid()
        ninterp = self._norm_rad_interp(sensor_index)
        return [ninterp(wav) for wav in pgrid]

    def norm_rad_nesr(self, sensor_index: int) -> np.ndarray:
        """Calculate the normalized radiance. This is for all data, so
        filtering out bad sample happens outside of this function.

        """
        sol_rad = self.solar_radiance(sensor_index)
        return np.array(
            [
                self._nesr[sensor_index][i] / sol_rad[i].value
                for i in range(len(self._nesr[sensor_index]))
            ]
        )

    def _norm_rad_interp(self, sensor_index: int) -> LinearInterpolate:
        """Calculate the interpolator used for the normalized
        radiance. This can't be done ahead of time, because the solar
        radiance used is the interpolated solar radiance.

        """
        solar_rad = self.solar_radiance(sensor_index)
        norm_rad_good = [
            rf.AutoDerivativeDouble(self._earth_rad[sensor_index][i]) / solar_rad[i]
            for i in range(len(self._earth_rad[sensor_index]))
            if self.bad_sample_mask(sensor_index)[i] != True
        ]
        orgwav_good = self._freq_data[sensor_index][
            self.bad_sample_mask(sensor_index) != True
        ]
        return LinearInterpolate2(orgwav_good, norm_rad_good)

    def snr_uplimit(self, sensor_index: int) -> float:
        """Upper limit for SNR, we adjust uncertainty if we are greater than this."""
        raise NotImplementedError()

    def spectrum_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> rf.Spectrum:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        nrad = self.norm_radiance(sensor_index)
        uncer = self.norm_rad_nesr(sensor_index)
        nrad_val = np.array([nrad[i].value for i in range(len(nrad))])
        snr = nrad_val / uncer
        uplimit = self.snr_uplimit(sensor_index)
        tind = np.asarray(snr > uplimit)
        uncer[tind] = nrad_val[tind] / uplimit
        uncer[self.bad_sample_mask(sensor_index) == True] = -999.0
        nrad_ad = rf.ArrayAd_double_1(len(nrad), self.coefficient.number_variable)
        for i, v in enumerate(self.bad_sample_mask(sensor_index)):
            nrad_ad[i] = rf.AutoDerivativeDouble(-999.0) if v == True else nrad[i]
        sr = rf.SpectralRange(nrad_ad, rf.Unit("sr^-1"), uncer)
        sd = self.spectral_domain_full(sensor_index)
        return rf.Spectrum(sd, sr)

    def radiance_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> np.ndarray | rf.ArrayAd_double_1:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        if skip_jacobian:
            return self.spectrum_full(
                sensor_index, skip_jacobian=True
            ).spectral_range.data
        else:
            return self.spectrum_full(
                sensor_index, skip_jacobian=False
            ).spectral_range.data_ad


class MusesTropomiObservation(MusesReflectanceObservation):
    """Observation for Tropomi"""

    def __init__(
        self,
        muses_py_dict: dict[str, Any],
        sdesc: dict[str, Any],
        filter_list: list[FilterIdentifier],
        existing_obs: Self | None = None,
        coeff: np.ndarray | None = None,
        mp: rf.StateMapping | None = None,
    ) -> None:
        """Note you don't normally create an object of this class with the
        __init__. Instead, call one of the create_xxx class methods."""
        super().__init__(
            muses_py_dict,
            sdesc,
            filter_list,
            existing_obs=existing_obs,
            coeff=coeff,
            mp=mp,
        )

    @classmethod
    def _read_data(
        cls,
        filename_dict: dict[str, str | os.PathLike[str]],
        xtrack_dict: dict[str, int],
        atrack_dict: dict[str, int],
        utc_time: str,
        filter_list: list[str],
        calibration_filename: str | None = None,
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Filter list should be in the same order as filename_list,
        # and should be things like "BAND3"
        if calibration_filename is not None:
            # The existing py-retrieve code doesn't actually work with
            # the calibration_filename. This needs to get added before
            # we can use this
            raise RuntimeError("We don't support TROPOMI calibration yet")
        i_windows = [
            {"instrument": "TROPOMI", "filter": str(flt)} for flt in filter_list
        ]
        o_tropomi = cls.read_tropomi(
            filename_dict, xtrack_dict, atrack_dict, utc_time, i_windows, osp_dir
        )
        sdesc = {
            "TROPOMI_ATRACK_INDEX_BAND1": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND1": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND1": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND2": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND2": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND2": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND3": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND3": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND3": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND4": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND4": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND4": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND5": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND5": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND5": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND6": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND6": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND6": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND7": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND7": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND7": -999.0,
            "TROPOMI_ATRACK_INDEX_BAND8": np.int16(-999),
            "TROPOMI_XTRACK_INDEX_BAND8": np.int16(-999),
            "POINTINGANGLE_TROPOMI_BAND8": -999.0,
        }
        # TODO Fill in POINTINGANGLE_TROPOMI
        for i, flt in enumerate([str(i) for i in filter_list]):
            sdesc[f"TROPOMI_XTRACK_INDEX_{flt}"] = np.int16(xtrack_dict[flt])
            sdesc[f"TROPOMI_ATRACK_INDEX_{flt}"] = np.int16(atrack_dict[flt])
            # Think this is right
            sdesc[f"POINTINGANGLE_TROPOMI_{flt}"] = o_tropomi["Earth_Radiance"][
                "ObservationTable"
            ]["ViewingZenithAngle"][i]
        return (o_tropomi, sdesc)

    @classmethod
    def read_tropomi(
        cls,
        filename_dict: dict[str, str | os.PathLike[str]],
        xtrack_dict: dict[str, int],
        atrack_dict: dict[str, int],
        utc_time: str,
        i_windows: list[dict[str, str]],
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        with osp_setup(osp_dir):
            o_tropomi = mpy_read_tropomi(
                {k: str(v) for (k, v) in filename_dict.items()},
                xtrack_dict,
                atrack_dict,
                utc_time,
                i_windows,
            )
            # Reading of the surface height is done separately. In muses-py this
            # was a side effect of setting up the StateInfo.
            for i in range(
                len(o_tropomi["Earth_Radiance"]["ObservationTable"]["ATRACK"])
            ):
                surfaceAltitude = mpy_read_tropomi_surface_altitude(
                    o_tropomi["Earth_Radiance"]["ObservationTable"]["Latitude"][i],
                    o_tropomi["Earth_Radiance"]["ObservationTable"]["Longitude"][i],
                )
                o_tropomi["Earth_Radiance"]["ObservationTable"]["TerrainHeight"][i] = (
                    surfaceAltitude
                )
        return o_tropomi

    @classmethod
    def combine_tropomi_erad(
        cls,
        tropomi_fns: dict[str, str],
        iXTracks: dict[str, str],
        iATracks: dict[str, str],
        windows: list[dict[str, str]],
    ) -> dict[str, Any]:
        o_ObservationTable: dict[str, Any] = {
            "ATRACK": [],
            "XTRACK": [],
            "Latitude": [],
            "Longitude": [],
            "Time": [],
            "SpacecraftLatitude": [],
            "SpacecraftLongitude": [],
            "SpacecraftAltitude": [],
            "TerrainHeight": [],
            "SolarAzimuthAngle": [],
            "SolarZenithAngle": [],
            "ViewingAzimuthAngle": [],
            "ViewingZenithAngle": [],
            "RelativeAzimuthAngle": [],
            "ScatteringAngle": [],
            "EarthSunDistance": [],
            "Filter_Band_Name": [],
        }

        radiance_total = []
        radiance_NSER_total = []
        wavelength_total = []
        EarthWavelength_Filter_total = []
        current_band = ""
        ##### EM - PLACE HOLDER FOR CO-ADDING PIXELS AND SPECTRAL PIXEL SAMPLING
        bandnames = []
        band_filenames = []
        for ii, bands in enumerate(windows):
            bandnames.append(bands["filter"])
        if "BAND1" and "BAND2" in bandnames:
            logger.info("Under construction")
        else:
            logger.info("No band resampling")

        for i, band in enumerate(windows):
            if band["instrument"] == "TROPOMI":
                if current_band != band["filter"]:
                    current_band = band["filter"]
                    band_filenames.append(tropomi_fns[current_band])
                    fh = netCDF4.Dataset(tropomi_fns[current_band])
                    try:
                        # Without this, various returned things are masked_array. We
                        # handle masking separately, so we want to skip this.
                        fh.set_auto_maskandscale(False)
                        eradd = cls.read_tropomi_erad(
                            fh,
                            int(float(iXTracks[current_band])),
                            int(iATracks[current_band]),
                            band["filter"],
                        )
                    finally:
                        fh.close()
                    if eradd is None:
                        raise RuntimeError("Trouble reading erad")
                    else:
                        erad = AttrDictAdapter(eradd)

                    EarthWavelength_Filter = np.asarray(
                        ["UUUUU" for ii in range(0, erad.PixelQualityFlags.shape[0])]
                    )
                    EarthWavelength_Filter[:] = band["filter"]
                    EarthWavelength_Filter_total.append(EarthWavelength_Filter)
                    wavelength_total.append(erad.Wavelength)
                    radiance_total.append(erad.Radiance)
                    radiance_NSER_total.append(erad.NESR)
                    # Viewing Angle Definition
                    raz = np.abs(erad.ViewingAzimuthAngle - erad.SolarAzimuthAngle)
                    if raz > 180.0:
                        raz = np.float64(360.0) - raz
                    raz = np.float64(180.0) - raz

                    # Compute scattering
                    sca = cls.compute_tropomi_sca(
                        erad.ViewingZenithAngle, erad.SolarZenithAngle, raz
                    )

                    # append data to output dictionaries
                    o_ObservationTable["ATRACK"].append(erad.iTrack)
                    o_ObservationTable["XTRACK"].append(erad.iXTrack)
                    o_ObservationTable["Latitude"].append(erad.Latitude)
                    o_ObservationTable["Longitude"].append(erad.Longitude)
                    o_ObservationTable["Time"].append(erad.Time)
                    o_ObservationTable["SpacecraftLatitude"].append(
                        erad.SpacecraftLatitude
                    )
                    o_ObservationTable["SpacecraftLongitude"].append(
                        erad.SpacecraftLongitude
                    )
                    o_ObservationTable["SpacecraftAltitude"].append(
                        erad.SpacecraftAltitude
                    )
                    o_ObservationTable["TerrainHeight"].append(erad.TerrainHeight)
                    o_ObservationTable["SolarAzimuthAngle"].append(
                        erad.SolarAzimuthAngle
                    )
                    o_ObservationTable["SolarZenithAngle"].append(erad.SolarZenithAngle)
                    o_ObservationTable["ViewingZenithAngle"].append(
                        erad.ViewingZenithAngle
                    )
                    o_ObservationTable["RelativeAzimuthAngle"].append(raz)
                    o_ObservationTable["ScatteringAngle"].append(sca)
                    o_ObservationTable["EarthSunDistance"].append(erad.EarthSunDistance)
                    o_ObservationTable["Filter_Band_Name"].append(band["filter"])

                else:
                    current_band = band["filter"]
                    logger.info(f"Repeat band skipping {current_band}")

            else:
                pass

        o_combined_erad_bands = {
            "omi_earth_rad_fn": band_filenames,
            "Wavelength": np.concatenate([i for i in wavelength_total], axis=0),
            "EarthRadiance": np.concatenate([i for i in radiance_total], axis=0),
            "EarthRadianceNESR": np.concatenate(
                [i for i in radiance_NSER_total], axis=0
            ),
            "EarthWavelength_Filter": np.concatenate(
                [i for i in EarthWavelength_Filter_total], axis=0
            ),
            "ObservationTable": o_ObservationTable,
        }

        return o_combined_erad_bands

    @classmethod
    def compute_tropomi_sca(cls, vza: float, sza: float, raz: float) -> float:
        temp1 = (
            -1
            * math.cos(math.pi * vza / np.float64(180.0))
            * math.cos(math.pi * sza / np.float64(180.0))
        )
        temp2 = math.sqrt(
            1
            - math.cos(math.pi * vza / np.float64(180.0))
            * math.cos(math.pi * vza / np.float64(180.0))
        )
        temp3 = math.sqrt(
            1
            - math.cos(math.pi * sza / np.float64(180.0))
            * math.cos(math.pi * sza / np.float64(180.0))
        )
        temp4 = math.cos(math.pi * (raz / np.float64(180.0)))
        o_sca = temp1 + (temp2 * temp3 * temp4)
        o_sca = np.float64(180.0) - math.acos(o_sca) * np.float64(180.0) / math.pi
        return o_sca

    @classmethod
    def read_tropomi_erad(
        cls, fh: netCDF4.Dataset, iXTrack: int, iTrack: int, iBand: str
    ) -> dict[str, Any]:
        geo_data_grp = f"{iBand}_RADIANCE/STANDARD_MODE/GEODATA"
        observations_data_grp = f"{iBand}_RADIANCE/STANDARD_MODE/OBSERVATIONS"
        instrument_data_grp = f"{iBand}_RADIANCE/STANDARD_MODE/INSTRUMENT"
        SpacecraftLatitude = fh[f"{geo_data_grp}/satellite_latitude"][0, iTrack]
        SpacecraftLongitude = fh[f"{geo_data_grp}/satellite_longitude"][0, iTrack]
        SpacecraftAltitude = fh[f"{geo_data_grp}/satellite_altitude"][0, iTrack]
        SolarAzimuthAngle = fh[f"{geo_data_grp}/solar_azimuth_angle"][
            0, iTrack, iXTrack
        ]
        SolarZenithAngle = fh[f"{geo_data_grp}/solar_zenith_angle"][0, iTrack, iXTrack]
        ViewingAzimuthAngle = fh[f"{geo_data_grp}/viewing_azimuth_angle"][
            0, iTrack, iXTrack
        ]
        ViewingZenithAngle = fh[f"{geo_data_grp}/viewing_zenith_angle"][
            0, iTrack, iXTrack
        ]
        Latitude = fh[f"{geo_data_grp}/latitude"][0, iTrack, iXTrack]
        Longitude = fh[f"{geo_data_grp}/longitude"][0, iTrack, iXTrack]
        Time = fh[f"{observations_data_grp}/time"]
        d_time = fh[f"{observations_data_grp}/delta_time"]
        # 536479200.0 is 17 years in seconds TROPOMI reference is Tropomi data is timed from 2010-01-01 00:00:00,
        # so must add constant to be constant with OMI, will revist.
        # JLL: d_time now seems to be in milliseconds, not seconds.
        Time = Time[:] + d_time[0, iTrack] / 1000 + 536479200.0
        # Terrain height is not included in TROPOMI products, leaving here as a place holder.
        TerrainHeight = 0
        EarthSunDistance = fh[f"{geo_data_grp}/earth_sun_distance"][0]
        GroundPixelQualityFlags = fh[f"{observations_data_grp}/ground_pixel_quality"][
            0, iTrack, iXTrack
        ]
        if GroundPixelQualityFlags != 0:
            raise RuntimeError(
                f"GroundPixelQualityFlag: {GroundPixelQualityFlags} is set to non-zero, {fh.filepath()}"
            )
        PixelQualityFlags = fh[f"{observations_data_grp}/spectral_channel_quality"][
            0, iTrack, iXTrack, :
        ]
        MeasurementQualityFlags = fh[f"{observations_data_grp}/measurement_quality"][
            :, iTrack
        ]
        if MeasurementQualityFlags != 0 and MeasurementQualityFlags != 16:
            raise RuntimeError(
                f"MeasurementQualityFlags: {MeasurementQualityFlags} is set to non-zero, {fh.filepath()}"
            )
        Radiance = fh[f"{observations_data_grp}/radiance"][0, iTrack, iXTrack, :]
        Radiance = Radiance.astype(np.float64)
        RadiancePrecision = fh[f"{observations_data_grp}/radiance_noise"][
            0, iTrack, iXTrack, :
        ]
        RadiancePrecision = RadiancePrecision.astype(np.float64)
        Wavelength = fh[f"{instrument_data_grp}/nominal_wavelength"][0, iXTrack, :]
        # TROPOMI stores noise in dB, must convert
        radiance_noise = Radiance / (10 ** (RadiancePrecision / 10))

        # compute the nesr
        # Radiance_err = np.sqrt(radiance_noise**2 + radiance_sys**2)
        Radiance_err = np.abs(radiance_noise)

        snr = np.mean(Radiance / Radiance_err)
        unit = "mol m-2 nm-1 s-1 sr-1"
        o_tropomi_rad = {
            "tropomi_file": fh.filepath(),
            "iTrack": iTrack,
            "iXTrack": iXTrack,
            "Latitude": Latitude,
            "Longitude": Longitude,
            "Time": Time,
            "SpacecraftLatitude": SpacecraftLatitude,
            "SpacecraftLongitude": SpacecraftLongitude,
            "SpacecraftAltitude": SpacecraftAltitude,
            "Wavelength": Wavelength,
            "Radiance": Radiance,
            "NESR": Radiance_err,
            "SNR": snr,
            "Unit": unit,
            "TerrainHeight": TerrainHeight,
            "SolarAzimuthAngle": SolarAzimuthAngle,
            "SolarZenithAngle": SolarZenithAngle,
            "ViewingAzimuthAngle": ViewingAzimuthAngle,
            "ViewingZenithAngle": ViewingZenithAngle,
            "EarthSunDistance": EarthSunDistance,
            "GroundPixelQualityFlags": GroundPixelQualityFlags,
            "PixelQualityFlags": PixelQualityFlags,
            "MeasurementQualityFlags": MeasurementQualityFlags,
        }
        return o_tropomi_rad

    def desc(self) -> str:
        return "MusesTropomiObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("TROPOMI")

    @classmethod
    def create_from_filename(
        cls,
        filename_dict: dict[str, str | os.PathLike[str]],
        xtrack_dict: dict[str, int],
        atrack_dict: dict[str, int],
        utc_time: str,
        filter_list: list[FilterIdentifier],
        calibration_filename: str | None = None,
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create from just the filenames. Note that spectral window
        doesn't get set here, but this can be useful if you just want
        access to the underlying data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).

        """
        o_tropomi, sdesc = cls._read_data(
            filename_dict,
            xtrack_dict,
            atrack_dict,
            utc_time,
            [str(i) for i in filter_list],
            calibration_filename=calibration_filename,
            osp_dir=osp_dir,
        )
        return cls(o_tropomi, sdesc, filter_list)

    @classmethod
    def create_from_id(
        cls,
        mid: MeasurementId,
        existing_obs: MusesTropomiObservation | None,
        current_state: CurrentState | None,
        spec_win: MusesSpectralWindow | None,
        fm_sv: rf.StateVector | None,
        osp_dir: str | os.PathLike[str] | None = None,
        write_tropomi_radiance_pickle: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Create from a MeasurementId. If this depends on any state
        information, you can pass in the CurrentState. This can be
        given as None if you just want to use default values, e.g. you
        aren't doing a retrieval. If the CurrentState is supplied, you
        can also pass a StateVector to add this class to as needed.

        Note that VLIDORT depends on having a pickle file
        created. This is a bad interface, basically this is like a
        hidden variable. But to support the old code, we can
        optionally generate that pickle file.

        """
        coeff = None
        mp = None
        if existing_obs is not None:
            # Take data from existing observation
            if current_state is not None:
                coeff, mp = current_state.object_state(
                    existing_obs.state_element_name_list()
                )
            obs = cls(
                existing_obs.muses_py_dict,
                existing_obs.sounding_desc,
                existing_obs.filter_list,
                existing_obs=existing_obs,
                coeff=coeff,
                mp=mp,
            )
        else:
            filter_list = mid.filter_list_dict[InstrumentIdentifier("TROPOMI")]
            if current_state is not None:
                coeff, mp = current_state.object_state(
                    cls.state_element_name_list_from_filter(filter_list)
                )
            if int(mid["TROPOMI_Rad_calRun_flag"]) != 1:
                # The current py-retrieve code just silently ignores calibration,
                # see about line 614 of script_retrieval_setup_ms. We duplicate
                # this behavior, but go ahead and warn that we are doing that.
                logger.warning(
                    "Don't support calibration files yet. Ignoring TROPOMI_Rad_calRun_flag"
                )
            filename_dict = {}
            xtrack_dict = {}
            atrack_dict = {}

            filename_dict["CLOUD"] = mid["TROPOMI_Cloud_filename"]
            # Note this is what mpy.get_tropomi_measurement_id_info requires. Not
            # sure if we don't have a band 3 here, but follow what that file does.
            xtrack_dict["CLOUD"] = mid["TROPOMI_XTrack_Index_BAND3"]
            atrack_dict["CLOUD"] = mid["TROPOMI_ATrack_Index"]
            for flt in [str(i) for i in filter_list]:
                filename_dict[flt] = mid[f"TROPOMI_filename_{flt}"]
                xtrack_dict[flt] = mid[f"TROPOMI_XTrack_Index_{flt}"]
                # We happen to have only one atrack in the file, but the
                # mpy.read_tropomi doesn't assume that so we have one entry
                # per filter here.
                atrack_dict[flt] = mid["TROPOMI_ATrack_Index"]
                if str(flt) in ("BAND7", "BAND8"):
                    filename_dict["IRR_BAND_7to8"] = mid["TROPOMI_IRR_SIR_filename"]
                    xtrack_dict["IRR_BAND_7to8"] = xtrack_dict[flt]
                else:
                    filename_dict["IRR_BAND_1to6"] = mid["TROPOMI_IRR_filename"]
                    xtrack_dict["IRR_BAND_1to6"] = xtrack_dict[flt]
            utc_time = mid["TROPOMI_utcTime"]
            o_tropomi, sdesc = cls._read_data(
                filename_dict,
                xtrack_dict,
                atrack_dict,
                utc_time,
                [str(i) for i in filter_list],
                osp_dir=osp_dir,
            )
            obs = cls(o_tropomi, sdesc, filter_list, coeff=coeff, mp=mp)

        if write_tropomi_radiance_pickle:
            # Save file needed by py-retrieve VLIDORT code
            pfname = os.path.normpath(
                f"{mid['initialGuessDirectory']}/../Radiance_TROPOMI_.pkl"
            )
            if not os.path.exists(pfname):
                subprocess.run(["mkdir", "-p", os.path.dirname(pfname)])
                pickle.dump(obs.muses_py_dict, open(pfname, "wb"))

        obs.spectral_window = (
            spec_win if spec_win is not None else MusesSpectralWindow(None, None)
        )
        obs.spectral_window.add_bad_sample_mask(obs)
        if fm_sv is not None:
            if current_state is None:
                raise RuntimeError(
                    "If fm_sv is not None, current_state needs to also be not None"
                )
            current_state.add_fm_state_vector_if_needed(
                fm_sv,
                obs.state_element_name_list(),
                [
                    obs,
                ],
            )
        return obs

    def snr_uplimit(self, sensor_index: int) -> float:
        """Upper limit for SNR, we adjust uncertainty is we are greater than this."""
        return 500.0

    @classmethod
    def state_element_name_list_from_filter(
        cls, filter_list: list[FilterIdentifier]
    ) -> list[StateElementIdentifier]:
        """List of state element names for this observation"""
        res = []
        for flt in filter_list:
            res.append(StateElementIdentifier(f"TROPOMISOLARSHIFT{str(flt)}"))
        for flt in filter_list:
            res.append(StateElementIdentifier(f"TROPOMIRADIANCESHIFT{str(flt)}"))
        for flt in filter_list:
            res.append(StateElementIdentifier(f"TROPOMIRADSQUEEZE{str(flt)}"))
        return res

    def state_vector_name_i(self, i: int) -> str:
        res = []
        for flt in self.filter_list:
            res.append(f"Solar Shift {str(flt)}")
        for flt in self.filter_list:
            res.append(f"Radiance Shift {str(flt)}")
        for flt in self.filter_list:
            res.append(f"Radiance Squeeze {str(flt)}")
        return res[i]

    def state_element_name_list(self) -> list[StateElementIdentifier]:
        """List of state element names for this observation"""
        return self.state_element_name_list_from_filter(self.filter_list)

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            float(
                self.muses_py_dict["Earth_Radiance"]["ObservationTable"][
                    "TerrainHeight"
                ][-1]
            ),
            "m",
        )


class MusesOmiObservation(MusesReflectanceObservation):
    """Observation for OMI"""

    def __init__(
        self,
        muses_py_dict: dict[str, Any],
        sdesc: dict[str, Any],
        filter_list: list[FilterIdentifier],
        existing_obs: Self | None = None,
        coeff: np.ndarray | None = None,
        mp: rf.StateMapping | None = None,
    ) -> None:
        """Note you don't normally create an object of this class with
        the __init__. Instead, call one of the create_xxx class
        methods.

        """
        super().__init__(
            muses_py_dict,
            sdesc,
            filter_list,
            existing_obs=existing_obs,
            coeff=coeff,
            mp=mp,
        )

    @property
    def monthly_minimum_surface_reflectance(self) -> float:
        return float(
            self.muses_py_dict["SurfaceAlbedo"]["MonthlyMinimumSurfaceReflectance"]
        )

    @classmethod
    def _read_data(
        cls,
        filename: str | os.PathLike[str],
        xtrack_uv1: int,
        xtrack_uv2: int,
        atrack: int,
        utc_time: str,
        calibration_filename: str | os.PathLike[str],
        cld_filename: str | os.PathLike[str] | None = None,
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        o_omi = cls.read_omi(
            filename,
            xtrack_uv2,
            atrack,
            utc_time,
            calibration_filename,
            cld_filename,
            osp_dir,
        )
        sdesc = {
            "OMI_ATRACK_INDEX": np.int16(atrack),
            "OMI_XTRACK_INDEX_UV1": np.int16(xtrack_uv1),
            "OMI_XTRACK_INDEX_UV2": np.int16(xtrack_uv2),
            "POINTINGANGLE_OMI": abs(
                np.mean(
                    o_omi["Earth_Radiance"]["ObservationTable"]["ViewingZenithAngle"][
                        1:3
                    ]
                )
            ),
        }
        t = re.sub(r"\.\d+", "", utc_time)
        dtime = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
        # We double the NESR for OMI from 2010 onward. Not sure of the
        # history of this, but this is in the muses-py code so we
        # duplicate this here.
        if dtime.year >= 2010:
            o_omi["Earth_Radiance"]["EarthRadianceNESR"][
                o_omi["Earth_Radiance"]["EarthRadianceNESR"] > 0
            ] *= 2
        return (o_omi, sdesc)

    @classmethod
    def read_omi(
        cls,
        filename: str | os.PathLike[str],
        xtrack_uv2: int,
        atrack: int,
        utc_time: str,
        calibration_filename: str | os.PathLike[str],
        cld_filename: str | os.PathLike[str] | None = None,
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        with osp_setup(osp_dir):
            o_omi = mpy_read_omi(
                str(filename),
                xtrack_uv2,
                atrack,
                utc_time,
                str(calibration_filename),
                cldFilename=str(cld_filename) if cld_filename is not None else None,
            )
        return o_omi

    def desc(self) -> str:
        return "MusesOmiObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("OMI")

    @classmethod
    def create_from_filename(
        cls,
        filename: str | os.PathLike[str],
        xtrack_uv1: int,
        xtrack_uv2: int,
        atrack: int,
        utc_time: str,
        calibration_filename: str,
        filter_list: list[FilterIdentifier],
        cld_filename: str | None = None,
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create from just the filenames. Note that spectral window
        doesn't get set here, but this can be useful if you just want
        access to the underlying data.

        You might also want to use create_from_id, which sets up
        everything (spectral window, coefficients, attaching to a
        fm_sv).

        """
        o_omi, sdesc = cls._read_data(
            str(filename),
            xtrack_uv1,
            xtrack_uv2,
            atrack,
            utc_time,
            calibration_filename,
            cld_filename=cld_filename,
            osp_dir=osp_dir,
        )
        return cls(o_omi, sdesc, filter_list)

    @classmethod
    def create_from_id(
        cls,
        mid: MeasurementId,
        existing_obs: MusesOmiObservation | None,
        current_state: CurrentState | None,
        spec_win: MusesSpectralWindow | None,
        fm_sv: rf.StateVector | None,
        osp_dir: str | os.PathLike[str] | None = None,
        write_omi_radiance_pickle: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Create from a MeasurementId. If this depends on any state
        information, you can pass in the CurrentState. This can be
        given as None if you just want to use default values, e.g. you
        aren't doing a retrieval. If the CurrentState is supplied, you
        can also pass a StateVector to add this class to as needed.

        """
        coeff = None
        mp = None
        if existing_obs is not None:
            if current_state is not None:
                coeff, mp = current_state.object_state(
                    existing_obs.state_element_name_list()
                )
            obs = cls(
                existing_obs.muses_py_dict,
                existing_obs.sounding_desc,
                existing_obs.filter_list,
                existing_obs=existing_obs,
                coeff=coeff,
                mp=mp,
            )
        else:
            filter_list = mid.filter_list_dict[InstrumentIdentifier("OMI")]
            if current_state is not None:
                coeff, mp = current_state.object_state(
                    cls.state_element_name_list_from_filter(filter_list)
                )
            xtrack_uv1 = int(mid["OMI_XTrack_UV1_Index"])
            xtrack_uv2 = int(mid["OMI_XTrack_UV2_Index"])
            atrack = int(mid["OMI_ATrack_Index"])
            filename = mid["OMI_filename"]
            cld_filename = mid["OMI_Cloud_filename"]
            utc_time = mid["OMI_utcTime"]
            if int(mid["OMI_Rad_calRun_flag"]) != 1:
                calibration_filename = mid["omi_calibrationFilename"]
            else:
                logger.info("Calibration run. Disabling EOF application.")
                calibration_filename = None
            o_omi, sdesc = cls._read_data(
                filename,
                xtrack_uv1,
                xtrack_uv2,
                atrack,
                utc_time,
                calibration_filename,
                cld_filename=cld_filename,
                osp_dir=osp_dir,
            )
            obs = cls(o_omi, sdesc, filter_list, coeff=coeff, mp=mp)

        if write_omi_radiance_pickle:
            # Save file needed by py-retrieve VLIDORT code
            pfname = os.path.normpath(
                f"{mid['initialGuessDirectory']}/../Radiance_OMI_.pkl"
            )
            if not os.path.exists(pfname):
                subprocess.run(["mkdir", "-p", os.path.dirname(pfname)])
                pickle.dump(obs.muses_py_dict, open(pfname, "wb"))

        obs.spectral_window = (
            spec_win if spec_win is not None else MusesSpectralWindow(None, None)
        )
        obs.spectral_window.add_bad_sample_mask(obs)
        if fm_sv is not None:
            if current_state is None:
                raise RuntimeError(
                    "If fm_sv is not None, current_state needs to also be not None"
                )
            current_state.add_fm_state_vector_if_needed(
                fm_sv,
                obs.state_element_name_list(),
                [
                    obs,
                ],
            )
        return obs

    def snr_uplimit(self, sensor_index: int) -> float:
        """Upper limit for SNR, we adjust uncertainty is we are greater than this."""
        if self.filter_list[sensor_index] == FilterIdentifier("UV2"):
            return 800.0
        return 500.0

    @classmethod
    def state_element_name_list_from_filter(
        cls, filter_list: list[FilterIdentifier]
    ) -> list[StateElementIdentifier]:
        """List of state element names for this observation"""
        res = []
        for flt in filter_list:
            res.append(StateElementIdentifier(f"OMINRADWAV{str(flt)}"))
        for flt in filter_list:
            res.append(StateElementIdentifier(f"OMIODWAV{str(flt)}"))
        for flt in filter_list:
            res.append(StateElementIdentifier(f"OMIODWAVSLOPE{str(flt)}"))
        return res

    def state_vector_name_i(self, i: int) -> str:
        res = []
        for flt in self.filter_list:
            res.append(f"Solar Shift {str(flt)}")
        for flt in self.filter_list:
            res.append(f"Radiance Shift {str(flt)}")
        for flt in self.filter_list:
            res.append(f"Radiance Squeeze {str(flt)}")
        return res[i]

    def state_element_name_list(self) -> list[StateElementIdentifier]:
        """List of state element names for this observation"""
        return self.state_element_name_list_from_filter(self.filter_list)

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            float(
                self.muses_py_dict["Earth_Radiance"]["ObservationTable"][
                    "TerrainHeight"
                ][0]
            ),
            "m",
        )


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("TROPOMI"), MusesTropomiObservation)
)
ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("OMI"), MusesOmiObservation)
)

__all__ = [
    "MusesReflectanceObservation",
    "MusesTropomiObservation",
    "MusesOmiObservation",
]
