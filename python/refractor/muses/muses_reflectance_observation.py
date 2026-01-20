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
from .misc import AttrDictAdapter
import os
import numpy as np
import scipy
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
from .input_file_helper import InputFileHelper, InputFilePath

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    import netCDF4
    import h5py  # type: ignore

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
            self._freq_data: list[np.ndarray] = existing_obs._freq_data  # noqa: SLF001
            self._nesr_data: list[np.ndarray] = existing_obs._nesr_data  # noqa: SLF001
            self._bsamp: list[np.ndarray] = existing_obs._bsamp  # noqa: SLF001
            self._solar_interp: list[LinearInterpolate] = existing_obs._solar_interp  # noqa: SLF001
            self._earth_rad: list[np.ndarray] = existing_obs._earth_rad  # noqa: SLF001
            self._nesr: list[np.ndarray] = existing_obs._nesr  # noqa: SLF001
            self._solar_spectrum: list[rf.Spectrum] = existing_obs._solar_spectrum  # noqa: SLF001
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

    def solar_interp_for_od(
        self,
        sensor_index: int,
        xsec_sd: rf.SpectralDomain,
        v1_mono: np.ndarray | None = None,
        v2_mono: np.ndarray | None = None,
    ) -> scipy.interpolate.interp1d:
        """The interpolation in MusesOpticalDepth has a specific form. This isn't
        substantially different than just our self._solar_interp, but the values
        calculated are different. In particular, the is a bug in the code where the
        wrong SpectralDomain is used. We match this wrong behavior to match the existing
        code, but this should get fixed at some point.

        We duplicate what was in the old code, so we can get the same results. We
        may want to just remove this at some point, or even better move the solar
        model in a rf.SolarModel.
        """
        # This logic is a bit convoluted to just find a subset of the solar data. Not
        # even clear why this needs to get subsetted, but we match what the software
        # in py-retrieve currently does.
        #
        # Note also the "magic" numbers to slightly widen the ranges. Again, we match
        # what py-retrieve is currently doing
        wn, sindex = self.wn_and_sindex(sensor_index)
        if v1_mono is not None and v2_mono is not None:
            start_wn = np.min(v1_mono) - 1.0
            end_wn = np.max(v2_mono) + 1.0
        else:
            start_wn = np.min(wn) - 1.0
            end_wn = np.max(wn) + 1.0
        swin = rf.SpectralWindowRange(
            rf.ArrayWithUnit_double_3([[[start_wn, end_wn]]], "nm")
        )
        sd = swin.apply(xsec_sd, 0)
        swin2 = rf.SpectralWindowRange(
            rf.ArrayWithUnit_double_3(
                [[[sd.data.min() - 0.01, sd.data.max() + 0.02]]], "nm"
            )
        )
        # TODO Fix this
        # Note that is actually *wrong*. The AdjustedSolarRadiance has already been
        # interpolated to the earth wavelengths. But match the old wrong behavior here,
        # so we can match py-retrieve
        erad = self._muses_py_dict["Earth_Radiance"]
        srad = self._muses_py_dict["Solar_Radiance"]
        flt_sub = erad["EarthWavelength_Filter"] == str(self.filter_list[sensor_index])
        wrong_sol_domain = rf.SpectralDomain(
            srad["Wavelength"][flt_sub], rf.Unit("nm")
        )  # Should be erad["Wavelength"]
        sol_range = rf.SpectralRange(
            srad["AdjustedSolarRadiance"][flt_sub], rf.Unit("ph / nm / s")
        )
        wrong_sol_spec = rf.Spectrum(wrong_sol_domain, sol_range)
        # After fixing, can just use self._solar_spectrum[sensor_index], or
        # probably just self._solar_interp[sensor_index] and skip the subsetting.
        sol_sub = swin2.apply(wrong_sol_spec, 0)
        # Exclude any bad points
        good_pt = sol_sub.spectral_range.data > 0.0
        return scipy.interpolate.interp1d(
            sol_sub.spectral_domain.data[good_pt],
            sol_sub.spectral_range.data[good_pt],
            fill_value="extrapolate",
        )

    @classmethod
    def idl_interpol_1d(
        cls,
        i_vector: np.ndarray,
        i_abscissaValues: np.ndarray,
        i_abscissaResult: np.ndarray,
    ) -> np.ndarray:
        return scipy.interpolate.interp1d(
            i_abscissaValues, i_vector, fill_value="extrapolate"
        )(i_abscissaResult)

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

    @classmethod
    def combine_omi_calibration(
        cls, iXtrack_all: list[int], calibrationFilename: str
    ) -> dict[str, Any]:
        from refractor.muses_py import read_omi_calibration

        # IDL_LEGACY_NOTE: This function combine_omi_calibration is the same as Combine_OMI_Calibration function in ms-setup/Combine_OMI_Calibration.pro file.
        ixtrack_uv1 = iXtrack_all[0]
        ixtrack = iXtrack_all[1]
        ixtrack_uv2_pair = iXtrack_all[2]

        # ===========================================
        # * Get Spectral Radiance Calibration Factors
        # ===========================================
        # For UV2

        iUV = 2
        cal_uv2 = read_omi_calibration(ixtrack, iUV, calibrationFilename)
        cal_uv2_pair = read_omi_calibration(ixtrack_uv2_pair, iUV, calibrationFilename)

        # * For UV1
        iUV = 1
        cal_uv1 = read_omi_calibration(ixtrack_uv1, iUV, calibrationFilename)

        # Convert to AttrDictAdapter so we can use the dot '.' notation.
        cal_uv2 = AttrDictAdapter(cal_uv2)
        cal_uv2_pair = AttrDictAdapter(cal_uv2_pair)
        cal_uv1 = AttrDictAdapter(cal_uv1)

        Wavelength_Filter_UV1 = np.asarray(["UV1" for ii in range(cal_uv1.nw)])
        Wavelength_Filter_UV2 = np.asarray(["UV2" for ii in range(cal_uv2.nw)])
        Wavelength_Filter = np.concatenate(
            (Wavelength_Filter_UV1, Wavelength_Filter_UV2), axis=0
        )

        Calibration = np.concatenate(
            (
                cal_uv1.calibration,
                (cal_uv2.calibration[:] + cal_uv2_pair.calibration[:])
                / np.float64(2.0),
            ),
            axis=0,
        )
        Wavelength = np.concatenate(
            (
                cal_uv1.wavelength,
                (cal_uv2.wavelength[:] + cal_uv2_pair.wavelength[:]) / np.float64(2.0),
            ),
            axis=0,
        )

        o_combined_radcal_bands = {
            "omi_rad_calibration_fn": cal_uv2.omi_file,  # Filename of radiance calibration factors
            "Wavelength": Wavelength,  # Wavelength Grid; Full Band
            "CalibrationFactor": Calibration,  # Calibration Factor; Full Band
            "Wavelength_Filter": Wavelength_Filter,  # Optical Filter Name for each Channel; Full Bands;
            "usage_pixels": iXtrack_all,  # index for healthy pixel
        }

        return o_combined_radcal_bands

    @classmethod
    def read_omi_surface_albedo(
        cls, f_alb: h5py.File, TLongitude: float, TLatitude: float, TMonth: int
    ) -> dict[str, Any]:
        from refractor.muses_py import get_distance

        # Get month index
        month_ind = TMonth - 1

        # Set wavelength index
        wave_ind = 0

        # Surface Albedo
        ScaleFactor = np.float64(0.0010)

        # Latitude
        Latitude = f_alb[
            "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/Latitude"
        ][:]

        # Longitude
        Longitude = f_alb[
            "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/Longitude"
        ][:]

        temp_lat_ind = np.where(
            (Latitude > (TLatitude - 2.0)) & (Latitude < (TLatitude + 2.0))
        )[0]
        temp_lon_ind = np.where(
            (Longitude > (TLongitude - 2.0)) & (Longitude < (TLongitude + 2.0))
        )[0]

        min_dist = 999.0
        min_temp_lat_ind = np.amin(temp_lat_ind)
        max_temp_lat_ind = np.amax(temp_lat_ind)
        min_temp_lon_ind = np.amin(temp_lon_ind)
        max_temp_lon_ind = np.amax(temp_lon_ind)

        lat_ind = 0
        lon_ind = 0
        for ilat in range(min_temp_lat_ind, max_temp_lat_ind):
            for ilon in range(min_temp_lon_ind, max_temp_lon_ind):
                (temp_dist, _, _) = get_distance(
                    Latitude[ilat], Longitude[ilon], TLatitude, TLongitude
                )
                # At this point, temp_dist is in Kilometers.
                if temp_dist <= min_dist:
                    lat_ind = ilat
                    lon_ind = ilon
                    min_dist = temp_dist

        Latitude = Latitude[lat_ind]
        Longitude = Longitude[lon_ind]

        # MonthlyMinimumSurfaceReflectance
        MonthlyMinimumSurfaceReflectance = (
            f_alb[
                "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/MonthlyMinimumSurfaceReflectance"
            ][month_ind, wave_ind, lat_ind, lon_ind]
            * ScaleFactor
        )

        # MonthlySurfaceReflectance
        MonthlySurfaceReflectance = (
            f_alb[
                "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/MonthlySurfaceReflectance"
            ][month_ind, wave_ind, lat_ind, lon_ind]
            * ScaleFactor
        )

        # MonthlySurfaceReflectanceFlag
        MonthlySurfaceReflectanceFlag = f_alb[
            "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/MonthlySurfaceReflectanceFlag"
        ][month_ind, lat_ind, lon_ind]

        # Wavelength
        Wavelength = f_alb[
            "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/Wavelength"
        ][wave_ind]

        # YearlyMinimumSurfaceReflectance
        YearlyMinimumSurfaceReflectance = (
            f_alb[
                "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/YearlyMinimumSurfaceReflectance"
            ][wave_ind, lat_ind, lon_ind]
            * ScaleFactor
        )

        # YearlySurfaceReflectance
        YearlySurfaceReflectance = (
            f_alb[
                "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/YearlySurfaceReflectance"
            ][wave_ind, lat_ind, lon_ind]
            * ScaleFactor
        )

        # YearlySurfaceReflectanceFlag
        YearlySurfaceReflectanceFlag = f_alb[
            "/HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields/YearlySurfaceReflectanceFlag"
        ][lat_ind, lon_ind]

        o_omi_SurfaceAlbedo = {
            "Latitude": Latitude,
            "Longitude": Longitude,
            "MonthlyMinimumSurfaceReflectance": MonthlyMinimumSurfaceReflectance,
            "MonthlySurfaceReflectance": MonthlySurfaceReflectance,
            "MonthlySurfaceReflectanceFlag": MonthlySurfaceReflectanceFlag,
            "Wavelength": Wavelength,
            "YearlyMinimumSurfaceReflectance": YearlyMinimumSurfaceReflectance,
            "YearlySurfaceReflectance": YearlySurfaceReflectance,
            "YearlySurfaceReflectanceFlag": YearlySurfaceReflectanceFlag,
        }

        return o_omi_SurfaceAlbedo


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
        filename_dict: dict[str, str | os.PathLike[str] | InputFilePath],
        xtrack_dict: dict[str, int],
        atrack_dict: dict[str, int],
        utc_time: str,
        filter_list: list[str],
        calibration_filename: str | None = None,
        ifile_hlp: InputFileHelper | None = None,
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
            filename_dict, xtrack_dict, atrack_dict, utc_time, i_windows, ifile_hlp
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
        filename_dict: dict[str, str | os.PathLike[str] | InputFilePath],
        xtrack_dict: dict[str, int],
        atrack_dict: dict[str, int],
        utc_time: str,
        i_windows: list[dict[str, str]],
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        if ifile_hlp is None:
            ifile_hlp = InputFileHelper()
        # TODO Remove osp_setup below
        with (
            ifile_hlp.open_ncdf(filename_dict["CLOUD"]) as f_cld,
            ifile_hlp.open_h5(
                ifile_hlp.osp_dir
                / "OMI"
                / "OMI_LER"
                / "OMI-Aura_L3-OMLER_2005m01-2009m12_v003-2010m0503t063707.he5"
            ) as f_alb,
            osp_setup(ifile_hlp),
        ):
            f_cld.set_auto_maskandscale(False)
            for k, v in filename_dict.items():
                ifile_hlp.notify_file_input(v)
            o_tropomi = cls.read_tropomi_l1b(
                ifile_hlp,
                f_cld,
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
                o_tropomi["Earth_Radiance"]["ObservationTable"]["TerrainHeight"][i] = (
                    cls.read_tropomi_surface_altitude(
                        ifile_hlp,
                        o_tropomi["Earth_Radiance"]["ObservationTable"]["Latitude"][i],
                        o_tropomi["Earth_Radiance"]["ObservationTable"]["Longitude"][i],
                    )
                )
        return o_tropomi

    @classmethod
    def read_tropomi_l1b(
        cls,
        ifile_hlp: InputFileHelper,
        f_cld: netCDF4.Dataset,
        filenames: dict[str, str],
        iXTracks: dict[str, int],
        iATracks: dict[str, int],
        UtcDateTime: str,
        windows: list[dict[str, str]],
        calibrationFilename: bool = False,
        albedo_from_dler: bool = False,
    ) -> dict[str, Any]:
        # EM - Will probs have to include a calibration file, not sure how to do that yet
        from refractor.muses_py import (
            read_tropomi_surface_albedo,
        )

        # ======================
        #  Earth Shine Radiances
        # ======================
        # EM - Deal with differing TROPOMI bands in this function
        erad = AttrDictAdapter(
            cls.combine_tropomi_erad(filenames, iXTracks, iATracks, windows, ifile_hlp)
        )

        # ============================
        #  For OMI, 3 Year Mean Solar Radiances used here
        #  For TROPOMI, trying daily solar irradiance files
        # ============================
        irad = AttrDictAdapter(cls.daily_tropomi_irad(filenames, iXTracks, windows))

        # ============================
        #  Get Calibration Factors
        # ============================
        if calibrationFilename:
            if True:
                raise RuntimeError("This hasn't been updated for tropomi")
            # We have the code that was in read_tropomi in py-retrieve, but you can
            # tell that this hasn't been updated for tropomi. Leave as a placeholder,
            # but this doesn't actually work for tropomi yet.
            logger.warning(
                "TROPOMI calibration functions likely need updated to work with the new filename/track index dictionary convention implemented for NIR. If you get an error in a few lines, update combine_tropomi_calibration first"
            )
            rad_cal = AttrDictAdapter(
                cls.combine_omi_calibration(iXTracks, calibrationFilename)
            )

            # ===============================================
            # * Earth Shine Radiances with calibration applied
            # ===============================================
            temp_ind1 = np.where(rad_cal.Wavelength_Filter == "UV1")[0]
            temp_ind2 = np.where(erad.EarthWavelength_Filter == "UV1")[0]

            temp_cf1 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )
            temp_cf2 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )

            # PYTHON_NOTE: It seems the Python version has more
            # precisions for temp_cf1 and temp_cf2 which causes
            # problems later.  PYTHON_NOTE: We are also changing to
            # float since it causes problem comparing to IDL.
            temp_cf1 = temp_cf1.astype(np.float64)
            temp_cf2 = temp_cf2.astype(np.float64)

            CALIBRATEDEARTHRADIANCE_uv1v = []
            for ii in range(0, len(temp_ind2)):
                CALIBRATEDEARTHRADIANCE_uv1v.append(
                    erad.EarthRadiance[temp_ind2[ii]] / temp_cf2[ii]
                )

            CALIBRATEDEARTHRADIANCE_uv1 = np.asarray(CALIBRATEDEARTHRADIANCE_uv1v)

            temp_ind1 = np.where(rad_cal.Wavelength_Filter == "UV2")[0]
            temp_ind2 = np.where(erad.EarthWavelength_Filter == "UV2")[0]

            temp_cf1 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )
            temp_cf2 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )

            temp_cf1 = temp_cf1.astype(np.float64)
            temp_cf2 = temp_cf2.astype(np.float64)

            CALIBRATEDEARTHRADIANCE_uv2 = erad.EarthRadiance[temp_ind2] / temp_cf2[:]
            CALIBRATEDEARTHRADIANCE = np.concatenate(
                (CALIBRATEDEARTHRADIANCE_uv1, CALIBRATEDEARTHRADIANCE_uv2), axis=0
            )

            erad.CalibratedEarthRadiance = CALIBRATEDEARTHRADIANCE  # Adding another key to ObjectView by merely using the dot '.' notation and an assignment.

        else:
            o_combined_radcal_bands = {
                "tropomi_rad_calibration_fn": "N/A",  # Filename of radiance calibration factors
                "Wavelength": erad.Wavelength,  # Wavelength Grid; Full Band
                "CalibrationFactor": np.ones(
                    len(erad.EarthRadiance)
                ),  # Calibration Factor; Full Band
                "Wavelength_Filter": erad.EarthWavelength_Filter,  # Optical Filter Name for each Channel; Full Bands;
                "usage_pixels": iXTracks,  # index for healthy pixel, JLL: update if a list is needed instead of a dict
            }
            rad_cal = AttrDictAdapter(
                o_combined_radcal_bands
            )  # No calibration, set calibration factor as 1s
            erad.CalibratedEarthRadiance = erad.EarthRadiance

        # ====================================================
        # * Solar IRRadiances with sunEarthDistance Adjustement
        # ====================================================
        # one_au = np.float64(149597890000.0)
        one_au = np.float64(1.0)
        temp_cf = (one_au / erad.ObservationTable["EarthSunDistance"][0]) ** 2.0
        ADJUSTEDSOLARRADIANCE = irad.SolarRadiance[:] * temp_cf
        irad.AdjustedSolarRadiance = ADJUSTEDSOLARRADIANCE.copy()

        # ===============================================
        # * align solar and earth radiance wavelength grid
        # ===============================================
        window_track = ""
        for ii2 in windows:
            if (
                ii2["instrument"] == "TROPOMI"
            ):  # EM - Necessary for dual band approaches
                if window_track != ii2["filter"]:
                    window_track = ii2["filter"]
                    temp_ind1 = np.where(erad.EarthWavelength_Filter == ii2["filter"])[
                        0
                    ]
                    temp_ind2 = np.where(irad.SolarWavelength_Filter == ii2["filter"])[
                        0
                    ]
                    irad.SolarRadiance[temp_ind1] = cls.idl_interpol_1d(
                        irad.SolarRadiance[temp_ind2],
                        irad.Wavelength[temp_ind2],
                        erad.Wavelength[temp_ind1],
                    )
                    irad.AdjustedSolarRadiance[temp_ind1] = cls.idl_interpol_1d(
                        ADJUSTEDSOLARRADIANCE[temp_ind2],
                        irad.Wavelength[temp_ind2],
                        erad.Wavelength[temp_ind1],
                    )
                else:
                    continue
            else:
                pass

        # ============================
        # * Get CloudInformation
        # ============================
        cloudInfo = cls.read_tropomi_cloud(
            f_cld, int(iXTracks["CLOUD"]), iATracks["CLOUD"]
        )

        # ======================
        # * Get TROPOMI SurfaceAlbedo
        # ======================
        temp = UtcDateTime.split("-")
        TMonth = int(temp[1])
        TLongitude = erad.ObservationTable["Longitude"][0]
        TLatitude = erad.ObservationTable["Latitude"][0]

        # JLL 2023-09-13: I've kept EM's implementation that uses the
        # OMI LER database for BAND3 (and bands 1--6) but added a
        # clause to read from a TROPOMI DLER database for Bands 7 &
        # 8. (NB: bands 1, 2, 4, 5, and 6 need to know which
        # wavelength to read from in the OMI LER database.)  Although
        # o_tropomi is returned and passed around the rest of
        # py-retrieve, it does not appear that the SurfaceAlbedo dict
        # is referenced anywhere in the code other than the initial
        # UIP setup below, so I've changed it to allow multiple bands'
        # albedos to be returned.
        tropomi_filters = sorted(
            {win["filter"] for win in windows if win["instrument"] == "TROPOMI"}
        )
        SurfaceAlbedo = read_tropomi_surface_albedo(
            TLongitude,
            TLatitude,
            TMonth,
            tropomi_filters,
            tropomi_radiances={
                "Earth_Radiance": erad.__dict__,
                "Solar_Radiance": irad.__dict__,
            },
            swir_from_radiances=not albedo_from_dler,
        )

        # * Define output parameter
        o_tropomi = {
            "Earth_Radiance": erad.__dict__,
            "Solar_Radiance": irad.__dict__,
            "Radiance_Calibration": rad_cal.__dict__,
            "SurfaceAlbedo": SurfaceAlbedo,
            "Cloud": cloudInfo,
        }

        if not np.all(
            np.isfinite(o_tropomi["Earth_Radiance"]["CalibratedEarthRadiance"])
        ):
            raise RuntimeError("CalibratedEarthRadiance not finite")
        if not np.all(np.isfinite(o_tropomi["Earth_Radiance"]["EarthRadiance"])):
            raise RuntimeError("EarthRadiance not finite")
        if not np.all(np.isfinite(o_tropomi["Earth_Radiance"]["Wavelength"])):
            raise RuntimeError("Wavelength not finite")
        if not np.all(
            np.isfinite(o_tropomi["Solar_Radiance"]["AdjustedSolarRadiance"])
        ):
            raise RuntimeError("AdjustedSolarRadiance not finite")

        return o_tropomi

    @classmethod
    def daily_tropomi_irad(cls, irrFilenames, iXtracks, windows) -> dict[str, Any]:
        from refractor.muses_py import read_tropomi_daily_irad, tropomi_constants

        # Define arrays for storing
        Radiance = []
        RadianceNESR = []
        Wavelength = []
        Wavelength_Filter_total = []
        current_band = []
        oXtrack = []

        for i, band in enumerate(windows):
            # ==============================
            # * Get Daily Solar Spec
            # ==============================
            if (
                band["instrument"] == "TROPOMI"
            ):  # EM Note - This is necessary for dual band retrievals, so non TROPOMI data is not passed.
                if current_band != band["filter"]:
                    current_band = band["filter"]
                    if current_band in tropomi_constants.UVN_BANDS:
                        key = "IRR_BAND_1to6"
                    elif current_band in tropomi_constants.SWIR_BANDS:
                        key = "IRR_BAND_7to8"
                    else:
                        raise NotImplementedError(
                            f"Reading TROPOMI irradiance for {current_band}"
                        )
                    irad = read_tropomi_daily_irad(
                        irrFilenames[key], int(float(iXtracks[key])), band["filter"]
                    )
                    if irad is None:
                        raise RuntimeError("Call to read_tropomi_daily_irad failed")
                    else:
                        # Convert our dictionaries so we use the dot '.' notation.
                        irad = AttrDictAdapter(irad)

                    Radiance.append(irad.Sol)
                    RadianceNESR.append(irad.Pre)
                    Wavelength.append(irad.Wav)
                    oXtrack.append(iXtracks[key])

                    Wavelength_Filter = np.asarray(
                        [band["filter"] for ii in range(0, len(irad.Sol))]
                    )
                    Wavelength_Filter_total.append(Wavelength_Filter)

                else:
                    current_band = band["filter"]
                    logger.info(f"Repeat band skipping {current_band}")
            else:
                pass

        o_combined_irad_bands = {
            "omi_solar_rad_fn": irad.tropomi_file[
                0
            ],  # L1B Earth Radiance Full Path and File Name
            "Wavelength": np.concatenate(
                [i for i in Wavelength], axis=0
            ),  #  Wavelength Grid; Full Band
            "SolarRadiance": np.concatenate(
                [i for i in Radiance], axis=0
            ),  #  Earth Shine Radiance; Full Band
            "SolarRadianceNESR": np.concatenate(
                [i for i in RadianceNESR], axis=0
            ),  #  NESR of Earth Shine Radiance ; Full Band
            "SolarWavelength_Filter": np.concatenate(
                [i for i in Wavelength_Filter_total], axis=0
            ),
            "usage_pixels": oXtrack,  #  index for healthy pixel
        }

        return o_combined_irad_bands

    @classmethod
    def read_tropomi_cloud(
        cls, fh_cloud: netCDF4.Dataset, iXtrack: int, iTrack: int
    ) -> dict[str, Any]:
        # GroundPixelQualityFlags
        GroundPixelQualityFlags = fh_cloud[
            "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/geolocation_flags"
        ][0, iTrack, iXtrack]

        # Latitude
        Latitude = fh_cloud["PRODUCT/latitude"][0, iTrack, iXtrack]

        # Longitude
        Longitude = fh_cloud["PRODUCT/longitude"][0, iTrack, iXtrack]

        # ProcessingQualityFlags
        ProcessingQualityFlags = fh_cloud[
            "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/processing_quality_flags"
        ][0, iTrack, iXtrack]

        # CloudFraction
        CloudFraction_crb = fh_cloud["PRODUCT/cloud_fraction"][0, iTrack, iXtrack]

        # VK Depending on the dataset version the cloud_fraction_nir variable may be available or not
        try:
            CloudFraction = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_nir"
            ][0, iTrack, iXtrack]
        except IndexError:
            logger.warning("Using cloud_fraction instead of cloud_fraction_nir")
            CloudFraction = CloudFraction_crb

        # EM NOTE - I inserted an arbitrary number, subject to change
        if np.abs(CloudFraction_crb - CloudFraction) > 0.05:
            logger.warning(
                "TROPOMI CAL and CRB cloud fraction values are significantly different"
            )

        if CloudFraction >= 9.0e36:
            logger.warning(
                "TROPOMI CloudFraction >= 9.0e+36. Likely a fill value. Assuming CloudFraction of 0.1"
            )
            CloudFraction = 0.1

        # CloudPressure
        CloudPressure = fh_cloud["PRODUCT/cloud_top_pressure"][0, iTrack, iXtrack]
        # NOTE - CTP is provided in Pa, need to convert to hPA
        CloudPressure = CloudPressure * 0.01

        if CloudPressure >= 9.0e34:
            CloudPressure = fh_cloud["PRODUCT/cloud_top_pressure"][
                0, iTrack, iXtrack + 1
            ]
            CloudPressure = CloudPressure * 0.01

        if CloudPressure >= 9.0e34:
            logger.warning(
                "TROPOMI CloudPressure >= 9.0e+36. Likely a fill value. Assuming CloudPressure of 1016.914 hPa"
            )
            # set to same surface pressure as OSP/L2_Setup/ops/L2_Setup/State_AtmProfiles.asc
            CloudPressure = 1016.914

        # CloudFractionPrecision
        try:
            CloudFractionPrecision = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_crb_precision_nir"
            ][0, iTrack, iXtrack]
        except IndexError:
            CloudFractionPrecision = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_crb_precision"
            ][0, iTrack, iXtrack]
            logger.warning(
                "using cloud_fraction_crb_precision instead of cloud_fraction_crb_precision_nir"
            )

        # CloudPressurePrecision
        CloudPressurePrecision = fh_cloud["PRODUCT/cloud_top_pressure_precision"][
            0, iTrack, iXtrack
        ]

        # Cloud Albedo
        try:
            CloudAlbedo = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_albedo_crb_nir"
            ][0, iTrack, iXtrack]
        except IndexError:
            CloudAlbedo = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_albedo_crb"
            ][0, iTrack, iXtrack]
            logger.warning("Using cloud_albedo_crb instead of cloud_albedo_crb_nir")

        if CloudAlbedo >= 9.0e36:
            logger.warning(
                "TROPOMI CloudAlbedo >= 9.0e+36. Likely a fill value. Assuming CloudAlbedo of 0.8"
            )
            CloudAlbedo = 0.8

        if np.isnan(CloudAlbedo):
            logger.warning("TROPOMI cloud product albedo is NaN, using 0.8")
            CloudAlbedo = 0.8

        # * Cloud Albedo Precision
        try:
            CloudAlbedoPrecision = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_albedo_crb_precision_nir"
            ][0, iTrack, iXtrack]
        except IndexError:
            logger.warning(
                "using cloud_albedo_crb_precision instead of cloud_albedo_crb_precision_nir"
            )
            CloudAlbedoPrecision = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_albedo_crb_precision"
            ][0, iTrack, iXtrack]

        # * Flag of interpolation
        interpolationflag = 0

        # interpolation when needed

        # On suggestion from Susan's email on 07/24/2019, instead of using scipy.interpolation.gridata, we will:
        # 1) find all "good" observations within 0.5 degrees.  If there are observations, use this selection.  If not use all "good" observations with 1.5 degrees.
        # 2) average CloudFraction_all and CloudPressure_all rather than trying to find the "best" interpolated result.

        if (CloudPressure < 0.0) or (CloudFraction < 0.0):
            CloudFraction_all = fh_cloud["PRODUCT/cloud_fraction"][:]
            CloudPressure_all = fh_cloud[
                "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_pressure_crb"
            ]
            Latitude_all = fh_cloud["PRODUCT/latitude"][:]
            Longitude_all = fh_cloud["PRODUCT/longitude"][:]

            lat_lon_ind = (
                (CloudFraction_all >= 0.0)
                & (Longitude_all > Longitude - 1.5)
                & (Longitude_all < Longitude + 1.5)
                & (Latitude_all > Latitude - 1.5)
                & (Latitude_all < Latitude + 1.5)
            )
            if np.count_nonzero(lat_lon_ind) > 0:
                # Try 0.5
                lat_lon_ind2 = (
                    (CloudFraction_all >= 0.0)
                    & (Longitude_all > Longitude - 0.5)
                    & (Longitude_all < Longitude + 0.5)
                    & (Latitude_all > Latitude - 0.5)
                    & (Latitude_all < Latitude + 0.5)
                )

                if np.count_nonzero(lat_lon_ind2) > 0:
                    lat_lon_ind = lat_lon_ind2

                CloudFraction = np.mean(CloudFraction_all[lat_lon_ind])
                CloudPressure = np.mean(CloudPressure_all[lat_lon_ind])

                interpolationflag = 1

        o_tropomi_cloud = {
            "Latitude": Latitude,
            "Longitude": Longitude,
            "CloudPressure": CloudPressure,
            "CloudFraction": CloudFraction,
            "CloudAlbedo": CloudAlbedo,
            "ProcessingQualityFlags": ProcessingQualityFlags,
            "GroundPixelQualityFlags": GroundPixelQualityFlags,
            "CloudFraction_err": CloudFractionPrecision,
            "CloudPressure_err": CloudPressurePrecision,
            "CloudAlbedo_err": CloudAlbedoPrecision,
            "Interpolationflag": interpolationflag,
        }

        return o_tropomi_cloud

    @classmethod
    def combine_tropomi_erad(
        cls,
        tropomi_fns: dict[str, str],
        iXTracks: dict[str, int],
        iATracks: dict[str, int],
        windows: list[dict[str, str]],
        ifile_hlp: InputFileHelper,
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
                    fh = ifile_hlp.open_ncdf(tropomi_fns[current_band])
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

    @classmethod
    def read_tropomi_surface_altitude(
        cls, ifile_hlp: InputFileHelper, Latitude: float, Longitude: float
    ) -> float:
        # The Surface Altitude product for TROPOMI
        # see: https://www.temis.nl/data/gmted2010/index.php
        with ifile_hlp.open_ncdf(
            ifile_hlp.osp_dir / "TROPOMI" / "DEM" / "GMTED2010_15n015_00625deg.nc"
        ) as fh:
            fh.set_auto_maskandscale(False)

            # DEM Latitude
            Latitude_DEM = fh["latitude"][:]

            # DEM Longitude
            Longitude_DEM = fh["longitude"][:]

            # Find closest coordinate
            xi = np.abs(Latitude_DEM - Latitude).argmin()
            yi = np.abs(Longitude_DEM - Longitude).argmin()

            o_tropomi_SurfaceAltitude = fh["elevation"][xi, yi]
            return o_tropomi_SurfaceAltitude

    def desc(self) -> str:
        return "MusesTropomiObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("TROPOMI")

    @property
    def radiance_for_uip(self) -> dict[str, Any]:
        # Explicitly create dict, so we document what is actually needed by
        # RefactorUip/MusesTropomiOrOmiForwardModelBase
        obs_table = {
            "Filter_Band_Name": self.observation_table["Filter_Band_Name"],
            "XTRACK": self.observation_table["XTRACK"],
            "SolarZenithAngle": self.observation_table["SolarZenithAngle"],
            "ViewingZenithAngle": self.observation_table["ViewingZenithAngle"],
            "RelativeAzimuthAngle": self.observation_table["RelativeAzimuthAngle"],
            "ScatteringAngle": self.observation_table["ScatteringAngle"],
            "TerrainHeight": self.observation_table["TerrainHeight"],
            "Latitude": self.observation_table["Latitude"],
        }
        erad = self._muses_py_dict["Earth_Radiance"]
        srad = self._muses_py_dict["Solar_Radiance"]
        return {
            "Earth_Radiance": {
                "EarthWavelength_Filter": erad["EarthWavelength_Filter"],
                "Wavelength": erad["Wavelength"],
                "ObservationTable": obs_table,
                "CalibratedEarthRadiance": erad["CalibratedEarthRadiance"],
                "EarthRadianceNESR": erad["EarthRadianceNESR"],
                "EarthRadiance": erad["EarthRadiance"],
            },
            "Solar_Radiance": {
                "AdjustedSolarRadiance": srad["AdjustedSolarRadiance"],
                "SolarRadiance": srad["SolarRadiance"],
                "SolarWavelength_Filter": srad["SolarWavelength_Filter"],
                "Wavelength": srad["Wavelength"],
            },
        }

    @classmethod
    def create_from_filename(
        cls,
        filename_dict: dict[str, str | os.PathLike[str] | InputFilePath],
        xtrack_dict: dict[str, int],
        atrack_dict: dict[str, int],
        utc_time: str,
        filter_list: list[FilterIdentifier],
        calibration_filename: str | None = None,
        ifile_hlp: InputFileHelper | None = None,
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
            ifile_hlp=ifile_hlp,
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
        ifile_hlp: InputFileHelper,
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
                existing_obs._muses_py_dict,  # noqa: SLF001
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
                ifile_hlp=ifile_hlp,
            )
            obs = cls(o_tropomi, sdesc, filter_list, coeff=coeff, mp=mp)

        if write_tropomi_radiance_pickle:
            # Save file needed by py-retrieve VLIDORT code. Note this *isn't*
            # needed by MusesForwardModel, rather this is to support older
            # old_py_retrieve_wrapper code if needed.
            pfname = os.path.normpath(
                f"{mid['initialGuessDirectory']}/../Radiance_TROPOMI_.pkl"
            )
            if not os.path.exists(pfname):
                subprocess.run(["mkdir", "-p", os.path.dirname(pfname)])
                pickle.dump(obs._muses_py_dict, open(pfname, "wb"))  # noqa: SLF001

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

    def monthly_minimum_surface_reflectance(self, band: int) -> float:
        return float(
            self._muses_py_dict["SurfaceAlbedo"][
                f"BAND{band}_MonthlyMinimumSurfaceReflectance"
            ]
        )

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            float(
                self._muses_py_dict["Earth_Radiance"]["ObservationTable"][
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
            self._muses_py_dict["SurfaceAlbedo"]["MonthlyMinimumSurfaceReflectance"]
        )

    @classmethod
    def _read_data(
        cls,
        filename: str | os.PathLike[str] | InputFilePath,
        xtrack_uv1: int,
        xtrack_uv2: int,
        atrack: int,
        utc_time: str,
        calibration_filename: str | os.PathLike[str] | InputFilePath,
        cld_filename: str | os.PathLike[str] | InputFilePath,
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        o_omi = cls.read_omi(
            filename,
            xtrack_uv2,
            atrack,
            utc_time,
            calibration_filename,
            cld_filename,
            ifile_hlp,
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
        filename: str | os.PathLike[str] | InputFilePath,
        xtrack_uv2: int,
        atrack: int,
        utc_time: str,
        calibration_filename: str | os.PathLike[str] | InputFilePath,
        cld_filename: str | os.PathLike[str] | InputFilePath,
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        if cld_filename is None:
            raise RuntimeError(
                "The old py-retrieve code had logic for searching for a cloud file if not supplied. However it isn't clear if this logic is still working, and in any case we don't have a test for this. So now we require cld_filename it be supplied. We can revisit this if it becomes an issue"
            )
        if ifile_hlp is None:
            ifile_hlp = InputFileHelper()
        # TODO Remove osp_setup below
        with (
            ifile_hlp.open_h5(cld_filename) as f_cld,
            ifile_hlp.open_h5(
                ifile_hlp.osp_dir
                / "OMI"
                / "OMI_LER"
                / "OMI-Aura_L3-OMLER_2005m01-2009m12_v003-2010m0503t063707.he5"
            ) as f_alb,
            osp_setup(ifile_hlp),
        ):
            ifile_hlp.notify_file_input(filename)
            ifile_hlp.notify_file_input(calibration_filename)
            o_omi = cls.read_omi_l1b(
                str(filename),
                xtrack_uv2,
                atrack,
                utc_time,
                str(calibration_filename),
                f_cld,
                f_alb,
            )
        return o_omi

    def desc(self) -> str:
        return "MusesOmiObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("OMI")

    @property
    def radiance_for_uip(self) -> dict[str, Any]:
        # Explicitly create dict, so we document what is actually needed by
        # RefactorUip/MusesTropomiOrOmiForwardModelBase
        obs_table = {
            "Filter_Band_Name": self.observation_table["Filter_Band_Name"],
            "XTRACK": self.observation_table["XTRACK"],
            "MeasurementMode": self.observation_table["MeasurementMode"],
            # Following needed for VLIDORT, this gets passed in the file
            # created for VLIDORT (print_omi_vga)
            "SolarZenithAngle": self.observation_table["SolarZenithAngle"],
            "ViewingZenithAngle": self.observation_table["ViewingZenithAngle"],
            "RelativeAzimuthAngle": self.observation_table["RelativeAzimuthAngle"],
            "ScatteringAngle": self.observation_table["ScatteringAngle"],
            "TerrainHeight": self.observation_table["TerrainHeight"],
            "Latitude": self.observation_table["Latitude"],
        }
        erad = self._muses_py_dict["Earth_Radiance"]
        srad = self._muses_py_dict["Solar_Radiance"]
        return {
            "Earth_Radiance": {
                "EarthWavelength_Filter": erad["EarthWavelength_Filter"],
                "Wavelength": erad["Wavelength"],
                "ObservationTable": obs_table,
                "CalibratedEarthRadiance": erad["CalibratedEarthRadiance"],
                "EarthRadianceNESR": erad["EarthRadianceNESR"],
                "EarthRadiance": erad["EarthRadiance"],
            },
            "Solar_Radiance": {
                "AdjustedSolarRadiance": srad["AdjustedSolarRadiance"],
                "SolarRadiance": srad["SolarRadiance"],
                "SolarWavelength_Filter": srad["SolarWavelength_Filter"],
                "Wavelength": srad["Wavelength"],
            },
        }

    @classmethod
    def create_from_filename(
        cls,
        filename: str | os.PathLike[str] | InputFilePath,
        xtrack_uv1: int,
        xtrack_uv2: int,
        atrack: int,
        utc_time: str,
        calibration_filename: str | os.PathLike[str] | InputFilePath,
        filter_list: list[FilterIdentifier],
        cld_filename: str | os.PathLike[str] | InputFilePath,
        ifile_hlp: InputFileHelper | None = None,
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
            cld_filename,
            ifile_hlp=ifile_hlp,
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
        ifile_hlp: InputFileHelper,
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
                existing_obs._muses_py_dict,  # noqa: SLF001
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
                cld_filename,
                ifile_hlp=ifile_hlp,
            )
            obs = cls(o_omi, sdesc, filter_list, coeff=coeff, mp=mp)

        if write_omi_radiance_pickle:
            # Save file needed by py-retrieve VLIDORT code. Note this *isn't*
            # needed by MusesForwardModel, rather this is to support older
            # old_py_retrieve_wrapper code if needed.
            pfname = os.path.normpath(
                f"{mid['initialGuessDirectory']}/../Radiance_OMI_.pkl"
            )
            if not os.path.exists(pfname):
                subprocess.run(["mkdir", "-p", os.path.dirname(pfname)])
                pickle.dump(obs._muses_py_dict, open(pfname, "wb"))  # noqa: SLF001

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
                self._muses_py_dict["Earth_Radiance"]["ObservationTable"][
                    "TerrainHeight"
                ][0]
            ),
            "m",
        )

    @classmethod
    def read_omi_l1b(
        cls,
        filename: str,
        iXTrack: int,
        iTrack: int,
        UtcDateTime: str,
        calibrationFilename: str,
        f_cld: h5py.File,
        f_alb: h5py.File,
    ) -> dict[str, Any]:
        # ======================
        #  Earth Shine Radiances
        # ======================
        eradd = cls.combine_omi_erad(filename, iXTrack, iTrack)
        if eradd is None:
            raise RuntimeError("combine_omi_erad failed")

        erad = AttrDictAdapter(eradd)

        # ============================
        #  3 Year Mean Solar Radiances
        # ============================
        iradd = cls.combine_omi_yearly_mean_irad(erad.iXtrack_all)
        irad = AttrDictAdapter(iradd)

        if calibrationFilename:
            # ============================
            #  Get Calibration Factors
            # ============================
            rad_cal = AttrDictAdapter(
                cls.combine_omi_calibration(erad.iXtrack_all, calibrationFilename)
            )

            # ===============================================
            # * Earth Shine Radiances with calibration applied
            # ===============================================
            temp_ind1 = np.where(rad_cal.Wavelength_Filter == "UV1")[0]
            temp_ind2 = np.where(erad.EarthWavelength_Filter == "UV1")[0]

            temp_cf1 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )
            temp_cf2 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )

            # PYTHON_NOTE: It seems the Python version has more
            # precisions for temp_cf1 and temp_cf2 which causes
            # problems later.  PYTHON_NOTE: We are also changing to
            # float since it causes problem comparing to IDL.
            temp_cf1 = temp_cf1.astype(np.float64)
            temp_cf2 = temp_cf2.astype(np.float64)

            CALIBRATEDEARTHRADIANCE_uv1v = []
            for ii in range(0, len(temp_ind2)):
                CALIBRATEDEARTHRADIANCE_uv1v.append(
                    erad.EarthRadiance[temp_ind2[ii]] / temp_cf2[ii]
                )

            CALIBRATEDEARTHRADIANCE_uv1 = np.asarray(CALIBRATEDEARTHRADIANCE_uv1v)

            temp_ind1 = np.where(rad_cal.Wavelength_Filter == "UV2")[0]
            temp_ind2 = np.where(erad.EarthWavelength_Filter == "UV2")[0]

            temp_cf1 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )
            temp_cf2 = cls.idl_interpol_1d(
                rad_cal.CalibrationFactor[temp_ind1],
                rad_cal.Wavelength[temp_ind1],
                erad.Wavelength[temp_ind2],
            )

            temp_cf1 = temp_cf1.astype(
                np.float64
            )  # PYTHON_NOTE: We are also changing to float since it causes problem comparing to IDL.
            temp_cf2 = temp_cf2.astype(
                np.float64
            )  # PYTHON_NOTE: We are also changing to float since it causes problem comparing to IDL.

            CALIBRATEDEARTHRADIANCE_uv2 = erad.EarthRadiance[temp_ind2] / temp_cf2[:]
            CALIBRATEDEARTHRADIANCE = np.concatenate(
                (CALIBRATEDEARTHRADIANCE_uv1, CALIBRATEDEARTHRADIANCE_uv2), axis=0
            )

            erad.CalibratedEarthRadiance = CALIBRATEDEARTHRADIANCE  # Adding another key to AttrDictAdapter by merely using the dot '.' notation and an assignment.
        else:
            # ===============================================
            # * Mimic Read_OMI_Without_RadCal.pro
            # ===============================================
            # Note: Continuing to use "Calibrated" naming though these will be uncalibrated radiances. Name change?
            temp_ind2 = np.where(erad.EarthWavelength_Filter == "UV1")[0]
            CALIBRATEDEARTHRADIANCE_uv1v = []
            for ii in range(0, len(temp_ind2)):
                CALIBRATEDEARTHRADIANCE_uv1v.append(erad.EarthRadiance[temp_ind2[ii]])
            CALIBRATEDEARTHRADIANCE_uv1 = np.asarray(CALIBRATEDEARTHRADIANCE_uv1v)

            temp_ind2 = np.where(erad.EarthWavelength_Filter == "UV2")[0]
            CALIBRATEDEARTHRADIANCE_uv2 = erad.EarthRadiance[temp_ind2]

            CALIBRATEDEARTHRADIANCE = np.concatenate(
                (CALIBRATEDEARTHRADIANCE_uv1, CALIBRATEDEARTHRADIANCE_uv2), axis=0
            )

            erad.CalibratedEarthRadiance = CALIBRATEDEARTHRADIANCE

            # Even though we don't use rad_cal, attempts to output it's __dict__ occur below. Create empty object.
            rad_cal = AttrDictAdapter({})

        # ====================================================
        # * Solar IRRadiances with sunEarthDistance Adjustement
        # ====================================================
        one_au = np.float64(149597890000.0)
        temp_cf = (one_au / erad.ObservationTable["EarthSunDistance"][0]) ** 2.0
        ADJUSTEDSOLARRADIANCE = irad.SolarRadiance[:] * temp_cf
        irad.AdjustedSolarRadiance = ADJUSTEDSOLARRADIANCE.copy()

        # ===============================================
        # * align solar and earth radiance wavelength grid
        # ===============================================
        temp_ind1 = np.where(erad.EarthWavelength_Filter == "UV1")[0]
        temp_ind2 = np.where(irad.SolarWavelength_Filter == "UV1")[0]
        irad.SolarRadiance[temp_ind1] = cls.idl_interpol_1d(
            irad.SolarRadiance[temp_ind2],
            irad.Wavelength[temp_ind2],
            erad.Wavelength[temp_ind1],
        )
        irad.AdjustedSolarRadiance[temp_ind1] = cls.idl_interpol_1d(
            ADJUSTEDSOLARRADIANCE[temp_ind2],
            irad.Wavelength[temp_ind2],
            erad.Wavelength[temp_ind1],
        )

        temp_ind1 = np.where(erad.EarthWavelength_Filter == "UV2")[0]
        temp_ind2 = np.where(irad.SolarWavelength_Filter == "UV2")[0]
        irad.SolarRadiance[temp_ind1] = cls.idl_interpol_1d(
            irad.SolarRadiance[temp_ind2],
            irad.Wavelength[temp_ind2],
            erad.Wavelength[temp_ind1],
        )
        irad.AdjustedSolarRadiance[temp_ind1] = cls.idl_interpol_1d(
            ADJUSTEDSOLARRADIANCE[temp_ind2],
            irad.Wavelength[temp_ind2],
            erad.Wavelength[temp_ind1],
        )

        # ============================
        # * Get CloudInformation
        # ============================
        cloudInfo = cls.read_omi_cloud(f_cld, iXTrack, iTrack)

        # ======================
        # * Get OMI SurfaceAlbedo
        # ======================
        temp = UtcDateTime.split("-")
        TMonth = int(temp[1])
        TLongitude = erad.ObservationTable["Longitude"][0]
        TLatitude = erad.ObservationTable["Latitude"][0]
        SurfaceAlbedo = cls.read_omi_surface_albedo(
            f_alb, TLongitude, TLatitude, TMonth
        )

        # * Define output parameter
        o_omi = {
            "Earth_Radiance": erad.__dict__,
            "Solar_Radiance": irad.__dict__,
            "Radiance_Calibration": rad_cal.__dict__,
            "SurfaceAlbedo": SurfaceAlbedo,
            "Cloud": cloudInfo,
        }

        if not np.all(np.isfinite(o_omi["Earth_Radiance"]["CalibratedEarthRadiance"])):
            raise RuntimeError("CalibratedEarthRadiance not finite")
        if not np.all(np.isfinite(o_omi["Earth_Radiance"]["EarthRadiance"])):
            raise RuntimeError("EarthRadiance not finite")
        if not np.all(np.isfinite(o_omi["Earth_Radiance"]["Wavelength"])):
            raise RuntimeError("Wavelength not finite")
        if not np.all(np.isfinite(o_omi["Solar_Radiance"]["AdjustedSolarRadiance"])):
            raise RuntimeError("AdjustedSolarRadiance not finite")

        return o_omi

    @classmethod
    def combine_omi_erad(cls, omi_fn: str, iXTrack: int, iTrack: int) -> dict[str, Any]:
        from refractor.muses_py import read_omi_erad, compute_omi_sca

        # Return a struture variable that has
        #   (1) OMI L1b measured earth spectral radiances/wavelength grid
        #   (2) OMI L1b viewing geometry

        o_combined_erad_bands = (
            None  # Set to None so we can return if something goes wrong.
        )

        iUV = 2  # filter band index uv2 = 2; uv1 = 1
        erad_uv2d = read_omi_erad(omi_fn, iXTrack, iTrack, iUV)
        if erad_uv2d is None:
            raise RuntimeError("error in read_omi_erad")

        erad_uv2 = AttrDictAdapter(erad_uv2d)

        erad_uv1d = {}
        erad_uv2_paird = {}
        if erad_uv2.MeasurementMode == "NORMAL":  # normal mode
            ixtrack_uv1 = math.floor(iXTrack / 2.0)

            if iXTrack == (2 * ixtrack_uv1):
                ixtrack_uv2_pair = 2 * ixtrack_uv1 + 1

            if iXTrack == (2 * ixtrack_uv1 + 1):
                ixtrack_uv2_pair = 2 * ixtrack_uv1

            iUV = 1  # filter band index uv2 = 2; uv1 = 1
            erad_uv1d = read_omi_erad(omi_fn, ixtrack_uv1, iTrack, iUV)
            if erad_uv1d is None:
                raise RuntimeError("error in read_omi_erad")

            iUV = 2  # filter band index uv2 = 2; uv1 = 1
            erad_uv2_paird = read_omi_erad(omi_fn, ixtrack_uv2_pair, iTrack, iUV)
        elif erad_uv2.MeasurementMode == "ZOOM":  # zoom mode
            if iXTrack == 29 or iXTrack == 0:
                raise RuntimeError(
                    "UV2 Pixel Index in ZOOM mode &  = 0 or 29, there is no UV1 Pair match .."
                )

            if iXTrack > 0 and iXTrack < 29:
                temp_mod = iXTrack % 2
                if temp_mod == 0:
                    ixtrack_uv1 = (iXTrack + 14) // 2
                    ixtrack_uv2_pair = 2 * ixtrack_uv1 - 15
                elif temp_mod == 1:
                    ixtrack_uv1 = (iXTrack + 15) // 2
                    ixtrack_uv2_pair = 2 * ixtrack_uv1 - 15 + 1

                iUV = 1  # filter band index uv2 = 2; uv1 = 1
                erad_uv1d = read_omi_erad(omi_fn, ixtrack_uv1, iTrack, iUV)
                if erad_uv1d is None:
                    raise RuntimeError("error in read_omi_erad")

                iUV = 2  # filter band index uv2 = 2; uv1 = 1
                erad_uv2_paird = read_omi_erad(omi_fn, ixtrack_uv2_pair, iTrack, iUV)
                if erad_uv2_paird is None:
                    raise RuntimeError("error in read_omi_erad")
        else:  # unknown mode.
            raise RuntimeError("OMI is not in NORMAL mode nor ZOOM mode")

        erad_uv1 = AttrDictAdapter(erad_uv1d)
        erad_uv2_pair = AttrDictAdapter(erad_uv2_paird)

        iXtrack_all = [ixtrack_uv1, iXTrack, ixtrack_uv2_pair]

        # * Evaluate the Quality of Earth Shine Radiances
        EarthWavelength_Filter_UV1 = np.asarray(
            ["UUU" for ii in range(0, erad_uv1.PixelQualityFlags.shape[0])]
        )
        EarthWavelength_Filter_UV2 = np.asarray(
            ["UUU" for ii in range(0, erad_uv2.PixelQualityFlags.shape[0])]
        )

        EarthWavelength_Filter_UV1[:] = "UV1"
        EarthWavelength_Filter_UV2[:] = "UV2"

        EarthWavelength_Filter = np.concatenate(
            (EarthWavelength_Filter_UV1, EarthWavelength_Filter_UV2), axis=0
        )

        uv1_erad_good_qf = np.where(erad_uv1.PixelQualityFlags == 0)[0]
        uv2_erad_good_qf = np.where(erad_uv2.PixelQualityFlags == 0)[0]
        uv2_erad_good_qf_pair = np.where(erad_uv2_pair.PixelQualityFlags == 0)[0]

        # * when both UV2 pixels are bad, will not process this target scene
        case_select = 0
        if len(uv2_erad_good_qf) > 0 and len(uv2_erad_good_qf_pair) > 0:
            case_select = 1

        if len(uv2_erad_good_qf) == 0 and len(uv2_erad_good_qf_pair) > 0:
            case_select = 2

        if len(uv2_erad_good_qf) > 0 and len(uv2_erad_good_qf_pair) == 0:
            case_select = 3

        if len(uv2_erad_good_qf == 0) and len(uv2_erad_good_qf_pair) == 0:
            case_select = 4

        if case_select == 1:
            usage_pixelsv = [iXTrack, ixtrack_uv2_pair]
        elif case_select == 2:
            erad_uv2 = erad_uv2_pair
            uv2_erad_good_qf = uv2_erad_good_qf_pair
            usage_pixelsv = [-999, ixtrack_uv2_pair]
        elif case_select == 3:
            erad_uv2_pair = erad_uv2
            uv2_erad_good_qf_pair = uv2_erad_good_qf
            usage_pixelsv = [iXTrack, -999]
        elif case_select == 4:
            raise RuntimeError("CASE_SELECT_4:OMI Bad Quality UV2 Spectrum")
        else:
            raise RuntimeError("Unexpected value for case_select")

        usage_pixels = np.asarray(usage_pixelsv)
        usage_pixels = np.concatenate((np.asarray([ixtrack_uv1]), usage_pixels), axis=0)

        # * when UV1 is bad, will continue processing but print out warning
        if len(uv1_erad_good_qf) == 0:
            logger.info(
                f"OMI Bad Quality UV1 Spectrum at ATRACK {iTrack} XTrack {iXTrack}"
            )
            logger.info(
                "Suggest to only use spectral region within the UV2 filter bands"
            )
            usage_pixels[0] = -999

        # Assigned the Earth Spectra --- Wavelength and Radiances
        uv1_erad_bad_qf = np.where(erad_uv1.PixelQualityFlags != 0)[0]
        uv2_erad_bad_qf = np.where(
            (erad_uv2.PixelQualityFlags != 0) | (erad_uv2_pair.PixelQualityFlags != 0)
        )[0]

        erad_uv1.Radiance[uv1_erad_bad_qf] = -999.0
        erad_uv1.NESR[uv1_erad_bad_qf] = -999.0

        erad_uv2.Radiance[uv2_erad_bad_qf] = -999.0
        erad_uv2.NESR[uv2_erad_bad_qf] = -999.0

        erad_uv2_pair.Radiance[uv2_erad_bad_qf] = -999.0
        erad_uv2_pair.NESR[uv2_erad_bad_qf] = -999.0

        temp_wav = (erad_uv2.Wavelength + erad_uv2_pair.Wavelength) / 2.0
        temp_rad = (erad_uv2.Radiance + erad_uv2_pair.Radiance) / 2.0
        temp_nesr = (erad_uv2.NESR + erad_uv2_pair.NESR) / 2.0

        temp_rad[uv2_erad_bad_qf] = np.float64(-999.0)
        temp_nesr[uv2_erad_bad_qf] = np.float64(-999.0)

        EarthWavelength = np.concatenate(
            (np.flip(erad_uv1.Wavelength, axis=0), temp_wav), axis=0
        )
        EarthRadiance = np.concatenate(
            (np.flip(erad_uv1.Radiance, axis=0), temp_rad), axis=0
        )
        EarthRadianceNESR = np.concatenate(
            (np.flip(erad_uv1.NESR, axis=0), temp_nesr), axis=0
        )

        # filter out bad channels using PIXELQUALITYFLAGS
        temp_bad = uv2_erad_bad_qf.copy()
        if len(temp_bad) > 0:
            EarthRadiance[temp_bad] = 0.0
            EarthRadianceNESR[temp_bad] = -999.0

        # filter out spectral spikes
        temp_bad = np.where((EarthRadianceNESR <= 0.0) | (EarthRadianceNESR > 1.0e12))[
            0
        ]
        if len(temp_bad) > 0:
            EarthRadiance[temp_bad] = 0.0
            EarthRadianceNESR[temp_bad] = -999.0

        # filter out secondary spectral spikes
        mean_nesr_ind = np.where((EarthRadianceNESR > 0.0) & (EarthWavelength < 350.0))[
            0
        ]
        mean_nesr = np.mean(EarthRadianceNESR[mean_nesr_ind])
        two_sigma_std_nesr = np.std(EarthRadianceNESR[mean_nesr_ind]) * 2.0

        temp_bad = np.where(
            (EarthRadianceNESR >= (mean_nesr + two_sigma_std_nesr))
            & (EarthWavelength < 350.0)
        )[0]
        if len(temp_bad) > 0:
            EarthRadiance[temp_bad] = 0.0
            EarthRadianceNESR[temp_bad] = -999.0

        # ===================================
        # Dejian Feb 17, 2017
        # base upon 2016 OMI L1B calibration
        # Manually filter out bad chanels
        # ===================================

        # UV1 pixel 3
        if ixtrack_uv1 == 3:
            temp_bad = np.where(
                (EarthWavelength > 293.00)
                & ((EarthWavelength <= 294.25) & (EarthWavelength_Filter == "UV1"))
            )[0]
            if len(temp_bad) > 0:
                EarthRadiance[temp_bad] = 0.0
                EarthRadianceNESR[temp_bad] = -999.0

        # UV1 pixel 7
        if ixtrack_uv1 == 7:
            temp_bad = np.where(
                (EarthWavelength > 310.40)
                & ((EarthWavelength <= 311.0) & (EarthWavelength_Filter == "UV1"))
            )[0]
            if len(temp_bad) > 0:
                EarthRadiance[temp_bad] = 0.0
                EarthRadianceNESR[temp_bad] = -999.0

        # UV1 pixel 21
        if ixtrack_uv1 == 21:
            temp_bad = np.where(
                (EarthWavelength > 294.9)
                & ((EarthWavelength <= 296.2) & (EarthWavelength_Filter == "UV1"))
            )[0]
            if len(temp_bad) > 0:
                EarthRadiance[temp_bad] = 0.0
                EarthRadianceNESR[temp_bad] = -999.0

        # Not implemented because the value of do_old_bad_pixel_map is 0.

        # Viewing Angle Definition

        # uv1 raz
        raz_uv1 = np.abs(erad_uv1.ViewingAzimuthAngle - erad_uv1.SolarAzimuthAngle)
        if raz_uv1 > 180.0:
            raz_uv1 = np.float64(360.0) - raz_uv1
        raz_uv1 = np.float64(180.0) - raz_uv1

        # uv1 sca
        sca_uv1 = compute_omi_sca(
            erad_uv1.ViewingZenithAngle, erad_uv1.SolarZenithAngle, raz_uv1
        )

        # uv2 raz
        raz_uv2 = abs(erad_uv2.ViewingAzimuthAngle - erad_uv2.SolarAzimuthAngle)
        raz_uv2_pair = abs(
            erad_uv2_pair.ViewingAzimuthAngle - erad_uv2_pair.SolarAzimuthAngle
        )

        if raz_uv2 > 180.0:
            raz_uv2 = np.float64(360.0) - raz_uv2

        if raz_uv2_pair > 180.0:
            raz_uv2_pair = np.float64(360.0) - raz_uv2_pair

        raz_uv2 = np.float64(180.0) - raz_uv2
        raz_uv2_pair = np.float64(180.0) - raz_uv2_pair

        # uv2 sca
        sca_uv2 = compute_omi_sca(
            erad_uv2.ViewingZenithAngle, erad_uv2.SolarZenithAngle, raz_uv2
        )
        sca_uv2_pair = compute_omi_sca(
            erad_uv2_pair.ViewingZenithAngle,
            erad_uv2_pair.SolarZenithAngle,
            raz_uv2_pair,
        )

        o_ObservationTable = {
            "ATRACK": [erad_uv1.iTrack, erad_uv2.iTrack, erad_uv2_pair.iTrack],
            "XTRACK": [erad_uv1.iXTrack, erad_uv2.iXTrack, erad_uv2_pair.iXTrack],
            "Latitude": [erad_uv1.Latitude, erad_uv2.Latitude, erad_uv2_pair.Latitude],
            "Longitude": [
                erad_uv1.Longitude,
                erad_uv2.Longitude,
                erad_uv2_pair.Longitude,
            ],
            "Time": [erad_uv1.Time, erad_uv2.Time, erad_uv2_pair.Time],
            "SpacecraftLatitude": [
                erad_uv1.SpacecraftLatitude,
                erad_uv2.SpacecraftLatitude,
                erad_uv2_pair.SpacecraftLatitude,
            ],
            "SpacecraftLongitude": [
                erad_uv1.SpacecraftLongitude,
                erad_uv2.SpacecraftLongitude,
                erad_uv2_pair.SpacecraftLongitude,
            ],
            "SpacecraftAltitude": [
                erad_uv1.SpacecraftAltitude,
                erad_uv2.SpacecraftAltitude,
                erad_uv2_pair.SpacecraftAltitude,
            ],
            "TerrainHeight": [
                erad_uv1.TerrainHeight,
                erad_uv2.TerrainHeight,
                erad_uv2_pair.TerrainHeight,
            ],  # Surface Altitude
            "SolarAzimuthAngle": [
                erad_uv1.SolarAzimuthAngle,
                erad_uv2.SolarAzimuthAngle,
                erad_uv2_pair.SolarAzimuthAngle,
            ],
            "SolarZenithAngle": [
                erad_uv1.SolarZenithAngle,
                erad_uv2.SolarZenithAngle,
                erad_uv2_pair.SolarZenithAngle,
            ],
            "ViewingAzimuthAngle": [
                erad_uv1.ViewingAzimuthAngle,
                erad_uv2.ViewingAzimuthAngle,
                erad_uv2_pair.ViewingAzimuthAngle,
            ],
            "ViewingZenithAngle": [
                erad_uv1.ViewingZenithAngle,
                erad_uv2.ViewingZenithAngle,
                erad_uv2_pair.ViewingZenithAngle,
            ],
            "RelativeAzimuthAngle": [raz_uv1, raz_uv2, raz_uv2_pair],
            "ScatteringAngle": [sca_uv1, sca_uv2, sca_uv2_pair],
            "EarthSunDistance": [
                erad_uv1.EarthSunDistance,
                erad_uv2.EarthSunDistance,
                erad_uv2_pair.EarthSunDistance,
            ],
            "MeasurementMode": [
                erad_uv1.MeasurementMode,
                erad_uv2.MeasurementMode,
                erad_uv2_pair.MeasurementMode,
            ],
            "Filter_Band_Name": ["UV1", "UV2", "UV2"],
        }

        # ----- Output Parameter
        o_combined_erad_bands = {
            "omi_earth_rad_fn": erad_uv2.omi_file,  # L1B Earth Radiance Full Path and File Name
            "Wavelength": EarthWavelength,  # Wavelength Grid; Full Band
            "EarthRadiance": EarthRadiance,  # Earth Shine Radiance; Full Band
            "EarthRadianceNESR": EarthRadianceNESR,  # NESR of Earth Shine Radiance ; Full Band;
            "EarthWavelength_Filter": EarthWavelength_Filter,  # Optical Filter Name for each Channel; Full Bands;
            "usage_pixels": usage_pixels,  # index for healthy pixel
            "ObservationTable": o_ObservationTable,  # Observation Geometry Including Pixel Indexa; Latitut and Longitude; Viewing Geometry
            "iXtrack_all": iXtrack_all,
        }

        return o_combined_erad_bands

    @classmethod
    def read_omi_cloud(
        cls, f_cld: h5py.File, iXtrack: int, iTrack: int
    ) -> dict[str, Any]:
        # * GroundPixelQualityFlags
        GroundPixelQualityFlags = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Geolocation Fields/GroundPixelQualityFlags"
        ][iTrack, iXtrack]

        # * Latitude
        Latitude = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Geolocation Fields/Latitude"
        ][iTrack, iXtrack]

        # * Longitude
        Longitude = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Geolocation Fields/Longitude"
        ][iTrack, iXtrack]

        # * ProcessingQualityFlags
        ProcessingQualityFlags = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/ProcessingQualityFlags"
        ][iTrack, iXtrack]

        # * CloudFraction
        CloudFraction = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/CloudFraction"
        ][iTrack, iXtrack]

        # * CloudPressure
        CloudPressure = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/CloudPressure"
        ][iTrack, iXtrack]

        # * CloudFractionPrecision
        CloudFractionPrecision = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/CloudFractionPrecision"
        ][iTrack, iXtrack]

        # * CloudPressurePrecision
        CloudPressurePrecision = f_cld[
            "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/CloudPressurePrecision"
        ][iTrack, iXtrack]

        # * Flag of interpolation
        interpolationflag = 0

        # interpolation when needed

        # On suggestion from Susan's email on 07/24/2019, instead of using scipy.interpolation.gridata, we will:
        # 1) find all "good" observations within 0.5 degrees.  If there are observations, use this selection.  If not use all "good" observations with 1.5 degrees.
        # 2) average CloudFraction_all and CloudPressure_all rather than trying to find the "best" interpolated result.

        if (CloudPressure < 0.0) or (CloudFraction < 0.0):
            CloudFraction_all = f_cld[
                "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/CloudFraction"
            ][:]
            CloudPressure_all = f_cld[
                "/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields/CloudPressure"
            ][:]
            Latitude_all = f_cld[
                "/HDFEOS/SWATHS/CloudFractionAndPressure/Geolocation Fields/Latitude"
            ][:]
            Longitude_all = f_cld[
                "/HDFEOS/SWATHS/CloudFractionAndPressure/Geolocation Fields/Longitude"
            ][:]

            lat_lon_ind = (
                (CloudFraction_all >= 0.0)
                & (Longitude_all > Longitude - 1.5)
                & (Longitude_all < Longitude + 1.5)
                & (Latitude_all > Latitude - 1.5)
                & (Latitude_all < Latitude + 1.5)
            )
            if np.count_nonzero(lat_lon_ind) > 0:
                # Try 0.5
                lat_lon_ind2 = (
                    (CloudFraction_all >= 0.0)
                    & (Longitude_all > Longitude - 0.5)
                    & (Longitude_all < Longitude + 0.5)
                    & (Latitude_all > Latitude - 0.5)
                    & (Latitude_all < Latitude + 0.5)
                )

                if np.count_nonzero(lat_lon_ind2) > 0:
                    lat_lon_ind = lat_lon_ind2

                CloudFraction = np.mean(CloudFraction_all[lat_lon_ind])
                CloudPressure = np.mean(CloudPressure_all[lat_lon_ind])

                interpolationflag = 1

        o_omi_cloud = {
            "Latitude": Latitude,
            "Longitude": Longitude,
            "CloudPressure": CloudPressure,
            "CloudFraction": CloudFraction,
            "ProcessingQualityFlags": ProcessingQualityFlags,
            "GroundPixelQualityFlags": GroundPixelQualityFlags,
            "CloudFraction_err": CloudFractionPrecision,
            "CloudPressure_err": CloudPressurePrecision,
            "Interpolationflag": interpolationflag,
        }

        return o_omi_cloud

    @classmethod
    def combine_omi_yearly_mean_irad(cls, iXtrack_all: list[int]) -> dict[str, Any]:
        from refractor.muses_py import read_omi_osp_irad
        # IDL_LEGACY_NOTE: This function combine_omi_yearly_mean_irad is the same as Combine_OMI_YearlyMeanIrad function in ms-setup/Combine_OMI_YearlyMeanIrad.pro file.
        # Description:
        # Return a struture variable that has
        #   (1) OMI L1b measured solar spectral radiances/wavelength grid
        #
        #
        # ==================================================================
        # Dependency:
        #  readhdfsd.pro   --- read maxtrix

        # Input: omi_fn: OMI L1B Earth Radiance file name
        #        iXtrack : CrossTrack Index
        #
        # Output:
        #       combined_irad_bands
        # =======================================================
        # Author:
        #
        #   Dejian Fu; November 17th, 2014
        #
        #           Dejian.fu@jpl.nasa.gov
        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        ixtrack_uv1 = iXtrack_all[0]
        ixtrack = iXtrack_all[1]
        ixtrack_uv2_pair = iXtrack_all[2]

        # ==============================
        # * Get 3 Yearly Mean Solar Spec
        # ==============================
        # For UV2
        iUV = 2
        irad_uv2 = read_omi_osp_irad(ixtrack, iUV)
        irad_uv2_pair = read_omi_osp_irad(ixtrack_uv2_pair, iUV)

        # * For uv1
        iUV = 1
        irad_uv1 = read_omi_osp_irad(ixtrack_uv1, iUV)

        # Convert our dictionaries so we use the dot '.' notation.
        irad_uv2 = AttrDictAdapter(irad_uv2)
        irad_uv2_pair = AttrDictAdapter(irad_uv2_pair)
        irad_uv1 = AttrDictAdapter(irad_uv1)

        # * Define output parameter
        Wavelength_Filter_UV1 = np.asarray(
            ["UV1" for ii in range(0, len(irad_uv1.Sol))]
        )
        Wavelength_Filter_UV2 = np.asarray(
            ["UV2" for ii in range(0, len(irad_uv2.Sol))]
        )
        Wavelength_Filter = np.concatenate(
            (Wavelength_Filter_UV1, Wavelength_Filter_UV2), axis=0
        )

        Radiance = np.concatenate(
            (irad_uv1.Sol, (irad_uv2.Sol[:] + irad_uv2_pair.Sol[:]) / np.float64(2.0)),
            axis=0,
        )
        RadianceNESR = np.concatenate(
            (irad_uv1.Pre, (irad_uv2.Pre[:] + irad_uv2_pair.Pre[:]) / np.float64(2.0)),
            axis=0,
        )
        Wavelength = np.concatenate(
            (irad_uv1.Wav, (irad_uv2.Wav[:] + irad_uv2_pair.Wav[:]) / np.float64(2.0)),
            axis=0,
        )

        o_combined_irad_bands = {
            "omi_solar_rad_fn": irad_uv2.omi_file,  # L1B Earth Radiance Full Path and File Name
            "Wavelength": Wavelength,  #  Wavelength Grid; Full Band
            "SolarRadiance": Radiance,  #  Earth Shine Radiance; Full Band
            "SolarRadianceNESR": RadianceNESR,  #  NESR of Earth Shine Radiance ; Full Band
            "SolarWavelength_Filter": Wavelength_Filter,  #  Optical Filter Name for each Channel; Full Bands
            "usage_pixels": iXtrack_all,  #  index for healthy pixel
        }

        return o_combined_irad_bands


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
