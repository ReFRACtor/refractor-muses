from __future__ import annotations
import numpy as np
import math
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier
from .input_file_helper import InputFileHelper
import os
import scipy
from functools import cache
import pandas
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import MusesObservation


class MusesOpticalDepth(rf.AbsorberXSec):
    """This is like MusesOpticalDepthFile, but we try to use as much
    of ReFRACtor as possible. The optical depth information is from
    py-retrieve, so this handling doing PRECONV. But most of the
    calculations are done using ReFRACtor code.

    """

    def __init__(
        self,
        pressure: rf.Pressure,
        temperature: rf.Temperature,
        altitude: rf.Altitude,
        absorber_vmr: rf.AbsorberVmr,
        obs: MusesObservation,
        ils_params_list: list[dict],
        ifile_hlp: InputFileHelper,
        osp_dir: str | os.PathLike[str],
    ) -> None:
        """Creator"""
        # Dummy since we are overwriting the optical_depth function
        self.osp_dir = Path(osp_dir)
        self.ifile_hlp = ifile_hlp
        self.obs = obs
        self.ils_params_list = ils_params_list
        xsec_tables = []

        spec_grid = rf.ArrayWithUnit(np.array([1, 2]), "nm")
        xsec_values = rf.ArrayWithUnit(np.zeros((2, 1)), "cm^2")
        xsec_tables.append(rf.XSecTableSimple(spec_grid, xsec_values, 0.0))

        # Register base director class
        rf.AbsorberXSec.__init__(
            self, absorber_vmr, pressure, temperature, altitude, xsec_tables
        )
        self.xsect_grid: list[np.ndarray] = []
        self.xsect_data: list[np.ndarray] = []
        for i in range(self.obs.num_channels):
            if self.obs.spectral_domain(i).data.shape[0] == 0:
                t1 = np.array([])
                t2 = np.array([])
            elif self.obs.instrument_name == InstrumentIdentifier("TROPOMI"):
                t1, t2 = self._xsect_tropomi_ils(i)
            elif self.obs.instrument_name == InstrumentIdentifier("OMI"):
                t1, t2 = self._xsect_omi_ils(i)
            else:
                raise RuntimeError(
                    f"Unrecognized instrument name {self.obs.instrument_name}"
                )
            self.xsect_grid.append(np.array(t1))
            self.xsect_data.append(np.array(t2))

    def _xsect_tropomi_ils(self, sensor_index: int) -> tuple[np.ndarray, np.ndarray]:
        # I don't think we ever have no2_col. We can add this in if needed later,
        # this would just be gas_number_density_layer for no2.
        tatm = self.temperature.temperature_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        ).value.value
        gas_col = self.gas_number_density_layer(sensor_index).value.value
        # Note that get_tropomi_o3xsec expects this in the opposite order (so decreasing
        # pressure) So we flip this
        o3_col = gas_col[::-1, 0]
        wn, sindex = self.obs.wn_and_sindex(sensor_index)
        o3_xsec = self.get_tropomi_o3xsec(
            sensor_index,
            tatm,
            wn,
            sindex,
            o3_col,
        )
        xsect_grid = o3_xsec["X0"][o3_xsec["freqIndex"]]
        xsect_data = o3_xsec["o3xsec"]
        return xsect_grid.astype(float), xsect_data.astype(float)

    def _xsect_omi_ils(self, sensor_index: int) -> tuple[np.ndarray, np.ndarray]:
        tatm = self.temperature.temperature_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        ).value.value
        gas_col = self.gas_number_density_layer(sensor_index).value.value
        # Note that get_tropomi_o3xsec expects this in the opposite order (so decreasing
        # pressure) So we flip this
        o3_col = gas_col[::-1, 0]
        wn, sindex = self.obs.wn_and_sindex(sensor_index)
        o3_xsec = self.get_omi_o3xsec(
            sensor_index,
            tatm,
            wn,
            sindex,
            o3_col,
        )
        xsect_grid = o3_xsec["X0"][o3_xsec["freqIndex"]]
        xsect_data = o3_xsec["o3xsec"]
        return xsect_grid.astype(float), xsect_data.astype(float)

    def optical_depth_each_layer(
        self, wn: float, sensor_index: int
    ) -> rf.ArrayAd_double_2:
        # Convert value to units of spectral points used in file
        spec_point = rf.DoubleWithUnit(wn, "cm^-1").convert_wave("nm").value

        # Find index of closest value
        od_index = np.searchsorted(
            self.xsect_grid[sensor_index], spec_point, side="left"
        )
        if od_index > 0 and (
            od_index == self.xsect_grid[sensor_index].shape[0]
            or math.fabs(spec_point - self.xsect_grid[sensor_index][od_index - 1])
            < math.fabs(spec_point - self.xsect_grid[sensor_index][od_index])
        ):
            od_index -= 1

        # Extra axis is the species index, not used since we only know about ozone
        wn_xsect_data = self.xsect_data[sensor_index][od_index, :]
        gdens = self.gas_number_density_layer(sensor_index).value
        # The number of layers may be different then the full wn_xsect_data if we
        # are handling clouds. So truncate to actual number of layers
        nlay = self.pressure.number_layer
        wn_od_data = wn_xsect_data[:nlay, np.newaxis] * gdens.value
        if gdens.is_constant:
            od_result = rf.ArrayAd_double_2(wn_od_data)
        else:
            wn_od_data_jac = (
                wn_xsect_data[:nlay, np.newaxis, np.newaxis] * gdens.jacobian
            )
            od_result = rf.ArrayAd_double_2(wn_od_data, wn_od_data_jac)
        return od_result

    def desc(self) -> str:
        s = "MusesOpticalDepth\n"
        s += self.print_parent()
        return s

    def get_tropomi_o3xsec(
        self,
        sensor_index: int,
        i_TATM: np.ndarray,
        i_tropomifreq: np.ndarray,
        i_tropomifreqIndex: int,
        i_o3_col: np.ndarray,
    ) -> dict[str, Any]:
        fn = self.osp_dir / "TROPOMI/Ozone_Xsec/serdyuchenkoo3temp.dat"
        self.ifile_hlp.notify_file_input(fn)
        (c0, c1, c2, temp_wav_all) = _tropomi_o3xsec(fn)

        start_freq = np.amin(i_tropomifreq[i_tropomifreqIndex]) - np.float64(1.0)
        endd_freq = np.amax(i_tropomifreq[i_tropomifreqIndex]) + np.float64(1.0)

        temp_ind = np.where((temp_wav_all >= start_freq) & (temp_wav_all <= endd_freq))[
            0
        ]
        start_ind = np.amin(temp_ind)

        count = len(temp_ind)

        freq_vector = temp_wav_all[temp_ind]

        c0 = c0[start_ind : start_ind + count]
        c1 = c1[start_ind : start_ind + count]
        c2 = c2[start_ind : start_ind + count]

        # get TATM layer
        TATM = np.flip(i_TATM, axis=0)

        nlevel = len(TATM)
        nlayer = len(TATM) - 1

        TATM_L = TATM[1:nlevel] * np.float64(0.5) + TATM[0 : nlevel - 1] * np.float64(
            0.5
        )

        o3_col = np.flip(i_o3_col, axis=0)

        tropomi = self.obs.muses_py_dict

        max_ind = np.amax(i_tropomifreqIndex)

        temp_filter = tropomi["Earth_Radiance"]["EarthWavelength_Filter"][max_ind]

        con1 = tropomi["Solar_Radiance"]["SolarWavelength_Filter"] == temp_filter
        con2 = tropomi["Solar_Radiance"]["Wavelength"] >= (
            np.amin(freq_vector) - np.float64(0.01)
        )
        con3 = tropomi["Solar_Radiance"]["Wavelength"] <= (
            np.amax(freq_vector) + np.float64(0.02)
        )

        temp_ind = np.where((con1 & con2) & con3)[0]

        sol_wav = tropomi["Solar_Radiance"]["Wavelength"][temp_ind]
        sol_int = tropomi["Solar_Radiance"]["AdjustedSolarRadiance"][temp_ind]

        temp_ind = np.where(sol_int > 0.0)[0]
        sol_wav = sol_wav[temp_ind]
        sol_int = sol_int[temp_ind]

        sol_interp = scipy.interpolate.interp1d(
            sol_wav, sol_int, fill_value="extrapolate"
        )
        central_wavelength = self.ils_params_list[sensor_index]["central_wavelength"]
        central_wavelength = central_wavelength[np.where(central_wavelength > -999)]
        num_points = len(central_wavelength)
        o3xsec = np.ndarray(shape=(num_points, nlayer), dtype=np.float64)
        abs_0_tatm = np.float64(273.15)
        convert_factor = 1e20
        for tempi in range(0, nlayer):
            o3xsec_mono = c0[:] * (
                1
                + c1[:] * (TATM_L[tempi] - abs_0_tatm)
                + c2[:] * (TATM_L[tempi] - abs_0_tatm) * (TATM_L[tempi] - abs_0_tatm)
            )
            o3xsec_mono = o3xsec_mono[:] / convert_factor
            o3xsec_interp = scipy.interpolate.interp1d(
                freq_vector, o3xsec_mono, fill_value="extrapolate"
            )

            for tempj in range(0, num_points):
                temp_delta_wavelength = self.ils_params_list[sensor_index][
                    "delta_wavelength"
                ][tempj, :].flatten()
                temp_delta_wavelength = temp_delta_wavelength[
                    np.where(temp_delta_wavelength > -999)
                ]
                ils_wavelength = central_wavelength[tempj] + temp_delta_wavelength
                temp_ils = self.ils_params_list[sensor_index]["isrf"][
                    tempj, :
                ].flatten()
                temp_ils = temp_ils[np.where(temp_ils > -999)]

                temp_o3xsec = o3xsec_interp(ils_wavelength)
                temp_solar = sol_interp(ils_wavelength)  # does this vary?

                o3xsec[tempj, tempi] = np.sum(
                    temp_solar[:]
                    * np.exp(np.float64(-1.0) * temp_o3xsec[:] * o3_col[tempi])
                    * temp_ils[:]
                ) / np.sum(temp_solar[:] * temp_ils[:])
                o3xsec[tempj, tempi] = (
                    np.float64(-1.0) * np.log(o3xsec[tempj, tempi]) / o3_col[tempi]
                )

        o_o3xsecInfo = {
            "X0": i_tropomifreq,
            "freqIndex": i_tropomifreqIndex,
            "o3xsec": o3xsec,
            "num_points": num_points,
            "nlayer": nlayer,
        }

        return o_o3xsecInfo

    def get_omi_o3xsec(
        self,
        sensor_index: int,
        i_TATM: np.ndarray,
        i_omifreq: np.ndarray,
        i_omifreqIndex: np.ndarray,
        i_o3_col: np.ndarray,
    ) -> dict[str, Any]:
        fn = self.osp_dir / "OMI/Ozone_Xsec/o3abs_brion_195_660_vacfinal.h5"
        self.ifile_hlp.notify_file_input(fn)
        c0, c1, c2, temp_wav_all = _omi_o3xsec(fn)

        start_freq = np.amin(
            self.ils_params_list[sensor_index]["v1_mono"]
        ) - np.float64(1.0)
        endd_freq = np.amax(self.ils_params_list[sensor_index]["v2_mono"]) + np.float64(
            1.0
        )

        temp_ind = np.where((temp_wav_all >= start_freq) & (temp_wav_all <= endd_freq))[
            0
        ]
        start_ind = np.amin(temp_ind)

        count = len(temp_ind)

        freq_vector = temp_wav_all[temp_ind]

        c0 = c0[start_ind : start_ind + count]  # Get a smaller slice of c0
        c1 = c1[start_ind : start_ind + count]  # Get a smaller slice of c1
        c2 = c2[start_ind : start_ind + count]  # Get a smaller slice of c2

        TATM = np.flip(i_TATM, axis=0)
        nlevel = len(TATM)
        nlayer = len(TATM) - 1

        TATM_L = TATM[1:nlevel] * np.float64(0.5) + TATM[0 : nlevel - 1] * np.float64(
            0.5
        )

        o3_col = np.flip(i_o3_col, axis=0)

        omi = self.obs.muses_py_dict

        max_ind = np.amax(i_omifreqIndex)

        if max_ind < 159:
            temp_filter = "UV1"

        if max_ind > 159 and max_ind < 716:
            temp_filter = "UV2"

        con1 = omi["Solar_Radiance"]["SolarWavelength_Filter"] == temp_filter
        con2 = omi["Solar_Radiance"]["Wavelength"] >= (
            np.amin(freq_vector) - np.float64(0.01)
        )
        con3 = omi["Solar_Radiance"]["Wavelength"] <= (
            np.amax(freq_vector) + np.float64(0.02)
        )

        temp_ind = np.where((con1 & con2) & con3)[0]

        sol_wav = omi["Solar_Radiance"]["Wavelength"][temp_ind]
        sol_int = omi["Solar_Radiance"]["AdjustedSolarRadiance"][temp_ind]

        temp_ind = np.where(sol_int > 0.0)[0]
        sol_wav = sol_wav[temp_ind]
        sol_int = sol_int[temp_ind]
        interpfunc_solar = scipy.interpolate.interp1d(
            sol_wav, sol_int, fill_value="extrapolate"
        )

        # compute I0-corrected ozone xcross sections
        num_points = len(self.ils_params_list[sensor_index]["X0_fm"])
        o3xsec = np.ndarray(shape=(num_points, nlayer), dtype=np.float64)
        abs_0_tatm = np.float64(273.15)
        convert_factor = 1e20
        for tempi in range(0, nlayer):
            o3xsec_mono = (
                c0[:]
                + c1[:] * (TATM_L[tempi] - abs_0_tatm)
                + c2[:] * (TATM_L[tempi] - abs_0_tatm) * (TATM_L[tempi] - abs_0_tatm)
            )

            o3xsec_mono = o3xsec_mono[:] / convert_factor

            interpfunc_o3xsec = scipy.interpolate.interp1d(
                freq_vector, o3xsec_mono, fill_value="extrapolate"
            )

            for tempj in range(0, num_points):
                x0 = i_omifreq[i_omifreqIndex[tempj]]
                NP_SINGLE_SIDE = self.ils_params_list[sensor_index]["NP_SINGLE_SIDE"][
                    tempj
                ]

                xcf = np.arange(0, NP_SINGLE_SIDE * 2 + 1) * np.float64(0.01)
                xcf[:] = xcf[:] - np.mean(xcf[:]) + x0

                temp_o3xsec = interpfunc_o3xsec(xcf)

                temp_solar = interpfunc_solar(xcf)

                temp_ils = self.ils_params_list[sensor_index]["ilsval"][:, tempj]
                temp_ils = temp_ils[0 : NP_SINGLE_SIDE * 2 + 1]

                o3xsec[tempj, tempi] = np.sum(
                    temp_solar[:]
                    * np.exp(np.float64(-1.0) * temp_o3xsec[:] * o3_col[tempi])
                    * temp_ils[:]
                ) / np.sum(temp_solar[:] * temp_ils[:])

                o3xsec[tempj, tempi] = (
                    np.float64(-1.0) * np.log(o3xsec[tempj, tempi]) / o3_col[tempi]
                )
        o_o3xsecInfo = {
            "X0": i_omifreq,
            "freqIndex": i_omifreqIndex,
            "o3xsec": o3xsec,
            "num_points": num_points,
            "nlayer": nlayer,
        }

        return o_o3xsecInfo


# Pulled out, just so we can cache this. Not sure how important that is, but original
# code has this saved, and it doesn't change to no reason not to cache it.
@cache
def _tropomi_o3xsec(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # We catch this file at a higher level
    header_skip = 20
    t = pandas.read_csv(
        fn,
        skiprows=header_skip,
        header=None,
        names=["wav", "c0", "c1", "c2"],
        sep=r"\s+",
    )
    return np.array(t["c0"]), np.array(t["c1"]), np.array(t["c2"]), np.array(t["wav"])


@cache
def _omi_o3xsec(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # We catch this file at a higher level
    with InputFileHelper.open_h5(fn, None) as f:
        wav = f["WAV_ALL"][:]
        c0 = f["C0_ALL"][:]
        c1 = f["C1_ALL"][:]
        c2 = f["C2_ALL"][:]
    return c0, c1, c2, wav


__all__ = [
    "MusesOpticalDepth",
]
