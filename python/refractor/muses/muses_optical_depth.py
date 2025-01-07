from __future__ import annotations
import numpy as np
import math
import refractor.framework as rf
import refractor.muses.muses_py as mpy
from pathlib import Path
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
        osp_dir: str | Path,
    ):
        """Creator"""
        # Dummy since we are overwriting the optical_depth function
        self.osp_dir = osp_dir
        self.obs = obs
        self.ils_params_list = ils_params_list
        xsec_tables = rf.vector_xsec_table()

        spec_grid = rf.ArrayWithUnit(np.array([1, 2]), "nm")
        xsec_values = rf.ArrayWithUnit(np.zeros((2, 1)), "cm^2")
        xsec_tables.push_back(rf.XSecTableSimple(spec_grid, xsec_values, 0.0))

        # Register base director class
        rf.AbsorberXSec.__init__(
            self, absorber_vmr, pressure, temperature, altitude, xsec_tables
        )
        self.xsect_grid = []
        self.xsect_data = []
        for i in range(self.obs.num_channels):
            if self.obs.spectral_domain(i).data.shape[0] == 0:
                t1 = []
                t2 = []
            elif self.obs.instrument_name == "TROPOMI":
                t1, t2 = self._xsect_tropomi_ils(i)
            elif self.obs.instrument_name == "OMI":
                t1, t2 = self._xsect_omi_ils(i)
            else:
                raise RuntimeError(
                    f"Unrecognized instrument name {self.obs.instrument_name}"
                )
            self.xsect_grid.append(t1)
            self.xsect_data.append(t2)

    def _xsect_tropomi_ils(self, sensor_index: int):
        uip = None
        do_temp_shift = False
        # I don't think we ever have no2_col. We can add this in if needed later,
        # this would just be gas_number_density_layer for no2.
        no2_col = []
        tatm = self.temperature.temperature_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        ).value.value
        gas_col = self.gas_number_density_layer(sensor_index).value.value
        # Note that get_tropomi_o3xsec expects this in the opposite order (so decreasing
        # pressure) So we flip this
        o3_col = gas_col[::-1, 0]
        wn, sindex = self.obs.wn_and_sindex(sensor_index)
        o3_xsec = mpy.get_tropomi_o3xsec(
            str(self.osp_dir),
            self.ils_params_list[sensor_index],
            tatm,
            wn,
            sindex,
            uip,
            do_temp_shift,
            o3_col,
            no2_col,
            self.obs.muses_py_dict,
        )
        xsect_grid = o3_xsec["X0"][o3_xsec["freqIndex"]]
        xsect_data = o3_xsec["o3xsec"]
        return xsect_grid.astype(float), xsect_data.astype(float)

    def _xsect_omi_ils(self, sensor_index: int):
        tatm = self.temperature.temperature_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        ).value.value
        gas_col = self.gas_number_density_layer(sensor_index).value.value
        # Note that get_tropomi_o3xsec expects this in the opposite order (so decreasing
        # pressure) So we flip this
        o3_col = gas_col[::-1, 0]
        wn, sindex = self.obs.wn_and_sindex(sensor_index)
        o3_xsec = mpy.get_omi_o3xsec(
            str(self.osp_dir),
            self.ils_params_list[sensor_index],
            tatm,
            wn,
            sindex,
            o3_col,
            self.obs.muses_py_dict,
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
