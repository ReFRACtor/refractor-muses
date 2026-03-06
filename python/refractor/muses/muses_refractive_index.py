from __future__ import annotations
import refractor.framework as rf  # type: ignore
import math
import numpy as np

class MusesRefractiveIndex:
    def __init__(self, pressure: rf.Pressure, temperature: rf.Temperature,
                 h2o_vmr : rf.AbsorberVmr, altitude: rf.Altitude) -> None:
        self.pressure = pressure
        self.temperature = temperature
        self.h2o_vmr = h2o_vmr
        self.altitude = altitude
        self.cache_observer = rf.CacheInvalidatedObserver()
        self.pressure.add_cache_invalidated_observer(self.cache_observer)
        self.temperature.add_cache_invalidated_observer(self.cache_observer)
        self.h2o_vmr.add_cache_invalidated_observer(self.cache_observer)
        self.altitude.add_cache_invalidated_observer(self.cache_observer)

    def _fill_in_cache(self):
        if self.cache_observer.cache_valid_flag:
            return
        self._pgrid = (
            self.pressure.pressure_grid(rf.Pressure.DECREASING_PRESSURE)
            .convert("Pa")
            .value.value
        )
        self._lnp = np.log(self._pgrid)
        self._tatm = self.temperature.temperature_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        ).convert("K").value.value
        self._h2ogrid = self.h2o_vmr.vmr_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        ).value
        self._altgrid = self.altitude.altitude_grid(self.pressure, rf.Pressure.DECREASING_PRESSURE).convert("m").value.value
        
        self.cache_observer.cache_valid_flag = True

    def refractive_index(self, alt: rf.DoubleWithUnit) -> float:
        self._fill_in_cache()
        av = alt.convert("m").value
        # Find the index that is just >= to av
        ind = np.searchsorted(self._altgrid, av)
        # Handle the == case, to give strictly >
        if(av >= self._altgrid[ind]):
            ind += 1
        ind -= 1
        # And handle end points, we just extrapolate if needed.
        if ind < 0:
            ind += 1
        if ind + 1 >= self._pgrid.shape[0]:
            ind -= 1

        # Interpolate levels to get data at the altitude
        hp = -(self._altgrid[ind + 1] - self._altgrid[ind]) / np.log(
            self._pgrid[ind + 1] / self._pgrid[ind]
        )
        p = self._pgrid[ind] * math.exp(-(av - self._altgrid[ind]) / hp)
        t = self._tatm[ind] + (av - self._altgrid[ind]) * (
            self._tatm[ind + 1] - self._tatm[ind]
        ) / (self._altgrid[ind + 1] - self._altgrid[ind])
        h2o = self._h2ogrid[ind] + (np.log(p) - self._lnp[ind]) * (
            self._h2ogrid[ind + 1] - self._h2ogrid[ind]
        ) / np.log(self._pgrid[ind + 1] / self._pgrid[ind])
        return self._ref_index(t, p, h2o)

    def _ref_index(self,
            i_temperature: float, i_pressure: float, i_H2O_vmr: float
                   ) -> float:
        frequency = 1050.0
        refractive_index = 0.0
        frequency_squared = 0.0
        temperature_centigrade = 0.0
        H2O_partial_pressure = 0.0

        a0 = 8342.54
        a1 = 2406147.0
        a2 = 130.0
        a3 = 15998.0
        a4 = 38.9
        a5 = 96095.43
        a6 = 0.601
        a7 = 0.00972
        a8 = 0.0036610
        a9 = 3.7345
        a10 = 0.0401

        TEMPERATURE_CONVERSION_FACTOR = 273.15

        frequency_squared = (frequency * 1.0e-4) * (frequency * 1.0e-4)
        temperature_centigrade = i_temperature - TEMPERATURE_CONVERSION_FACTOR
        
        H2O_partial_pressure = i_H2O_vmr * i_pressure

        b0 = 1.0e-8 * (a0 + a1 / (a2 - frequency_squared) + a3 / (a4 - frequency_squared))
        b1 = (
            i_pressure
            * b0
            / a5
            * (1.0 + 1.0e-8 * (a6 - a7 * temperature_centigrade) * i_pressure)
            / (1.0 + a8 * temperature_centigrade)
        )

        refractive_index = (
            1.0 + b1 - H2O_partial_pressure * (a9 - a10 * frequency_squared) * 1.0e-10
        )

        return refractive_index
    

__all__ = ["MusesRefractiveIndex",]    
