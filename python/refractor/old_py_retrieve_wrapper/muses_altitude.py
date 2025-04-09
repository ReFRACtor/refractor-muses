from __future__ import annotations
import numpy as np
import refractor.framework as rf  # type: ignore
from .muses_ray_info import MusesRayInfo
from typing import cast, Self


class MusesAltitude(rf.Altitude):
    """This uses the py-retrieve MusesRayInfo to calculate Altitude information.

    This was used initially to match how py-retrieve did the calculation, so we could
    compare the forward model runs with ReFRACtor without having the minor differences in
    the altitude calculation enter into the differences.

    This is not something we normally use, instead rf.AltitudeHydrostatic is used. But
    this is class is useful if we want to compare against old py-retrieve results
    """

    def __init__(
        self, ray_info: MusesRayInfo, pressure: rf.Pressure, latitude: float
    ) -> None:
        # Initialize director
        super().__init__()

        self.ray_info = ray_info
        self._pressure = pressure
        self._latitude = latitude

        self.altitude_grid: None | np.ndarray = None

    def cache_altitude(self) -> None:
        nlev = self._pressure.number_level

        # Recompute if value is undefined or if number of layers in pressure grid has changed
        if self.altitude_grid is not None and self.altitude_grid.shape[0] == nlev:
            return

        self.altitude_grid = self.ray_info.altitude_grid()

    def altitude(
        self, pressure_value: rf.DoubleWithUnit
    ) -> rf.AutoDerivativeWithUnitDouble:
        self.cache_altitude()

        pgrid_ad = self._pressure.pressure_grid().convert("Pa").value
        pgrid_val: np.ndarray = pgrid_ad.value

        alt_value = np.interp(
            cast(float, pressure_value.convert("Pa").value.value),
            pgrid_val,
            cast(np.ndarray, self.altitude_grid),
        )

        return rf.AutoDerivativeWithUnitDouble(
            rf.AutoDerivativeDouble(alt_value, np.zeros(pgrid_ad.number_variable)), "m"
        )

    def gravity(self, pressure_value: rf.DoubleWithUnit) -> rf.AutoDerivativeDouble:
        "Gravity as implemented by vlidort_cli"

        # constants
        a1 = 0.0026373
        a2 = 0.0000059
        a3 = 3.085462e-4
        a4 = 2.27e-7
        a5 = 7.254e-11
        a6 = 1.0e-13
        a7 = 1.517e-17
        a8 = 6.0e-20

        cos2p = np.cos(np.radians(2 * self._latitude))

        # sea level acceleration due to gravity

        g0 = 980.616 * (1.0 - a1 * cos2p + a2 * cos2p * cos2p)

        # acceleration due to gravity
        alt = self.altitude(pressure_value).convert("m").value.value

        grav = (
            g0
            - (a3 + a4 * cos2p) * alt
            + (a5 + a6 * cos2p) * alt**2
            + (a7 + a8 * cos2p) * alt**3
        )

        # Convert from cm/s^2 to m/s^2
        grav *= 1e-2

        return rf.AutoDerivativeWithUnitDouble(rf.AutoDerivativeDouble(grav), "m/s^2")

    def clone(self) -> Self:
        return MusesAltitude(self.ray_info, self._pressure, self._latitude)

    def desc(self) -> str:
        return "MusesAltitude"


__all__ = [
    "MusesAltitude",
]
