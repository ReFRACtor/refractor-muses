from __future__ import annotations
import numpy as np
import math


class MusesAltitudePge:
    """We already have height objects in our ReFRACtor forward model, but muses-py
    had it own calculation of this. So we can match old data, we duplicate this
    calculation.

    It is possible this can go away in the future.

    Note this is similar but distinct from the MusesAltitude found in old_py_retrieve_wrapper.
    This isn't actually a rf.Altitude object, although it wouldn't take much to make
    this into one. Instead, this calculates a few altitude related things.

    This was the function mpy.compute_altitude_pge, and uses slightly different values
    based on the flag tes_pge flag.
    """

    def __init__(
        self,
        pressure: np.ndarray,
        tatm: np.ndarray,
        h2o: np.ndarray,
        surface_altitude: float,
        latitude: float,
        tes_pge: bool = False,
        h2o_type: int = 1,
    ) -> None:
        # VMR wrt dry air
        # pressure in hPa
        # surfaceAltitude in meters
        # TATM kelvin
        # H2O in VMR
        # latitude in degrees

        Avo = 6.0225e23
        kb = 1.380622e-23

        self.air_density = pressure * 1e-4 / (kb * tatm)

        # h2o_type = 0    h2o is fractional mixing ratio (nh2o/airDensity) relative to total air
        # h2o_type = 1    h2o is fractional mixing ratio relative to dry air
        # h2o_type = 2    h2o is in g/kg with respect to dry air
        # Use either or.. makes a negligble difference (i.e., 1 cm difference)

        if h2o_type == 0:
            chi = h2o
            self.air_density_dry = (
                self.air_density - self.air_density * h2o
            )  # dry air number density
            chid = (
                chi * self.air_density / self.air_density_dry
            )  # mixing ratio of h2o relative to dry air
        elif h2o_type == 1:
            chid = h2o
            self.air_density_dry = self.air_density / (1 + chid)
            chi = chid * self.air_density_dry / self.air_density
        elif h2o_type == 2:
            chid = (
                h2o / 1000.0 / 0.6220
            )  # .6220 is the ratio of dry air mass to h2o mass.
            self.air_density_dry = self.air_density / (1 + chid)
            chi = chid * self.air_density_dry / self.air_density
        else:
            raise RuntimeError("h2o_type needs to be between 0 and 2")

        mh2o = 0.018015  # molar mass of h2o
        mdry = 0.0289654  # molar mass of dry air
        mr = mh2o / mdry  # mass ratio

        # Set up the altitude grid.

        self.altitude = np.ndarray(shape=pressure.shape, dtype=np.float64)
        self.altitude[0] = surface_altitude

        # compression factor, deviation from ideal gas
        # see PGE L2_C_Atmosphere.cpp
        a0 = 1.58123e-6
        a1 = -2.9331e-8
        a2 = 1.1043e-10
        b0 = 5.707e-6
        b1 = -2.051e-8
        c0 = 1.9898e-4
        c1 = -2.376e-6
        d = 1.83e-11
        e = -0.0765e-8
        dt = tatm - 273.15
        chim = 0.6223 * chid
        ratio = pressure * 100 / tatm

        # We need to do some rounding to match IDL.
        ratio = np.round(ratio, 9)
        comp_factor = (
            1.0
            - ratio
            * (
                a0
                + a1 * dt
                + a2 * np.square(dt)
                + (b0 + b1 * dt) * chim
                + (c0 + c1 * dt) * np.square(chim)
            )
            + (d + e * np.square(chim)) * (np.square(ratio))
        )

        # We need to do some rounding to match IDL.
        comp_factor = np.round(comp_factor, 8)

        self.air_density = 1e6 * self.air_density / comp_factor
        self.air_density_dry = 1e6 * self.air_density_dry / comp_factor
        lnp = np.log(pressure)
        rho = mdry * self.air_density_dry * (1 + mr * chid) / Avo

        # We need to do some rounding to match IDL.
        lnp = np.round(lnp, 7)
        rho = np.round(rho, 10)

        nump = len(pressure)
        self.layer_pressure = np.ndarray(shape=(nump - 1), dtype=np.float64)
        self.layer_temperature = np.ndarray(shape=(nump - 1), dtype=np.float64)

        # Go through each pressure/temperature level and calculate F/cm2 of the air
        # Note that we start the for loop with 1.

        for ii in range(1, nump):
            dlnp = lnp[ii] - lnp[ii - 1]
            gravity = self.gravity(latitude, self.altitude[ii - 1], tes_pge)
            self.altitude[ii] = (
                self.altitude[ii - 1]
                - 0.5
                * (
                    pressure[ii] / (rho[ii] * gravity)
                    + pressure[ii - 1] / (rho[ii - 1] * gravity)
                )
                * 100.0
                * dlnp
            )

            # Get layer Pressure ssk 2008
            self.layer_pressure[ii - 1] = (
                1.0
                / (
                    1.0
                    + math.log(pressure[ii] / pressure[ii - 1])
                    / math.log(rho[ii] / rho[ii - 1])
                )
                * (pressure[ii - 1] * rho[ii - 1] - pressure[ii] * rho[ii])
                / (rho[ii - 1] - rho[ii])
            )

            # Get layer temperature ssk 2008
            t2 = tatm[ii]
            t1 = tatm[ii - 1]
            if t1 == t2:
                self.layer_temperature[ii - 1] = t1
            else:
                self.layer_temperature[ii - 1] = (
                    1.0
                    / (1.0 + math.log(t2 / t1) / math.log(rho[ii] / rho[ii - 1]))
                    * (t1 * rho[ii - 1] - t2 * rho[ii])
                    / (rho[ii - 1] - rho[ii])
                )
        # end for ii in range(1,num):

        self.air_density = self.air_density / 1000000.0  # convert to molecules/cm3
        self.air_density_dry = (
            self.air_density_dry / 1000000.0
        )  # convert to molecules/cm3

    def gravity(self, latitude: float, altitude: float, tes_pge: bool = False) -> float:
        rad_earth = self.earth_radius(latitude, tes_pge)
        if not tes_pge:
            g0 = 9.80612 - 0.02586 * math.cos(2 * math.pi * 45 / 180.0)  # gravity
            res = g0 * np.square(rad_earth / (rad_earth + altitude))
        else:
            cosine = math.cos(2.0 * math.radians(latitude))
            gravity_at_surface = (
                980.612 - (2.5865 * cosine) + (0.0058 * cosine * cosine)
            ) / 100.0
            res = (
                gravity_at_surface
                * (rad_earth / (rad_earth + altitude))
                * (rad_earth / (rad_earth + altitude))
            )
        return res

    def earth_radius(self, latitude: float, tes_pge: bool = False) -> float:
        if tes_pge:
            # TES pge (note TES PGE in km).
            polar_radius = 6.356779e06
            equatorial_radius = 6.378160e06

            ratio = (
                polar_radius * polar_radius / (equatorial_radius * equatorial_radius)
            )
            sine = math.sin(math.radians(latitude))

            res = equatorial_radius * math.sqrt(
                (1.0 + (ratio * ratio - 1.0) * sine * sine)
                / (1.0 + (ratio - 1.0) * sine * sine)
            )
        else:
            res = 6.356779e06 + (6.37816e6 - 6.356779e06) * math.cos(
                math.radians(latitude)
            )
        return res


__all__ = ["MusesAltitudePge"]
