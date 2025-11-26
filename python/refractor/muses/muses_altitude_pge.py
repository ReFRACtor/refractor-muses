from __future__ import annotations
import numpy as np
import math
import sys
import scipy
from typing import Any


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
        self.pressure = pressure

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

    def cloud_factor(self, pcloud: float, scale_pressure: float) -> float:
        """This is compute_cloud_factor from muses_py"""
        z = self.altitude / 1000
        cloud_ext_level = np.array(
            [
                1
                * math.exp(
                    -(
                        ((math.log(pcloud / self.pressure[i])) / abs(scale_pressure))
                        ** 2
                    )
                )
                for i in range(self.pressure.shape[0])
            ]
        )

        cloud_od_layer = np.zeros(cloud_ext_level.shape[0])
        cloud_ext_layer = np.zeros(cloud_ext_level.shape[0])
        for jj in range(1, cloud_ext_level.shape[0]):
            c1 = cloud_ext_level[jj - 1]
            c2 = cloud_ext_level[jj]
            p1 = math.log(self.pressure[jj - 1])
            p2 = math.log(self.pressure[jj])
            p = math.log(self.layer_pressure[jj - 1])

            cloud_ext_layer[jj - 1] = c1 + (p - p1) / (p2 - p1) * (c2 - c1)
            cloud_od_layer[jj - 1] = cloud_ext_layer[jj - 1] * (z[jj] - z[jj - 1])
            if cloud_od_layer[jj - 1] < 1e-7:
                cloud_od_layer[jj - 1] = 0
        return np.sum(cloud_od_layer)

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

    def surface_pressure(self, surface_altitude: float) -> float:
        """Calculate surface pressure, the way mpy.supplier_surface_pressure does."""
        spress = np.exp(
            idl_interpol_1d(
                np.log(self.pressure), self.altitude, np.array([surface_altitude])
            )[0]
        )
        # TODO - do we really want to round this? This is what muses-py does
        return np.round(spress, 4)

    def column_integrate(
        self,
        vmr: np.ndarray,
        min_index: int = 0,
        max_index: int = 0,
        linear_flag: bool = False,
    ) -> dict[str, Any]:
        if max_index == 0:
            max_index = len(self.altitude) - 1

        if max_index > len(self.altitude) - 1:
            raise RuntimeError("max_index must be less than number altitude elements")

        columnAirTotal = 0
        columnTotal = 0
        columnLayer = np.zeros(shape=(len(self.altitude) - 1), dtype=np.float32)
        columnAirLayer = np.zeros(shape=(len(self.altitude) - 1), dtype=np.float32)
        vmrLayer = np.zeros(shape=(len(self.altitude) - 1), dtype=np.float32)

        for jj in range(min_index + 1, max_index + 1):
            x1 = self.air_density[jj - 1] * np.float64(vmr[jj - 1])
            x2 = self.air_density[jj] * np.float64(vmr[jj])
            dz = self.altitude[jj] - self.altitude[jj - 1]
            dz *= 100  # Convert to centimeters
            if x1 == x2:
                x1 = x2 * 1.0001

            # column for species
            columnLayer[jj - 1] = dz / np.log(np.abs(x1 / x2)) * (x1 - x2)

            # column for air
            x1d = self.air_density[jj - 1]
            x2d = self.air_density[jj]
            columnAirLayer[jj - 1] = dz / np.log(np.abs(x1d / x2d)) * (x1d - x2d)

            if linear_flag:
                HV = (vmr[jj] - vmr[jj - 1]) / dz
                HP = (
                    np.log(x2d, dtype=np.float64) - np.log(x1d, dtype=np.float64)
                ) / dz

                # sometimes HP will be very small, i.e. practically 0.0 and that trips the calculation below
                if HP == 0.0:
                    HP = sys.float_info.min

                # override log calculations
                columnLayer[jj - 1] = (x2 - x1) / HP - HV * (x2d - x1d) / HP / HP
            # end: if linear_flag:

            columnAirTotal = columnAirTotal + columnAirLayer[jj - 1]
            columnTotal = columnTotal + columnLayer[jj - 1]
            vmrLayer[jj - 1] = columnLayer[jj - 1] / columnAirLayer[jj - 1]
            if not np.isfinite(columnLayer[jj - 1]):
                raise RuntimeError("NaN in column")

        n = len(vmr)
        derivative = np.zeros(shape=(n), dtype=np.float64)  # derivative of dcolumn/dvmr
        level_to_layer = np.zeros(
            shape=(n, n - 1), dtype=np.float64
        )  # map from levels to layers

        for jj in range(1, n):
            x1 = self.air_density[jj - 1] * vmr[jj - 1]
            x2 = self.air_density[jj] * vmr[jj]
            dz = self.altitude[jj] - self.altitude[jj - 1]
            dz *= 100  # Convert to centimeters

            term1 = np.log(np.abs(x1 / x2))
            term2 = x1 - x2
            derivative[jj] = (
                derivative[jj]
                - dz / term1 * self.air_density[jj]
                + dz / term1 / term1 * term2 / vmr[jj]
            )
            derivative[jj - 1] = (
                derivative[jj - 1]
                + dz / term1 * self.air_density[jj - 1]
                - dz / term1 / term1 * term2 / vmr[jj - 1]
            )

            # since above is d(column gas)/dvmr, change to
            # d(layer # vmr) / d(levelvmr) by dividing by the air density for this layer.
            # air density for this layer: take above column measurements where vmr = 1.
            # The dz cancels
            x1d = self.air_density[jj - 1]
            x2d = self.air_density[jj]
            factor = np.log(np.abs(x1d / x2d)) / (x1d - x2d)

            level_to_layer[jj, jj - 1] = (
                level_to_layer[jj, jj - 1]
                + (
                    1 / term1 / term1 * term2 / vmr[jj]
                    - 1 / term1 * self.air_density[jj]
                )
                * factor
            )
            level_to_layer[jj - 1, jj - 1] = (
                level_to_layer[jj - 1, jj - 1]
                + (
                    1 / term1 * self.air_density[jj - 1]
                    - 1 / term1 / term1 * term2 / vmr[jj - 1]
                )
                * factor
            )

        derivativeLayer = np.matmul(derivative, level_to_layer)

        result = {
            "columnLayer": columnLayer,
            "column": columnTotal,
            "derivative": derivative,
            "level_to_layer": level_to_layer,
            "derivativeLayer": derivativeLayer,
            "columnAirLayer": columnAirLayer,
            "columnAir": columnAirTotal,
        }

        return result

    # This can almost just be a member function of MusesAltitudePge. But we may adjust
    # the layers, adding some if needed. So instead, we have this as a class method since
    # it is so closely related.
    @classmethod
    def column(
        cls,
        vmr_in: np.ndarray,
        pressure_in: np.ndarray,
        tatm_in: np.ndarray,
        h2o_in: np.ndarray,
        surface_altitude: float,
        latitude: float,
        min_pressure: float | None = None,
        max_pressure: float | None = None,
        linear: bool = False,
    ) -> dict[str, Any]:
        # We may add layers if needed, so copy input in case we change it.
        pressure = pressure_in.copy()
        vmr = vmr_in.copy()
        tatm = tatm_in.copy()
        h2o = h2o_in.copy()

        ind = np.where(h2o < 1e-20)[0]
        if len(ind) > 0:
            h2o[ind] = 1e-20
        ind = np.where(pressure < 1e-20)[0]
        if len(ind) > 0:
            pressure[ind] = 1e-20

        # if specify min and max pressures, find corresponding indices
        if min_pressure is None:
            min_pressure = np.amin(pressure)
        if max_pressure is None:
            max_pressure = np.amax(pressure)

        if min_pressure == 0:
            min_pressure = np.amin(pressure)
        if max_pressure > np.amax(pressure):
            max_pressure = np.max(pressure)

        ind_min = int(np.argmin(np.abs(min_pressure - pressure)))
        mindp = np.amin(np.abs(min_pressure - pressure))
        ind_max = int(np.argmin(np.abs(max_pressure - pressure)))
        maxdp = np.amin(np.abs(max_pressure - pressure))

        if maxdp > 0.1:
            # add in new pressure, interpolate to it
            # note need to make changes if layervmr=1
            pressure0 = np.copy(pressure)
            ind = np.where(pressure > max_pressure)[0]
            from_max_pres = pressure[np.amax(ind) + 1 :]
            pressure = pressure[ind]
            pressure = np.append(pressure, max_pressure)
            pressure = np.append(pressure, from_max_pres)
            tatm = idl_interpol_1d(tatm, np.log(pressure0), np.log(pressure))
            h2o = np.exp(
                idl_interpol_1d(np.log(h2o), np.log(pressure0), np.log(pressure))
            )

            if linear:
                vmr = idl_interpol_1d(vmr, np.log(pressure0), np.log(pressure))
            else:
                vmr = np.exp(
                    idl_interpol_1d(np.log(vmr), np.log(pressure0), np.log(pressure))
                )

        if mindp > 0.1:
            # add in new pressure, interpolate to it
            pressure0 = pressure.copy()  # Make a copy and leave original pressure alone because we will be modifying the left hand side.
            ind = np.where(pressure > min_pressure)[0]
            if (
                len(ind) == 0
            ):  # min_pressure could be e.g. 500, larger than 485, a possible surface pressure, so use surface value so as not to get an error.
                ind = np.array([0])
            from_max_pres = pressure[
                np.amax(ind) + 1 :
            ]  # Get a view of portion of pressure array this before it gets destroyed.

            pressure = pressure[ind]
            pressure = np.append(pressure, min_pressure)
            pressure = np.append(pressure, from_max_pres)

            tatm = idl_interpol_1d(tatm, np.log(pressure0), np.log(pressure))
            h2o = np.exp(
                idl_interpol_1d(np.log(h2o), np.log(pressure0), np.log(pressure))
            )

            if linear:
                vmr = idl_interpol_1d(vmr, np.log(pressure0), np.log(pressure))
            else:
                vmr = np.exp(
                    idl_interpol_1d(np.log(vmr), np.log(pressure0), np.log(pressure))
                )

        ind_max = int(np.argmin(np.abs(min_pressure - pressure)))
        ind_min = int(np.argmin(np.abs(max_pressure - pressure)))

        result = MusesAltitudePge(
            pressure, tatm, h2o, surface_altitude, latitude, tes_pge=True
        )

        result2 = result.column_integrate(
            vmr, min_index=ind_min, max_index=ind_max, linear_flag=linear
        )

        myresult = result.__dict__ | result2

        return myresult


def idl_interpol_1d(
    i_vector: np.ndarray, i_abscissaValues: np.ndarray, i_abscissaResult: np.ndarray
) -> np.ndarray:
    return scipy.interpolate.interp1d(
        i_abscissaValues, i_vector, fill_value="extrapolate"
    )(i_abscissaResult)


__all__ = ["MusesAltitudePge"]
