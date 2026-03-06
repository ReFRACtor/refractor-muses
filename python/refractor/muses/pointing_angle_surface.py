from __future__ import annotations
import refractor.framework as rf  # type: ignore
import typing
import math
import numpy as np

if typing.TYPE_CHECKING:
    from .muses_altitude_pge import MusesAltitudePge


def ref_index(
    i_temperature: np.ndarray, i_pressure: np.ndarray, i_H2O_vmr: np.ndarray
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


def pointing_angle_surface(
    sat_radius: rf.DoubleWithUnit,
    pointing_angle: rf.DoubleWithUnit,
    alt: MusesAltitudePge,
) -> rf.DoubleWithUnit:
    # These parameters are needed for the atmospheric equation of state
    pressure = alt.pressure
    lnp = np.log(pressure)
    temperature = alt.tatm
    h2o = alt.h2o

    radius = alt.radius
    nlayers = pressure.shape[0] - 1

    ds_fix = 500.0

    # spherical snells law with n = 1

    sin_theta_u = (
        sat_radius.convert("m").value
        * math.sin(pointing_angle.convert("rad").value)
        / radius[nlayers]
    )

    snells_constant = radius[nlayers] * sin_theta_u

    cos_theta_u = math.sqrt(1.0 - sin_theta_u**2)

    for jj in reversed(range(0, nlayers)):  # go from top to bottom
        hp = -(radius[jj + 1] - radius[jj]) / np.log(pressure[jj + 1] / pressure[jj])
        r_u = radius[jj + 1]
        flag = 0
        while flag == 0:  # sub layer loop
            dr = ds_fix * cos_theta_u
            # This while loop only exit if the following condition is true.
            if (r_u - dr) < radius[jj]:
                dr = r_u - radius[jj]
                flag = 1
            r_l = r_u - dr
            p_l = pressure[jj] * math.exp(-(r_l - radius[jj]) / hp)
            t_l = temperature[jj] + (r_l - radius[jj]) * (
                temperature[jj + 1] - temperature[jj]
            ) / (radius[jj + 1] - radius[jj])
            h2o_l = h2o[jj] + (np.log(p_l) - lnp[jj]) * (
                h2o[jj + 1] - h2o[jj]
            ) / np.log(pressure[jj + 1] / pressure[jj])
            n_l = ref_index(t_l, p_l * 100.0, h2o_l)

            sin_theta_u = snells_constant / r_l / n_l
            cos_theta_u = math.sqrt(1 - sin_theta_u**2)
            r_u = r_l

    return rf.DoubleWithUnit(math.asin(sin_theta_u), "rad")


__all__ = ["pointing_angle_surface"]
