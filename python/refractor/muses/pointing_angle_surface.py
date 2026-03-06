from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .misc import AttrDictAdapter
import typing
import math
import numpy as np

if typing.TYPE_CHECKING:
    from refractor.muses_py_fm import RefractorUip
    from .identifier import InstrumentIdentifier
    from .muses_altitude_pge import MusesAltitudePge

def pointing_angle_surface(sat_radius: rf.DoubleWithUnit,
                           pointing_angle: rf.DoubleWithUnit,
                           alt: MusesAltitudePge):
    from refractor.muses_py import ref_index
    
    # These parameters are needed for the atmospheric equation of state
    pressure = alt.pressure
    lnp = np.log(pressure)
    temperature = alt.tatm
    h2o = alt.h2o


    radius = alt.radius
    nlayers = pressure.shape[0]-1

    ds_fix = 500.0

    # spherical snells law with n = 1

    sin_theta_u = sat_radius.convert("m").value * math.sin(pointing_angle.convert("rad").value) / radius[nlayers]

    snells_constant = radius[nlayers] * sin_theta_u

    cos_theta_u = math.sqrt(1.0 - sin_theta_u**2)
    
    for jj in reversed(range(0, nlayers)): # go from top to bottom
        hp = -(radius[jj+1] - radius[jj]) / np.log(pressure[jj+1] / pressure[jj])
        r_u = radius[jj+1]
        flag = 0
        while flag == 0: # sub layer loop
            dr = ds_fix * cos_theta_u
            # This while loop only exit if the following condition is true.
            if (r_u - dr) < radius[jj]:
                dr = r_u - radius[jj]
                flag = 1
            r_l = r_u - dr
            p_l = pressure[jj] * math.exp(-(r_l - radius[jj]) / hp)
            t_l = temperature[jj] + (r_l - radius[jj]) * (temperature[jj+1] - temperature[jj]) / (radius[jj+1] - radius[jj])
            h2o_l = h2o[jj] + (np.log(p_l) - lnp[jj]) * (h2o[jj+1] - h2o[jj]) / np.log(pressure[jj+1] / pressure[jj])
            n_l = ref_index(t_l, p_l * 100., h2o_l)

            sin_theta_u = snells_constant / r_l / n_l
            cos_theta_u = math.sqrt(1 - sin_theta_u**2)
            r_u = r_l

    return rf.DoubleWithUnit(math.asin(sin_theta_u), "rad")

__all__ = ["pointing_angle_surface"]        
