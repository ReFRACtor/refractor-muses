from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .misc import AttrDictAdapter
import typing
import math
import numpy as np

if typing.TYPE_CHECKING:
    from refractor.muses_py_fm import RefractorUip
    from .identifier import InstrumentIdentifier

def raylayer_nadir(i_uip, i_atmparams):
    from refractor.muses_py import idl_tag_names, ref_index
    
    # These parameters are needed for the atmospheric equation of state
    pressure = i_atmparams.pressure
    lnp = np.log(pressure)
    temperature = i_atmparams.tatm
    h2o = i_atmparams.h2o


    # AT_LINE 28 ELANOR/raylayer_nadir.pro
    radius = i_atmparams.radius
    nlayers = i_atmparams.nlayers


    # AT_LINE 42 ELANOR/raylayer_nadir.pro
    ds_fix = 500.0

    if 'SUB_LAYER_DIST' in idl_tag_names(i_uip):
        ds_fix = i_uip.SUB_LAYER_DIST # in meteres.


    # spherical snells law with n = 1
    # took out call to earth_radius() b/c variable was not used

    # PYTHON_NOTE: There is only one obs_table so we cannot use the index.
    radiusSat = i_uip.obs_table['sat_radius']  # There is only one obs_table so we cannot use the index.

    sin_theta_u = radiusSat * math.sin(i_uip.obs_table['pointing_angle']) / radius[nlayers]

    snells_constant = radius[nlayers] * sin_theta_u

    cos_theta_u = math.sqrt(1.0 - sin_theta_u**2)
    
    x_u = radius[nlayers] * cos_theta_u
    
    for jj in reversed(range(0, nlayers)): # go from top to bottom
        n_u = ref_index(temperature[jj+1], pressure[jj+1] * 100., h2o[jj+1])

        hp = -(radius[jj+1] - radius[jj]) / np.log(pressure[jj+1] / pressure[jj])
        
        # AT_LINE 137 ELANOR/raylayer_nadir.pro

        r_u = radius[jj+1]

        flag = 0

        # AT_LINE 148 ELANOR/raylayer_nadir.pro
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

            sin_theta_l = snells_constant / r_l / n_l
            cos_theta_l = math.sqrt(1 - sin_theta_l**2)
            x_l = r_l * cos_theta_l

            
            cos_theta_u = cos_theta_l
            sin_theta_u = sin_theta_l
            r_u = r_l
            n_u = n_l

        # end while (flag == 0): # sub layer loop
        
    o_results = {
        'ray_angle_surface': math.asin(sin_theta_u),         
    }                   

    return o_results
    

def pointing_angle_surface(rf_uip: RefractorUip,
                     instrument_name: InstrumentIdentifier,
                     pointing_angle: rf.DoubleWithUnit):
    import refractor.muses_py as mpy
    uall = rf_uip.uip_all(instrument_name)
    uall["obs_table"]["pointing_angle"] = pointing_angle.convert("rad").value
    t = raylayer_nadir(
        AttrDictAdapter(uall), AttrDictAdapter(mpy.atmosphere_level(uall))
    )
    
    return rf.DoubleWithUnit(t["ray_angle_surface"], "rad")

__all__ = ["pointing_angle_surface"]        
