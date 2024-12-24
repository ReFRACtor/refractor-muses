from test_support import *
from test_support.old_py_retrieve_test_support import *
import numpy as np
from refractor.old_py_retrieve_wrapper import (
    OmiRadiancePyRetrieve, OmiRadianceToUip)
import refractor.framework as rf
import pandas as pd

@old_py_retrieve_test
def test_omi_radiance(omi_uip_step_2):
    # The initial shift for everything is 0. Change to something so we can test that
    # this actually gets used.
    print(omi_uip_step_2.uip_omi['jacobians'])
    omi_uip_step_2.uip_omi['jacobians'] = ["OMINRADWAVUV1",
                                       "OMINRADWAVUV2",
                                       "OMIODWAVUV1",
                                       "OMIODWAVUV2",
                                       "OMIODWAVSLOPEUV1",
                                       "OMIODWAVSLOPEUV2"]
    mrad = OmiRadiancePyRetrieve(omi_uip_step_2)
    sv = rf.StateVector()
    sv.add_observer(mrad)
    omi_rad_to_uip = OmiRadianceToUip(omi_uip_step_2, mrad)
    x = [0.01, 0.02, 0.03, 0.04, 0.001, 0.002]
    sv.update_state(x)
    r = mrad.radiance_all()
    # Expect jacobian to be different, because we are fixing this in
    # the refractor version.
    if False:
        print(r.spectral_range.data_ad.jacobian)
    fdlist = [0.001, 0.001, 0.001, 0.001, 0.0001, 0.001]
    y0 = r.spectral_range.data
    print("I think this is wrong for index 0 and 1 (the normwav_jac part). We'll come back to this.")
    for i,fd in enumerate(fdlist):
        xdelta = x.copy()
        xdelta[i] += fd
        sv.update_state(xdelta)
        rdelta = mrad.radiance_all()
        yd = rdelta.spectral_range.data
        jfd = (yd - y0) / fd
        jcalc = r.spectral_range.data_ad.jacobian[:,i]
        reldiff = np.abs((jfd-jcalc)/jcalc)
        reldiff[jcalc == 0] = 0
        print(f"Find difference jacobian for index {i}")
        if False:
            print("   Jac finite difference: ", jfd)
            print("   Jac calculate: ", jcalc)
            print("   Jac relative difference: ", reldiff)
        print("   Summary abs difference")
        print(pd.DataFrame(np.abs(jfd-jcalc)).describe())
        print("   Summary relative difference")
        print(pd.DataFrame(reldiff).describe())
        
        

        
