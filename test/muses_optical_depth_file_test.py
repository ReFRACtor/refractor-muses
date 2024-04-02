import numpy as np
import numpy.testing as npt
from refractor.muses import (MusesOpticalDepthFile, AbsorberVmrToUip)
from test_support import *
import refractor.framework as rf

@require_muses_py
def test_muses_optical_depth_file(tropomi_uip_step_2, tropomi_obs_step_2,
                                  clean_up_replacement_function):
    try:
        from refractor.tropomi import TropomiFmObjectCreator
    except ImportError:
        raise pytest.skip("test requires tropomi to be available")
    
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_2, tropomi_obs_step_2)
    # Don't look at temperature jacobian right now, it doesn't actually
    # work correctly and has been removed from the production strategy tables.
    # Our older test data has this in, but just remove it
    obj_creator.rf_uip.uip_tropomi["jacobians"] = ["O3",]
    mod  = MusesOpticalDepthFile(obj_creator.rf_uip, "TROPOMI",
                                 obj_creator.pressure,
                                 obj_creator.temperature, obj_creator.altitude,
                                 obj_creator.absorber_vmr,
                                 obj_creator.num_channels)
    # TODO Note this is currently printing the wrong thing, this
    # is "Absorber" rather than "MusesOpticalDepthFile". Not super urgent,
    # but should fix this.
    print(mod)

    sv = rf.StateVector()
    sv.add_observer(obj_creator.absorber_vmr[0])
    sv_val = []
    sv_val = np.log(obj_creator.rf_uip.atmosphere_column("O3"))
    sv.update_state(sv_val)
    # Make sure update to sv gets reflected in UIP
    obj_creator.rf_uip.refractor_cache["atouip_O3"] =  \
        AbsorberVmrToUip(obj_creator.rf_uip, obj_creator.pressure,
                         obj_creator.absorber_vmr[0], "O3")
    
    # Pick a wavenumber in the forward model, but not at the edge. Pretty
    # arbitrary, we are just looking for "a typical wave number".
    spec_index = 0
    wn = obj_creator.underlying_forward_model.spectral_domain(spec_index).data[10]
    print(mod.optical_depth_each_layer(wn, spec_index))
    svinitial = np.copy(sv.state)
    odinitial = mod.optical_depth_each_layer(wn, spec_index).value
    jac = mod.optical_depth_each_layer(wn, spec_index).jacobian
    fdjac = np.zeros((odinitial.shape[0], 1, svinitial.shape[0]))
    delta = 0.0001
    for i in range(svinitial.shape[0]):
        svtemp = np.copy(svinitial)
        svtemp[i] += delta
        sv.update_state(svtemp)
        od = mod.optical_depth_each_layer(wn, spec_index).value
        fdjac[:,:, i] = (od - odinitial) / delta
    npt.assert_allclose(jac, fdjac, atol=4e-6)

    
    
