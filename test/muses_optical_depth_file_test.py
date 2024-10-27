import numpy as np
import numpy.testing as npt
from refractor.muses import (MusesOpticalDepthFile, AbsorberVmrToUip, MusesRayInfo,
                             RetrievalConfiguration, CurrentStateUip, MeasurementIdFile)
from refractor.tropomi import TropomiFmObjectCreator
from test_support import *
import refractor.framework as rf

@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_2(tropomi_uip_step_2, tropomi_obs_step_2, osp_dir):
    '''Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/tropomi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'TROPOMI' : ['BAND3']}
    mid = MeasurementIdFile(f"{test_base_path}/tropomi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    return TropomiFmObjectCreator(CurrentStateUip(tropomi_uip_step_2), mid,
                                  tropomi_obs_step_2, rf_uip_func=lambda **kwargs: tropomi_uip_step_2)


@require_muses_py
def test_muses_optical_depth_file(tropomi_fm_object_creator_step_2,
                                  tropomi_uip_step_2):
    obj_creator = tropomi_fm_object_creator_step_2
    # Don't look at temperature jacobian right now, it doesn't actually
    # work correctly and has been removed from the production strategy tables.
    # Our older test data has this in, but just remove it
    tropomi_uip_step_2.uip_tropomi["jacobians"] = ["O3",]
    mod  = MusesOpticalDepthFile(obj_creator.ray_info,
                                 obj_creator.pressure,
                                 obj_creator.temperature, obj_creator.altitude,
                                 obj_creator.absorber_vmr,
                                 obj_creator.num_channels,
                                 "./Step01_O3-Band3/vlidort/input")
    print(mod)

    sv = rf.StateVector()
    obj_creator.fm_sv.remove_observer(obj_creator.absorber_vmr[0])
    sv.add_observer(obj_creator.absorber_vmr[0])
    sv_val = []
    sv_val = np.log(tropomi_uip_step_2.atmosphere_column("O3"))
    sv.update_state(sv_val)
    # Make sure update to sv gets reflected in UIP
    tropomi_uip_step_2.refractor_cache["atouip_O3"] =  \
        AbsorberVmrToUip(tropomi_uip_step_2, obj_creator.pressure,
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

    
    
