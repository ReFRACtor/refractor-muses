from test_support import *
from refractor.muses import (RefractorUip,
                             MusesCrisForwardModel, MusesCrisObservation, 
                             MusesAirsForwardModel, MusesAirsObservation,
                             MusesTropomiForwardModel, 
                             MusesOmiForwardModel, 
                             )
import refractor.framework as rf
import copy
import pickle

@require_muses_py
def test_muses_cris_forward_model(joint_tropomi_uip_step_12, joint_tropomi_obs_step_12):
    rf_uip = joint_tropomi_uip_step_12
    obs, obs_tropomi = joint_tropomi_obs_step_12
    obs.spectral_window.include_bad_sample = True
    fm = MusesCrisForwardModel(lambda **kwargs : rf_uip, obs)
    print(pickle.loads(pickle.dumps(obs)))
    # Pickle doesn't work with lambda functions out of the box. We can
    # import dill, but normally our rf_uip_func is actually
    # CostFunctionCreator._rf_uip_func_wrap which can be filtered. So just
    # skip this for now, we can always either handle lambda separately or
    # use the dill package instead.
    #print(pickle.loads(pickle.dumps(fm)))
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 216
    assert jac.shape[0] == 216
    assert jac.shape[1] == 285
    if False:
        print(rad)
        print(jac)
        print(rad.shape)
        print(jac.shape)
    s = obs.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 216
    assert uncer.shape[0] == 216
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)

@require_muses_py
def test_muses_tropomi_forward_model(joint_tropomi_uip_step_12, joint_tropomi_obs_step_12,
                                     vlidort_cli):
    rf_uip = joint_tropomi_uip_step_12
    obs_cris, obs = joint_tropomi_obs_step_12
    obs_cris.spectral_window.include_bad_sample = True
    fm = MusesTropomiForwardModel(lambda **kwargs : rf_uip,
                                  obs, vlidort_cli=vlidort_cli)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 53
    assert jac.shape[0] == 53
    assert jac.shape[1] == 285
    if False:
        print(rad)
        print(jac)
        print(rad.shape)
        print(jac.shape)
    s = obs.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 53
    assert uncer.shape[0] == 53
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)

@require_muses_py
def test_muses_airs_forward_model(joint_omi_uip_step_8, joint_omi_obs_step_8):
    rf_uip = joint_omi_uip_step_8
    obs, obs_omi = joint_omi_obs_step_8
    obs.spectral_window.include_bad_sample = True
    fm = MusesAirsForwardModel(lambda **kwargs : rf_uip, obs)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 150
    assert jac.shape[0] == 150
    assert jac.shape[1] == 168
    if False:
        print(rad)
        print(jac)
        print(rad.shape)
        print(jac.shape)
    s = obs.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 150
    assert uncer.shape[0] == 150
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)

@require_muses_py
def test_muses_omi_forward_model(joint_omi_uip_step_8, joint_omi_obs_step_8, vlidort_cli):
    rf_uip = joint_omi_uip_step_8
    obs_airs, obs =joint_omi_obs_step_8
    obs.spectral_window.include_bad_sample = True
    fm = MusesOmiForwardModel(lambda **kwargs : rf_uip, obs,
                              vlidort_cli=vlidort_cli)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 219
    assert jac.shape[0] == 219
    assert jac.shape[1] == 168
    if False:
        print(rad)
        print(jac)
        print(rad.shape)
        print(jac.shape)
    s = obs.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 219
    assert uncer.shape[0] == 219
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)
        
    
    
