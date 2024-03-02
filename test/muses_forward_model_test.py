from test_support import *
from refractor.muses import (RefractorUip, StateVectorPlaceHolder,
                             MusesCrisForwardModel, MusesCrisObservation, 
                             MusesAirsForwardModel, MusesAirsObservation,
                             MusesTropomiForwardModel, MusesTropomiObservation,
                             MusesOmiForwardModel, MusesOmiObservation,
                             StateVectorHandleSet, MusesStateVectorObserverHandle
                             )
import refractor.framework as rf
import copy
import pickle

@require_muses_py
def test_muses_cris_forward_model(joint_tropomi_uip_step_12):
    rf_uip = joint_tropomi_uip_step_12
    obs_rad_fake = np.zeros((len(rf_uip.instrument_list),))
    meas_err_fake = np.ones(obs_rad_fake.shape)
    obs = MusesCrisObservation(rf_uip, obs_rad = obs_rad_fake, meas_err=meas_err_fake)
    fm = MusesCrisForwardModel(rf_uip, obs)
    print(pickle.loads(pickle.dumps(obs)))
    print(pickle.loads(pickle.dumps(fm)))
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
    s = obs.radiance(0)
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
def test_muses_tropomi_forward_model(joint_tropomi_uip_step_12, vlidort_cli):
    rf_uip = joint_tropomi_uip_step_12
    fm = MusesTropomiForwardModel(rf_uip, vlidort_cli=vlidort_cli)
    obs = MusesTropomiObservation(fm)
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
    s = obs.radiance(0)
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 53
    assert uncer.shape[0] == 53
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)
    # Check that cache properly gets invalidated with state vector changes
    state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
    state_vector_handle_set.add_handle(MusesStateVectorObserverHandle(fm),
                                       priority_order=1000)
    sv = state_vector_handle_set.create_state_vector(rf_uip)
    assert fm.cache_valid_flag
    sv.update_state(rf_uip.current_state_x)
    assert not fm.cache_valid_flag
        
@require_muses_py
def test_muses_airs_forward_model(joint_omi_uip_step_8):
    rf_uip = joint_omi_uip_step_8
    obs_rad_fake = np.zeros((len(rf_uip.instrument_list),))
    meas_err_fake = np.ones(obs_rad_fake.shape)
    obs = MusesAirsObservation(rf_uip, obs_rad = obs_rad_fake,
                               meas_err=meas_err_fake)
    fm = MusesAirsForwardModel(rf_uip,obs)
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
    s = obs.radiance(0)
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
def test_muses_omi_forward_model(joint_omi_uip_step_8, vlidort_cli):
    rf_uip = joint_omi_uip_step_8
    fm = MusesOmiForwardModel(rf_uip, vlidort_cli=vlidort_cli,
                              include_bad_sample=True)
    obs = MusesOmiObservation(fm, include_bad_sample=True)
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
    s = obs.radiance(0)
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 219
    assert uncer.shape[0] == 219
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)
    # Check that cache properly gets invalidated with state vector changes
    state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
    state_vector_handle_set.add_handle(MusesStateVectorObserverHandle(fm),
                                       priority_order=1000)
    sv = state_vector_handle_set.create_state_vector(rf_uip)
    assert fm.cache_valid_flag
    sv.update_state(rf_uip.current_state_x)
    assert not fm.cache_valid_flag
        
@require_muses_py
def test_state_vector_placeholder():
    sv = rf.StateVector()
    sv1 = StateVectorPlaceHolder(0,3,"O3")
    sv2 = StateVectorPlaceHolder(3,5,"Bob")
    sv.add_observer(sv1)
    sv.add_observer(sv2)
    sv.observer_claimed_size = 8
    sv.update_state([1,2,3,4,5,6,7,8])
    npt.assert_allclose(sv1.coeff, [1,2,3])
    npt.assert_allclose(sv2.coeff, [4,5,6,7,8])
    if False:
        print(sv)
        print(sv1.coeff)
        print(sv2.coeff)
    
    
