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

@require_muses_py
def test_muses_cris_forward_model(joint_tropomi_uip_step_10):
    rf_uip = joint_tropomi_uip_step_10
    fm = MusesCrisForwardModel(rf_uip)
    obs = MusesCrisObservation(rf_uip, obs_rad = None, meas_err=None)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 216
    assert jac.shape[0] == 216
    assert jac.shape[1] == 92
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
def test_muses_tropomi_forward_model(joint_tropomi_uip_step_10, vlidort_cli):
    rf_uip = joint_tropomi_uip_step_10
    fm = MusesTropomiForwardModel(rf_uip, vlidort_cli=vlidort_cli)
    obs = MusesTropomiObservation(fm)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 52
    assert jac.shape[0] == 52
    assert jac.shape[1] == 92
    if False:
        print(rad)
        print(jac)
        print(rad.shape)
        print(jac.shape)
    s = obs.radiance(0)
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 52
    assert uncer.shape[0] == 52
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
    sv.update_state([1,2,3])
    assert not fm.cache_valid_flag
        
@require_muses_py
def test_muses_airs_forward_model(joint_omi_uip_step_7):
    rf_uip = joint_omi_uip_step_7
    fm = MusesAirsForwardModel(rf_uip)
    obs = MusesAirsObservation(rf_uip, obs_rad = None, meas_err=None)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 150
    assert jac.shape[0] == 150
    assert jac.shape[1] == 62
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
def test_muses_omi_forward_model(joint_omi_uip_step_7, vlidort_cli):
    rf_uip = joint_omi_uip_step_7
    fm = MusesOmiForwardModel(rf_uip, vlidort_cli=vlidort_cli)
    obs = MusesOmiObservation(fm)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    # Basically just making sure we can run this, no easy way to check
    # if the results are correct or not.
    assert rad.shape[0] == 221
    assert jac.shape[0] == 221
    assert jac.shape[1] == 62
    if False:
        print(rad)
        print(jac)
        print(rad.shape)
        print(jac.shape)
    s = obs.radiance(0)
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 221
    assert uncer.shape[0] == 221
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
    sv.update_state([1,2,3])
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
    
    
