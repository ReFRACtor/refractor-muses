from test_support import *
from refractor.muses import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
    MusesTropomiForwardModel,
    MusesOmiForwardModel,
    MeasurementIdDict,
)
import pickle


def test_muses_cris_forward_model(joint_tropomi_step_12, osp_dir):
    rs, rstep, _ = joint_tropomi_step_12
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs_cris = rs.observation_handle_set.observation(
        "CRIS",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["CRIS"],
        None,
        osp_dir=osp_dir,
    )
    obs_tropomi = rs.observation_handle_set.observation(
        "TROPOMI",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["TROPOMI"],
        None,
        osp_dir=osp_dir,
        write_tropomi_radiance_pickle=True,
    )
    obs_cris.spectral_window.include_bad_sample = True
    mid = MeasurementIdDict({}, {})
    fm = MusesCrisForwardModel(rf_uip, obs_cris, mid)
    print(pickle.loads(pickle.dumps(obs_cris)))
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
    s = obs_cris.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 216
    assert uncer.shape[0] == 216
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)


def test_muses_tropomi_forward_model(joint_tropomi_step_12, vlidort_cli, osp_dir):
    rs, rstep, _ = joint_tropomi_step_12
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs_cris = rs.observation_handle_set.observation(
        "CRIS",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["CRIS"],
        None,
        osp_dir=osp_dir,
    )
    obs_tropomi = rs.observation_handle_set.observation(
        "TROPOMI",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["TROPOMI"],
        None,
        osp_dir=osp_dir,
        write_tropomi_radiance_pickle=True,
    )
    obs_tropomi.spectral_window.include_bad_sample = True
    mid = MeasurementIdDict({}, {})
    fm = MusesTropomiForwardModel(rf_uip, obs_tropomi, mid, vlidort_cli=vlidort_cli)
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
    s = obs_tropomi.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 53
    assert uncer.shape[0] == 53
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)


def test_muses_airs_forward_model(joint_omi_step_8, osp_dir):
    rs, rstep, _ = joint_omi_step_8
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs_airs = rs.observation_handle_set.observation(
        "AIRS",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["AIRS"],
        None,
        osp_dir=osp_dir,
    )
    obs_omi = rs.observation_handle_set.observation(
        "OMI",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["OMI"],
        None,
        osp_dir=osp_dir,
        write_omi_radiance_pickle=True,
    )
    obs_airs.spectral_window.include_bad_sample = True
    mid = MeasurementIdDict({}, {})
    fm = MusesAirsForwardModel(rf_uip, obs_airs, mid)
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
    s = obs_airs.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 150
    assert uncer.shape[0] == 150
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)


def test_muses_omi_forward_model(joint_omi_step_8, vlidort_cli, osp_dir):
    rs, rstep, _ = joint_omi_step_8
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs_airs = rs.observation_handle_set.observation(
        "AIRS",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["AIRS"],
        None,
        osp_dir=osp_dir,
    )
    obs_omi = rs.observation_handle_set.observation(
        "OMI",
        rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["OMI"],
        None,
        osp_dir=osp_dir,
        write_omi_radiance_pickle=True,
    )
    obs_omi.spectral_window.include_bad_sample = True
    mid = MeasurementIdDict({}, {})
    fm = MusesOmiForwardModel(rf_uip, obs_omi, mid, vlidort_cli=vlidort_cli)
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
    s = obs_omi.radiance_all()
    rad = s.spectral_range.data
    uncer = s.spectral_range.uncertainty
    assert rad.shape[0] == 219
    assert uncer.shape[0] == 219
    if False:
        print(rad)
        print(uncer)
        print(rad.shape)
        print(uncer.shape)
