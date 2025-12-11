from refractor.muses import (
    MeasurementIdDict,
    InstrumentIdentifier,
)
from refractor.muses_py_fm import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
    MusesTropomiForwardModel,
    MusesOmiForwardModel,
    RefractorUip,
)
import pickle
import tempfile


def test_muses_cris_forward_model(joint_tropomi_step_12_no_run_dir, osp_dir):
    rs, rstep, _ = joint_tropomi_step_12_no_run_dir
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
    )
    obs_cris.spectral_window.include_bad_sample = True
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        [obs_cris],
        rs.current_state,
        rs.retrieval_config,
    )
    mid = MeasurementIdDict({}, {}, osp_dir=osp_dir)
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


# Remove subprocess.run before isolated_dir
def test_muses_tropomi_forward_model(joint_tropomi_step_12_no_run_dir, osp_dir):
    rs, rstep, _ = joint_tropomi_step_12_no_run_dir
    obs_tropomi = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
    )
    obs_tropomi.spectral_window.include_bad_sample = True
    vlidort_temp_dir = tempfile.TemporaryDirectory()
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        [obs_tropomi],
        rs.current_state,
        rs.retrieval_config,
        vlidort_dir=vlidort_temp_dir.name,
    )
    mid = MeasurementIdDict({}, {}, osp_dir=osp_dir)
    fm = MusesTropomiForwardModel(
        rf_uip, obs_tropomi, mid, vlidort_temp_dir=vlidort_temp_dir
    )
    vlidort_temp_dir = None
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


def test_muses_airs_forward_model(joint_omi_step_8_no_run_dir, osp_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    obs_airs.spectral_window.include_bad_sample = True
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        [obs_airs],
        rs.current_state,
        rs.retrieval_config,
    )
    mid = MeasurementIdDict({}, {}, osp_dir=osp_dir)
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


# Remove subprocess.run before isolated_dir
def test_muses_omi_forward_model(joint_omi_step_8_no_run_dir, osp_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_omi = rs.observation_handle_set.observation(
        InstrumentIdentifier("OMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
        None,
    )
    obs_omi.spectral_window.include_bad_sample = True
    vlidort_temp_dir = tempfile.TemporaryDirectory()
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        [obs_omi],
        rs.current_state,
        rs.retrieval_config,
        vlidort_dir=vlidort_temp_dir.name,
    )
    mid = MeasurementIdDict({}, {}, osp_dir=osp_dir)
    fm = MusesOmiForwardModel(rf_uip, obs_omi, mid, vlidort_temp_dir=vlidort_temp_dir)
    vlidort_temp_dir = None
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
