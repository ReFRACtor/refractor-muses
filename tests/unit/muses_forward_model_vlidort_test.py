from refractor.muses import (
    InstrumentIdentifier,
)
from refractor.omi import OmiFmObjectCreator
from refractor.tropomi import TropomiFmObjectCreator
from refractor.muses_py_fm import (
    MusesTropomiForwardModel,
    MusesOmiForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import numpy.testing as npt


@require_muses_py_fm
def test_muses_tropomi_forward_model_vlidort(joint_tropomi_step_12_no_run_dir):
    rs, rstep, _ = joint_tropomi_step_12_no_run_dir
    obs_tropomi = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
    )
    obs_tropomi.spectral_window.include_bad_sample = True
    ocreator = TropomiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.retrieval_config,
        obs_tropomi,
        use_vlidort=True,
        match_py_retrieve=True,
    )
    fm = ocreator.forward_model
    # Set up jacobians
    ocreator.fm_sv.update_state(ocreator.fm_sv.state, ocreator.fm_sv.state_covariance)

    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesTropomiForwardModel(
        rs.current_state,
        obs_tropomi,
        rs.retrieval_config,
    )
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian.copy()
    assert rad.shape == radcmp.shape
    assert jac.shape == jaccmp.shape
    npt.assert_allclose(rad, radcmp)
    # Note there are numerous problems with the py-retrieve jacobians. Don't compare,
    # because it is wrong.
    #npt.assert_allclose(jac, jaccmp)


@require_muses_py_fm
def test_muses_omi_forward_model_vlidort(joint_omi_step_8_no_run_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_omi = rs.observation_handle_set.observation(
        InstrumentIdentifier("OMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
        None,
    )
    obs_omi.spectral_window.include_bad_sample = True
    ocreator = OmiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.retrieval_config,
        obs_omi,
        use_vlidort=True,
        match_py_retrieve=True,
    )
    fm = ocreator.forward_model
    # Set up jacobians
    ocreator.fm_sv.update_state(ocreator.fm_sv.state, ocreator.fm_sv.state_covariance)

    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesOmiForwardModel(rs.current_state, obs_omi, rs.retrieval_config)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian.copy()
    assert rad.shape == radcmp.shape
    assert jac.shape == jaccmp.shape
    npt.assert_allclose(rad, radcmp)
    # Note there are numerous problems with the py-retrieve jacobians. Don't compare,
    # because it is wrong.
    #npt.assert_allclose(jac, jaccmp)
