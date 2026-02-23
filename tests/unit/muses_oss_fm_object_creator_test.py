from refractor.muses import (
    InstrumentIdentifier,
    CrisFmObjectCreator,
    AirsFmObjectCreator,
)
from refractor.muses_py_fm import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import numpy.testing as npt


@require_muses_py_fm
def test_muses_cris_forward_model_oss(joint_tropomi_step_12_no_run_dir):
    rs, rstep, _ = joint_tropomi_step_12_no_run_dir
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
    )
    obs_cris.spectral_window.include_bad_sample = True
    ocreator = CrisFmObjectCreator(rs.current_state, rs.retrieval_config, obs_cris)
    fm = ocreator.forward_model
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesCrisForwardModel(rs.current_state, obs_cris, rs.retrieval_config)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian
    npt.assert_allclose(rad, radcmp)
    npt.assert_allclose(jac, jaccmp)


@require_muses_py_fm
def test_muses_airs_forward_model_oss(joint_omi_step_8_no_run_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    obs_airs.spectral_window.include_bad_sample = True
    ocreator = AirsFmObjectCreator(rs.current_state, rs.retrieval_config, obs_airs)
    fm = ocreator.forward_model
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesAirsForwardModel(rs.current_state, obs_airs, rs.retrieval_config)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian
    npt.assert_allclose(rad, radcmp)
    npt.assert_allclose(jac, jaccmp)
