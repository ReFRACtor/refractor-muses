from refractor.muses import (
    InstrumentIdentifier,
    StateElementIdentifier,
    CrisFmObjectCreator,
    AirsFmObjectCreator,
)
from refractor.muses_py_fm import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import numpy as np
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
    # For the purpose of testing, add some extra jacobians in so we can check
    # that part of the code.
    rs.current_state.testing_add_retrieval_state_element_id(
        StateElementIdentifier("TATM")
    )
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
def test_muses_cris_forward_model_pan_oss(joint_tropomi_step_4_no_run_dir):
    # Step 4 in the pan step. There is special handling for negative PAN VMR,
    # so we want to test this out specifically
    rs, rstep, _ = joint_tropomi_step_4_no_run_dir
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
    )
    # One of the steps with negative values we happened to encounter in a test run.
    # Not overly important the actual values, just that some of them are negative.
    parm_with_neg = np.array([1.96855068e-11, -1.28673082e-11,  5.93128416e-12, -2.55182263e-12,
                     -4.08883901e-11, -5.01950030e-11, -4.33600751e-11, -5.53636433e-11,
                     -1.01035998e-10, -1.35665252e-10,  5.19130457e-11,  1.59050278e-11,
                     7.56316404e-14,  7.66969517e-14,  8.82406249e-14])
    # Default forward model is MusesCrisForwardModelOss
    cfunc = rs.strategy_executor.create_cost_function()
    t1 = cfunc.fm_sv.state
    cfunc.parameters = parm_with_neg
    print(cfunc.fm_sv)
    fm = cfunc.fm_list[0]
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesCrisForwardModel(rs.current_state, obs_cris, rs.retrieval_config)
    fmcmp.update_uip(parm_with_neg)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian
    npt.assert_allclose(rad, radcmp)
    npt.assert_allclose(jac, jaccmp, 2e-7)
    

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
