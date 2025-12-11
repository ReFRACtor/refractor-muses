from refractor.muses import InstrumentIdentifier
from refractor.muses_py_fm import RefractorUip


def test_fm_obs_creator_tropomi(joint_tropomi_step_12, osp_dir):
    rs, rstep, _ = joint_tropomi_step_12
    obs = [
        rs.observation_handle_set.observation(
            InstrumentIdentifier("CRIS"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
            None,
        ),
        rs.observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[
                InstrumentIdentifier("TROPOMI")
            ],
            None,
        ),
    ]
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        obs,
        rs.current_state,
        rs.retrieval_config,
    )

    # We test using this else where. Here, just make sure we can
    # call the creation function.
    cf = rs.cost_function_creator.cost_function_from_uip(rf_uip, obs, None)
    print(cf)


def test_fm_obs_creator_omi(joint_omi_step_8, osp_dir):
    rs, rstep, _ = joint_omi_step_8
    obs = [
        rs.observation_handle_set.observation(
            InstrumentIdentifier("AIRS"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
            None,
        ),
        rs.observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
            None,
        ),
    ]
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        obs,
        rs.current_state,
        rs.retrieval_config,
    )

    # We test using this else where. Here, just make sure we can
    # call the creation function.
    cf = rs.cost_function_creator.cost_function_from_uip(rf_uip, obs, None)
    print(cf)
