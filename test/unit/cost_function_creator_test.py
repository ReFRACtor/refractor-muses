from refractor.muses import CostFunctionCreator, InstrumentIdentifier


def test_fm_obs_creator_tropomi(joint_tropomi_step_12, osp_dir):
    rs, rstep, _ = joint_tropomi_step_12
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs = [
        rs.observation_handle_set.observation(
            InstrumentIdentifier("CRIS"),
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
            None,
            osp_dir=osp_dir,
        ),
        rs.observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict[
                InstrumentIdentifier("TROPOMI")
            ],
            None,
            osp_dir=osp_dir,
        ),
    ]
    creator = CostFunctionCreator()

    # We test using this else where. Here, just make sure we can
    # call the creation function.
    cf = creator.cost_function_from_uip(rf_uip, obs, None)
    print(cf)
    breakpoint()


def test_fm_obs_creator_omi(joint_omi_step_8, osp_dir):
    rs, rstep, _ = joint_omi_step_8
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs = [
        rs.observation_handle_set.observation(
            InstrumentIdentifier("AIRS"),
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
            None,
            osp_dir=osp_dir,
        ),
        rs.observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
            None,
            osp_dir=osp_dir,
        ),
    ]
    creator = CostFunctionCreator()

    # We test using this else where. Here, just make sure we can
    # call the creation function.
    cf = creator.cost_function_from_uip(rf_uip, obs, None)
    print(cf)
