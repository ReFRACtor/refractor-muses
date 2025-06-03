from refractor.old_py_retrieve_wrapper import MusesStrategyOldStrategyTable
import pytest


@pytest.mark.old_py_retrieve_test
def test_muses_strategy_old(joint_omi_current_state, osp_dir):
    cstate, cstate_record, strategy, rconfig, measurement_id = joint_omi_current_state
    stable = MusesStrategyOldStrategyTable(
        rconfig["run_dir"] / "Table.asc", osp_dir
    )
    stable.notify_update_target(measurement_id)
    print(stable.filter_list_dict)
    print(stable.retrieval_elements)
    print(stable.error_analysis_interferents)
    print(stable.instrument_name)
    # In general, we need to use the right current_state in next_step below. However, for
    # this particular unit test we know that only the cstate.brightness_temperature_data
    # affects the MusesStrategyOldStrategyTable, so we get away with only using one
    # cstate since brightness_temperature_data doesn't change after the step it is
    # calculated.
    cstate_record.play(cstate, strategy, rconfig, 8)
    stable.restart()
    while not stable.is_done():
        print(stable.current_strategy_step())
        stable.next_step(cstate)
