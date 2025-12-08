from refractor.old_py_retrieve_wrapper import MusesStrategyOldStrategyTable
from fixtures.require_check import require_muses_py
import os
import pytest


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_muses_strategy_old(joint_omi_step_8):
    rs, rsetp, kwargs = joint_omi_step_8
    # Old strategy table code assumes we are in the run dir. Our new
    # refractor version doesn't have that limit, but we are testing the
    # old one here.
    os.chdir(rs.run_dir)
    stable = MusesStrategyOldStrategyTable(
        rs.run_dir / "Table.asc", rs.osp_dir, rs.spectral_window_handle_set
    )
    stable.notify_update_target(rs.measurement_id)
    print(stable.filter_list_dict)
    print(stable.retrieval_elements)
    print(stable.error_analysis_interferents)
    print(stable.instrument_name)
    # In general, we need to use the right current_state in next_step below. However, for
    # this particular unit test we know that only the cstate.brightness_temperature_data
    # affects the MusesStrategyOldStrategyTable, so we get away with only using one
    # cstate since brightness_temperature_data doesn't change after the step it is
    # calculated.
    cstate = rs.current_state
    stable.restart()
    while not stable.is_done():
        print(stable.current_strategy_step())
        stable.next_step(cstate)
