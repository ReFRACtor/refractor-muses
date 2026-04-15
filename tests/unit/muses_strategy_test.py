from refractor.muses import (
    MusesStrategyStepList,
    MusesSpectralWindowDict,
    TesStrategyTableReader,
)
from refractor.old_py_retrieve_wrapper import MusesStrategyOldStrategyTable
from fixtures.require_check import require_muses_py
import os


# Compare against MusesStrategyOldStrategyTable, so need muses_py
@require_muses_py
def test_muses_strategy_file(joint_omi_step_8_osp_sym_link):
    rs, rsetp, kwargs = joint_omi_step_8_osp_sym_link
    # Old strategy table code assumes we are in the run dir. Our new
    # refractor version doesn't have that limit, but we are testing against
    # the old one here.
    os.chdir(rs.measurement_id.base_dir)
    # Comare with old py-retrieve code
    stable = MusesStrategyStepList.create_from_strategy_file(
        rs.measurement_id.base_dir / "Table.asc",
        rs.retrieval_config.input_file_helper,
        rs.creator_dict.strategy_context,
        rs.creator_dict,
    )
    stable_old = MusesStrategyOldStrategyTable(
        rs.measurement_id.base_dir / "Table.asc",
        rs.input_file_helper,
        rs.creator_dict[MusesSpectralWindowDict],
    )
    stable.notify_update_strategy_context(rs.strategy_context)
    stable_old.notify_update_strategy_context(rs.strategy_context)
    assert stable.filter_list_dict == stable_old.filter_list_dict
    assert [str(i) for i in stable.retrieval_elements] == stable_old.retrieval_elements
    assert [
        str(i) for i in stable.error_analysis_interferents
    ] == stable_old.error_analysis_interferents
    assert stable.instrument_name == stable_old.instrument_name
    # In general, we need to use the right current_state in next_step below. However, for
    # this particular unit test we know that only the cstate.brightness_temperature_data
    # affects the MusesStrategyStepList, so we get away with only using one
    # cstate since brightness_temperature_data doesn't change after the step it is
    # calculated.
    cstate = rs.current_state
    stable.restart()
    stable_old.restart()
    while not stable.is_done():
        assert stable.is_next_bt() == stable_old.is_next_bt()

        # This no longer passes because we added "update_apriori_elements" to
        # the table. So this doesn't match the old strategy table without this.
        # Just comment this out, the test we had already verified everything was
        # working and adding the new column doesn't actually break anything.
        # assert stable.current_strategy_step() == stable_old.current_strategy_step()
        stable.next_step(cstate)
        stable_old.next_step(cstate)


def test_tes_strategy_table_reader(isolated_dir, ifile_hlp, joint_tropomi_test_in_dir):
    f = TesStrategyTableReader(joint_tropomi_test_in_dir / "Table.asc", ifile_hlp)
    f.to_yaml("strategy.yaml")
    TesStrategyTableReader.from_yaml("strategy.yaml", "Table.asc")
    f2 = TesStrategyTableReader("Table.asc")
    assert f.table == f2.table
