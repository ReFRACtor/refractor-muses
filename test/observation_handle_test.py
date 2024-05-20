from test_support import *
from refractor.muses import (StateInfo, RetrievalStrategy, MusesRunDir,
                             ObservationHandleSet, CostFunctionCreator)

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "initial set up done"):
            raise StopIteration()

@require_muses_py
def test_radiance(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    # TODO - We should have a constructor for StateInfo. Don't currently,
    # so we just run RetrievalStrategy to the beginning and stop
    try:
        with all_output_disabled():
            r = MusesRunDir(joint_omi_test_in_dir,
            #r = MusesRunDir(joint_tropomi_test_in_dir,
                            osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    oset = rs.observation_handle_set
    rad = oset.mpy_radiance_full_band(None, rs.strategy_table)
    cfunc = CostFunctionCreator()
    cfunc.update_target(rs.measurement_id, rs)
    rad2 = cfunc.radiance(rs.state_info, rs.strategy_table.instrument_name(all_step=True))
    npt.assert_allclose(rad["frequency"], rad2["frequency"])
    npt.assert_allclose(rad["radiance"], rad2["radiance"])
    npt.assert_allclose(rad["NESR"], rad2["NESR"])
    
    
