from test_support import *
from refractor.muses import RefractorStateInfo, RetrievalStrategy, MusesRunDir

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "initial set up done"):
            raise StopIteration()

@require_muses_py
def test_refractor_state_info(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    # TODO - We should have a constructor for RefractorStateInfo. Don't currently,
    # so we just run RetrievalStrategy to the beginning and stop
    try:
        with all_output_disabled():
            r = MusesRunDir(joint_tropomi_test_in_dir,
                            osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    sinfo = rs.state_info
    print(sinfo)
