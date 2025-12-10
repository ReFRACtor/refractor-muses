from refractor.muses import (
    RetrievalStrategy,
    StateElementIdentifier,
    modify_strategy_table,
)
from refractor.tropomi import TropomiSwirForwardModelHandle
import pytest

# ----------------------------------------------------------------
# These tests were all in development. I don't think they are currently
# working, we'll want to get Josh to clean this up when things settle
# down. But for now, skip all these
# ----------------------------------------------------------------


@pytest.mark.skip
@pytest.mark.long_test
def test_band8_retrieval(tropomi_swir, josh_osp_dir):
    """Work through issues to do a band 8 retrieval, without making
    any py-retrieve modifications"""
    rs = RetrievalStrategy(None, osp_dir=josh_osp_dir)
    # Just retrieve CO
    modify_strategy_table(
        rs,
        0,
        [
            StateElementIdentifier("CO"),
        ],
    )
    # Grab each step so we can separately test output
    # rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
    # rs.add_observer(rscap)
    ihandle = TropomiSwirForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False, osp_dir=josh_osp_dir
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{tropomi_swir.run_dir}/Table.asc")
    # This doesn't execute yet for band 8. We'll work through issues here by
    # debugging, and put the first problems in the next section to work through
    # them
    if False:
        rs.retrieval_ms()

    # Do all the setup etc., but stop the retrieval at step 0 (i.e., before we
    # do the first retrieval step). We then grab things to check stuff out
    rs.strategy_executor.execute_retrieval(stop_at_step=0)
    # Currently fails, we want to get this working
    # flist_dict = rs.strategy_executor.filter_list_dict
    # assert flist_dict == {"TROPOMI": ["BAND8"]}
