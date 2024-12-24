from test_support import *
from refractor.muses import (ErrorAnalysis, RetrievalStrategy, MusesRunDir)
import copy

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "initial set up done"):
            raise StopIteration()

@skip        
def test_error_analysis_init(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    if False:
        try:
            with all_output_disabled():
                #r = MusesRunDir(joint_omi_test_in_dir,
                r = MusesRunDir(joint_tropomi_test_in_dir,
                                osp_dir, gmao_dir, path_prefix=".")
                rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
                rs.clear_observers()
                rs.add_observer(RetrievalStrategyStop())
                rs.retrieval_ms()
        except StopIteration:
            pass
        # Save a copy of the error_intial, so we can compare as we try rewriting
        # this
        rs.save_pickle("/home/smyth/Local/refractor-muses/error_analysis.pkl")
    rs, _ = RetrievalStrategy.load_retrieval_strategy("/home/smyth/Local/refractor-muses/error_analysis.pkl", change_to_dir=True, osp_dir=osp_dir, gmao_dir=gmao_dir)
    einitial_compare = rs.error_analysis.error_initial.__dict__
    enalysis = ErrorAnalysis(rs.strategy_table, rs.state_info)
    struct_compare(enalysis.error_initial.__dict__, einitial_compare,
                   skip_list=["preferences"], verbose=True)
    

def test_error_analysis_update_retrieval_results(isolated_dir):
    rs, rstep, _ = set_up_run_to_location(joint_tropomi_test_in_dir, 10,
                                          "systematic_jacobian")
    # TODO Not really sure what the results are suppose to be, or even easily what
    # gets changed. So we just check that the call is successful.
    # Note we do check this indirectly by our end to end runs and comparison
    # to expected results, but it would be good to check this is more detail
    # in this test.
    before = copy.deepcopy(rstep.results)
    rs.error_analysis.update_retrieval_result(rstep.results)
    
