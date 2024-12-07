from test_support import *
from refractor.muses import (ErrorAnalysis, RetrievalStrategy, MusesRunDir,
                             order_species)
import copy

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "initial set up done"):
            raise StopIteration()

@skip        
@require_muses_py
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
    

def test_error_analysis_update_retrieval_results(isolated_dir, osp_dir, gmao_dir):
    # Error analysis constructor depends on accessing "../OSP". We'll hopefully
    # clean this up in the future, but for now just unpack the directory structure
    # we have stored in retrieval_strategy_retrieval_step_10.pkl.
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        f"{joint_tropomi_test_in_dir}/retrieval_strategy_retrieval_step_10.pkl",
        osp_dir=osp_dir, gmao_dir=gmao_dir,change_to_dir=True)
    
    retrieval_result = pickle.load(open(f"{joint_tropomi_test_in_dir}/retrieval_result_10.pkl", "rb"))
    retrieval_result.state_info.retrieval_config.osp_dir = osp_dir
    retrieval_result.state_info.retrieval_config.base_dir = os.path.abspath(".")
    # This gets figured out in MusesStrategyExecutor. To make a stand alone
    # unit test, we just hard code the results that MusesStrategyExecutor
    # calculates
    #covariance_state_element_name = order_species(
    #        set(strategy_table.retrieval_elements_all_step) |
    #        set(strategy_table.error_analysis_interferents_all_step))
    covariance_state_element_name = ['TATM', 'H2O', 'O3', 'N2O', 'CO', 'CH4', 'NH3', 'TSUR', 'EMIS', 'CLOUDEXT', 'PCLOUD', 'HDO', 'CH3OH', 'PAN', 'TROPOMICLOUDFRACTION', 'TROPOMISURFACEALBEDOBAND3', 'TROPOMISURFACEALBEDOSLOPEBAND3', 'TROPOMISURFACEALBEDOSLOPEORDER2BAND3', 'TROPOMISOLARSHIFTBAND3', 'TROPOMIRADIANCESHIFTBAND3', 'TROPOMIRINGSFBAND3']
    enalysis = ErrorAnalysis(retrieval_result.current_strategy_step,
                             retrieval_result.state_info,
                             covariance_state_element_name)
    before = copy.deepcopy(retrieval_result)
    enalysis.update_retrieval_result(retrieval_result)
    # TODO Not really sure what the results are suppose to be, or even easily what
    # gets changed. So we just check that the call is successful.
    # Note we do check this indirectly by our end to end runs and comparison
    # to expected results, but it would be good to check this is more detail
    # in this test.
    
