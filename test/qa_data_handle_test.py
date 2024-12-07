from test_support import *
from refractor.muses import (ErrorAnalysis, RetrievalStrategy, MusesRunDir,
                             order_species,
                             MeasurementIdFile, RetrievalConfiguration,
                             QaDataHandleSet)

def test_qa_data_update_retrieval_results(isolated_dir, osp_dir, gmao_dir):
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
    enalysis.update_retrieval_result(retrieval_result)
    qaset = QaDataHandleSet.default_handle_set()
    rconfig = RetrievalConfiguration.create_from_strategy_file("./Table.asc",
                                                               osp_dir=osp_dir)
    flist = {'TROPOMI': ['BAND3'], 'CRIS': ['2B1', '1B2', '2A1', '1A1']}
    mid = MeasurementIdFile("./Measurement_ID.asc", rconfig, flist)
    qaset.notify_update_target(mid)
    qaset.qa_update_retrieval_result(retrieval_result)
