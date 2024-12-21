from test_support import *
from refractor.muses import (ErrorAnalysis, RetrievalStrategy, MusesRunDir,
                             order_species,
                             MeasurementIdFile, RetrievalConfiguration,
                             QaDataHandleSet)


def test_qa_data_update_retrieval_results(isolated_dir):
    rs, rstep, _ = set_up_run_to_location(joint_tropomi_test_in_dir, 10,
                                          "systematic_jacobian")
    rs.error_analysis.update_retrieval_result(rstep.results)
    rs.qa_data_handle_set.qa_update_retrieval_result(rstep.results)
    
