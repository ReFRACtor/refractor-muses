from test_support import *
from refractor.muses import (ErrorAnalysis, RetrievalStrategy, MusesRunDir,
                             order_species,
                             MeasurementIdFile, RetrievalConfiguration,
                             QaDataHandleSet)


def test_qa_data_update_retrieval_results(isolated_dir, osp_dir, gmao_dir,
                                          vlidort_cli):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rstep = run_step_to_location(rs, 10, joint_tropomi_test_in_dir,
                                 "systematic_jacobian")
    rs.error_analysis.update_retrieval_result(rstep.results)
    rs.qa_data_handle_set.qa_update_retrieval_result(rstep.results)
    
