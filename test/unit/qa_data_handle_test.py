from refractor.muses import QaFlagValueFile, MusesPyQaDataHandle
from fixtures.retrieval_step_fixture import set_up_run_to_location


def test_qa_data_update_retrieval_results(
    isolated_dir, joint_tropomi_test_in_dir, osp_dir, gmao_dir, vlidort_cli
):
    rs, rstep, _ = set_up_run_to_location(
        joint_tropomi_test_in_dir,
        10,
        "systematic_jacobian",
        osp_dir,
        gmao_dir,
        vlidort_cli,
    )
    # Temp, this will probably go away
    rstep.results.update_error_analysis()
    rs.qa_data_handle_set.qa_update_retrieval_result(
        rstep.results, rs.current_strategy_step
    )
    # Check reading the file directly
    qa_handle = MusesPyQaDataHandle()
    qa_handle.notify_update_target(rs.measurement_id)
    f = QaFlagValueFile(qa_handle.quality_flag_file_name(rs.current_strategy_step))
    print(f.qa_flag_name)
    print(f.cutoff_min)
    print(f.cutoff_max)
    print(f.use_for_master)
