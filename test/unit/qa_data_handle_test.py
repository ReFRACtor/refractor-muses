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
    rs.error_analysis.update_retrieval_result(rstep.results)
    rs.qa_data_handle_set.qa_update_retrieval_result(rstep.results)
