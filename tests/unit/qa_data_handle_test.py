from refractor.muses import QaFlagValueFile, MusesPyQaDataHandle
from fixtures.retrieval_step_fixture import set_up_run_to_location


def test_qa_data_qa_flag(isolated_dir, joint_tropomi_test_in_dir, osp_dir, gmao_dir):
    rs, rstep, _ = set_up_run_to_location(
        joint_tropomi_test_in_dir,
        10,
        "retrieval step",
        osp_dir,
        gmao_dir,
    )
    res = rs.qa_data_handle_set.qa_flag(rstep.results, rs.current_strategy_step)
    assert res == "BAD"
    # Check reading the file directly
    qa_handle = MusesPyQaDataHandle()
    qa_handle.notify_update_target(rs.measurement_id, rs.retrieval_config)
    f = QaFlagValueFile(
        qa_handle.quality_flag_file_name(rs.current_strategy_step),
        rs.retrieval_config.input_file_helper,
    )
    print(f.qa_flag_name)
    print(f.cutoff_min)
    print(f.cutoff_max)
    print(f.use_for_master)
