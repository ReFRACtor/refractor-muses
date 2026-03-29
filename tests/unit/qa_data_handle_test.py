from refractor.muses import QaFlagValueFile, MusesPyQaDataHandle, QaFlag
from fixtures.retrieval_step_fixture import set_up_run_to_location
import itertools


def test_qa_data_qa_flag(isolated_dir, joint_tropomi_test_in_dir, ifile_hlp):
    rs, rstep, _ = set_up_run_to_location(
        joint_tropomi_test_in_dir,
        10,
        "retrieval step",
        ifile_hlp,
    )
    res = rs.creator_dict[QaFlag].qa_flag(rstep.results, rs.current_strategy_step)
    assert res.master_flag == "BAD"
    # Check reading the file directly
    # This is a little convoluted, but it finds the MusesPyQaDataHandle in our
    # handle set
    qa_handle = [
        t
        for t in itertools.chain(*rs.qa_data_handle_set.handle_set.values())
        if isinstance(t, MusesPyQaDataHandle)
    ][0]
    f = QaFlagValueFile(
        qa_handle.quality_flag_file_name(rs.current_strategy_step),
        rs.retrieval_config.input_file_helper,
    )
    print(f.qa_flag_name)
    print(f.cutoff_min)
    print(f.cutoff_max)
    print(f.use_for_master)
