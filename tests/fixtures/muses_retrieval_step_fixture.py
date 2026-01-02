import pytest
from refractor.old_py_retrieve_wrapper import MusesRetrievalStep


def load_muses_retrieval_step(
    dir_in, step_number=1, ifile_hlp=None, change_to_dir=False
):
    """This reads parameters that can be use to call the py-retrieve function
    run_retrieval. See muses_capture in refractor-muses for collecting this.
    """
    return MusesRetrievalStep.load_retrieval_step(
        f"{dir_in}/run_retrieval_step_{step_number}.pkl",
        ifile_hlp=ifile_hlp,
        change_to_dir=change_to_dir,
    )


@pytest.fixture(scope="function")
def joint_tropomi_muses_retrieval_step_12(
    isolated_dir, ifile_hlp, joint_tropomi_test_in_dir
):
    """Joint tropomi muses retrieval step"""
    return load_muses_retrieval_step(
        joint_tropomi_test_in_dir, step_number=12, ifile_hlp=ifile_hlp
    )


@pytest.fixture(scope="function")
def joint_omi_muses_retrieval_step_8(isolated_dir, ifile_hlp, joint_omi_test_in_dir):
    """Joint omi muses retrieval step"""
    return load_muses_retrieval_step(
        joint_omi_test_in_dir, step_number=8, ifile_hlp=ifile_hlp
    )


@pytest.fixture(scope="function")
def tropomi_muses_retrieval_step_2(isolated_dir, ifile_hlp, tropomi_test_in_dir):
    """Tropomi muses retrieval step"""
    return load_muses_retrieval_step(
        tropomi_test_in_dir, step_number=2, ifile_hlp=ifile_hlp
    )


@pytest.fixture(scope="function")
def omi_muses_retrieval_step_2(isolated_dir, ifile_hlp, omi_test_in_dir):
    """Omi muses retrieval step"""
    return load_muses_retrieval_step(
        omi_test_in_dir, step_number=2, ifile_hlp=ifile_hlp
    )
