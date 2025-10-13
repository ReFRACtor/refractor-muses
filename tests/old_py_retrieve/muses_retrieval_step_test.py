from refractor.old_py_retrieve_wrapper import MusesRetrievalStep
import pytest


@pytest.mark.old_py_retrieve_test
def test_muse_retrieval_step(isolated_dir, omi_test_in_dir, osp_dir, gmao_dir):
    m = MusesRetrievalStep.load_retrieval_step(
        omi_test_in_dir / "run_retrieval_step_2.pkl",
        change_to_dir=False,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )
    print(m.params)


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
def test_muse_retrieval_run_step(isolated_dir, osp_dir, gmao_dir, omi_test_in_dir):
    m = MusesRetrievalStep.load_retrieval_step(
        omi_test_in_dir / "run_retrieval_step_2.pkl",
        change_to_dir=False,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )
    m.run_retrieval()
