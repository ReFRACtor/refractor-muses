from refractor.old_py_retrieve_wrapper import MusesRetrievalStep
from fixtures.require_check import require_muses_py, require_muses_py_fm
import pytest


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_muse_retrieval_step(isolated_dir, omi_test_in_dir, ifile_hlp):
    m = MusesRetrievalStep.load_retrieval_step(
        omi_test_in_dir / "run_retrieval_step_2.pkl",
        change_to_dir=False,
        ifile_hlp=ifile_hlp,
    )
    print(m.params)


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
@require_muses_py_fm
def test_muse_retrieval_run_step(isolated_dir, ifile_hlp, omi_test_in_dir):
    m = MusesRetrievalStep.load_retrieval_step(
        omi_test_in_dir / "run_retrieval_step_2.pkl",
        change_to_dir=True,
        ifile_hlp=ifile_hlp,
    )
    m.run_retrieval()
