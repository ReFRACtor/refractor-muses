from test_support import *
from test_support.old_py_retrieve_test_support import old_py_retrieve_test
from refractor.old_py_retrieve_wrapper import MusesRetrievalStep


@old_py_retrieve_test
def test_muse_retrieval_step(isolated_dir):
    m = MusesRetrievalStep.load_retrieval_step(
        f"{test_base_path}/omi/in/sounding_1/run_retrieval_step_2.pkl",
        change_to_dir=False,
    )
    print(m.params)


@long_test
@old_py_retrieve_test
def test_muse_retrieval_run_step(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    m = MusesRetrievalStep.load_retrieval_step(
        f"{test_base_path}/omi/in/sounding_1/run_retrieval_step_2.pkl",
        change_to_dir=False,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )
    m.run_retrieval(vlidort_cli=vlidort_cli)
