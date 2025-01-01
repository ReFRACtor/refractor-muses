from test_support import *
from refractor.muses import RetrievalInfo, RetrievalStrategy, MusesRunDir
from refractor.old_py_retrieve_wrapper import RetrievalInfoOld
import pickle


# Temporary, depends on our test run
@skip
@pytest.mark.parametrize("step_number", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_retrieval_info(isolated_dir, vlidort_cli, osp_dir, gmao_dir, step_number):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    pname = f"/home/smyth/Local/refractor-muses/retrieval_strategy_cris_tropomi/20190807_065_04_08_5/retrieval_step_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(
        pname, vlidort_cli=vlidort_cli
    )
    # Currently depends on being in local directory, we can try to relax
    # that as we work through this.
    with rs.chdir_run_dir():
        rinfo = RetrievalInfo(rs.error_analysis, rs.strategy_table, rs.state_info)
    if False:
        with rs.chdir_run_dir():
            rinfo2 = RetrievalInfoOld(rs.strategy_table, rs.state_info)
        pickle.dump(
            rinfo2,
            open(
                f"/home/smyth/Local/refractor-muses/rinfo_base_{step_number}.pkl", "wb"
            ),
        )
    rinfo_expect = pickle.load(
        open(f"/home/smyth/Local/refractor-muses/rinfo_base_{step_number}.pkl", "rb")
    )
    struct_compare(rinfo.retrieval_dict, rinfo_expect.retrieval_dict)
