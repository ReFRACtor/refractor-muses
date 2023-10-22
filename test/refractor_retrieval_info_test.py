from test_support import *
from refractor.muses import (RefractorRetrievalInfo,
                             RetrievalStrategy,
                             MusesRunDir)
import subprocess
import glob
import pickle

def struct_compare(s1, s2):
    for k in s1.keys():
        print(k)
        if(isinstance(s1[k], np.ndarray) and
           np.can_cast(s1[k], np.float64)):
           npt.assert_allclose(s1[k], s2[k])
        elif(isinstance(s1[k], np.ndarray)):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]

# Temporary, depends on our test run
@require_muses_py
def test_retrieval_info(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    pname = f"/home/smyth/Local/refractor-muses/retrieval_strategy_cris_tropomi/20190807_065_04_08_5/retrieval_step_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    # Currently depends on being in local directory, we can try to relax
    # that as we work through this.
    with rs.chdir_run_dir():
        rinfo = RefractorRetrievalInfo(rs.strategy_table, rs.state_info)
    if False:
        pickle.dump(rinfo, open("/home/smyth/Local/refractor-muses/rinfo_base.pkl", "wb"))
    rinfo_expect = pickle.load(open("/home/smyth/Local/refractor-muses/rinfo_base.pkl", "rb"))
    struct_compare(rinfo.retrieval_dict, rinfo_expect.retrieval_dict)

    

