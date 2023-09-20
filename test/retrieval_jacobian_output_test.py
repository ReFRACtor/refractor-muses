from test_support import *
from refractor.muses import (RetrievalJacobianOutput, RetrievalStrategy,
                             MusesRunDir)

@require_muses_py
def test_retrieval_jacobian_outpu(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    step_number = 2
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy.load_retrieval_strategy(f"{joint_tropomi_test_in_dir}/retrieval_strategy_retrieval_step_{step_number}.pkl", vlidort_cli=vlidort_cli)
    rs.clear_observers()
    jout = RetrievalJacobianOutput()
    rs.add_observer(jout)
    assert os.path.basename(jout.out_fname) == "Products_Jacobian-TATM,H2O,HDO,N2O,CH4,O3,TSUR,CLOUDEXT-bar_land"
    rs.notify_update("retrieval step")
