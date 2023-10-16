from test_support import *
from refractor.muses import (RetrievalJacobianOutput, RetrievalStrategy,
                             RetrievalL2Output, RetrievalRadianceOutput,
                             MusesRunDir)
import subprocess

@skip
@require_muses_py
def test_retrieval_l2_output(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    step_number = 2
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy.load_retrieval_strategy(f"{joint_tropomi_test_in_dir}/retrieval_strategy_retrieval_step_{step_number}.pkl", vlidort_cli=vlidort_cli)
    rs.clear_observers()
    jout = RetrievalL2Output()
    rs.add_observer(jout)
    rs.notify_update("retrieval step")
    

# Temporary, depends on our test run
@require_muses_py
def test_retrieval_radiance_output(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    pname = f"/home/smyth/Local/refractor-muses/retrieval_strategy_cris_tropomi/20190807_065_04_08_5/retrieval_step_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    rs.clear_observers()
    jout = RetrievalRadianceOutput()
    rs.add_observer(jout)
    rs.notify_update("retrieval step", **kwarg)
    f1 = "20190807_065_04_08_5/Products/Products_Radiance-H2O,O3,_TROPOMI-joint.nc"
    f2 = f"/home/smyth/Local/refractor-muses/original_retrieval_cris_tropomi/{f1}"
    cmd = f"h5diff --relative 1e-8 {f1} {f2}"
    print(os.getcwd())
    print(cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

@require_muses_py
def test_retrieval_jacobian_output(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    pname = f"/home/smyth/Local/refractor-muses/retrieval_strategy_cris_tropomi/20190807_065_04_08_5/retrieval_step_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    rs.clear_observers()
    jout = RetrievalJacobianOutput()
    rs.add_observer(jout)
    rs.notify_update("retrieval step", **kwarg)
    f1 = "20190807_065_04_08_5/Products/Products_Jacobian-H2O,O3,_TROPOMI-joint.nc"
    f2 = f"/home/smyth/Local/refractor-muses/original_retrieval_cris_tropomi/{f1}"
    cmd = f"h5diff --relative 1e-8 {f1} {f2}"
    print(os.getcwd())
    print(cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
    
