from test_support import *
from refractor.muses import (RetrievalJacobianOutput, RetrievalStrategy,
                             RetrievalL2Output, RetrievalRadianceOutput,
                             MusesRunDir)
import subprocess
import glob

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
    for f1 in glob.glob("20190807_065_04_08_5/Products/*.nc"):
        f2 = f"/home/smyth/Local/refractor-muses/original_retrieval_cris_tropomi/{f1}"
        cmd = f"h5diff --relative 1e-8 {os.path.abspath(f1)} {f2}"
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
    for f1 in glob.glob("20190807_065_04_08_5/Products/*.nc"):
        f2 = f"/home/smyth/Local/refractor-muses/original_retrieval_cris_tropomi/{f1}"
        cmd = f"h5diff --relative 1e-8 {os.path.abspath(f1)} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=True)
    
@require_muses_py
def test_retrieval_l2_output(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    do_cris=True
    if do_cris:
        step_number = 12
        r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
        pname = f"/home/smyth/Local/refractor-muses/retrieval_strategy_cris_tropomi/20190807_065_04_08_5/retrieval_step_{step_number}.pkl"
    else:
        step_number = 8
        r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
        pname = f"/home/smyth/Local/refractor-muses/retrieval_strategy_airs_omi/20160401_231_049_29/retrieval_step_{step_number}.pkl"

    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    rs.clear_observers()
    jout = RetrievalL2Output()
    rs.add_observer(jout)
    rs.notify_update("retrieval step", **kwarg)
    for f1 in glob.glob("20190807_065_04_08_5/Products/*.nc") if do_cris else glob.glob("20160401_231_049_29/Products/*.nc"):
        if do_cris:
            f2 = f"/home/smyth/Local/refractor-muses/original_retrieval_cris_tropomi/{f1}"
        else:
            f2 = f"/home/smyth/Local/refractor-muses/original_retrieval_airs_omi/{f1}"
        cmd = f"h5diff --relative 1e-8 {os.path.abspath(f1)} {f2}"
        print(os.getcwd())
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=True)
    

