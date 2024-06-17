from test_support import *
from refractor.muses import (RetrievalJacobianOutput, RetrievalStrategy,
                             RetrievalL2Output, RetrievalRadianceOutput,
                             MusesRunDir,
                             RetrievalStrategy, RetrievalStrategyCaptureObserver,)
from refractor.tropomi import TropomiForwardModelHandle
import subprocess
import glob

# It is a bit hard to test the various output functions, because rightly they are pretty
# coupled to the actual retrieval run. We may be able to pull this apart to some extent,
# but on the other hand some level of coupling is real.
#
# Right now, we do this testing by capturing when the retrieval step is done. We don't have
# this in our normal test data because this interface is a bit unstable right now. So we can
# do a capture run here and use in the other tests.
#
# This tests are mostly skipped, since we don't want to depend on having this capture data
# available. But you can comment out the skip to run this, and in the future when the interface
# is stable we can have these run all the time.

def compare_run(expected_dir, run_dir, diff_is_error=True):
    '''Compare products from two runs.'''
    for f in glob.glob(f"{run_dir}/*/Products/Products_L2*.nc"):
        f2 = f.replace(run_dir, expected_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{run_dir}/*/Products/Lite_Products_*.nc"):
        f2 = f.replace(run_dir, expected_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{run_dir}/*/Products/Products_Radiance*.nc"):
        f2 = f.replace(run_dir, expected_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{run_dir}/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace(run_dir, expected_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
        
#@skip
@capture_test
@require_muses_py
def test_capture_cris_tropomi_output(osp_dir, gmao_dir):
    subprocess.run("rm -r cris_tropomi_ready_output", shell=True)
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="cris_tropomi_ready_output")
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rscap = RetrievalStrategyCaptureObserver("retrieval_ready_output", "retrieval step")
    rs.add_observer(rscap)
    ihandle = TropomiForwardModelHandle(use_pca=True, use_lrad=False,
                                        lrad_second_order=False)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.retrieval_ms()
    
    
#@skip
@require_muses_py
def test_retrieval_radiance_output(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalRadianceOutput()
    jout.notify_update(rs, "retrieval step", **kwarg)
    compare_dir = joint_tropomi_test_refractor_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20190807_065_04_08_5",
                diff_is_error=diff_is_error)

@skip        
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
@skip
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
    

