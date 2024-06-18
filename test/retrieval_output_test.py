from test_support import *
from refractor.muses import (RetrievalJacobianOutput, RetrievalStrategy,
                             RetrievalL2Output, RetrievalRadianceOutput,
                             MusesRunDir, RetrievalPickleResult, RetrievalPlotRadiance,
                             RetrievalPlotResult, RetrievalInputOutput,
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

run_retrieval_output_test = False
# Only run these if you have done the capture
#run_retrieval_output_test = True

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
        
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
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
    

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
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

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_jacobian_output(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalJacobianOutput()
    jout.notify_update(rs, "retrieval step", **kwarg)
    compare_dir = joint_tropomi_test_refractor_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20190807_065_04_08_5",
                diff_is_error=diff_is_error)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_l2_output(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalL2Output()
    jout.notify_update(rs, "retrieval step", **kwarg)
    compare_dir = joint_tropomi_test_refractor_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20190807_065_04_08_5",
                diff_is_error=diff_is_error)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_pickle_results(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalPickleResult()
    jout.notify_update(rs, "retrieval step", **kwarg)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    assert os.path.exists("20190807_065_04_08_5/Step12_H2O,O3,EMIS_TROPOMI/Diagnostics/results.pkl")
    
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_plot_radiance(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalPlotRadiance()
    jout.notify_update(rs, "retrieval step", **kwarg)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("radiance_fit_diff.png", "radiance_fit_diff_vs_radiance.png",
                  "radiance_fit_initial_diff.png", "radiance_fit_initial_diff_vs_radiance.png",
                  "radiance_fit_initial.png", "radiance_fit.png"):
        assert os.path.exists(f"20190807_065_04_08_5/Step12_H2O,O3,EMIS_TROPOMI/StepAnalysis/{fname}")
    
# Currently not working, too tightly coupled to retrieval strategy so we have this
# turned out
@skip
@require_muses_py
def test_retrieval_input_output(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalInputOutput()
    jout.notify_update(rs, "retrieval step", **kwarg)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same

# Currently not working, too tightly coupled to retrieval strategy so we have this
# turned out
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_plot_results(isolated_dir, osp_dir, gmao_dir):
    step_number = 12
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    originaldir = isolated_dir
    pname = f"{originaldir}/cris_tropomi_ready_output/20190807_065_04_08_5/retrieval_ready_output_{step_number}.pkl"
    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    jout = RetrievalPlotResult()
    jout.notify_update(rs, "retrieval step", **kwarg)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("ak_full.png", "plot_H2O.png", "plot_O3.png"):
        assert os.path.exists(f"20190807_065_04_08_5/Step12_H2O,O3,EMIS_TROPOMI/{fname}")

    

