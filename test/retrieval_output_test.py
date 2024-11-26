from test_support import *
from refractor.muses import (RetrievalJacobianOutput, RetrievalStrategy,
                             RetrievalL2Output, RetrievalRadianceOutput,
                             RetrievalIrkOutput,
                             MusesRunDir, RetrievalPickleResult, RetrievalPlotRadiance,
                             RetrievalPlotResult, RetrievalInputOutput,
                             RetrievalStrategy, RetrievalStrategyCaptureObserver,)
from refractor.tropomi import TropomiForwardModelHandle
import subprocess
import glob

# It is a bit hard to test the various output functions, because
# rightly they are pretty coupled to the actual retrieval run. We may
# be able to pull this apart to some extent, but on the other hand
# some level of coupling is real.
#
# Right now, we do this testing by capturing when the retrieval step
# is done. We don't have this in our normal test data because this
# interface is a bit unstable right now. So we can do a capture run
# and use in the other tests. The run is found in capture_data_test.py
# and is test_capture_tropomi_cris_retrieval_strategy and
# test_captures_airs_irk
#
# This tests are mostly skipped now, since we don't want to depend on
# having this capture data available. But you can comment out the skip
# to run this, and in the future when the interface is stable we can
# have these run all the time.
#
# TODO Longer term we would like a more stable way to save state. Perhaps just
# pickle/save StateInfo, and recreate everything else from scratch? 

#run_retrieval_output_test = False
# Only run these if you have done the capture
run_retrieval_output_test = True

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
    for f in glob.glob(f"{run_dir}/*/Products/Products_IRK.nc"):
        f2 = f.replace(run_dir, expected_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)

@pytest.fixture(scope="function")
def joint_tropomi_output(isolated_dir, osp_dir, gmao_dir):
    '''Common part of out output tests.'''
    step_number = 12
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        f"{joint_tropomi_test_in_dir}/retrieval_ready_output_{step_number}.pkl",
        osp_dir=osp_dir, gmao_dir=gmao_dir, change_to_dir=True)
    yield rs, kwargs
    compare_dir = joint_tropomi_test_refractor_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20190807_065_04_08_5",
                diff_is_error=diff_is_error)
    
        
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_radiance_output(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalRadianceOutput()
    jout.notify_update(rs, "retrieval step", **kwargs)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_jacobian_output(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalJacobianOutput()
    jout.notify_update(rs, "retrieval step", **kwargs)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_l2_output(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalL2Output()
    jout.notify_update(rs, "retrieval step", **kwargs)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_pickle_results(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalPickleResult()
    jout.notify_update(rs, "retrieval step", **kwargs)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    assert os.path.exists("Step12_H2O,O3,EMIS_TROPOMI/Diagnostics/results.pkl")
    
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_plot_radiance(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalPlotRadiance()
    jout.notify_update(rs, "retrieval step", **kwargs)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("radiance_fit_diff.png", "radiance_fit_diff_vs_radiance.png",
                  "radiance_fit_initial_diff.png", "radiance_fit_initial_diff_vs_radiance.png",
                  "radiance_fit_initial.png", "radiance_fit.png"):
        assert os.path.exists(f"Step12_H2O,O3,EMIS_TROPOMI/StepAnalysis/{fname}")

# Doesn't work, too tightly coupled to StrategyTable. We can perhaps get this
# working, but this is a diagnostic anyways so not overly important
@skip        
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_input_output(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalInputOutput()
    jout.notify_update(rs, "retrieval step", **kwargs)
    print(os.path.abspath("."))
    breakpoint()

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_plot_results(joint_tropomi_output):
    rs, kwargs = joint_tropomi_output
    jout = RetrievalPlotResult()
    jout.notify_update(rs, "retrieval step", **kwargs)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("ak_full.png", "plot_H2O.png", "plot_O3.png"):
        assert os.path.exists(f"Step12_H2O,O3,EMIS_TROPOMI/{fname}")


@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
@require_muses_py
def test_retrieval_irk_output(isolated_dir, osp_dir, gmao_dir):
    step_number = 6
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        f"{airs_irk_test_in_dir}/retrieval_irk_{step_number}.pkl",
        osp_dir=osp_dir, gmao_dir=gmao_dir, change_to_dir=True)
    jout = RetrievalIrkOutput()
    jout.notify_update(rs, "IRK step", **kwargs)
    compare_dir = airs_irk_test_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20160401_231_049_29",
                diff_is_error=diff_is_error)
