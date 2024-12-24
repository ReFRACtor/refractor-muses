from test_support import *
from refractor.muses import (
    RetrievalJacobianOutput, 
    RetrievalL2Output, RetrievalRadianceOutput, RetrievalIrkOutput,
    RetrievalPickleResult, RetrievalPlotRadiance, RetrievalPlotResult,
    RetrievalInputOutput)
import subprocess
import glob

#run_retrieval_output_test = False
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
def joint_tropomi_output(isolated_dir):
    '''Common part of out output tests.'''
    rs, rstep, kwargs = set_up_run_to_location(joint_tropomi_test_in_dir, 12,
                                               "retrieval step")
    yield rs, rstep, kwargs
    compare_dir = joint_tropomi_test_refractor_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20190807_065_04_08_5",
                diff_is_error=diff_is_error)
    
        
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_radiance_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalRadianceOutput()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_jacobian_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalJacobianOutput()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_l2_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalL2Output()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_pickle_results(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalPickleResult()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    assert os.path.exists(f"{rs.run_dir}/Step12_H2O,O3,EMIS_TROPOMI/Diagnostics/results.pkl")
    
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_plot_radiance(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalPlotRadiance()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("radiance_fit_diff.png", "radiance_fit_diff_vs_radiance.png",
                  "radiance_fit_initial_diff.png", "radiance_fit_initial_diff_vs_radiance.png",
                  "radiance_fit_initial.png", "radiance_fit.png"):
        assert os.path.exists(f"{rs.run_dir}/Step12_H2O,O3,EMIS_TROPOMI/StepAnalysis/{fname}")

# Doesn't work, too tightly coupled to StrategyTable. We can perhaps get this
# working, but this is a diagnostic anyways so not overly important
@skip        
@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_input_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalInputOutput()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)
    print(os.path.abspath("."))
    breakpoint()

@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_plot_results(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalPlotResult()
    jout.notify_update(rs, "retrieval step", retrieval_strategy_step=rstep,
                       **kwargs)
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("ak_full.png", "plot_H2O.png", "plot_O3.png"):
        assert os.path.exists(f"{rs.run_dir}/Step12_H2O,O3,EMIS_TROPOMI/{fname}")


@pytest.mark.skipif(not run_retrieval_output_test,
                    reason="skipped because retrieval_output_test is False")
def test_retrieval_irk_output(isolated_dir):
    rs, rstep, kwargs = set_up_run_to_location(airs_irk_test_in_dir, 6,
                                               "IRK step")
    jout = RetrievalIrkOutput()
    jout.notify_update(rs, "IRK step", retrieval_strategy_step=rstep, **kwargs)
    compare_dir = airs_irk_test_expected_dir
    diff_is_error = True
    compare_run(compare_dir, "20160401_231_049_29",
                diff_is_error=diff_is_error)
