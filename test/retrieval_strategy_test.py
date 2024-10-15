from test_support import *
from refractor.muses import (MusesRunDir, RetrievalStrategy, RetrievalStrategyCaptureObserver,
                             CurrentStateUip)
from refractor.omi import OmiForwardModelHandle
from refractor.tropomi import TropomiForwardModelHandle
import refractor.muses.muses_py as mpy
import subprocess
import pprint
import glob
import shutil
import copy
from loguru import logger

# Use refractor forward model. We default to not, because we are
# mostly testing everything *other* than the forward model with this
# test. But can be useful to run with this occasionally.
# Note that there is a separate set of expected results for a refractor run.
#run_refractor = False
run_refractor = True

def compare_run(expected_dir, run_dir, diff_is_error=True):
    '''Compare products from two runs.'''
    for f in glob.glob(f"{expected_dir}/*/Products/Products_L2*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Lite_Products_*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Products_Radiance*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    
# This test was used to generate the original test data using py-retrieve. We have
# tweaked the expected output slightly for test_retrieval_strategy_cris_tropomi (so
# minor round off differeneces). But leave this code around, it can still be useful to
# have the full original run if we need to dig into and issue with
# test_retrieval_strategy_cris_tropomi. But note the output isn't identical, just pretty
# close.
@skip
@long_test
@require_muses_py
def test_original_retrieval_cris_tropomi(osp_dir, gmao_dir, vlidort_cli,
                                         clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This uses our RefractorMusesIntegration, but muses-py version of
    script_retrieval_ms. There are pretty minor differences (which we have checked out
    separately) with the original muses-py run - we just pull these out here so
    we can focus on any difference with our RetrievalStrategy'''
    subprocess.run("rm -r original_retrieval_cris_tropomi", shell=True)
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="original_retrieval_cris_tropomi")
    #rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    #rmi.register_with_muses_py()
    r.run_retrieval(vlidort_cli=vlidort_cli, debug=True, plots=True)

@long_test
@require_muses_py
def test_retrieval_strategy_cris_tropomi(osp_dir, gmao_dir, vlidort_cli,
                                         clean_up_replacement_function):
    '''Full run, that we then compare the output files to expected results.
    This is not really a unit test, but for convenience we have it here.
    Note that a "failure" in the comparison might not actually indicate a problem, just
    that the output changed. You may need to look into detail and decide that the
    run was successful and we just want to update the expected results.

    Data goes in the local directory, rather than an isolated one. We can change this
    in the future if desired, but  for now it is useful to be able to look into the directory
    if some kind of a problem arises.'''
    subprocess.run("rm -r retrieval_strategy_cris_tropomi", shell=True)
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="retrieval_strategy_cris_tropomi")
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", writeOutput=True, writePlots=True,
                           vlidort_cli=vlidort_cli,
                           write_tropomi_radiance_pickle=not run_refractor)
    try:
        lognum = logger.add("retrieval_strategy_cris_tropomi/retrieve.log")
        # Grab each step so we can separately test output
        rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
        rs.add_observer(rscap)
        compare_dir = joint_tropomi_test_expected_dir
        if run_refractor:
            # Use refractor forward model. 
            ihandle = TropomiForwardModelHandle(use_pca=True, use_lrad=False,
                                                lrad_second_order=False)
            rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
            # Different expected results. Close, but not identical to VLIDORT version
            compare_dir = joint_tropomi_test_refractor_expected_dir
            rs.update_target(f"{r.run_dir}/Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    diff_is_error = True
    compare_run(compare_dir, "retrieval_strategy_cris_tropomi",
                diff_is_error=diff_is_error)

@long_test
@require_muses_py
def test_compare_retrieval_cris_tropomi(osp_dir, gmao_dir, vlidort_cli):
    '''The test_retrieval_strategy_cris_tropomi already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_strategy_cris_tropomi already having been run.'''
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    #diff_is_error = True
    diff_is_error = False
    compare_dir = joint_tropomi_test_expected_dir
    if run_refractor:
        compare_dir = joint_tropomi_test_refractor_expected_dir
    compare_run(compare_dir, "retrieval_strategy_cris_tropomi",
                diff_is_error=diff_is_error)

# This test was used to generate the original test data using py-retrieve. We have
# tweaked the expected output slightly for test_retrieval_strategy_airs_omi (so
# minor round off differeneces). But leave this code around, it can still be useful to
# have the full original run if we need to dig into and issue with
# test_retrieval_strategy_airs_omi. But note the output isn't identical, just pretty
# close.
@skip
@long_test
@require_muses_py
def test_original_retrieval_airs_omi(osp_dir, gmao_dir, vlidort_cli,
                                         clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This uses our RefractorMusesIntegration, but muses-py version of
    script_retrieval_ms. There are pretty minor differences (which we have checked out
    separately) with the original muses-py run - we just pull these out here so
    we can focus on any difference with our RetrievalStrategy'''
    subprocess.run("rm -r original_retrieval_airs_omi", shell=True)
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="original_retrieval_airs_omi")
    #rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    #rmi.register_with_muses_py()
    r.run_retrieval(vlidort_cli=vlidort_cli)

@long_test
@require_muses_py
def test_retrieval_strategy_airs_omi(osp_dir, gmao_dir, vlidort_cli,
                                         clean_up_replacement_function):
    '''Full run, that we then compare the output files to expected results.
    This is not really a unit test, but for convenience we have it here.
    Note that a "failure" in the comparison might not actually indicate a problem, just
    that the output changed. You may need to look into detail and decide that the
    run was successful and we just want to update the expected results.

    Data goes in the local directory, rather than an isolated one. We can change this
    in the future if desired, but  for now it is useful to be able to look into the directory
    if some kind of a problem arises.'''
    subprocess.run("rm -r retrieval_strategy_airs_omi", shell=True)
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="retrieval_strategy_airs_omi")
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli,
                           write_omi_radiance_pickle=not run_refractor)
    try:
        lognum = logger.add("retrieval_strategy_airs_omi/retrieve.log")
        # Grab each step so we can separately test output
        rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
        rs.add_observer(rscap)
        compare_dir = joint_omi_test_expected_dir
        if run_refractor:
            # Use refractor forward model.
            ihandle = OmiForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False, use_eof=False)
            rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
            # Different expected results. Close, but not identical to VLIDORT version
            compare_dir = joint_omi_test_refractor_expected_dir
            rs.update_target(f"{r.run_dir}/Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)

    diff_is_error = True
    compare_run(compare_dir, "retrieval_strategy_airs_omi",
                diff_is_error=diff_is_error)
    
@long_test
@require_muses_py
def test_compare_retrieval_airs_omi(osp_dir, gmao_dir, vlidort_cli):
    '''The test_retrieval_strategy_airs_omi already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_strategy_airs_omi already having been run.'''
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    #diff_is_error = True
    diff_is_error = False
    compare_dir = joint_omi_test_expected_dir
    if run_refractor:
        compare_dir = joint_omi_test_refractor_expected_dir
    compare_run(compare_dir, "retrieval_strategy_airs_omi",
                diff_is_error=diff_is_error)

@long_test
@require_muses_py
def test_two_tropomi(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    '''Run two soundings, using a single RetrievalStrategy. This is how the
    MPI version of py-retrieve works, and we need to make sure any caching
    etc. gets cleared out from the first sounding to the second.'''
    r = MusesRunDir(tropomi_test_in_dir,
                    osp_dir, gmao_dir)
    r2 = MusesRunDir(tropomi_test_in_dir3,
                     osp_dir, gmao_dir, skip_sym_link=True)
    
    rs = RetrievalStrategy(None, writeOutput=True, writePlots=True,
                           vlidort_cli=vlidort_cli)
    rs.forward_model_handle_set.add_handle(TropomiForwardModelHandle(use_pca=True,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)
    rs.script_retrieval_ms(f"{r.run_dir}/Table.asc")
    rs.script_retrieval_ms(f"{r2.run_dir}/Table.asc")

# Sample of how to load a saved step (by RetrievalStrategyCaptureObserver) and
# start running it again. This  test depends on a specific run, and a hard coded path,
# so we don't normally run. But it is a basic example of how to do this.
@skip    
def test_run_fabiano_refractor(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy("/home/smyth/muses/refractor-muses/compare_fabiano_refractor/20200701_204_05_29_0/retrieval_step_10.pkl",osp_dir=osp_dir,gmao_dir=gmao_dir,vlidort_cli=vlidort_cli, change_to_dir=True)
    rs.continue_retrieval()

@skip    
def test_run_fabiano_vlidort(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    print(vlidort_cli)
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy("/home/smyth/muses/refractor-muses/compare_fabiano_refractor_vlidort/20200701_204_05_29_0/retrieval_step_10.pkl",osp_dir=osp_dir,gmao_dir=gmao_dir,vlidort_cli=vlidort_cli, change_to_dir=True)
    rs.continue_retrieval()
    
