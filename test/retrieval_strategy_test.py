from test_support import *
from refractor.muses import (FmObsCreator, CostFunction,
                             MusesForwardModelStep,
                             MusesRunDir,
                             RefractorMusesIntegration,
                             RetrievalStrategy)
import refractor.muses.muses_py as mpy
import subprocess
import pprint
import glob

def struct_compare(s1, s2):
    for k in s1.keys():
        #print(k)
        if(isinstance(s1[k], np.ndarray) and
           np.can_cast(s1[k], np.float64)):
           npt.assert_allclose(s1[k], s2[k])
        elif(isinstance(s1[k], np.ndarray)):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]

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
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r retrieval_strategy_cris_tropomi", shell=True)
    # Think we can remove this
    #rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    #rmi.register_with_muses_py()
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="retrieval_strategy_cris_tropomi")
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", writeOutput=True, writePlots=True,
                           vlidort_cli=vlidort_cli)
    rs.retrieval_ms()

    # Temp, do compare right after
    diff_is_error = True
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Products_L2*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Lite_Products_*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Products_Radiance*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    
@long_test
@require_muses_py
def test_compare_retrieval_cris_tropomi(osp_dir, gmao_dir, vlidort_cli):
    '''Quick test to compare cris_tropomi runs. This assumes they are
    already done. This is just h5diff, but this figures out the path
    for each of the tests so we don't have to.'''
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    #diff_is_error = True
    diff_is_error = False
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Products_L2*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Lite_Products_*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Products_Radiance*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_cris_tropomi/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace("original_retrieval_cris_tropomi", "retrieval_strategy_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)

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
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    rmi.register_with_muses_py()
    r.run_retrieval(vlidort_cli=vlidort_cli)

@long_test
@require_muses_py
def test_retrieval_strategy_airs_omi(osp_dir, gmao_dir, vlidort_cli,
                                         clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r retrieval_strategy_airs_omi", shell=True)
    #rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    #rmi.register_with_muses_py()
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="retrieval_strategy_airs_omi")
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rs.retrieval_ms()

    # Temp, compare right after
    diff_is_error = True
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Products_L2*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Lite_Products_*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Products_Radiance*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    
@long_test
@require_muses_py
def test_compare_retrieval_airs_omi(osp_dir, gmao_dir, vlidort_cli):
    '''Quick test to compare airs_omi runs. This assumes they are
    already done. This is just h5diff, but this figures out the path
    for each of the tests so we don't have to.'''
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    diff_is_error = True
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Products_L2*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Lite_Products_*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Products_Radiance*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob("original_retrieval_airs_omi/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace("original_retrieval_airs_omi", "retrieval_strategy_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
        
