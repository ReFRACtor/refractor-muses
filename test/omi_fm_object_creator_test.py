import numpy as np
import numpy.testing as npt

from test_support import *
import refractor.framework as rf
import glob
from refractor.muses import (MusesRunDir, CostFunctionCreator, CostFunction, 
                             CurrentStateUip, RetrievalConfiguration, MeasurementIdFile)
from refractor.omi import (OmiFmObjectCreator, OmiForwardModelHandle)
from refractor.old_py_retrieve_wrapper import (RefractorMusesIntegration,
                                               RefractorOmiFm,MusesForwardModelStep,)
import subprocess

DEBUG = False

@pytest.fixture(scope="function")
def omi_fm_object_creator_step_1(omi_uip_step_1, omi_obs_step_1, osp_dir):
    '''Fixture for OmiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/omi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'OMI' : ['UV1', 'UV2']}
    mid = MeasurementIdFile(f"{test_base_path}/omi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    return OmiFmObjectCreator(CurrentStateUip(omi_uip_step_1), mid, omi_obs_step_1,
                              rf_uip=omi_uip_step_1)

@pytest.fixture(scope="function")
def omi_fm_object_creator_step_2(omi_uip_step_2, omi_obs_step_2, osp_dir):
    '''Fixture for OmiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/omi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'OMI' : ['UV1', 'UV2']}
    mid = MeasurementIdFile(f"{test_base_path}/omi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    return OmiFmObjectCreator(CurrentStateUip(omi_uip_step_2), mid, omi_obs_step_2,
                              rf_uip=omi_uip_step_2)

def test_solar_model(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.omi_solar_model[0])

def test_spec_win(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.spec_win)

def test_spectrum_sampling(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.spectrum_sampling)

def test_instrument(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.instrument)

@require_muses_py
def test_atmosphere(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.atmosphere)

@require_muses_py
def test_radiative_transfer(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.radiative_transfer)

@require_muses_py
def test_forward_model(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.forward_model)

class PrintSpectrum(rf.ObserverPtrNamedSpectrum):
    def notify_update(self, o):
        print("---------")
        print(o.name)
        print(o.spectral_domain.wavelength("nm"))
        print(o.spectral_range.data)
        print("---------")


class SaveSpectrum(rf.ObserverPtrNamedSpectrum):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def notify_update(self, o):
        import pickle
        fn = self.filename.format(name=o.name.replace(" ", "_"))
        with open(fn, "wb") as out:
            print(f"Saving {o.name} to {fn}")
            data = { "name": o.name,
                     "wavelength": o.spectral_domain.wavelength("nm"),
                     "radiance": o.spectral_range.data,
                    }
            pickle.dump(data, out)


@require_muses_py
def test_fm_run(omi_fm_object_creator_step_1):
    fm = omi_fm_object_creator_step_1.forward_model
    fm.add_observer_and_keep_reference(PrintSpectrum())
    print(fm.radiance(1, True).value)

def test_state_vector(omi_fm_object_creator_step_1, omi_uip_step_1):
    omi_fm_object_creator_step_1.fm_sv.update_state(omi_uip_step_1.current_state_x_fm)
    print(omi_fm_object_creator_step_1.fm_sv)

@require_muses_py
def test_state_vector_step2(omi_fm_object_creator_step_2, omi_uip_step_2):
    omi_fm_object_creator_step_2.fm_sv.update_state(omi_uip_step_2.current_state_x_fm)
    print(omi_fm_object_creator_step_2.fm_sv)

@require_muses_py
def test_raman_effect(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.raman_effect)

@require_muses_py
def test_forward_model_step2(omi_fm_object_creator_step_2):
    '''Step 2, which has two microwindows'''
    print(omi_fm_object_creator_step_2.forward_model)


@require_muses_py
def test_fm_run_step2(omi_fm_object_creator_step_2, omi_uip_step_2, omi_obs_step_2,
                      osp_dir):
    omi_fm_object_creator_step_2.use_pca = False
    omi_fm_object_creator_step_2.use_lrad = False
    omi_fm_object_creator_step_2.lrad_second_order = False
    fm = omi_fm_object_creator_step_2.forward_model
    if DEBUG:
        fm.add_observer_and_keep_reference(SaveSpectrum("/home/mcduffie/Temp/lidort_radiance_{name}.pkl"))
        fm.add_observer_and_keep_reference(PrintSpectrum())

    spectrum_lidort = fm.radiance(0, True)

    
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/omi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    omi_fm_object_creator_step_2.fm_sv.remove_observer(omi_obs_step_2)
    fm = OmiFmObjectCreator(CurrentStateUip(omi_uip_step_2), rconf, omi_obs_step_2,
                            rf_uip=omi_uip_step_2, use_pca=True, use_lrad=False,
                            lrad_second_order=False).forward_model

    if DEBUG:
        fm.add_observer_and_keep_reference(SaveSpectrum("/home/mcduffie/Temp/pca_radiance_{name}.pkl"))
        fm.add_observer_and_keep_reference(PrintSpectrum())

    spectrum_pca = fm.radiance(0, True)

    npt.assert_allclose(spectrum_lidort.spectral_domain.data, spectrum_pca.spectral_domain.data, rtol=1e-10)
    npt.assert_allclose(spectrum_lidort.spectral_range.data, spectrum_pca.spectral_range.data, rtol=2e-2)

@require_muses_py
def test_residual_fm_jac_omi(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                             joint_omi_obs_step_8):
    '''Test out the CostFunction residual_fm_jacobian using our forward model. Note
    that this just tests that we can make the call, to debug any problems there. The
    actual comparison on results is done in full run tests below.'''
    step_number = 8
    iteration = 2
    
    curdir = os.path.curdir
    rrefractor = muses_residual_fm_jac(joint_omi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = CostFunctionCreator()
    ihandle = OmiForwardModelHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False)
    creator.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    cfunc = creator.cost_function_from_uip(rf_uip, joint_omi_obs_step_8,
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)
    
@long_test
@require_muses_py
def test_omi_fm_object_creator_airs_omi(osp_dir, gmao_dir, vlidort_cli,
                                        clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r omi_fm_object_creator_airs_omi", shell=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rmi.register_with_muses_py()
    ihandle = OmiForwardModelHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False)
    rmi.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="omi_fm_object_creator_airs_omi")
    r.run_retrieval(vlidort_cli=vlidort_cli)

@long_test
@require_muses_py
def test_refractor_py_fm_airs_omi(osp_dir, gmao_dir, vlidort_cli,
                                        clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This uses our older RefractorOmiFm code. The point of this is that
    we've validated RefractorOmiFm against the original muses-py version.
    We expect the results of this to be nearly identical to our newer
    RefractorMusesIntegration version.'''
    subprocess.run("rm -r refractor_py_fm_airs_omi", shell=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rmi.register_with_muses_py()
    rfm = RefractorOmiFm(use_pca=False, use_lrad=False,
                         lrad_second_order=False)
    rfm.register_with_muses_py()
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="refractor_py_fm_airs_omi")
    r.run_retrieval(vlidort_cli=vlidort_cli)
    
@long_test
@require_muses_py
def test_compare_airs_omi(osp_dir, gmao_dir, vlidort_cli):
    '''Quick test to compare airs_omi runs. This assumes they are
    already done. This is just h5diff, but this figures out the path
    for each of the tests so we don't have to.'''
    for f in glob.glob("refractor_py_fm_airs_omi/*/Products/Products_L2*.nc"):
        f2 = f.replace("refractor_py_fm_airs_omi", "omi_fm_object_creator_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)
    for f in glob.glob("refractor_py_fm_airs_omi/*/Products/Products_Radiance*.nc"):
        f2 = f.replace("refractor_py_fm_airs_omi", "omi_fm_object_creator_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)
    for f in glob.glob("refractor_py_fm_airs_omi/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace("refractor_py_fm_airs_omi", "omi_fm_object_creator_airs_omi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)
        
# It isn't necessary to run this test regularly, the full runs above already
# do this. But if we need to debug an issue, it can be useful to do this
@skip
@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_run_forward_model_joint_omi(call_num,
                                     isolated_dir, osp_dir, gmao_dir,
                                     vlidort_cli):
    pfile = f"{joint_omi_test_in_dir}/run_forward_model_call_{call_num}.pkl"
    curdir = os.path.abspath(os.path.curdir)
    rrefractor = MusesForwardModelStep.load_forward_model_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    ihandle = OmiForwardModelHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False)
    rmi.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    (uip, o_radianceOut, o_jacobianOut) = rmi.run_forward_model(**rrefractor.params)

    # Compare against our older tropomi interface
    rmi2 = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rfm = RefractorOmiFm(use_pca=False, use_lrad=False,
                         lrad_second_order=False)
    rfm.register_with_muses_py()
    (uip2, o_radianceOut2, o_jacobianOut2) = rmi2.run_forward_model(**rrefractor.params)

    struct_compare(o_radianceOut, o_radianceOut2)
    struct_compare(o_jacobianOut, o_jacobianOut2)
    
