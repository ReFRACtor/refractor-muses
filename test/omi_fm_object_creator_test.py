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
                              rf_uip_func=lambda **kwargs: omi_uip_step_1)

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
                              rf_uip_func=lambda **kwargs: omi_uip_step_2)

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
                            rf_uip_func=lambda **kwargs: omi_uip_step_2, use_pca=True, use_lrad=False,
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
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/omi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'OMI' : ['UV1', 'UV2']}
    mid = MeasurementIdFile(f"{test_base_path}/omi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(rf_uip, joint_omi_obs_step_8,
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)
    
    
