import numpy as np
import numpy.testing as npt

from test_support import *
import refractor.framework as rf
import glob
from refractor.muses import (MusesRunDir, CostFunctionCreator, CostFunction, 
                             RetrievalConfiguration, MeasurementIdFile)
from refractor.omi import (OmiFmObjectCreator, OmiForwardModelHandle)
from refractor.old_py_retrieve_wrapper import (RefractorMusesIntegration,
                                               RefractorOmiFm,MusesForwardModelStep,)
import subprocess

DEBUG = False

@pytest.fixture(scope="function")
def omi_fm_object_creator_step_1(isolated_dir, osp_dir):
    '''Fixture for OmiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rs, rstep, _ = set_up_run_to_location(omi_test_in_dir, 0, "retrieval input",
                                          include_result=False)
    res = OmiFmObjectCreator(
        rs.current_state(), rs.measurement_id,
        rs.observation_handle_set.observation(
            "OMI", rs.current_state(),
            rs.current_strategy_step.spectral_window_dict["OMI"],
            None,osp_dir=osp_dir), osp_dir=osp_dir)
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res

@pytest.fixture(scope="function")
def omi_fm_object_creator_step_2(isolated_dir, osp_dir):
    '''Fixture for OmiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rs, rstep, _ = set_up_run_to_location(omi_test_in_dir, 1, "retrieval input",
                                          include_result=False)
    res = OmiFmObjectCreator(
        rs.current_state(), rs.measurement_id,
        rs.observation_handle_set.observation(
            "OMI", rs.current_state(),
            rs.current_strategy_step.spectral_window_dict["OMI"],
            None,osp_dir=osp_dir), osp_dir=osp_dir)
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res

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

def test_state_vector(omi_fm_object_creator_step_1):
    omi_fm_object_creator_step_1.fm_sv.update_state(
        omi_fm_object_creator_step_1.current_state.initial_guess_fm)
    print(omi_fm_object_creator_step_1.fm_sv)

@require_muses_py
def test_state_vector_step2(omi_fm_object_creator_step_2):
    omi_fm_object_creator_step_2.fm_sv.update_state(
        omi_fm_object_creator_step_2.current_state.initial_guess_fm)
    print(omi_fm_object_creator_step_2.fm_sv)

@require_muses_py
def test_raman_effect(omi_fm_object_creator_step_1):
    print(omi_fm_object_creator_step_1.raman_effect)

@require_muses_py
def test_forward_model_step2(omi_fm_object_creator_step_2):
    '''Step 2, which has two microwindows'''
    print(omi_fm_object_creator_step_2.forward_model)

@require_muses_py
def test_fm_run_step2(omi_fm_object_creator_step_2):
    omi_fm_object_creator_step_2.use_pca = False
    omi_fm_object_creator_step_2.use_lrad = False
    omi_fm_object_creator_step_2.lrad_second_order = False
    fm = omi_fm_object_creator_step_2.forward_model

    spectrum_lidort = fm.radiance(0, True)

    fm = OmiFmObjectCreator(
        omi_fm_object_creator_step_2.current_state,
        omi_fm_object_creator_step_2.measurement_id,
        omi_fm_object_creator_step_2.rs.observation_handle_set.observation(
            "OMI", omi_fm_object_creator_step_2.current_state,
            omi_fm_object_creator_step_2.rs.current_strategy_step.spectral_window_dict["OMI"],
            None,osp_dir=omi_fm_object_creator_step_2.osp_dir),
        osp_dir=omi_fm_object_creator_step_2.osp_dir,
        use_pca=True, use_lrad=False, lrad_second_order=False).forward_model

    spectrum_pca = fm.radiance(0, True)

    npt.assert_allclose(spectrum_lidort.spectral_domain.data, spectrum_pca.spectral_domain.data, rtol=1e-10)
    npt.assert_allclose(spectrum_lidort.spectral_range.data, spectrum_pca.spectral_range.data, rtol=2e-2)

@require_muses_py
    
    
