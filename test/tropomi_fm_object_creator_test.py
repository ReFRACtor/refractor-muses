import numpy as np
import numpy.testing as npt
from test_support import *
import refractor.framework as rf
import glob
from refractor.tropomi import (TropomiFmObjectCreator, 
                               TropomiForwardModelHandle)
from refractor.muses import (MusesRunDir, CostFunctionCreator, CostFunction, 
                             CurrentStateUip, RetrievalConfiguration, MeasurementIdFile)
from refractor.old_py_retrieve_wrapper import (RefractorTropOmiFm, RefractorMusesIntegration,
                                               MusesForwardModelStep)
import subprocess

DEBUG = False


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_1(tropomi_uip_step_1, tropomi_obs_step_1, osp_dir):
    '''Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/tropomi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'TROPOMI' : ['BAND3']}
    mid = MeasurementIdFile(f"{test_base_path}/tropomi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    return TropomiFmObjectCreator(CurrentStateUip(tropomi_uip_step_1), mid,
                                  tropomi_obs_step_1,
                                  rf_uip_func=lambda instrument_name: tropomi_uip_step_1)

@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_2(tropomi_uip_step_2, tropomi_obs_step_2, osp_dir):
    '''Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests'''
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/tropomi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'TROPOMI' : ['BAND3']}
    mid = MeasurementIdFile(f"{test_base_path}/tropomi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    return TropomiFmObjectCreator(CurrentStateUip(tropomi_uip_step_2), mid,
                                  tropomi_obs_step_2,
                                  rf_uip_func=lambda **kwargs:tropomi_uip_step_2)



def test_spec_win(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.spec_win)

def test_ils_params_postconv(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.ils_params_postconv(0))

def test_ils_params_fastconv(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.ils_params_fastconv(0))
    
def test_spectrum_sampling(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.spectrum_sampling)

def test_instrument(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.instrument)

def test_ground_albedo(tropomi_fm_object_creator_step_1, tropomi_uip_step_1):
    """Test that the object creator reads the correct albedo
    parameters from the UIP for Band 3

    This is to test that changes to add new bands do not cause it to
    accidentally get the wrong values.
    """
    obj_albedo_coeffs = tropomi_fm_object_creator_step_1.ground_clear.albedo_coefficients(0).value
    expected = [
        tropomi_uip_step_1.tropomi_params['surface_albedo_BAND3'], # 0.896 as of 2023-10-03
        tropomi_uip_step_1.tropomi_params['surface_albedo_slope_BAND3'], # 0.0 as of 2023-10-03
        tropomi_uip_step_1.tropomi_params['surface_albedo_slope_order2_BAND3'], # 0.0 as of 2023-10-03
    ]
    assert np.allclose(obj_albedo_coeffs, expected)

    # Now check the state mapping indices. Since none of the albedo terms are in step 1 of this UIP,
    # this should be an empty array.
    obj_state_map = tropomi_fm_object_creator_step_1.ground_clear.state_mapping.retrieval_indexes
    assert np.array_equal(obj_state_map, [])




def test_absorber(tropomi_fm_object_creator_step_1):
    assert 'O3' == tropomi_fm_object_creator_step_1.absorber.gas_name(0)

    # JLL: nothing special about choosing 330 nm or the step size of 15 - the idea was just to (a) get
    # optical depths somewhat in the middle of the O3 window for Band 3 over a range of levels without
    # having to type too many check values. Weirdly the OD doesn't seem to change with wavelength, which
    # seems wrong. Test values gotten on 2023-10-04.
    optical_depths = tropomi_fm_object_creator_step_1.absorber.optical_depth_each_layer(330.0, 0).value[::15].flatten()
    expected = [
        3.88059557e-07,
        4.75033709e-06,
        1.35658586e-05,
        4.43398085e-06,
        8.64108049e-06]
    assert np.allclose(optical_depths, expected)


def test_vmr(tropomi_fm_object_creator_step_1, tropomi_uip_step_1):
    obj_vmrs = tropomi_fm_object_creator_step_1.absorber_vmr[0].vmr_profile
    uip_vmrs = tropomi_uip_step_1.atmosphere_column('O3')
    assert np.allclose(obj_vmrs, uip_vmrs)


@require_muses_py
def test_atmosphere(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.atmosphere)

@require_muses_py
def test_radiative_transfer(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.radiative_transfer)


@require_muses_py
def test_forward_model(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.forward_model)


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
def test_fm_run(tropomi_fm_object_creator_step_1):
    fm = tropomi_fm_object_creator_step_1.forward_model
    rf.write_shelve("fm.xml", fm)
    fm.add_observer_and_keep_reference(PrintSpectrum())
    print(fm.radiance(0, True).value)


def test_state_vector(tropomi_fm_object_creator_step_1, tropomi_uip_step_1):
    tropomi_fm_object_creator_step_1.fm_sv.update_state(tropomi_uip_step_1.current_state_x_fm)
    print(tropomi_fm_object_creator_step_1.fm_sv)


@require_muses_py
def test_state_vector_step2(tropomi_fm_object_creator_step_2, tropomi_uip_step_2):
    tropomi_fm_object_creator_step_2.fm_sv.update_state(tropomi_uip_step_2.current_state_x_fm)
    print(tropomi_fm_object_creator_step_2.fm_sv)


@require_muses_py
def test_raman_effect(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.raman_effect)

@require_muses_py
def test_forward_model_step2(tropomi_fm_object_creator_step_2):
    '''Step 2, which has two microwindows'''
    print("Start of test, ignore valgrind errors before this", flush=True)
    fmodel = tropomi_fm_object_creator_step_2.forward_model
    print(fmodel)
    atm = tropomi_fm_object_creator_step_2.underlying_forward_model.radiative_transfer.atmosphere
    # This use to be a bug, we fixed this in framework so it works now
    if True:
        # This combination causes an use to cause an invalid read error with
        # valgrind. This is fixed now, but leave test here to demonstrate this
        # is fixed.
        #
        # BTW, to run with valgrind do something like;
        # PYTHONMALLOC=malloc valgrind --track-origins=yes --suppressions=valgrind-python.supp $(which python) $(which pytest) -s test/tropomi_fm_object_creator_test.py -k test_forward_model_step2
        #
        # The PYTHONMALLOC is important, see https://stackoverflow.com/questions/20112989/how-to-use-valgrind-with-python.
        # Without PYTHONMALLOC you will get a zillion valgrind errors.
        # The valgrind-python.supp comes from python source code.
        #
        # There are a number of errors unrelated to our code (triggered by
        # __mpn_construct_long_double). I think this is numpy or scipy. In
        # any case, ignore errors before the message "Start of test", they
        # aren't ours
        absorber = atm.absorber
    else:
        # This is an alternative which didn't cause the read error initially
        # (and of course still doesn't)
        absorber = tropomi_fm_object_creator_step_2.absorber
    
    
@require_muses_py
def test_species_basis(tropomi_uip_step_2):
    # Check that we are consistent with our species_basis_matrix
    # and atmosphere_retrieval_level_subset.
    npt.assert_allclose(tropomi_uip_step_2.species_basis_matrix("O3"),
                        tropomi_uip_step_2.species_basis_matrix_calc("O3"))


@require_muses_py
def test_residual_fm_jac_tropomi(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                                 joint_tropomi_obs_step_12):
    '''Test out the CostFunction residual_fm_jacobian using our forward model. Note
    that this just tests that we can make the call, to debug any problems there. The
    actual comparison on results is done in full run tests below.'''
    step_number = 12
    iteration = 2
    
    curdir = os.path.curdir
    rrefractor = muses_residual_fm_jac(joint_tropomi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = CostFunctionCreator()
    ihandle = TropomiForwardModelHandle(use_pca=False, use_lrad=False,
                                      lrad_second_order=False)
    creator.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    obslist = joint_tropomi_obs_step_12
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{joint_tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    flist = {'TROPOMI' : ['BAND3']}
    mid = MeasurementIdFile(f"{joint_tropomi_test_in_dir}/Measurement_ID.asc",
                            rconf, flist)
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(rf_uip, obslist,
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)


def test_compare_altitude(tropomi_fm_object_creator_step_1):
    '''Compare MuseAltitude and ReFRACtor altitude'''
    alt1 = tropomi_fm_object_creator_step_1.altitude_muses[0]
    alt2 = tropomi_fm_object_creator_step_1.altitude_refractor[0]
    p = tropomi_fm_object_creator_step_1.pressure.pressure_grid()
    print(alt1.gravity(p[0]).units.name)
    print(alt2.gravity(p[0]).units.name)
    print(alt1.altitude(p[0]).units.name)
    print(alt2.altitude(p[0]).units.name)
    gdifper = []
    adifper = []
    for i in range(p.rows):
        print(f"gravity {i}: {alt1.gravity(p[i]).value.value} {alt2.gravity(p[i]).value.value} diff: {(alt1.gravity(p[i]).value.value - alt2.gravity(p[i]).value.value) / alt1.gravity(p[i]).value.value * 100} %")
        print(f"altitude {i}: {alt1.altitude(p[i]).value.value} {alt2.altitude(p[i]).value.value*1000} diff: {(alt1.altitude(p[i]).value.value - alt2.altitude(p[i]).value.value*1000) / max(alt1.altitude(p[i]).value.value,1) * 100} %")
        gdifper.append((alt1.gravity(p[i]).value.value - alt2.gravity(p[i]).value.value) / alt1.gravity(p[i]).value.value * 100)
        adifper.append((alt1.altitude(p[i]).value.value - alt2.altitude(p[i]).value.value*1000) / max(alt1.altitude(p[i]).value.value,1) * 100)

    # Check that we are close. Gravity is almost identical, altitude varies a little more
    # near the top of the atmosphere but is still pretty close. These are percent differences
    assert np.abs(gdifper).max() < 0.005
    assert np.abs(adifper).max() < 0.55
    
