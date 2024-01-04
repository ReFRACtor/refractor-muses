import numpy as np
import numpy.testing as npt

from refractor.tropomi import (TropomiFmObjectCreator, RefractorTropOmiFm,
                               TropomiInstrumentHandle)
from test_support import *
import refractor.framework as rf
import glob
from refractor.muses import (RefractorMusesIntegration, MusesRunDir,
                             FmObsCreator, CostFunction, MusesForwardModelStep)
import subprocess

DEBUG = False


def test_spec_win(tropomi_uip_step_1):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.spec_win)

def test_spectrum_sampling(tropomi_uip_step_1):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.spectrum_sampling)


def test_instrument(tropomi_uip_step_1):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.instrument)


def test_band3_ground_albedo(tropomi_uip_step_1):
    """Test that the object creator reads the correct albedo parameters from the UIP for Band 3

    This is to test that changes to add new bands do not cause it to accidentally get the wrong values.
    """
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    obj_albedo_coeffs = obj_creator.ground_clear.albedo_coefficients(0).value
    expected = [
        tropomi_uip_step_1.tropomi_params['surface_albedo_BAND3'], # 0.896 as of 2023-10-03
        tropomi_uip_step_1.tropomi_params['surface_albedo_slope_BAND3'], # 0.0 as of 2023-10-03
        tropomi_uip_step_1.tropomi_params['surface_albedo_slope_order2_BAND3'], # 0.0 as of 2023-10-03
    ]
    assert np.allclose(obj_albedo_coeffs, expected)

    # Now check the state mapping indices. Since none of the albedo terms are in step 1 of this UIP,
    # this should be an empty array.
    obj_state_map = obj_creator.ground_clear.state_mapping.retrieval_indexes
    assert np.array_equal(obj_state_map, [])


def test_band7_ground_albedo(tropomi_uip_band7_swir_step):
    """Test that the object creator reads the correct albedo parameters from the UIP for Band 7

    This is to test that changes to add new bands do not cause it to accidentally get the wrong values.
    """
    uip = tropomi_uip_band7_swir_step
    obj_creator = TropomiFmObjectCreator(uip)
    obj_albedo_coeffs = obj_creator.ground_clear.albedo_coefficients(0).value
    expected = [
        uip.tropomi_params['surface_albedo_BAND7'], # 0.00169 as of 2023-10-03
        uip.tropomi_params['surface_albedo_slope_BAND7'], # 0.0 as of 2023-10-03
        uip.tropomi_params['surface_albedo_slope_order2_BAND7'], # 0.0 as of 2023-10-03
    ]
    assert np.allclose(obj_albedo_coeffs, expected)

    # Now check the state mapping indices. Since all three of the albedo terms are in step 1 of this UIP,
    # this should be an array with indices 0 to 2.
    obj_state_map = obj_creator.ground_clear.state_mapping.retrieval_indexes
    assert np.array_equal(obj_state_map, [0, 1, 2])


def test_band3_absorber(tropomi_uip_step_1):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    assert 'O3' == obj_creator.absorber.gas_name(0)

    # JLL: nothing special about choosing 330 nm or the step size of 15 - the idea was just to (a) get
    # optical depths somewhat in the middle of the O3 window for Band 3 over a range of levels without
    # having to type too many check values. Weirdly the OD doesn't seem to change with wavelength, which
    # seems wrong. Test values gotten on 2023-10-04.
    optical_depths = obj_creator.absorber.optical_depth_each_layer(330.0, 0).value[::15].flatten()
    expected = [
        3.43152898e-7,
        4.70869935e-6,
        1.34748075e-05,
        4.38287067e-06,
        8.47041057e-06,
    ]
    assert np.allclose(optical_depths, expected)


def test_band7_absorber(tropomi_uip_band7_swir_step, osp_dir):
    # The osp_dir fixture is just to ensure that the OSP environmental variable is set 
    # since this needs the ABSCO tables and the creator can only infer their location
    # if it's in a sounding directory with "OSP" linked in the parent directory.
    obj_creator = TropomiFmObjectCreator(tropomi_uip_band7_swir_step)

    # JLL: I chose these values of pressure, temperature, and H2O VMR
    # because they are points in the ABSCO table that I can just extract
    # to compare against what the absorber returns without any interpolation
    # (constant value or otherwise). The 2330 nm (= 4291.8 cm-1) is the middle
    # of the SWIR band we're interested in.
    test_pres = rf.DoubleWithUnit(1000.024, 'mbar')
    test_temp = rf.DoubleWithUnit(310.0, 'K')
    test_h2o = rf.ArrayWithUnit(np.array([10.0e-6]), 'mol/mol')
    test_freq = 1. / 2330.0e-7  # convering 2330 nm -> cm-1
    expected_xsec = {
        'CO': 5.005110392346992e-22,
        'CH4': 2.9172434973940954e-22,
        'H2O': 2.6598953312203934e-25,
        'HDO': 6.214096110477619e-23
    }
    
    for igas in range(4):
        gas = obj_creator.absorber.gas_name(igas)
        obj_xsec = obj_creator.absorber.gas_absorption(gas).absorption_cross_section(
            test_freq, test_pres, test_temp, test_h2o
        ).value
        assert np.isclose(obj_xsec, expected_xsec[gas]), f'{gas} xsec does not match'


def test_band3_vmr(tropomi_uip_step_1):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    obj_vmrs = obj_creator.absorber_vmr[0].vmr_profile
    uip_vmrs = tropomi_uip_step_1.atmosphere_column('O3')
    assert np.allclose(obj_vmrs, uip_vmrs)


def test_band7_vmr(tropomi_uip_band7_swir_step):
    uip = tropomi_uip_band7_swir_step
    obj_creator = TropomiFmObjectCreator(uip)
    for i, name in enumerate(uip.atm_params('TROPOMI')['species']):
        obj_vmrs = obj_creator.absorber_vmr[i].vmr_profile
        uip_vmrs = uip.atmosphere_column(name)
        assert np.allclose(obj_vmrs, uip_vmrs), f'{name} VMRs differ in the object creator and UIP'


def test_band7_ils_simple(tropomi_uip_band7_swir_step, tropomi_band7_simple_ils_test_data):
    uip = tropomi_uip_band7_swir_step
    obj_creator = TropomiFmObjectCreator(uip)
    inner_ils_obj = obj_creator.instrument.ils(0)

    nchan = inner_ils_obj.sample_grid().pixel_grid.data.size
    test_conv_spec = inner_ils_obj.apply_ils(
        tropomi_band7_simple_ils_test_data['hi_res_freq'],
        tropomi_band7_simple_ils_test_data['hi_res_spec'],
        list(range(nchan))
    )

    # JLL: I checked a plot of the differences, and there is some structure, but they are about 1000x
    # smaller than the test radiances. I think this is okay, but we could revist this if needed.
    assert np.allclose(test_conv_spec, tropomi_band7_simple_ils_test_data['convolved_spec'], rtol=1e-3)
    

@require_muses_py
def test_atmosphere(tropomi_uip_step_1, clean_up_replacement_function):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.atmosphere)


@require_muses_py
def test_radiative_transfer(tropomi_uip_step_1, clean_up_replacement_function):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.radiative_transfer)


@require_muses_py
def test_forward_model(tropomi_uip_step_1, clean_up_replacement_function):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.forward_model)


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
def test_fm_run(tropomi_uip_step_1, clean_up_replacement_function):
    fm = TropomiFmObjectCreator(tropomi_uip_step_1).forward_model
    fm.add_observer_and_keep_reference(PrintSpectrum())
    print(fm.radiance(0, True).value)


def test_state_vector(tropomi_uip_step_1):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.state_vector_for_testing)


@require_muses_py
def test_state_vector_step2(tropomi_uip_step_2, clean_up_replacement_function):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_2)
    print(obj_creator.state_vector_for_testing)


@require_muses_py
def test_raman_effect(tropomi_uip_step_1, clean_up_replacement_function):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_1)
    print(obj_creator.raman_effect)

@require_muses_py
def test_forward_model_step2(tropomi_uip_step_2, clean_up_replacement_function):
    '''Step 2, which has two microwindows'''
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_2)
    fmodel = obj_creator.forward_model
    print(fmodel)
    atm = obj_creator.underlying_forward_model.radiative_transfer.atmosphere
    if True:
        # This combination causes an invalid read error with valgrind. Leave
        # this off and it doesn't, something about this combination causes
        # the problem. The issue seems to be the round tripping through
        # swig. With valgrind, we get a "Invalid read" when we attempt to
        # read data from freeded memory.
        #
        # BTW, to run with valgrind do something like;
        # PYTHONMALLOC=malloc valgrind $(which python) $(which pytest) -s test/tropomi_fm_object_creator_test.py -k test_forward_model_step2
        #
        # The PYTHONMALLOC is important, see https://stackoverflow.com/questions/20112989/how-to-use-valgrind-with-python.
        # Without PYTHONMALLOC you will get a zillion valgrind errors.
        absorber = atm.absorber
    else:
        # This on the other hand is fine.
        #
        # We should perhaps find a fix for this at some point, but swig
        # with directors combined with shared_ptr has had issues. Perhaps a
        # newer version of swig will fix this, but for now we can just
        # work around this by keeping a copy of the object we pass to C++
        # in python, and just using that version.
        absorber = obj_creator.absorber
    
    
@require_muses_py
def test_species_basis(tropomi_uip_step_2, clean_up_replacement_function):
    obj_creator = TropomiFmObjectCreator(tropomi_uip_step_2)
    # Check that we are consistent with our species_basis_matrix
    # and atmosphere_retrieval_level_subset.
    npt.assert_allclose(obj_creator.rf_uip.species_basis_matrix("O3"),
                        obj_creator.rf_uip.species_basis_matrix_calc("O3"))
    


@require_muses_py
def test_residual_fm_jac_tropomi(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
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
    creator = FmObsCreator()
    ihandle = TropomiInstrumentHandle(use_pca=False, use_lrad=False,
                                      lrad_second_order=False)
    creator.instrument_handle_set.add_handle(ihandle, priority_order=100)
    cfunc = CostFunction(*creator.fm_and_obs(rf_uip,
                                             rrefractor.params["ret_info"],
                                             vlidort_cli=vlidort_cli))
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)

@long_test
@require_muses_py
def test_tropomi_fm_object_creator_cris_tropomi(osp_dir, gmao_dir, vlidort_cli,
                                        clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r tropomi_fm_object_creator_cris_tropomi", shell=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rmi.register_with_muses_py()
    ihandle = TropomiInstrumentHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False)
    rmi.instrument_handle_set.add_handle(ihandle, priority_order=100)
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="tropomi_fm_object_creator_cris_tropomi")
    r.run_retrieval(vlidort_cli=vlidort_cli)

@long_test
@require_muses_py
def test_refractor_py_fm_cris_tropomi(osp_dir, gmao_dir, vlidort_cli,
                                        clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This uses our older RefractorTropOmiFm code. The point of this is that
    we've validated RefractorOmiFm against the original muses-py version.
    We expect the results of this to be nearly identical to our newer
    RefractorMusesIntegration version.'''
    subprocess.run("rm -r refractor_py_fm_cris_tropomi", shell=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rmi.register_with_muses_py()
    rfm = RefractorTropOmiFm(use_pca=False, use_lrad=False,
                         lrad_second_order=False)
    rfm.register_with_muses_py()
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="refractor_py_fm_cris_tropomi")
    r.run_retrieval(vlidort_cli=vlidort_cli)
    
@long_test
@require_muses_py
def test_compare_cris_tropomi(osp_dir, gmao_dir, vlidort_cli):
    '''Quick test to compare cris_tropomi runs. This assumes they are
    already done. This is just h5diff, but this figures out the path
    for each of the tests so we don't have to.'''
    for f in glob.glob("refractor_py_fm_cris_tropomi/*/Products/Products_L2*.nc"):
        f2 = f.replace("refractor_py_fm_cris_tropomi", "tropomi_fm_object_creator_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)
    for f in glob.glob("refractor_py_fm_cris_tropomi/*/Products/Products_Radiance*.nc"):
        f2 = f.replace("refractor_py_fm_cris_tropomi", "tropomi_fm_object_creator_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)
    for f in glob.glob("refractor_py_fm_cris_tropomi/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace("refractor_py_fm_cris_tropomi", "tropomi_fm_object_creator_cris_tropomi")
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)
    
