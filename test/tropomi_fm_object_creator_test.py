import numpy as np
import numpy.testing as npt
from test_support import *
import refractor.framework as rf
from refractor.tropomi import TropomiFmObjectCreator

DEBUG = False


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_1(isolated_dir, osp_dir):
    """Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests"""
    rs, rstep, _ = set_up_run_to_location(
        tropomi_test_in_dir, 0, "retrieval input", include_ret_state=False
    )
    res = TropomiFmObjectCreator(
        rs.current_state(),
        rs.measurement_id,
        rs.observation_handle_set.observation(
            "TROPOMI",
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict["TROPOMI"],
            None,
            osp_dir=osp_dir,
        ),
        rf_uip_func=rs.strategy_executor.rf_uip_func_cost_function(False, None),
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_2(isolated_dir, osp_dir):
    """Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests"""
    rs, rstep, _ = set_up_run_to_location(
        tropomi_test_in_dir, 1, "retrieval input", include_ret_state=False
    )
    res = TropomiFmObjectCreator(
        rs.current_state(),
        rs.measurement_id,
        rs.observation_handle_set.observation(
            "TROPOMI",
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict["TROPOMI"],
            None,
            osp_dir=osp_dir,
        ),
        rf_uip_func=rs.strategy_executor.rf_uip_func_cost_function(False, None),
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


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


def test_ground_albedo(tropomi_fm_object_creator_step_1):
    """Test that the object creator reads the correct albedo
    parameters from the UIP for Band 3

    This is to test that changes to add new bands do not cause it to
    accidentally get the wrong values.
    """
    obj_albedo_coeffs = (
        tropomi_fm_object_creator_step_1.ground_clear.albedo_coefficients(0).value
    )
    uip = tropomi_fm_object_creator_step_1.rf_uip_func("TROPOMI")
    expected = [
        uip.tropomi_params["surface_albedo_BAND3"],  # 0.896 as of 2023-10-03
        uip.tropomi_params["surface_albedo_slope_BAND3"],  # 0.0 as of 2023-10-03
        uip.tropomi_params["surface_albedo_slope_order2_BAND3"],  # 0.0 as of 2023-10-03
    ]
    assert np.allclose(obj_albedo_coeffs, expected)

    # Now check the state mapping indices. Since none of the albedo terms are in step 1 of this UIP,
    # this should be an empty array.
    obj_state_map = (
        tropomi_fm_object_creator_step_1.ground_clear.state_mapping.retrieval_indexes
    )
    assert np.array_equal(obj_state_map, [])


def test_absorber(tropomi_fm_object_creator_step_1):
    assert "O3" == tropomi_fm_object_creator_step_1.absorber.gas_name(0)

    # JLL: nothing special about choosing 330 nm or the step size of 15 - the idea was just to (a) get
    # optical depths somewhat in the middle of the O3 window for Band 3 over a range of levels without
    # having to type too many check values. Weirdly the OD doesn't seem to change with wavelength, which
    # seems wrong. Test values gotten on 2023-10-04.
    optical_depths = (
        tropomi_fm_object_creator_step_1.absorber.optical_depth_each_layer(330.0, 0)
        .value[::15]
        .flatten()
    )
    expected = [
        3.88059557e-07,
        4.75033709e-06,
        1.35658586e-05,
        4.43398085e-06,
        8.64108049e-06,
    ]
    assert np.allclose(optical_depths, expected)


def test_vmr(tropomi_fm_object_creator_step_1):
    obj_vmrs = tropomi_fm_object_creator_step_1.absorber_vmr[0].vmr_profile
    uip_vmrs = tropomi_fm_object_creator_step_1.rf_uip_func(
        "TROPOMI"
    ).atmosphere_column("O3")
    assert np.allclose(obj_vmrs, uip_vmrs)


def test_atmosphere(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.atmosphere)


def test_radiative_transfer(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.radiative_transfer)


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
            data = {
                "name": o.name,
                "wavelength": o.spectral_domain.wavelength("nm"),
                "radiance": o.spectral_range.data,
            }
            pickle.dump(data, out)


def test_fm_run(tropomi_fm_object_creator_step_1):
    fm = tropomi_fm_object_creator_step_1.forward_model
    rf.write_shelve("fm.xml", fm)
    fm.add_observer_and_keep_reference(PrintSpectrum())
    print(fm.radiance(0, True).value)


def test_state_vector(tropomi_fm_object_creator_step_1):
    tropomi_fm_object_creator_step_1.fm_sv.update_state(
        tropomi_fm_object_creator_step_1.current_state.initial_guess_fm
    )
    print(tropomi_fm_object_creator_step_1.fm_sv)


def test_state_vector_step2(tropomi_fm_object_creator_step_2):
    tropomi_fm_object_creator_step_2.fm_sv.update_state(
        tropomi_fm_object_creator_step_2.current_state.initial_guess_fm
    )
    print(tropomi_fm_object_creator_step_2.fm_sv)


def test_raman_effect(tropomi_fm_object_creator_step_1):
    print(tropomi_fm_object_creator_step_1.raman_effect)


def test_forward_model_step2(tropomi_fm_object_creator_step_2):
    """Step 2, which has two microwindows"""
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


def test_species_basis(tropomi_fm_object_creator_step_2):
    uip = tropomi_fm_object_creator_step_2.rf_uip_func("TROPOMI")
    # Check that we are consistent with our species_basis_matrix
    # and atmosphere_retrieval_level_subset.
    npt.assert_allclose(
        uip.species_basis_matrix("O3"), uip.species_basis_matrix_calc("O3")
    )


def test_compare_altitude(tropomi_fm_object_creator_step_1):
    """Compare MuseAltitude and ReFRACtor altitude"""
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
        print(
            f"gravity {i}: {alt1.gravity(p[i]).value.value} {alt2.gravity(p[i]).value.value} diff: {(alt1.gravity(p[i]).value.value - alt2.gravity(p[i]).value.value) / alt1.gravity(p[i]).value.value * 100} %"
        )
        print(
            f"altitude {i}: {alt1.altitude(p[i]).value.value} {alt2.altitude(p[i]).value.value*1000} diff: {(alt1.altitude(p[i]).value.value - alt2.altitude(p[i]).value.value*1000) / max(alt1.altitude(p[i]).value.value,1) * 100} %"
        )
        gdifper.append(
            (alt1.gravity(p[i]).value.value - alt2.gravity(p[i]).value.value)
            / alt1.gravity(p[i]).value.value
            * 100
        )
        adifper.append(
            (alt1.altitude(p[i]).value.value - alt2.altitude(p[i]).value.value * 1000)
            / max(alt1.altitude(p[i]).value.value, 1)
            * 100
        )

    # Check that we are close. Gravity is almost identical, altitude varies a little more
    # near the top of the atmosphere but is still pretty close. These are percent differences
    assert np.abs(gdifper).max() < 0.005
    assert np.abs(adifper).max() < 0.55
