import numpy as np
import numpy.testing as npt
from test_support import *
import refractor.framework as rf
from refractor.tropomi import TropomiSwirFmObjectCreator


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_swir_step(isolated_dir, josh_osp_dir):
    """Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests"""
    rs, rstep, _ = set_up_run_to_location(
        tropomi_band7_test_in_dir, 0, "retrieval input", include_ret_state=False
    )
    res = TropomiSwirFmObjectCreator(
        rs.current_state(),
        rs.measurement_id,
        rs.observation_handle_set.observation(
            "TROPOMI",
            rs.current_state(),
            rs.current_strategy_step.spectral_window_dict["TROPOMI"],
            None,
            osp_dir=josh_osp_dir,
        ),
        rf_uip_func=rs.strategy_executor.rf_uip_func_cost_function(False, None),
        osp_dir=josh_osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


def test_ground_albedo(tropomi_fm_object_creator_swir_step):
    """Test that the object creator reads the correct albedo
    parameters from the UIP for Band 7

    This is to test that changes to add new bands do not cause it to
    accidentally get the wrong values.
    """
    uip = tropomi_fm_object_creator_swir_step.rf_uip_func("TROPOMI")
    obj_albedo_coeffs = (
        tropomi_fm_object_creator_swir_step.ground_clear.albedo_coefficients(0).value
    )
    expected = [
        uip.tropomi_params["surface_albedo_BAND7"],  # 0.00169 as of 2023-10-03
        uip.tropomi_params["surface_albedo_slope_BAND7"],  # 0.0 as of 2023-10-03
        uip.tropomi_params["surface_albedo_slope_order2_BAND7"],  # 0.0 as of 2023-10-03
    ]
    assert np.allclose(obj_albedo_coeffs, expected)

    # Now check the state mapping indices. Since all three of the albedo terms are in step 1 of this UIP,
    # this should be an array with indices 0 to 2.
    obj_state_map = (
        tropomi_fm_object_creator_swir_step.ground_clear.state_mapping.retrieval_indexes
    )
    assert np.array_equal(obj_state_map, [0, 1, 2])


def test_absorber(tropomi_fm_object_creator_swir_step):
    # JLL: I chose these values of pressure, temperature, and H2O VMR
    # because they are points in the ABSCO table that I can just extract
    # to compare against what the absorber returns without any interpolation
    # (constant value or otherwise). The 2330 nm (= 4291.8 cm-1) is the middle
    # of the SWIR band we're interested in.
    test_pres = rf.DoubleWithUnit(1000.024, "mbar")
    test_temp = rf.DoubleWithUnit(310.0, "K")
    test_h2o = rf.ArrayWithUnit(np.array([10.0e-6]), "mol/mol")
    test_freq = 1.0 / 2330.0e-7  # convering 2330 nm -> cm-1
    expected_xsec = {
        "CO": 5.005110392346992e-22,
        "CH4": 2.9172434973940954e-22,
        "H2O": 2.6598953312203934e-25,
        "HDO": 6.214096110477619e-23,
    }

    for igas in range(4):
        gas = tropomi_fm_object_creator_swir_step.absorber.gas_name(igas)
        obj_xsec = (
            tropomi_fm_object_creator_swir_step.absorber.gas_absorption(gas)
            .absorption_cross_section(test_freq, test_pres, test_temp, test_h2o)
            .value
        )
        assert np.isclose(obj_xsec, expected_xsec[gas]), f"{gas} xsec does not match"


def test_vmr(tropomi_fm_object_creator_swir_step):
    uip = tropomi_fm_object_creator_swir_step.rf_uip_func("TROPOMI")
    for i, name in enumerate(tropomi_fm_object_creator_swir_step.absorption_gases):
        obj_vmrs = tropomi_fm_object_creator_swir_step.absorber_vmr[i].vmr_profile
        uip_vmrs = uip.atmosphere_column(name)
        assert np.allclose(
            obj_vmrs, uip_vmrs
        ), f"{name} VMRs differ in the object creator and UIP"


# Doesn't currently work, I think a spectral window is missing here
@skip
def test_ils_simple(
    tropomi_fm_object_creator_swir_step, tropomi_band7_simple_ils_test_data
):
    inner_ils_obj = tropomi_fm_object_creator_swir_step.instrument.ils(0)

    nchan = inner_ils_obj.sample_grid().pixel_grid.data.size
    test_conv_spec = inner_ils_obj.apply_ils(
        tropomi_band7_simple_ils_test_data["hi_res_freq"],
        tropomi_band7_simple_ils_test_data["hi_res_spec"],
        list(range(nchan)),
    )

    # JLL: I checked a plot of the differences, and there is some structure, but they are about 1000x
    # smaller than the test radiances. I think this is okay, but we could revist this if needed.
    npt.assert_allclose(
        test_conv_spec, tropomi_band7_simple_ils_test_data["convolved_spec"], rtol=1e-2
    )
