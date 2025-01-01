import numpy as np
import numpy.testing as npt
from refractor.muses import MusesOpticalDepthFile, AbsorberVmrToUip, MusesOpticalDepth
from refractor.tropomi import TropomiFmObjectCreator
from test_support import *
import refractor.framework as rf


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_2(isolated_dir, osp_dir):
    """Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests"""
    rs, rstep, _ = set_up_run_to_location(
        tropomi_test_in_dir, 1, "retrieval input", include_ret_state=False
    )
    os.chdir(rs.run_dir)
    uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)("TROPOMI")
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
        rf_uip_func=lambda instrument: uip,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


def test_muses_optical_depth_file(tropomi_fm_object_creator_step_2, osp_dir):
    obj_creator = tropomi_fm_object_creator_step_2
    # Don't look at temperature jacobian right now, it doesn't actually
    # work correctly and has been removed from the production strategy tables.
    # Our older test data has this in, but just remove it
    uip = tropomi_fm_object_creator_step_2.rf_uip_func("TROPOMI")
    uip.uip_tropomi["jacobians"] = [
        "O3",
    ]
    mod = MusesOpticalDepthFile(
        obj_creator.ray_info,
        obj_creator.pressure,
        obj_creator.temperature,
        obj_creator.altitude,
        obj_creator.absorber_vmr,
        obj_creator.num_channels,
        "./Step01_O3-Band3/vlidort/input",
    )
    print(mod)
    mod2 = MusesOpticalDepth(
        obj_creator.pressure,
        obj_creator.temperature,
        obj_creator.altitude,
        obj_creator.absorber_vmr,
        obj_creator.observation,
        [
            obj_creator.ils_params(0),
        ],
        osp_dir,
    )

    print(mod2)
    sv = rf.StateVector()
    obj_creator.fm_sv.remove_observer(obj_creator.absorber_vmr[0])
    sv.add_observer(obj_creator.absorber_vmr[0])
    sv_val = []
    sv_val = np.log(uip.atmosphere_column("O3"))
    sv.update_state(sv_val)
    # Make sure update to sv gets reflected in UIP
    uip.refractor_cache["atouip_O3"] = AbsorberVmrToUip(
        uip, obj_creator.pressure, obj_creator.absorber_vmr[0], "O3"
    )

    # Pick a wavenumber in the forward model, but not at the edge. Pretty
    # arbitrary, we are just looking for "a typical wave number".
    spec_index = 0
    wn = obj_creator.underlying_forward_model.spectral_domain(spec_index).data[10]
    mod_a = mod.total_air_number_density_layer(0).value.value
    mod2_a = mod2.total_air_number_density_layer(0).value.value
    # We have a bit of a larger difference with some of the layers that have
    # a smaller value. So compare against the max value, just to have a reasonable
    # comparision
    aden_diff = (mod_a - mod2_a) / mod_a.max() * 100
    assert np.abs(aden_diff).max() < 0.2
    mod_g = mod.gas_number_density_layer(0).value.value
    mod2_g = mod2.gas_number_density_layer(0).value.value
    gden_diff = (mod_g - mod2_g) / mod_g * 100
    # Actually have a few larger differences, although most of them are pretty similar.
    # Not sure of the reason, but refractor is probably the more accurate.
    assert np.abs(gden_diff).max() < 13.0
    assert np.median(np.abs(gden_diff)) < 0.1
    mod_od = mod.optical_depth_each_layer(wn, spec_index).value
    mod2_od = mod2.optical_depth_each_layer(wn, spec_index).value
    od_diff = (mod_od - mod2_od) / mod_od * 100
    # Same here, since the gden_diff is larger
    assert np.abs(od_diff).max() < 13.0
    assert np.median(np.abs(od_diff)) < 1.0
    svinitial = np.copy(sv.state)
    odinitial = mod.optical_depth_each_layer(wn, spec_index).value
    od2initial = mod2.optical_depth_each_layer(wn, spec_index).value
    jac = mod.optical_depth_each_layer(wn, spec_index).jacobian
    jac2 = mod2.optical_depth_each_layer(wn, spec_index).jacobian
    fdjac = np.zeros((odinitial.shape[0], 1, svinitial.shape[0]))
    fdjac2 = np.zeros((odinitial.shape[0], 1, svinitial.shape[0]))
    delta = 0.0001
    for i in range(svinitial.shape[0]):
        svtemp = np.copy(svinitial)
        svtemp[i] += delta
        sv.update_state(svtemp)
        od = mod.optical_depth_each_layer(wn, spec_index).value
        fdjac[:, :, i] = (od - odinitial) / delta
        od2 = mod2.optical_depth_each_layer(wn, spec_index).value
        fdjac2[:, :, i] = (od2 - od2initial) / delta
    npt.assert_allclose(jac, fdjac, atol=4e-6)
    # Refractor is a more accurate jacobian
    npt.assert_allclose(jac2, fdjac2, atol=2e-9)
