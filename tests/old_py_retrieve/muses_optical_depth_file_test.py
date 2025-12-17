import numpy as np
import numpy.testing as npt
from refractor.muses import MusesOpticalDepth
from refractor.old_py_retrieve_wrapper import MusesOpticalDepthFile
import refractor.framework as rf
from fixtures.require_check import require_muses_py, require_muses_py_fm
import pytest


class AbsorberVmrToUip(rf.CacheInvalidatedObserver):
    def __init__(self, rf_uip, pressure, absorber_vmr, species_name):
        super().__init__()
        self.rf_uip = rf_uip
        self.pressure = pressure
        self.absorber_vmr = absorber_vmr
        self.species_name = species_name
        self.absorber_vmr.add_cache_invalidated_observer(self)
        # Make sure data is synchronized initially
        self.invalidate_cache()

    def invalidate_cache(self):
        """Called when self.absorber_vmr changes"""
        # Get the VMR, and put into the right place in the UIP.
        # Note that the UIP always has just the VMR, so even if absorber_vmr
        # uses a state vector of log(vmr) or something like that we want
        # the UIP to have the VMR.
        # Note the UIP goes from surface to TOA, so we want
        # the DECREASING_PRESSURE order here.
        vgrid = self.absorber_vmr.vmr_grid(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        )
        self.rf_uip.atmosphere_column(self.species_name)[:] = vgrid.value
        self.cache_valid_flag = True


@pytest.mark.old_py_retrieve_test
@require_muses_py
@require_muses_py_fm
def test_muses_optical_depth_file(tropomi_fm_object_creator_step_1, osp_dir):
    obj_creator = tropomi_fm_object_creator_step_1
    # Don't look at temperature jacobian right now, it doesn't actually
    # work correctly and has been removed from the production strategy tables.
    # Our older test data has this in, but just remove it
    uip = obj_creator.ray_info.rf_uip
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
        uip.run_dir / "Step01_O3-Band3/vlidort/input",
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
        obj_creator.retrieval_config.input_file_helper,
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
