try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from refractor.muses import (RefractorFmObjectCreator,
                             RefractorUip, 
                             O3Absorber, SwirAbsorber,
                             ForwardModelHandle,
                             MusesRaman, CurrentState)
import refractor.framework as rf
import logging
import numpy as np
import re
import glob
import copy

logger = logging.getLogger("py-retrieve")

class TropomiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(self, rf_uip : RefractorUip,
                 observation : 'MusesObservation', **kwargs):
        super().__init__(rf_uip, "TROPOMI", observation, **kwargs)
        unique_filters = set(self.filter_list)
        if len(unique_filters) != 1:
            raise NotImplementedError('Cannot handle multiple bands yet (requires different absorbers per band)')
        unique_filters = unique_filters.pop()
        if unique_filters == 'BAND3':
            self._inner_absorber = O3Absorber(self)
        elif unique_filters == 'BAND7':
            self._inner_absorber = SwirAbsorber(self)
        else:
            raise NotImplementedError(f'No absorber class defined for filter "{unique_filters}" on instrument {self.instruument_name}')
        

    @cached_property
    def instrument_correction(self):
        res = rf.vector_vector_instrument_correction()
        for i in range(self.num_channels):
            v = rf.vector_instrument_correction()
            v.push_back(self.radiance_scaling[i])
            res.push_back(v)
        return res
    
    def instrument_hwhm(self, ii_mw: int) -> rf.DoubleWithUnit:
        band_name = self.filter_list[ii_mw]
        if band_name == 'BAND7':
            # JLL: testing different values of HWHM with the IlsGrating component,
            # this value (= a 0.2 nm difference at 2330 nm) gave output spectra that
            # compared well with TROPOMI radiances. This is wider than the HWHM in
            # the Landgraf CO paper (doi: 10.5194/amt-9-4955-2016), which gives 0.25 nm
            # as the FULL width half max in Appendix B, but a HWHM of 0.25/2 nm didn't
            # compare as well. (NB in their Appendix B, there is a 0.1 nm FWHM, but that
            # is only part of the ISRF).
            return rf.DoubleWithUnit(0.36, 'cm^-1')
        else:
            raise NotImplementedError(f'HWHM for band {band_name} not defined')
    
    @cached_property
    def radiance_scaling(self):
        # By convention, the reference band is the middle of the full
        # frequency (see rev_and_fm_map
        res = []
        for i in range(self.num_channels):
            band_name = self.filter_list[i]
            t = self.rf_uip.full_band_frequency(self.instrument_name)[self.rf_uip.mw_fm_slice(i, self.instrument_name)]
            ref_wav = (t[0] + t[-1])/2
            coeff = [self.rf_uip.tropomi_params[f"resscale_O0_{band_name}"],
                     self.rf_uip.tropomi_params[f"resscale_O1_{band_name}"],
                     self.rf_uip.tropomi_params[f"resscale_O2_{band_name}"]]
            rscale = rf.RadianceScalingSvMusesFit(coeff, rf.DoubleWithUnit(ref_wav,"nm"), "BAND3")
            res.append(rscale)
        return res

    @property
    def uip_params(self):
        return self.rf_uip.tropomi_params

    @cached_property
    def temperature(self):
        tlev_fm = self.rf_uip.atmosphere_column("TATM")

        tlevel = rf.TemperatureLevel(tlev_fm, self.pressure_fm)
        t = rf.TemperatureLevelOffset(self.pressure_fm,
                                      tlevel.temperature_profile(),
                                      self.uip_params['temp_shift_BAND3'])
        return t

    @cached_property
    def ground_clear(self):
        albedo = np.zeros((self.num_channels, 3))
        which_retrieved = np.full((self.num_channels, 3), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)

        for i in range(self.num_channels):
            filt_name = self.filter_list[i]
            if re.match(r'BAND\d$', filt_name) is not None:
                # This duplicates the calculation in
                # print_tropomi_surface_albedo.py in py_retrieve
                slc = self.rf_uip.mw_fm_slice(i, self.instrument_name)
                wave_arr = self.rf_uip.full_band_frequency(self.instrument_name)[slc]
                band_reference[i] = (wave_arr[-1] + wave_arr[0]) / 2
                albedo[i, 0] = self.uip_params[f'surface_albedo_{filt_name}']
                albedo[i, 1] = self.uip_params[f'surface_albedo_slope_{filt_name}']
                albedo[i, 2] = self.uip_params[f'surface_albedo_slope_order2_{filt_name}']

                if(f'TROPOMISURFACEALBEDO{filt_name}' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[i, 0] = True
                if(f'TROPOMISURFACEALBEDOSLOPE{filt_name}' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[i, 1] = True
                if(f'TROPOMISURFACEALBEDOSLOPEORDER2{filt_name}' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[i, 2] = True
            else:
                raise RuntimeError("Don't recognize filter name")

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      rf.Unit("nm"),
                      self.filter_list,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
        
    @cached_property
    def ground_cloud(self):
        albedo = np.zeros((self.num_channels, 1))
        which_retrieved = np.full((self.num_channels, 1), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)
        band_reference[:] = 1000

        albedo[:,0] = self.uip_params['cloud_Surface_Albedo']
        if("TROPOMICLOUDSURFACEALBEDO" in self.rf_uip.state_vector_params(self.instrument_name)):
            which_retrieved[0,0] = True

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      ["Cloud",] * self.num_channels,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))

    @cached_property
    def cloud_fraction(self):
        # JLL: got a float32 for the cloud fraction in one case which made the C++ unhappy,
        # so force conversion to a double
        return rf.CloudFractionFromState(float(self.rf_uip.tropomi_cloud_fraction))

    def add_to_sv(self, current_state: CurrentState, fm_sv : rf.StateVector):
        # TODO We have this hardcoded now. We'll rework this, adding to the state
        # vector should get moved into the object creation. But we'll have this in
        # place for now.
        current_state.add_fm_state_vector_if_needed(
            fm_sv, ["TROPOMICLOUDFRACTION",], [self.cloud_fraction,])
        current_state.add_fm_state_vector_if_needed(
            fm_sv, ["O3",], [self.absorber.absorber_vmr("O3"),])
        current_state.add_fm_state_vector_if_needed(
            fm_sv, [f"TROPOMICLOUDSURFACEALBEDO",], [self.ground_cloud,])
        for b in (3,):
            current_state.add_fm_state_vector_if_needed(
                fm_sv, [f"TROPOMISURFACEALBEDOBAND{b}",
                        f"TROPOMISURFACEALBEDOSLOPEBAND{b}",
                        f"TROPOMISURFACEALBEDOSLOPEORDER2BAND{b}"], [self.ground_clear,])
            current_state.add_fm_state_vector_if_needed(
                fm_sv, [f"TROPOMIRINGSFBAND{b}",], [self.raman_effect[0],])
            current_state.add_fm_state_vector_if_needed(
                fm_sv, [f"TROPOMIRESSCALEO0BAND{b}",
                        f"TROPOMIRESSCALEO1BAND{b}",
                        f"TROPOMIRESSCALEO2BAND{b}"], self.radiance_scaling[0])

    @cached_property
    def state_vector_for_testing(self):
        '''Create a state vector for just this forward model. This is really
        meant more for unit tests, during normal runs CostFunctionCreator handles
        this (including the state vector element for other instruments).'''
        current_state = CurrentState(self.rf_uip)
        fm_sv = rf.StateVector()
        self.add_to_sv(current_state, fm_sv)
        fm_sv.observer_claimed_size = current_state.fm_state_vector_size
        return fm_sv

    @cached_property
    def raman_effect(self):
        # Note we should probably look at this sample grid, and
        # make sure it goes RamanSioris.ramam_edge_wavenumber past
        # the edges of our spec_win. Also there isn't any particular
        # reason that the solar data/optical depth should be calculated
        # on the muses_fm_spectral_domain. But this is what muses-py
        # does, so we'll match that for now.
        res = []
        for i in range(self.num_channels):
            if(self.filter_list[i] in ("BAND1", "BAND2", "BAND3")):
                scale_factor = self.uip_params[f"ring_sf_{self.filter_list[i]}"]
            elif(self.filter_list[i] in ("BAND7", "BAND8")):
                # JLL: The SWIR bands should not need to account for Raman scattering -
                # Vijay has never seen Raman scattering accounted for in the CO band.
                scale_factor = None
            else:
                raise RuntimeError("Unrecognized filter_list")
            if scale_factor is None:
                # JLL: As of 2023-11-02, it's important that there be an entry for each channel
                # in res, otherwise the indexing in the `spectrum_effect` method of `RefractorFmObjectCreator`
                # won't match. It might be better long term to store these in a dictionary with
                # the microwindows as keys, but need to make sure that this method isn't called anywhere
                # else first.
                res.append(None)
            else:
                res.append(MusesRaman(self.rf_uip, self.instrument_name,
                    self.rf_uip.rad_wavelength(i, self.instrument_name),
                    float(scale_factor),
                    i,
                    self.filter_list[i],
                    self.sza_with_unit[i],
                    self.oza_with_unit[i],
                    self.raz_with_unit[i],
                    self.atmosphere,
                    self.solar_model(i),
                    rf.StateMappingLinear()))
        return res


class TropomiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      spec_win : rf.SpectralWindowRange,
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      include_bad_sample=False,
                      **kwargs):
        if(instrument_name != "TROPOMI"):
            return None
        obj_creator = TropomiFmObjectCreator(rf_uip_func(), obs,
                                             include_bad_sample=include_bad_sample,
                                             **self.creator_kwargs)
        fm = obj_creator.forward_model
        obj_creator.add_to_sv(current_state, fm_sv)
        return fm

__all__ = ["TropomiFmObjectCreator", "TropomiForwardModelHandle"]
