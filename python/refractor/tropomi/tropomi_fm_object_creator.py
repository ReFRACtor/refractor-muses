from functools import cached_property, lru_cache
from refractor.muses import (RefractorFmObjectCreator,
                             RefractorUip, 
                             O3Absorber, SwirAbsorber,
                             ForwardModelHandle,
                             MusesRaman, CurrentState, CurrentStateUip,
                             SurfaceAlbedo)
import refractor.framework as rf
import logging
import numpy as np
import re
import glob
import copy

logger = logging.getLogger("py-retrieve")

class TropomiSurfaceAlbedo(SurfaceAlbedo):
    def __init__(self, ground : rf.GroundWithCloudHandling, spec_index : int):
        self.ground = ground
        self.spec_index = spec_index
    def surface_albedo(self):
        # We just directly use the coefficients for the constant term. Could
        # do something more clever, but this is what py-retrieve does
        if(self.ground.do_cloud):
            return self.ground.ground_cloud.albedo_coefficients(self.spec_index)[0].value
        else:
            return self.ground.ground_clear.albedo_coefficients(self.spec_index)[0].value
            
class TropomiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(self, current_state : 'CurrentState',
                 measurement_id : 'MeasurementId',
                 observation : 'MusesObservation',
                 **kwargs):
        super().__init__(current_state, measurement_id, "TROPOMI", observation,
                         **kwargs)
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
        # Temp, until we get this all in place
        _ = self.forward_model
        self.add_to_sv(self.fm_sv)
        

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
            filter_name = self.filter_list[i]
            t = self.rf_uip.full_band_frequency(self.instrument_name)[self.rf_uip.mw_fm_slice(filter_name, self.instrument_name)]
            ref_wav = (t[0] + t[-1])/2
            selem = [f"TROPOMIRESSCALEO0{filter_name}",
                     f"TROPOMIRESSCALEO1{filter_name}",
                     f"TROPOMIRESSCALEO2{filter_name}"]
            coeff,mp = self.current_state.object_state(selem)
            rscale = rf.RadianceScalingSvMusesFit(coeff, rf.DoubleWithUnit(ref_wav,"nm"), filter_name)
            self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, [rscale,])
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
        band_reference = np.zeros(self.num_channels)
        selem = [ ] 
        for i in range(self.num_channels):
            filt_name = self.filter_list[i]
            if re.match(r'BAND\d$', filt_name) is not None:
                # This duplicates the calculation in
                # print_tropomi_surface_albedo.py in py_retrieve
                slc = self.rf_uip.mw_fm_slice(filt_name, self.instrument_name)
                wave_arr = self.rf_uip.full_band_frequency(self.instrument_name)[slc]
                band_reference[i] = (wave_arr[-1] + wave_arr[0]) / 2
                selem.extend([f"TROPOMISURFACEALBEDO{filt_name}",
                              f"TROPOMISURFACEALBEDOSLOPE{filt_name}",
                              f"TROPOMISURFACEALBEDOSLOPEORDER2{filt_name}"])
            else:
                raise RuntimeError("Don't recognize filter name")

        coeff,mp = self.current_state.object_state(selem)
        albedo[:, :] = np.reshape(coeff,albedo.shape)
        res = rf.GroundLambertian(albedo, rf.ArrayWithUnit(band_reference, "nm"),
                                  rf.Unit("nm"), self.filter_list, mp)
        self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, [res,])
        return res
        
    @cached_property
    def ground_cloud(self):
        albedo = np.zeros((self.num_channels, 1))
        which_retrieved = np.full((self.num_channels, 1), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)
        band_reference[:] = 1000
        selem = ["TROPOMICLOUDSURFACEALBEDO",]
        coeff,mp = self.current_state.object_state(selem)
        albedo[:,0] = coeff[0]
        res = rf.GroundLambertian(
            albedo, rf.ArrayWithUnit(band_reference, "nm"),
            ["Cloud",] * self.num_channels,
            mp)
        self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, [res,])
        return res

    @cached_property
    def cloud_fraction(self):
        selem = [f"TROPOMICLOUDFRACTION",]
        coeff,mp = self.current_state.object_state(selem)
        cf = rf.CloudFractionFromState(float(coeff[0]))
        self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, [cf,])
        return cf

    def add_to_sv(self, fm_sv : rf.StateVector):
        # TODO We have this hardcoded now. We'll rework this, adding to the state
        # vector should get moved into the object creation. But we'll have this in
        # place for now.
        if("O3" in self.current_state.fm_sv_loc):
            self.current_state.add_fm_state_vector_if_needed(
                fm_sv, ["O3",], [self.absorber.absorber_vmr("O3"),])
        for b in (3,):
            self.current_state.add_fm_state_vector_if_needed(
                fm_sv, [f"TROPOMIRINGSFBAND{b}",], [self.raman_effect(0),])

    @lru_cache(maxsize=None)
    def raman_effect(self, i):
        # Note we should probably look at this sample grid, and
        # make sure it goes RamanSioris.ramam_edge_wavenumber past
        # the edges of our spec_win. Also there isn't any particular
        # reason that the solar data/optical depth should be calculated
        # on the muses_fm_spectral_domain. But this is what muses-py
        # does, so we'll match that for now.
        if(self.filter_list[i] in ("BAND1", "BAND2", "BAND3")):
            scale_factor = self.uip_params[f"ring_sf_{self.filter_list[i]}"]
        elif(self.filter_list[i] in ("BAND7", "BAND8")):
            # JLL: The SWIR bands should not need to account for Raman scattering -
            # Vijay has never seen Raman scattering accounted for in the CO band.
            scale_factor = None
        else:
            raise RuntimeError("Unrecognized filter_list")
        if scale_factor is None:
            return None
        else:
            with self.observation.modify_spectral_window(do_raman_ext=True):
                wlen = self.observation.spectral_domain(i)
            # This is short if we aren't actually running this filter
            if(wlen.data.shape[0] < 2):
                return None
            salbedo = TropomiSurfaceAlbedo(self.ground, i)
            return MusesRaman(salbedo, self.rf_uip, self.instrument_name,
                              wlen,
                              float(scale_factor),
                              i,
                              self.sza_with_unit[i],
                              self.oza_with_unit[i],
                              self.raz_with_unit[i],
                              self.atmosphere,
                              self.solar_model(i),
                              rf.StateMappingLinear())


class TropomiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        self.measurement_id = None
        
    def notify_update_target(self, measurement_id : 'MeasurementId'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        self.measurement_id = measurement_id
        
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      **kwargs):
        if(instrument_name != "TROPOMI"):
            return None
        obj_creator = TropomiFmObjectCreator(current_state, self.measurement_id, obs,
                                             rf_uip=rf_uip_func(),
                                             fm_sv=fm_sv,
                                             **self.creator_kwargs)
        fm = obj_creator.forward_model
        logger.info(f"Tropomi Forward model\n{fm}")
        return fm

__all__ = ["TropomiFmObjectCreator", "TropomiForwardModelHandle"]
