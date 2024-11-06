from functools import cached_property, lru_cache
from refractor.muses import (RefractorFmObjectCreator,
                             RefractorUip, 
                             ForwardModelHandle,
                             MusesRaman, CurrentState, CurrentStateUip,
                             SurfaceAlbedo)
from refractor.muses import muses_py as mpy
import refractor.framework as rf
from loguru import logger
import numpy as np
import re
import glob
import copy
import os

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
                 use_raman=True,
                 **kwargs):
        super().__init__(current_state, measurement_id, "TROPOMI", observation,
                         use_raman=use_raman, **kwargs)

    @cached_property
    def instrument_correction(self):
        res = rf.vector_vector_instrument_correction()
        for i in range(self.num_channels):
            v = rf.vector_instrument_correction()
            v.push_back(self.radiance_scaling[i])
            res.push_back(v)
        return res

    def ils_params_preconv(self, sensor_index : int):
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        num_fwhm_srf=4.0
        with self.observation.modify_spectral_window(include_bad_sample = True,
                                                     do_raman_ext = True):
            sindex = self.observation.spectral_domain(sensor_index).sample_index - 1
        return mpy.get_tropomi_ils(
            self.osp_dir,
            self.observation.frequency_full(sensor_index), sindex,
            self.observation.wavelength_filter, 
            self.observation.observation_table,
            num_fwhm_srf)

    def ils_params_postconv(self, sensor_index : int):
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        # Place holder, this doesn't work yet. Copy of what is
        # done in make_uip_tropomi.
        num_fwhm_srf=4.0

        # Kind of a convoluted way to just get the monochromatic list, but this is what
        # make_uip_tropomi does, so we duplicate it here. We have the observation frequencies,
        # and the monochromatic for all the windows added into one long list, and then give
        # the index numbers that are actually relevant for the ILS. 
        mono_list, mono_filter_list, mono_list_length = self.observation.spectral_window.muses_monochromatic()
        wn_list = np.concatenate([self.observation.frequency_full(sensor_index), mono_list],
                                 axis=0)
        wn_filter = np.concatenate([self.observation.wavelength_filter, mono_filter_list],
                                   axis=0)
        startmw_fm = len(self.observation.frequency_full(sensor_index)) + sum(mono_list_length[:sensor_index])
        
        sindex = np.arange(0,mono_list_length[sensor_index]) + startmw_fm
        return mpy.get_tropomi_ils(self.osp_dir, wn_list, sindex, wn_filter,
                                   self.observation.observation_table,
                                   num_fwhm_srf)
    
    def ils_params_fastconv(self, sensor_index : int):
        # Note, I'm not sure of fastconv actually works. This fails in muses-py if
        # we try to use fastconv. From the code, I *think* this is what was intended,
        # but I don't know if this was actually tested anywhere since you can't actuall
        # run this. But put this in here for completeness.
        
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        num_fwhm_srf=4.0 
        with self.observation.modify_spectral_window(include_bad_sample = True,
                                                     do_raman_ext = True):
            sindex = self.observation.spectral_domain(sensor_index).sample_index - 1
        # Kind of a convoluted way to just get the monochromatic list, but this is what
        # make_uip_tropomi does, so we duplicate it here. We have the observation frequencies,
        # and the monochromatic for all the windows added into one long list, and then give
        # the index numbers that are actually relevant for the ILS. 
        mono_list, mono_filter_list, mono_list_length = self.observation.spectral_window.muses_monochromatic()
        wn_list = np.concatenate([self.observation.frequency_full(sensor_index), mono_list],
                                 axis=0)
        wn_filter = np.concatenate([self.observation.wavelength_filter, mono_filter_list],
                                   axis=0)
        startmw_fm = len(self.observation.frequency_full(sensor_index)) + sum(mono_list_length[:sensor_index])
        
        sindex2 = np.arange(0,mono_list_length[sensor_index]) + startmw_fm
        
        return mpy.get_tropomi_ils_fastconv.get_tropomi_ils_fastconv(
            self.osp_dir, wn_list, sindex, wn_filter,
            self.observation.observation_table,
            num_fwhm_srf,
            i_monochromfreq=wn_list[sindex2],
            i_interpmethod="INTERP_MONOCHROM")

    def ils_params(self, sensor_index : int):
        '''ILS parameters'''
        if self.ils_method(sensor_index) == "APPLY":
            return self.ils_params_preconv(sensor_index)
        elif (self.ils_method(sensor_index) == "POSTCONV"):
            return self.ils_params_postconv(sensor_index)
        elif (self.ils_method(sensor_index) == "FASTCONV"):
            return self.ils_params_fastconv(sensor_index)
        else:
            raise RuntimeError(f"Unrecognized ils_method {self.ils_method(sensor_index)}")

    def ils_method(self, sensor_index : int) -> str:
        '''Return the ILS method to use. This is APPLY, POSTCONV, or FASTCONV'''
        # Note in principle we could have this be a function of the sensor band,
        # however the current implementation just has one value set here.
        #
        # This is currently the same for all sensor_index values. We can extend this
        # if needed, but we also need to change the absorber to handle this, since the
        # selection is based off the ils_method. We could probably wrap the different
        # absorber types and select which one is used by some kind of logic.
        #
        # We really need a test case to work through the logic here before changing this
        return self.measurement_id["ils_tropomi_xsection"]
    
    def instrument_hwhm(self, sensor_index: int) -> rf.DoubleWithUnit:
        band_name = self.filter_list[sensor_index]
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

    def reference_wavelength(self, sensor_index : int):
        '''Calculate the reference wavelength used in couple of places. By convention, this
        is the middle of the band used for MusesRaman, including bad samples.'''
        with self.observation.modify_spectral_window(do_raman_ext=True,
                                                     include_bad_sample=True):
            t = self.observation.spectral_domain(sensor_index).data
            return (t[0] + t[-1])/2
    
    @cached_property
    def radiance_scaling(self):
        # By convention, the reference band is the middle of the full
        # frequency (see rev_and_fm_map
        res = []
        for i in range(self.num_channels):
            filter_name = self.filter_list[i]
            
            selem = [f"TROPOMIRESSCALEO0{filter_name}",
                     f"TROPOMIRESSCALEO1{filter_name}",
                     f"TROPOMIRESSCALEO2{filter_name}"]
            coeff,mp = self.current_state.object_state(selem)
            rscale = rf.RadianceScalingSvMusesFit(coeff, rf.DoubleWithUnit(self.reference_wavelength(i),"nm"), filter_name)
            self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, [rscale,])
            res.append(rscale)
        return res

    @cached_property
    def temperature(self):
        tlev_fm, _ = self.current_state.object_state(["TATM",])
        selem = ["TROPOMITEMPSHIFTBAND3",]
        coeff, _ = self.current_state.object_state(selem)
        tlevel = rf.TemperatureLevel(tlev_fm, self.pressure_fm)
        t = rf.TemperatureLevelOffset(self.pressure_fm,
                                      tlevel.temperature_profile(),
                                      coeff[0])
        self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem,
                                                         [t,])        
        return t

    @cached_property
    def ground_clear(self):
        albedo = np.zeros((self.num_channels, 3))
        band_reference = np.zeros(self.num_channels)
        selem = [ ] 
        for i in range(self.num_channels):
            filt_name = self.filter_list[i]
            if re.match(r'BAND\d$', filt_name) is not None:
                band_reference[i] = self.reference_wavelength(i)
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

    @lru_cache(maxsize=None)
    def raman_effect(self, i):
        # Note we should probably look at this sample grid, and
        # make sure it goes RamanSioris.ramam_edge_wavenumber past
        # the edges of our spec_win. Also there isn't any particular
        # reason that the solar data/optical depth should be calculated
        # on the muses_fm_spectral_domain. But this is what muses-py
        # does, so we'll match that for now.
        selem = [f"TROPOMIRINGSF{self.filter_list[i]}",]
        if(self.filter_list[i] in ("BAND1", "BAND2", "BAND3")):
            coeff,mp = self.current_state.object_state(selem)
            scale_factor = float(coeff[0])
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
            ram = MusesRaman(salbedo, self.ray_info,
                             wlen,
                             float(scale_factor),
                             i,
                             rf.DoubleWithUnit(self.sza[i], "deg"),
                             rf.DoubleWithUnit(self.oza[i], "deg"),
                             rf.DoubleWithUnit(self.raz[i], "deg"),
                             self.atmosphere,
                             self.solar_model(i),
                             rf.StateMappingLinear())
            self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, [ram,])
            return ram


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
                      rf_uip_func : "Optional(Callable[{instrument:None}, RefractorUip])",
                      **kwargs):
        if(instrument_name != "TROPOMI"):
            return None
        obj_creator = TropomiFmObjectCreator(current_state, self.measurement_id, obs,
                                             rf_uip_func=rf_uip_func,
                                             fm_sv=fm_sv,
                                             **self.creator_kwargs)
        fm = obj_creator.forward_model
        logger.info(f"Tropomi Forward model\n{fm}")
        return fm

__all__ = ["TropomiFmObjectCreator", "TropomiForwardModelHandle"]
