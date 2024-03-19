try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from refractor.muses import (RefractorFmObjectCreator,
                             RefractorUip, StateVectorHandle,
                             O3Absorber, SwirAbsorber,
                             StateVectorHandleSet,
                             InstrumentHandle,
                             MusesRaman)
import refractor.framework as rf
import logging
import numpy as np
import re
import glob
import copy

logger = logging.getLogger("py-retrieve")

class TropomiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(self, rf_uip : RefractorUip, **kwargs):
        super().__init__(rf_uip, "TROPOMI", **kwargs)
        unique_filters = set(self.filter_name)
        if len(unique_filters) != 1:
            raise NotImplementedError('Cannot handle multiple bands yet (requires different absorbers per band)')
        unique_filters = unique_filters.pop()
        if unique_filters == 'BAND3':
            self._inner_absorber = O3Absorber(self)
        elif unique_filters == 'BAND7':
            self._inner_absorber = SwirAbsorber(self)
        else:
            raise NotImplementedError(f'No absorber class defined for filter "{unique_filters}" on instrument {self.instruument_name}')
        

    # This will go away in a bit
    @property
    def observation(self):
        return None

    @cached_property
    def instrument_correction(self):
        res = rf.vector_vector_instrument_correction()
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            v = rf.vector_instrument_correction()
            v.push_back(self.radiance_scaling[fm_idx])
            res.push_back(v)
        return res
    
    def instrument_hwhm(self, ii_mw: int) -> rf.DoubleWithUnit:
        band_name = self.rf_uip.filter_name(ii_mw)
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
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            band_name = self.rf_uip.filter_name(ii_mw)
            t = self.rf_uip.full_band_frequency(self.instrument_name)[self.rf_uip.mw_fm_slice(fm_idx, self.instrument_name)]
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
        albedo = np.zeros((self.num_channel, 3))
        which_retrieved = np.full((self.num_channel, 3), False, dtype=bool)
        band_reference = np.zeros(self.num_channel)

        for fm_idx, ii_mw in enumerate(self.channel_list()):
            filt_name = self.filter_name[fm_idx]
            if re.match(r'BAND\d$', filt_name) is not None:
                # This duplicates the calculation in
                # print_tropomi_surface_albedo.py in py_retrieve
                slc = self.rf_uip.mw_fm_slice(fm_idx, self.instrument_name)
                wave_arr = self.rf_uip.full_band_frequency(self.instrument_name)[slc]
                band_reference[fm_idx] = (wave_arr[-1] + wave_arr[0]) / 2
                albedo[fm_idx, 0] = self.uip_params[f'surface_albedo_{filt_name}']
                albedo[fm_idx, 1] = self.uip_params[f'surface_albedo_slope_{filt_name}']
                albedo[fm_idx, 2] = self.uip_params[f'surface_albedo_slope_order2_{filt_name}']

                if(f'TROPOMISURFACEALBEDO{filt_name}' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[fm_idx, 0] = True
                if(f'TROPOMISURFACEALBEDOSLOPE{filt_name}' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[fm_idx, 1] = True
                if(f'TROPOMISURFACEALBEDOSLOPEORDER2{filt_name}' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[fm_idx, 2] = True
            else:
                raise RuntimeError("Don't recognize filter name")

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      rf.Unit("nm"),
                      self.filter_name,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
        
    @cached_property
    def ground_cloud(self):
        albedo = np.zeros((self.num_channel, 1))
        which_retrieved = np.full((self.num_channel, 1), False, dtype=bool)
        band_reference = np.zeros(self.num_channel)
        band_reference[:] = 1000

        albedo[:,0] = self.uip_params['cloud_Surface_Albedo']
        if("TROPOMICLOUDSURFACEALBEDO" in self.rf_uip.state_vector_params(self.instrument_name)):
            which_retrieved[0,0] = True

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      ["Cloud",] * self.num_channel,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))

    @cached_property
    def cloud_fraction(self):
        # JLL: got a float32 for the cloud fraction in one case which made the C++ unhappy,
        # so force conversion to a double
        return rf.CloudFractionFromState(float(self.rf_uip.tropomi_cloud_fraction))

    @cached_property
    def state_vector_for_testing(self):
        '''Create a state vector for just this forward model. This is really
        meant more for unit tests, during normal runs CostFuncCreator handles
        this (including the state vector element for other instruments).'''
        svhandle = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        svhandle.add_handle(TropomiStateVectorHandle(self))
        sv = svhandle.create_state_vector(self.rf_uip,
                                          self.use_full_state_vector)
        if(not self.use_full_state_vector):
            if sv.observer_claimed_size != len(self.rf_uip.current_state_x):
                raise RuntimeError(f"Number of state vector elements {sv.observer_claimed_size} does not match number of expected MUSES jacobians parameters {len(self.rf_uip.current_state_x)}")
        else:
            if sv.observer_claimed_size != len(self.rf_uip.current_state_x_fm):
                raise RuntimeError(f"Number of state vector elements {sv.observer_claimed_size} does not match number of expected MUSES jacobians parameters {len(self.rf_uip.current_state_x_fm)}")

        if(not self.use_full_state_vector):
            sv.update_state(self.rf_uip.current_state_x)
        else:
            sv.update_state(self.rf_uip.current_state_x_fm)
        logger.info(f"Created ReFRACtor state vector:\n{sv}")
        return sv

    @cached_property
    def raman_effect(self):
        # Note we should probably look at this sample grid, and
        # make sure it goes RamanSioris.ramam_edge_wavenumber past
        # the edges of our spec_win. Also there isn't any particular
        # reason that the solar data/optical depth should be calculated
        # on the muses_fm_spectral_domain. But this is what muses-py
        # does, so we'll match that for now.
        res = []
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            if(self.filter_name[fm_idx] in ("BAND1", "BAND2", "BAND3")):
                scale_factor = self.uip_params[f"ring_sf_{self.filter_name[fm_idx]}"]
            elif(self.filter_name[fm_idx] in ("BAND7", "BAND8")):
                # JLL: The SWIR bands should not need to account for Raman scattering -
                # Vijay has never seen Raman scattering accounted for in the CO band.
                scale_factor = None
            else:
                raise RuntimeError("Unrecognized filter_name")
            if scale_factor is None:
                # JLL: As of 2023-11-02, it's important that there be an entry for each channel
                # in res, otherwise the indexing in the `spectrum_effect` method of `RefractorFmObjectCreator`
                # won't match. It might be better long term to store these in a dictionary with
                # the microwindows as keys, but need to make sure that this method isn't called anywhere
                # else first.
                res.append(None)
            else:
                res.append(MusesRaman(self.rf_uip, self.instrument_name,
                    self.rf_uip.rad_wavelength(fm_idx, self.instrument_name),
                    float(scale_factor),
                    fm_idx,
                    ii_mw,
                    self.sza_with_unit[fm_idx],
                    self.oza_with_unit[fm_idx],
                    self.raz_with_unit[fm_idx],
                    self.atmosphere,
                    self.solar_model(fm_idx),
                    rf.StateMappingLinear()))
        return res


class TropomiStateVectorHandle(StateVectorHandle):
    def __init__(self, obj_creator):
        self.obj_creator = obj_creator

    def add_sv(self, sv, species_name, ptart, plen, **kwargs):
        if(species_name == "TROPOMICLOUDFRACTION"):
            sv.add_observer(self.obj_creator.cloud_fraction)
        elif(species_name in ("O3", "CO", "CH4", "H2O", "HDO")):
            # MMS, Right now we only deal with O3. Silently ignore the other
            # species, we'll presumably get those in place in the future
            if(species_name == "O3"):
                sv.add_observer(self.obj_creator.absorber.absorber_vmr(species_name))
        elif(species_name.startswith(("TROPOMISURFACEALBEDOBAND", "TROPOMISURFACEALBEDOSLOPEBAND", "TROPOMISURFACEALBEDOSLOPEORDER2BAND"))):
            # JLL: this should match any band's albedo variables.
            self.add_sv_once(sv, self.obj_creator.ground_clear)
        elif(species_name.startswith(("TROPOMISOLARSHIFTBAND", "TROPOMIRADIANCESHIFTBAND", "TROPOMIRADSQUEEZEBAND"))):
            # JLL: this should match any band's radiance/irradiance shift/squeeze variables.
            #self.add_sv_once(sv, self.obj_creator.observation)
            pass
        elif(species_name.startswith("TROPOMIRINGSFBAND")):
            # JLL: this should match any band's ring scale factor
            sv.add_observer(self.obj_creator.raman_effect[0])
        elif(species_name.startswith("TROPOMITEMPSHIFTBAND")):
            # JLL: this should match any band's temperature shift
            sv.add_observer(self.obj_creator.temperature)
        elif(species_name == "TROPOMICLOUDSURFACEALBEDO"):
            sv.add_observer(self.obj_creator.ground_cloud)
        elif(species_name.startswith(("TROPOMIRESSCALEO0BAND", "TROPOMIRESSCALEO1BAND", "TROPOMIRESSCALEO2BAND"))):
            # JLL: this should match any band's radiance scaling
            self.add_sv_once(sv, self.obj_creator.radiance_scaling[0])
        else:
            # Didn't recognize the species_name, so we didn't handle this
            return False
        return True
    
class TropomiInstrumentHandle(InstrumentHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, svhandle,
                   use_full_state_vector=False, include_bad_sample=False, **kwargs):
        if(instrument_name != "TROPOMI"):
            return (None, None)
        obj_creator = TropomiFmObjectCreator(rf_uip, use_full_state_vector=use_full_state_vector, include_bad_sample=include_bad_sample, **self.creator_kwargs)
        svhandle.add_handle(TropomiStateVectorHandle(obj_creator),
                            priority_order=100)
        return (obj_creator.forward_model, obj_creator.observation)

__all__ = ["TropomiFmObjectCreator", "TropomiInstrumentHandle"]
