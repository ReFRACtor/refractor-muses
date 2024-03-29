try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from refractor.muses import (RefractorFmObjectCreator,
                             RefractorUip, ForwardModelHandle,
                             MusesRaman)
import refractor.framework as rf
import os
import logging
import numpy as np
import h5py
import copy
# for netCDF3 support which is not supported in h5py
from netCDF4 import Dataset

logger = logging.getLogger("py-retrieve")

class OmiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(self, rf_uip : RefractorUip,
                 observation : 'MusesObservation',
                 use_eof=False, eof_dir=None,
                 **kwargs):
        super().__init__(rf_uip, "OMI", observation, **kwargs)
        self.use_eof = use_eof
        self.eof_dir = eof_dir
        
    @cached_property
    def instrument_correction(self):
        res = rf.vector_vector_instrument_correction()
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            v = rf.vector_instrument_correction()
            if(self.use_eof):
                for e in self.eof[self.rf_uip.filter_name(ii_mw)]:
                    v.push_back(e)
            res.push_back(v)
        return res

    @cached_property
    def eof(self):
        res = {}
        for band in ("UV1", "UV2"):
            selem = self.state_info.state_element(f"OMIEOF{band}")
            r = []
            # This is the full instrument size of the given band. We only run
            # the forward model on a subset of these, but the eof is in terms of
            # the full instrument size. Note the values outside of what we actually
            # run the forward model on don't really matter - we don't end up using
            # them. We can set these to zero if desired.
            npixel = np.count_nonzero(self.rf_uip.uip_omi["frequencyfilterlist"] == band)
            # Note: npixel is 577 for UV2, which doesn't make sense given indexes in EOF
            # Muses uses fullbandfrequency subset by microwindow info (shown below) instead

            if(self.eof_dir is not None):
                uv1_index = 0
                uv2_index = 0
                uv_basename = "EOF_xtrack_{0}-{1:02d}_window_{0}.nc"
                uv1_fname = f"{os.path.join(self.eof_dir, uv_basename.format('uv1', uv1_index))}"
                uv2_fname = f"{os.path.join(self.eof_dir, uv_basename.format('uv2', uv2_index))}"

                if band == "UV1":
                    eof_fname = uv1_fname
                    mw_index = 0
                elif band == "UV2":
                    eof_fname = uv2_fname
                    mw_index = 1

                # Note: We hit this code when retrieving cloud fraction which uses 7 freq. and 1 microwin
                if len(self.rf_uip.uip_omi["microwindows"]) == 1:
                    res["UV2"] = []
                    continue

                # Enabling the code below leads to Out of range error in FullPhysics::NLLSMaxAPosteriori::residual()
                # startmw = self.rf_uip.uip_omi["microwindows"][mw_index]["startmw"][mw_index]
                # endmw = self.rf_uip.uip_omi["microwindows"][mw_index]["enddmw"][mw_index]
                # wave_arr =self.rf_uip.uip_omi["fullbandfrequency"][startmw:endmw+1]
                # npixel = len(wave_arr)

                eof_path = "/eign_vector"
                eof_index_path = "/Index"
                with Dataset(eof_fname) as eof_ds:
                    eofs = eof_ds[eof_path][:]
                    pixel_indexes = eof_ds[eof_index_path][:]
            
                for basis_index in range(selem.number_eof):
                    wform_data = np.zeros(npixel)
                    for eof_index, pixel_index in enumerate(pixel_indexes):
                        wform_data[pixel_index] = eofs[basis_index,eof_index]
                    # Not sure about the units, but this is what we assigned to our
                    # forward model, and the EOF needs to match
                    wform = rf.ArrayWithUnit(wform_data, rf.Unit("sr^-1"))
                    r.append(rf.EmpiricalOrthogonalFunction(selem.value[basis_index], wform, basis_index+1, band))
                res[band] = r
            else:
                # If we don't have EOF data, use all zeros. This should go
                # away, this is just so we can start using the EOF before
                # we have all the data sorted out
                wform = rf.ArrayWithUnit(np.zeros(npixel), rf.Unit("sr^-1"))
                for i in range(selem.number_eof):
                    r.append(rf.EmpiricalOrthogonalFunction(selem.value[i], wform, i+1, band))               
                res[band] = r
        return res
        
    
    @cached_property
    def solar_reference_filename(self):
        return os.path.join(self.input_dir, "OMI_Solar/omisol_v003_avg_nshi_backup.h5")


    @cached_property
    def omi_solar_model(self):
        '''We read a 3 year average solar file HDF file for omi. This
        duplicates what mpy.read_omi does, which is then stored in the pickle
        file that solar_model uses.'''
        f = h5py.File(self.solar_reference_filename, "r")
        res = []
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            ind = self.rf_uip.across_track_indexes(ii_mw, self.instrument_name)[0]
            wav_vals = f[f"WAV_{self.filter_name[fm_idx]}"][:, ind]
            irad_vals = f[f"SOL_{self.filter_name[fm_idx]}"][:, ind]
            one_au = 149597870691
            irad_vals *= (one_au / self.rf_uip.earth_sun_distance(self.instrument_name)) ** 2
            # File does not have units contained within it
            # Same units as the OMI L1B files, but use irradiance units
            sol_domain = rf.SpectralDomain(wav_vals, rf.Unit("nm"))
            sol_range = rf.SpectralRange(irad_vals, rf.Unit("ph / nm / s"))
            sol_spec = rf.Spectrum(sol_domain, sol_range)
            ref_spec = rf.SolarReferenceSpectrum(sol_spec, None)
            res.append(ref_spec)
        return res

    def instrument_hwhm(self, ii_mw: int) -> rf.DoubleWithUnit:
        band_name = self.rf_uip.filter_name(ii_mw)
        raise NotImplementedError(f'HWHM for band {band_name} not defined')
        
    @property
    def uip_params(self):
        return self.rf_uip.omi_params
    
    @cached_property
    def ground_clear(self):
        albedo = np.zeros((self.num_channel, 3))
        which_retrieved = np.full((self.num_channel, 3), False, dtype=bool)
        band_reference = np.zeros(self.num_channel)

        for fm_idx, ii_mw in enumerate(self.channel_list()):
            if(self.filter_name[fm_idx] == "UV1"):
                band_reference[fm_idx] = (315 + 262) / 2.0
                albedo[fm_idx, 0] = self.uip_params['surface_albedo_uv1']
                if('OMISURFACEALBEDOUV1' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[fm_idx, 0] = True
            elif(self.filter_name[fm_idx] == "UV2"):
                # Note this value is hardcoded in print_omi_surface_albedo
                band_reference[fm_idx] = 320.0
                albedo[fm_idx, 0] = self.uip_params['surface_albedo_uv2']
                albedo[fm_idx, 1] = self.uip_params['surface_albedo_slope_uv2']
                if('OMISURFACEALBEDOUV2' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[fm_idx, 0] = True
                if('OMISURFACEALBEDOSLOPEUV2' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[fm_idx, 1] = True
            else:
                raise RuntimeError("Don't recognize filter name")

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      rf.Unit("nm"),
                      self.filter_name,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
                    
    @cached_property
    def cloud_fraction(self):
        if(self.state_info is not None):
            cfrac = self.state_info.state_element("OMICLOUDFRACTION").value[0]
            return rf.CloudFractionFromState(cfrac)
        else:
            return rf.CloudFractionFromState(self.rf_uip.omi_cloud_fraction)

    @cached_property
    def state_vector_for_testing(self):
        '''Create a state vector for just this forward model. This is really
        meant more for unit tests, during normal runs CostFunctionCreator handles
        this (including the state vector element for other instruments).'''
        svhandle = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        svhandle.add_handle(OmiStateVectorHandle(self))
        sv = svhandle.create_state_vector(self.rf_uip)
        if sv.observer_claimed_size != len(self.rf_uip.current_state_x_fm):
            raise RuntimeError(f"Number of state vector elements {sv.observer_claimed_size} does not match number of expected MUSES jacobians parameters {len(self.rf_uip.current_state_x_fm)}")

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
            if(self.filter_name[fm_idx] in ("UV1", "UV2")):
                scale_factor = self.uip_params[f"ring_sf_{str.lower(self.filter_name[fm_idx])}"]
            else:
                raise RuntimeError("Unrecognized filter_name")
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
    
#class OmiStateVectorHandle(StateVectorHandle):
class OmiStateVectorHandle:
    def __init__(self, obj_creator):
        self.obj_creator = obj_creator

    def add_sv(self, sv, species_name, ptart, plen, **kwargs):
        if(species_name == "OMICLOUDFRACTION"):
            sv.add_observer(self.obj_creator.cloud_fraction)
        elif(species_name == "O3"):
            sv.add_observer(self.obj_creator.absorber.absorber_vmr("O3"))
        elif(species_name in ("OMISURFACEALBEDOUV1",
                              "OMISURFACEALBEDOUV2",
                              "OMISURFACEALBEDOSLOPEUV2"
                              )):
            self.add_sv_once(sv, self.obj_creator.ground_clear)
        elif(species_name in ("OMINRADWAVUV1",
                              "OMINRADWAVUV2",
                              "OMIODWAVUV1",
                              'OMIODWAVUV2',
                              "OMIODWAVSLOPEUV1",
                              "OMIODWAVSLOPEUV2",
                              )):
            #self.add_sv_once(sv, self.obj_creator.observation)
            pass
        elif(species_name == "OMIEOFUV1"):
            for eof in self.obj_creator.eof["UV1"]:
                sv.add_observer(eof)
        elif(species_name == "OMIEOFUV2"):
            for eof in self.obj_creator.eof["UV2"]:
                sv.add_observer(eof)
        else:
            # Didn't recognize the species_name, so we didn't handle this
            return False
        return True

class OmiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def forward_model(self, instrument_name, rf_uip, obs, svhandle,
                      include_bad_sample=False,
                      **kwargs):
        if(instrument_name != "OMI"):
            return None
        obj_creator = OmiFmObjectCreator(rf_uip, obs, include_bad_sample=include_bad_sample,
                                         **self.creator_kwargs)
        svhandle.add_handle(OmiStateVectorHandle(obj_creator),
                            priority_order=100)
        return obj_creator.forward_model

__all__ = ["OmiFmObjectCreator", "OmiForwardModelHandle"]
    
