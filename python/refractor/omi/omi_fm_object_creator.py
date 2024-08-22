from functools import cached_property, lru_cache
from refractor.muses import (RefractorFmObjectCreator, ForwardModelHandle,
                             MusesRaman, SurfaceAlbedo)
import refractor.framework as rf
import os
import logging
import numpy as np
import h5py
import copy
# for netCDF3 support which is not supported in h5py
from netCDF4 import Dataset

logger = logging.getLogger("py-retrieve")

class OmiSurfaceAlbedo(SurfaceAlbedo):
    def __init__(self, ground : rf.GroundWithCloudHandling, spec_index : int):
        self.ground = ground
        self.spec_index = spec_index
        
    def surface_albedo(self):
        # We just directly use the coefficients for the constant term. Could
        # do something more clever, but this is what py-retrieve does
        if(self.ground.do_cloud):
            # TODO Reevaluate using a fixed value here
            # py-retrieve returns a hard coded value. Not sure why we don't
            # just use the cloud albedo, but for now match the old code
            #return self.ground_cloud.coefficient[0].value
            return 0.80
        else:
            return self.ground.ground_clear.albedo_coefficients(self.spec_index)[0].value

class OmiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(self, current_state : 'CurrentState',
                 measurement_id : 'MeasurementId',
                 observation : 'MusesObservation',
                 use_eof=False, eof_dir=None,
                 **kwargs):
        super().__init__(current_state, measurement_id, "OMI", observation,
                         **kwargs)
        self.use_eof = use_eof
        self.eof_dir = eof_dir
        # Temp, until we get this updated to use current_state
        if(hasattr(self.rf_uip, "state_info")):
            self.state_info = self.rf_uip.state_info
        else:
            self.state_info = None
        # Temp, until we get this all in place
        self.add_to_sv(self.fm_sv)
        
    @cached_property
    def instrument_correction(self):
        res = rf.vector_vector_instrument_correction()
        for i in range(self.num_channels):
            v = rf.vector_instrument_correction()
            if(self.use_eof):
                if self.observation.filter_list[i] in self.eof:
                    for e in self.eof[self.observation.filter_list[i]]:
                        v.push_back(e)
            res.push_back(v)
        return res

    # This should go away, I think we can get this from the observation. Curently just
    # used by eof below
    def channel_list(self):
        '''This is list of microwindows relevant to self.instrument_name

        Note that there are two microwindow indexes floating around. We have
        ii_mw which goes through all the instruments, so for step 7 in
        AIRS+OMI ii_mw goes through 12 values (only 10 and 11 are OMI).
        mw_index (also call fm_idx) is relative to a instrument,
        so if we are working with OMI the first microwindow has ii_mw = 10, but
        mw_index is 0 (UV1, with the second UV2).
        
        The contents of channel_list() are ii_mw (e.g., 10 and 11 in our 
        AIRS+OMI example), and the index into channel_list() is fm_idx
        (also called mw_index). So you might loop with something like:

        for fm_idx, ii_mw in enumerate(self.channel_list()):
             blah blah
        '''
        chan_list = []
        for ii_mw in range(self.rf_uip.number_micro_windows):
            if self.rf_uip.instrument_name(ii_mw) == self.instrument_name:
                chan_list.append(ii_mw)

        return chan_list

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
                uv1_index, uv2_index, _uv2_pair_index = self.rf_uip.omi_obs_table['XTRACK']
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

                eof_path = "/eign_vector"
                eof_index_path = "/Index"
                with Dataset(eof_fname) as eof_ds:
                    eofs = eof_ds[eof_path][:]
                    pixel_indexes = eof_ds[eof_index_path][:]
            
                for basis_index in range(selem.number_eof):
                    offset = 0
                    findex = self.rf_uip.freq_index("OMI")
                    for fm_idx, ii_mw in enumerate(self.channel_list()):
                        sg = self.rf_uip.sample_grid(fm_idx, ii_mw)
                        full_instrument_size = len(sg.data)
                        # mod 10 is for joint retrievals where we come back with channel_list() == [10, 11]
                        if (mw_index == (ii_mw % 10)):
                            eof_full = np.zeros((full_instrument_size,))
                            nonzero_eof_index = findex[np.logical_and(findex >= offset,
                                                        findex < offset+full_instrument_size)] - offset
                            eof_channel = np.zeros((len(nonzero_eof_index),))
                            eof_channel[pixel_indexes] = eofs[basis_index, :]
                            eof_full[nonzero_eof_index] = eof_channel
                        offset += full_instrument_size

                    # Not sure about the units, but this is what we assigned to our
                    # forward model, and the EOF needs to match
                    wform = rf.ArrayWithUnit(eof_full, rf.Unit("sr^-1"))
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
    def omi_solar_model(self):
        '''We read a 3 year average solar file HDF file for omi. This
        duplicates what mpy.read_omi does, which is then stored in the pickle
        file that solar_model uses.'''
        f = h5py.File(self.measurement_id["omiSolarReference"], "r")
        res = []
        for i in range(self.num_channels):
            ind = self.rf_uip.across_track_indexes(self.filter_list[i],
                                                   self.instrument_name)[0]
            wav_vals = f[f"WAV_{self.filter_list[i]}"][:, ind]
            irad_vals = f[f"SOL_{self.filter_list[i]}"][:, ind]
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
        band_name = self.rf_uip.filter_list(ii_mw)
        raise NotImplementedError(f'HWHM for band {band_name} not defined')
        
    @property
    def uip_params(self):
        return self.rf_uip.omi_params

    @cached_property
    def ground_clear(self):
        albedo = np.zeros((self.num_channels, 3))
        which_retrieved = np.full((self.num_channels, 3), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)

        for i in range(self.num_channels):
            if(self.filter_list[i] == "UV1"):
                band_reference[i] = (315 + 262) / 2.0
                albedo[i, 0] = self.uip_params['surface_albedo_uv1']
                if('OMISURFACEALBEDOUV1' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[i, 0] = True
            elif(self.filter_list[i] == "UV2"):
                # Note this value is hardcoded in print_omi_surface_albedo
                band_reference[i] = 320.0
                albedo[i, 0] = self.uip_params['surface_albedo_uv2']
                albedo[i, 1] = self.uip_params['surface_albedo_slope_uv2']
                if('OMISURFACEALBEDOUV2' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[i, 0] = True
                if('OMISURFACEALBEDOSLOPEUV2' in self.rf_uip.state_vector_params(self.instrument_name)):
                    which_retrieved[i, 1] = True
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

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      ["Cloud",] * self.num_channels,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
    
                    
    @cached_property
    def cloud_fraction(self):
        if(self.state_info is not None):
            cfrac = self.state_info.state_element("OMICLOUDFRACTION").value[0]
            return rf.CloudFractionFromState(cfrac)
        else:
            return rf.CloudFractionFromState(self.rf_uip.omi_cloud_fraction)

    def add_to_sv(self, fm_sv : rf.StateVector):
        # TODO We have this hardcoded now. We'll rework this, adding to the state
        # vector should get moved into the object creation. But we'll have this in
        # place for now.
        self.current_state.add_fm_state_vector_if_needed(
            fm_sv, ["OMICLOUDFRACTION",], [self.cloud_fraction,])
        self.current_state.add_fm_state_vector_if_needed(
            fm_sv, ["O3",], [self.absorber.absorber_vmr("O3"),])
        self.current_state.add_fm_state_vector_if_needed(
            fm_sv, ["OMISURFACEALBEDOUV1",
                    "OMISURFACEALBEDOUV2",
                    "OMISURFACEALBEDOSLOPEUV2"], [self.ground_clear,])
        # Temp, the EOF required state_info. Just allow this to fail, we
        # are planning on reworking this anyways.
        try:
            self.current_state.add_fm_state_vector_if_needed(
                fm_sv, ["OMIEOFUV1",], self.eof["UV1"])
            self.current_state.add_fm_state_vector_if_needed(
                fm_sv, ["OMIEOFUV2",], self.eof["UV2"])
        except (AttributeError, KeyError):
            pass
        
    @lru_cache(maxsize=None)
    def raman_effect(self, i):
        # Note we should probably look at this sample grid, and
        # make sure it goes RamanSioris.ramam_edge_wavenumber past
        # the edges of our spec_win. Also there isn't any particular
        # reason that the solar data/optical depth should be calculated
        # on the muses_fm_spectral_domain. But this is what muses-py
        # does, so we'll match that for now.
        if(self.filter_list[i] in ("UV1", "UV2")):
            scale_factor = self.uip_params[f"ring_sf_{str.lower(self.filter_list[i])}"]
        else:
            raise RuntimeError("Unrecognized filter name")
        with self.observation.modify_spectral_window(do_raman_ext=True):
            wlen = self.observation.spectral_domain(i)
        # This is short if we aren't actually running this filter
        if(wlen.data.shape[0] < 2):
            return None
        salbedo = OmiSurfaceAlbedo(self.ground, i)
        return MusesRaman(salbedo, self.ray_info,
                          wlen,
                          float(scale_factor),
                          i,
                          self.sza_with_unit[i],
                          self.oza_with_unit[i],
                          self.raz_with_unit[i],
                          self.atmosphere,
                          self.solar_model(i),
                          rf.StateMappingLinear())
    

class OmiForwardModelHandle(ForwardModelHandle):
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
        if(instrument_name != "OMI"):
            return None
        obj_creator = OmiFmObjectCreator(current_state, self.measurement_id, obs,
                                         rf_uip = rf_uip_func(),
                                         fm_sv = fm_sv, **self.creator_kwargs)
        fm = obj_creator.forward_model
        logger.info(f"OMI Forward model\n{fm}")
        return fm

__all__ = ["OmiFmObjectCreator", "OmiForwardModelHandle"]
    
