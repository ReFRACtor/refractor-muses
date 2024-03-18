from .misc import osp_setup
from .priority_handle_set import PriorityHandleSet
from .cost_function_creator import ObservationHandle, ObservationHandleSet, StateVectorHandle
import refractor.muses.muses_py as mpy
import os
import numpy as np
import refractor.framework as rf
import abc

class StateVectorHandleName(StateVectorHandle):
    '''I think most of the StateVector attachment will be pretty similar. Not
    exactly sure of the interface, but we'll start with this and see if we need
    to change over time.'''
    def __init__(self, obj_to_add, state_element_name_list):
        self.obj_to_add = obj_to_add
        self.state_element_name_list = state_element_name_list
        
    def add_sv(self, sv: rf.StateVector, state_element_name : str, 
               pstart : int, plen : int, **kwargs):
        if(state_element_name in self.state_element_name_list):
            self.add_sv_once(sv, self.obj_to_add)
        else:
            # Didn't recognize the state_element_name, so we didn't handle this
            return False
        return True

class MusesObservation(rf.ObservationSvImpBase):
    '''The Observation for MUSES is a standard ReFRACtor Observation, with a few
    extra pieces needed by the MUSES code.

    This class specifies the interface needed. Note like most of the time we use
    standard duck typing, so a MUSES observation doesn't actually need to inherit
    from this class if for whatever reason that isn't convenient. But is still useful
    to know that the interface is.

    The things added are:
    
    sounding_desc - this is a dictionary with the instrument specific way of describing
        what sounding we are using. This is used in the product output files (so stuff
        in RetrievalOutput)
    spectral_window - an rf.Observation needs to supply the radiance/reflectance data to
        match against the forward model. Often this is a subset of the full instrument
        data (e.g., removing bad samples, applying microwindows). The rf.Observation class
        doesn't specify how this is done. The input data might have just already been subsetted,
        or we might apply a SpectralWindow to a larger set of data. For MusesObservation we
        always use a SpectralWindow. Further, the SpectralWindow can be updated, which is
        common from one retrieval step to the other. This spectral_window should also
        filter out all the bad samples.
    spectral_window_with_bad_sample - a peculiarity of py-retrieve is that some of the
        output files actually include bad sample data. I think this is actually a bad idea,
        but none the less is how the current py-retrieve works. So we have a separate version
        of the spectral_window that does not include removing bad samples.
    radiance_all_with_bad_sample - variation of the normal rf.Observation.radiance_all,
        but includes bad samples
    muses_py_dict - not sure how much longer we will need this, but currently have code
        that depends on the older python dict in muses_py (e.g. o_tropomi). This provides
        access to that.
    
    We have all the normal rf.Observation stuff, plus what is found in this class.
    '''

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        raise NotImplementedError()

    @property
    def spectral_window_with_bad_sample(self):
        '''SpectralWindow to apply to the observation data, which include bad samples. I'm not
        sure how much sense it makes, but some of the output includes bad sample data. This
        applies the spectral window, but not the bad sample removal'''
        raise NotImplementedError()

    @spectral_window_with_bad_sample.setter
    def spectral_window_with_bad_sample(self, val):
        raise NotImplementedError()

    @property
    def spectral_window(self):
        '''SpectralWindow to apply to the observation data.'''
        raise NotImplementedError()

    @spectral_window.setter
    def spectral_window(self, val):
        '''Set the SpectralWindow to apply to the observation data. Note this can be updated,
        e.g., a MusesObservationBase used for one strategy step with a set of microwindows and
        then updated to another set.'''
        raise NotImplementedError()

    def radiance_all_with_bad_sample(self):
        '''The radiance for all the sensor indexes, but including bad samples also.'''
        raise NotImplementedError()
    
class MusesObservationImp(MusesObservation):
    '''Common behavior for each of the MusesObservation classes we have'''
    def __init__(self, muses_py_dict, sdesc, num_channels=1, coeff=None,which_retrieved=None):
        if(coeff is None):
            super().__init__([])
        else:
            super().__init__(coeff, rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
        self.muses_py_dict = muses_py_dict
        self._spectral_window = None
        self._spectral_window_with_bad_sample = None
        self._num_channels = num_channels
        self._sd = [None,] * self.num_channels
        self._spec = [None,] * self.num_channels
        self._sounding_desc = sdesc
        self._force_no_bad_sample = False

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        return self._sounding_desc
        
    def _v_num_channels(self):
        return self._num_channels
    
    @property
    def spectral_window_with_bad_sample(self):
        '''SpectralWindow to apply to the observation data, which include bad samples. I'm not
        sure how much sense it makes, but some of the output includes bad sample data. This
        applies the spectral window, but not the bad sample removal'''
        return self._spectral_window_with_bad_sample

    @spectral_window_with_bad_sample.setter
    def spectral_window_with_bad_sample(self, val):
        self._spectral_window_with_bad_sample = val
        
    @property
    def spectral_window(self):
        '''SpectralWindow to apply to the observation data.'''
        return self._spectral_window

    @spectral_window.setter
    def spectral_window(self, val):
        '''Set the SpectralWindow to apply to the observation data. Note this can be updated,
        e.g., a MusesObservationBase used for one strategy step with a set of microwindows and
        then updated to another set.'''
        self._spectral_window = val
        self._sd = [None,] * self.num_channels
        self._spec = [None,] * self.num_channels

    def spectral_domain(self, sensor_index):
        return self.radiance(sensor_index).spectral_domain

    def radiance_all_with_bad_sample(self):
        try:
            # "Trick" to leverage radiance_all we already have in StackedRadianceMixin.
            # We could implement this functionality again, but it is nice just to use
            # the already tested and implemented StackedRadianceMixin
            self._force_no_bad_sample = True
            return self.radiance_all()
        finally:
            self._force_no_bad_sample = False

    def radiance(self, sensor_index, skip_jacobian=False):
        # "Trick" to get radiance_all_with_bad_sample for free. We use the normal
        # StackedRadianceMixin support, and call radiance_with_bad_sample instead.
        if(self._force_no_bad_sample):
            return self.radiance_with_bad_sample(sensor_index)
        if(self._spec[sensor_index] is None):
            # By convention, sample index starts with 1. This was from OCO-2, I'm not
            # sure if that necessarily makes sense here or not. But I think we have code
            # that depends on the 1 base.
            freq = self.frequency_full(sensor_index)
            sindex = np.array(list(range(len(freq)))) + 1
            sd = rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
            sr = rf.SpectralRange(self.radiance_full(sensor_index, skip_jacobian=skip_jacobian),
                                  rf.Unit("sr^-1"), self.nesr_full(sensor_index))
            sp = rf.Spectrum(sd, sr)
            self._spec[sensor_index] = self.spectral_window.apply(sp, sensor_index)
        return self._spec[sensor_index]

    def radiance_with_bad_sample(self, sensor_index):
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        sd = rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
        sr = rf.SpectralRange(self.radiance_full(sensor_index), rf.Unit("sr^-1"),
                              self.nesr_full(sensor_index))
        sp = rf.Spectrum(sd, sr)
        return self.spectral_window_with_bad_sample.apply(sp, sensor_index)
        
    def radiance_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def frequency_full(self, sensor_index):
        '''The full list of frequency, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def nesr_full(self, sensor_index):
        '''The full list of NESR, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def bad_sample_mask(self, sensor_index):
        # Default way to find bad samples is to look for negative NESR. Some of the
        # the derived objects override this (e.g., Tropomi also check the solar model
        # for negative values).
        return np.array(self.nesr_full(sensor_index) < 0)

    def notify_update_rs(self, rs: 'RetrievalStategy'):
        '''Do anything needed when we are on a new retrieval step. Default is to do nothing.'''
        pass

    def state_element_name_list(self):
        '''List of state element names for this observation'''
        return []
    
    def add_state_vector_handle(self, svhandle : StateVectorHandle):
        '''Add a handle for attaching to the state vector.'''
        slist = self.state_element_name_list()
        if(len(slist) > 0):
            svhandle.add_handle(StateVectorHandleName(self, slist))
        

class MusesObservationHandle(ObservationHandle):
    '''A lot of our observation classes just map a name to
    a object of a specific class. This handles this generic construction.'''
    def __init__(self, instrument_name, obs_cls):
        self.instrument_name = instrument_name
        self.obs_cls = obs_cls
        # Keep the same observation around as long as the target doesn't
        # change - we just update the spectral windows.
        self.obs = None

    def __getstate__(self):
        # If we pickle, don't include the stashed obs
        attributes = self.__dict__.copy()
        del attributes['obs']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.obs = None

    def notify_update_target(self, rs):
        # Need to read new data when the target changes
        self.obs = None

    def observation(self, instrument_name : str, rs: 'RetrievalStategy',
                    svhandle: 'StateVectorHandleSet', include_bad_sample=False,
                    **kwargs):
        if(instrument_name != self.instrument_name):
            return None
        if(self.obs is None):
            self.obs = self.obs_cls.create_from_rs(rs)
        # Nothing done with svhandle - most of our observations don't have
        # state elements in them.

        # Update the spectral window
        
        # Always include a spectral window which doesn't remove bad samples. I'm not
        # sure how much sense this makes, but py-retrieve has various output that include
        # bad samples. We might want to remove this in the future, but for now keep the
        # option of handling this
        swin = rs.strategy_table.spectral_window(self.instrument_name)
        self.obs.spectral_window_with_bad_sample = swin
        swin = rs.strategy_table.spectral_window(self.instrument_name)
        if(not include_bad_sample):
            # Unless we are told otherwise, also give a spectral window that removes
            # bad samples
            for i in range(self.obs.num_channels):
                swin.bad_sample_mask(self.obs.bad_sample_mask(i), i)
        self.obs.spectral_window = swin
        self.obs.notify_update_rs(rs)
        self.obs.add_state_vector_handle(svhandle)
        return self.obs
        

# muses_forward_model has an older observation class named MusesAirsObservation.
# Short term we add a "New" here. We should sort that out -
# we are somewhat rewriting MusesObservationBase to not use a UIP. This will
# probably get married into one clases, but we aren't ready to do that yet.

class MusesAirsObservationNew(MusesObservationImp):
    def __init__(self, filename, granule, xtrack, atrack, filter_list, osp_dir=None):
        i_fileid = {}
        i_fileid['preferences'] = {'AIRS_filename' : os.path.abspath(filename),
                                   'AIRS_XTrack_Index' : xtrack,
                                   'AIRS_ATrack_Index' : atrack}
        i_window = []
        for cname in filter_list:
            i_window.append({'filter': cname})
        with(osp_setup(osp_dir)):
            o_airs = mpy.read_airs(i_fileid, i_window)
        sdesc = {
            "AIRS_GRANULE" : np.int16(granule),
            "AIRS_ATRACK_INDEX" : np.int16(atrack),
            "AIRS_XTRACK_INDEX" : np.int16(xtrack),
            "POINTINGANGLE_AIRS" : abs(o_airs['scanAng'])
        }
        super().__init__(o_airs, sdesc)

    def desc(self):
        return "MusesAirsObservationNew"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            filter_list = rs.strategy_table.filter_list("AIRS")
            p = rs.measurement_id_file['preferences']
            filename = p['AIRS_filename']
            granule = p['AIRS_Granule']
            xtrack = p['AIRS_XTrack_Index']
            atrack = p['AIRS_ATrack_Index']
            return cls(filename, granule, xtrack, atrack, filter_list)
    
    def radiance_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict['radiance']['radiance']

    def frequency_full(self, sensor_index):
        '''The full list of frequency, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict['radiance']['frequency']

    def nesr_full(self, sensor_index):
        '''The full list of NESR, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict['radiance']['NESR']


class MusesCrisObservationNew(MusesObservationImp):
    def __init__(self, filename, granule, xtrack, atrack, pixel_index, osp_dir=None):
        i_fileid = {'CRIS_filename' : os.path.abspath(filename),
                    'CRIS_XTrack_Index' : xtrack,
                    'CRIS_ATrack_Index' : atrack,
                    'CRIS_Pixel_Index' : pixel_index,
                    }
        self.filename = os.path.abspath(filename)
        with(osp_setup(osp_dir)):
            if(self.l1b_type in ('snpp_fsr', 'noaa_fsr')):
               o_cris = read_noaa_cris_fsr(i_fileid)
            else:
               o_cris = mpy.read_nasa_cris_fsr(i_fileid)
        # Leaving RADIANCESTRUCT out of o_cris, I don't think this is actually
        # used anywhere
        #
        # We can perhaps clean this up, but for now there is some metadata  written
        # in the output file that depends on getting the l1b_type through o_cris,
        # so set that up
        o_cris['l1bType'] = self.l1b_type
        sdesc = {
            "CRIS_GRANULE" : np.int16(granule),
            "CRIS_ATRACK_INDEX" : np.int16(atrack),
            "CRIS_XTRACK_INDEX" : np.int16(xtrack),
            "CRIS_PIXEL_INDEX" : np.int16(pixel_index),
            "POINTINGANGLE_CRIS" : abs(o_cris['SCANANG']),
            "CRIS_L1B_TYPE" : np.int16(self.l1b_type_int)
        }
        
        super().__init__(o_cris, sdesc)


    @property
    def l1b_type_int(self):
        '''Enumeration used in output metadata for the l1b_type'''
        return ['suomi_nasa_nsr', 'suomi_nasa_fsr', 'suomi_nasa_nomw',
                'jpss1_nasa_fsr', 'suomi_cspp_fsr','jpss1_cspp_fsr',
                'jpss2_cspp_fsr'].index(self.l1b_type)
        
    @property
    def l1b_type(self):
        '''There are a number of sources for the CRIS data, and two different file format
        types. This determines the l1b_type by looking at the path/filename. This isn't
        particularly robust, it depends on the specific directory structure. However it
        isn't clear what a better way to handle this would be - this is really needed
        metadata that isn't included in the Measurement_ID file but inferred by where the
        CRIS data comes from.'''
        if 'nasa_nsr' in self.filename:
            return 'suomi_nasa_nsr'
        elif 'nasa_fsr' in self.filename:
            return 'suomi_nasa_fsr'
        elif 'jpss_1_fsr' in filename:
            return 'jpss1_nasa_fsr'
        elif 'snpp_fsr' in filename:
            return 'suomi_cspp_fsr'
        elif 'noaa_fsr' in filename:
            return 'suomi_noaa_fsr'
        else:
            raise RuntimeError(f"Don't recognize CRIS file type from path/filename {self.filename}")

    def desc(self):
        return "MusesCrisObservationNew"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            p = rs.measurement_id_file['preferences']
            filename = p['CRIS_filename']
            granule = p['CRIS_Granule']
            xtrack = p['CRIS_XTrack_Index']
            atrack = p['CRIS_ATrack_Index']
            pixel_index = p['CRIS_Pixel_Index']
            return cls(filename, granule, xtrack, atrack, pixel_index)
    
    def radiance_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict['RADIANCE']

    def frequency_full(self, sensor_index):
        '''The full list of frequency, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict['FREQUENCY']

    def nesr_full(self, sensor_index):
        '''The full list of NESR, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict['NESR']

class MusesDispersion:    
    '''Helper class, just pull out the calculation of the wavelength at the pixel grid.
    This is pretty similar to rf.DispersionPolynomial, but there are enough differences
    that it is worth pulling this out.
    Note that for convenience we don't actually handle the StateVector here, instead we
    just handle the routing from the classes that use this.

    Also, we include all pixels, including bad samples. Filtering of bad sample happens
    outside of this class.
    '''
    def __init__(self, original_wav, bad_sample_mask, offset_func, slope_func, order):
        '''For convenience, we take the offset and slope as a function. This allows
        us to directly use data from the Observation class without needing to worry
        about routing this'''
        self.orgwav = rf.vector_auto_derivative()
        for v in original_wav:
            self.orgwav.append(rf.AutoDerivativeDouble(float(v)))
        self.offset_func = offset_func
        self.slope_func = slope_func
        self.order = order
        self.orgwav_mean = np.mean(original_wav[bad_sample_mask != True])

    def pixel_grid(self):
        '''Return the pixel grid. This is in "nm", although for convenience we just return
        the data.'''
        if(self.order == 1):
            offset = self.offset_func()
            return [self.orgwav[i] - offset for i in range(self.orgwav.size())]
        elif(self.order == 2):
            offset = self.offset_func()
            slope = self.slope_func()
            return [self.orgwav[i] - (offset+(self.orgwav[i]-self.orgwav_mean) * slope) 
                    for i in range(self.orgwav.size())]
        else:
            raise RuntimeError("order needs to be 1 or 2.")
        
    
class MusesObservationReflectance(MusesObservationImp):
    '''Both omi and tropomi actually use reflectance rather than radiance. In additon,
    both the solar model and the radiance data have state elements that control the
    Dispersion for the data.

    This object captures the common behavior between the two.
    '''
    def __init__(self, muses_py_dict, sdesc, filter_list):
        self.filter_list = filter_list
        coeff = np.zeros((len(self.filter_list)*3))
        which_retrieved = [True,]*(len(self.filter_list)*3)
        super().__init__(muses_py_dict, sdesc, num_channels=len(self.filter_list),
                         coeff=coeff, which_retrieved=which_retrieved)

        # Stash some values we use in later calculations. Note that the radiance data
        # is all smooshed together, so we separate this.
        #
        # It isn't clear here if the best indexing is the full instrument (so 8 bands) with
        # only some of the bands filled in, or instead the index number into the passed
        # in filter_list. For now, we are using the index into the filter_list. We can possibly
        # reevaluate this - it wouldn't be huge change in the code we have here.
        self._freq_data = []
        self._nesr_data = []
        self._bsamp = []
        self._solar_wav = []
        self._norm_rad_wav = []
        self._solar_interp = []
        self._earth_rad = []
        self._nesr = []
        for i,flt in enumerate(filter_list):
            flt_sub = (muses_py_dict['Earth_Radiance']['EarthWavelength_Filter'] == flt)
            self._freq_data.append(muses_py_dict['Earth_Radiance']['Wavelength'][flt_sub])
            self._nesr_data.append(muses_py_dict['Earth_Radiance']['EarthRadianceNESR'][flt_sub])
            self._bsamp.append(
                (muses_py_dict['Earth_Radiance']['EarthRadianceNESR'][flt_sub] <= 0.0)  |
                 (muses_py_dict['Solar_Radiance']['AdjustedSolarRadiance'][flt_sub]<=0.0))
            self._earth_rad.append(muses_py_dict['Earth_Radiance']['CalibratedEarthRadiance'][flt_sub])
            self._nesr.append(muses_py_dict['Earth_Radiance']['EarthRadianceNESR'][flt_sub])
            self._solar_wav.append(MusesDispersion(self._freq_data[i], self.bad_sample_mask(i),
                                  lambda:self.mapped_state[0*len(self.filter_list)+i], None,
                                  order=1))
            self._norm_rad_wav.append(MusesDispersion(self._freq_data[i], self.bad_sample_mask(i),
                                  lambda:self.mapped_state[1*len(self.filter_list)+i],
                                  lambda:self.mapped_state[2*len(self.filter_list)+i],
                                  order=2))
            # Create a interpolator for the solar model, only using good data.
            orgwav_good = rf.vector_auto_derivative()
            for wav in self._freq_data[i][self.bad_sample_mask(i) != True]:
                orgwav_good.append(rf.AutoDerivativeDouble(float(wav)))
            solar_data = muses_py_dict['Solar_Radiance']['AdjustedSolarRadiance'][flt_sub]
            solar_good = rf.vector_auto_derivative()
            for v in solar_data[self.bad_sample_mask(i) != True]:
                solar_good.append(rf.AutoDerivativeDouble(float(v)))
            self._solar_interp.append(rf.LinearInterpolateAutoDerivative(orgwav_good, solar_good))

    def desc(self):
        return "MusesObservationReflectance"
    
    def notify_update_rs(self, rs: 'RetrievalStategy'):
        # Grab all the coefficients from the StateInfo (since some of them might not
        # be in the state vector), and determine which set will be retrieved.
        coeff = []
        which_retrieved = []
        for nm in self.state_element_name_list():
            coeff.extend(rs.state_info.state_element(nm).value)
            which_retrieved.append(nm in rs.strategy_table.retrieval_elements())
        # Update the coefficients and mapping for this object
        self.init(np.array(coeff), rf.StateMappingAtIndexes(np.ravel(np.array(which_retrieved))))
        self._spec = [None,] * self.num_channels
        
    def notify_update(self, sv):
        # Not positive about this for more than one channel
        super().notify_update(sv)
        self._spec = [None,] * self.num_channels
        
    def spectral_domain(self, sensor_index):
        # Since self.radiance involves more calculation, give a optimized version of
        # the spectral_domain function for when we just want this.
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        sd = rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
        if(self._force_no_bad_sample):
            return self.spectral_window_with_bad_sample.apply(sd, sensor_index)
        else:
            return self.spectral_window.apply(sd, sensor_index)

    def radiance(self, sensor_index, skip_jacobian=False):
        # "Trick" to get radiance_all_with_bad_sample for free. We use the normal
        # StackedRadianceMixin support, and call radiance_with_bad_sample instead.
        if(self._force_no_bad_sample):
            return self.radiance_with_bad_sample(sensor_index)
        if(self._spec[sensor_index] is None):
            self._spec[sensor_index] = self.spectral_window.apply(self.spectrum_full(sensor_index),
                                                                  sensor_index)
        return self._spec[sensor_index]

    def radiance_with_bad_sample(self, sensor_index):
        return self.spectral_window_with_bad_sample.apply(self.spectrum_full(sensor_index),
                                                          sensor_index)
        
    def frequency_full(self, sensor_index):
        '''The full list of frequency, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self._freq_data[sensor_index]

    def nesr_full(self, sensor_index):
        '''The full list of NESR, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self._nesr_data[sensor_index]

    def bad_sample_mask(self, sensor_index):
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        return self._bsamp[sensor_index]

    def solar_radiance(self, sensor_index):
        '''Use our interpolator to get the solar model at the
        shifted spectrum. This is for all data, so filtering out bad sample happens outside
        of this function. '''
        pgrid = self._solar_wav[sensor_index].pixel_grid()
        return [self._solar_interp[sensor_index](wav) for wav in pgrid]

    def norm_radiance(self, sensor_index):
        '''Calculate the normalized radiance. This is for all data, so filtering out bad
        sample happens outside of this function. '''
        pgrid = self._norm_rad_wav[sensor_index].pixel_grid()
        ninterp = self._norm_rad_interp(sensor_index)
        return [ninterp(wav) for wav in pgrid]

    def norm_rad_nesr(self, sensor_index):
        '''Calculate the normalized radiance. This is for all data, so filtering out bad
        sample happens outside of this function.'''
        sol_rad = self.solar_radiance(sensor_index)
        return np.array([self._nesr[sensor_index][i] / sol_rad[i].value for i in
                         range(len(self._nesr[sensor_index]))])
        
    def _norm_rad_interp(self, sensor_index):
        '''Calculate the interpolator used for the normalized radiance. This can't be
        done ahead of time, because the solar radiance used is the interpolated solar
        radiance.'''
        orgwav_good = rf.vector_auto_derivative()
        for wav in self._freq_data[sensor_index][self.bad_sample_mask(sensor_index) != True]:
            orgwav_good.append(rf.AutoDerivativeDouble(float(wav)))
        bmask = self.bad_sample_mask(sensor_index)
        solar_rad = self.solar_radiance(sensor_index)
        norm_rad_good = rf.vector_auto_derivative()
        for i in range(len(self._earth_rad[sensor_index])):
            if(bmask[i] != True):
                norm_rad_good.append(rf.AutoDerivativeDouble(float(self._earth_rad[sensor_index][i])) / solar_rad[i])
        return rf.LinearInterpolateAutoDerivative(orgwav_good, norm_rad_good)

    def snr_uplimit(self,sensor_index):
        '''Upper limit for SNR, we adjust uncertainty is we are greater than this.'''
        raise NotImplementedError()

    def spectrum_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        nrad = self.norm_radiance(sensor_index)
        uncer = self.norm_rad_nesr(sensor_index)
        nrad_val = np.array([nrad[i].value for i in range(len(nrad))])
        snr = nrad_val / uncer
        uplimit = self.snr_uplimit(sensor_index)
        tind = np.asarray(snr > uplimit)
        uncer[tind] = nrad_val[tind] / uplimit
        uncer[self.bad_sample_mask(sensor_index) == True] = -999.0
        nrad_ad = rf.ArrayAd_double_1(len(nrad), self.coefficient.number_variable)
        for i,v in enumerate(self.bad_sample_mask(sensor_index)):
            nrad_ad[i] = rf.AutoDerivativeDouble(-999.0) if v == True else nrad[i]
        sr = rf.SpectralRange(nrad_ad, rf.Unit("sr^-1"), uncer)
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        sd = rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
        return rf.Spectrum(sd, sr)
    

class MusesTropomiObservationNew(MusesObservationReflectance):
    '''Observation for Tropomi'''
    def __init__(self, filename_list, irr_filename, cld_filename, xtrack_list, atrack,
                 utc_time, filter_list, calibration_filename=None, osp_dir=None):
        # Filter list should be in the same order as filename_list, and should be
        # things like "BAND3"
        if(calibration_filename is not None):
            raise RuntimeError("We don't support TROPOMI calibration yet")
        i_windows = [{'instrument' : 'TROPOMI', 'filter' : flt} for flt in filter_list]
        with(osp_setup(osp_dir)):
            o_tropomi = mpy.read_tropomi(filename_list, irr_filename, cld_filename,
                                         xtrack_list, atrack, utc_time, i_windows)
        sdesc = {
            "TROPOMI_ATRACK_INDEX" : np.int16(atrack),
            'TROPOMI_XTRACK_INDEX_BAND1': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND1': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND2': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND2': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND3': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND3': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND4': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND4': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND5': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND5': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND6': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND6': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND7': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND7': -999.0,
            'TROPOMI_XTRACK_INDEX_BAND8': np.int16(-999),
            'POINTINGANGLE_TROPOMI_BAND8': -999.0,
        }
        # TODO Fill in POINTINGANGLE_TROPOMI
        for i,flt in enumerate(filter_list):
            sdesc[f'TROPOMI_XTRACK_INDEX_{flt}'] = np.int16(xtrack_list[i])
            # Think this is right
            sdesc[f'POINTINGANGLE_TROPOMI_{flt}'] = o_tropomi["Earth_Radiance"]["ObservationTable"]["ViewingZenithAngle"][i]
        super().__init__(o_tropomi, sdesc, filter_list)

    def desc(self):
        return "MusesTropomiObservationNew"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            p = rs.measurement_id_file['preferences']
            if(int(p['TROPOMI_Rad_calRun_flag']) != 1):
                raise RuntimeError("Don't support calibration files yet")
            filter_list = rs.strategy_table.filter_list("TROPOMI")
            irr_filename = p['TROPOMI_IRR_filename']
            cld_filename = p['TROPOMI_Cloud_filename']
            atrack = int(p['TROPOMI_ATrack_Index'])
            utc_time = p['TROPOMI_utcTime']
            filename_list = [p[f"TROPOMI_filename_{flt}"] for flt in filter_list]
            xtrack_list = [int(p[f"TROPOMI_XTrack_Index_{flt}"]) for flt in filter_list]
            return cls(filename_list, irr_filename, cld_filename, xtrack_list, atrack,
                       utc_time, filter_list)

    def snr_uplimit(self, sensor_index):
        '''Upper limit for SNR, we adjust uncertainty is we are greater than this.'''
        return 500.0
    
    def state_element_name_list(self):
        '''List of state element names for this observation'''
        res = []
        for flt in self.filter_list:
            res.append(f"TROPOMISOLARSHIFT{flt}")
        for flt in self.filter_list:
            res.append(f"TROPOMIRADIANCESHIFT{flt}")
        for flt in self.filter_list:
            res.append(f"TROPOMIRADSQUEEZE{flt}")
        return res

class MusesOmiObservationNew(MusesObservationReflectance):
    '''Observation for OMI'''
    def __init__(self, filename, xtrack_uv1, xtrack_uv2, atrack, utc_time, calibration_filename,
                 filter_list, cld_filename=None, osp_dir=None):
        with(osp_setup(osp_dir)):
            o_omi = mpy.read_omi(filename, xtrack_uv2, atrack, utc_time, calibration_filename,
                                 cldFilename=cld_filename)
        sdesc = {
            "OMI_ATRACK_INDEX": np.int16(atrack),
            "OMI_XTRACK_INDEX_UV1": np.int16(xtrack_uv1),
            "OMI_XTRACK_INDEX_UV2": np.int16(xtrack_uv2),
            "POINTINGANGLE_OMI" : abs(np.mean(o_omi["Earth_Radiance"]["ObservationTable"]["ViewingZenithAngle"][1:3]))
        }
        dstruct = mpy.utc_from_string(utc_time)
        # We double the NESR for OMI from 2010 onward. Not sure of the history of this,
        # but this is in the muses-py code so we duplicate this here.
        if(dstruct['utctime'].year >= 2010):
            o_omi['Earth_Radiance']['EarthRadianceNESR'][o_omi['Earth_Radiance']['EarthRadianceNESR'] > 0] *= 2
        super().__init__(o_omi, sdesc, filter_list)

    def desc(self):
        return "MusesOmiObservationNew"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            p = rs.measurement_id_file['preferences']
            filter_list = rs.strategy_table.filter_list("OMI")
            xtrack_uv1 = int(p["OMI_XTrack_UV1_Index"])
            xtrack_uv2 = int(p["OMI_XTrack_UV2_Index"])
            atrack = int(p['OMI_ATrack_Index'])
            filename = p["OMI_filename"]
            cld_filename = p['OMI_Cloud_filename']
            utc_time = p['OMI_utcTime']
            calibration_filename = rs.strategy_table.preferences["omi_calibrationFilename"]
            return cls(filename, xtrack_uv1, xtrack_uv2, atrack, utc_time,
                       calibration_filename, filter_list, cld_filename=cld_filename)

    def snr_uplimit(self, sensor_index):
        '''Upper limit for SNR, we adjust uncertainty is we are greater than this.'''
        if(self.filter_list[sensor_index] == "UV2"):
            return 800.0
        return 500.0
        
    def state_element_name_list(self):
        '''List of state element names for this observation'''
        res = []
        for flt in self.filter_list:
            res.append(f"OMINRADWAV{flt}")
        for flt in self.filter_list:
            res.append(f"OMIODWAV{flt}")
        for flt in self.filter_list:
            res.append(f"OMIODWAVSLOPE{flt}")
        return res
    
    
ObservationHandleSet.add_default_handle(MusesObservationHandle("AIRS", MusesAirsObservationNew))
ObservationHandleSet.add_default_handle(MusesObservationHandle("CRIS", MusesCrisObservationNew))
ObservationHandleSet.add_default_handle(MusesObservationHandle("TROPOMI",
                                                               MusesTropomiObservationNew))
ObservationHandleSet.add_default_handle(MusesObservationHandle("OMI",
                                                               MusesOmiObservationNew))

__all__ = ["MusesAirsObservationNew", "MusesObservation", "MusesObservationHandle",
           "MusesCrisObservationNew", "MusesObservationReflectance",
           "MusesTropomiObservationNew", "MusesOmiObservationNew"]
