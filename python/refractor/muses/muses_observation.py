from .misc import osp_setup
from .cost_function_creator import ObservationHandle, ObservationHandleSet
import refractor.muses.muses_py as mpy
import os
import numpy as np
import refractor.framework as rf
import abc
import copy

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class MeasurementId(object, metaclass=abc.ABCMeta):
    '''py-retrieve uses a file called Measurement_ID.asc. This files contains
    information about the soundings we use. This is mostly just a standard
    keyword/value set, however there are a few complications:

    1. The names may be relative to the directory that the Measurement_ID.asc file
       is in, so we need to handle translating this to a full path since we aren't
       in general in the Measurement_ID.asc directory.
    2. There may be "associated" files that really logically should live in the
       Measurement_ID.asc file but don't because it is convenient to store
       them elsewhere - for example the omi_calibration_filename which comes
       from the strategy file.
    3. When reading the data, we often need to know the specific filters we will
       be working with, e.g., so we only read that data out of the sounding files.

    This class brings this stuff together. It is mostly just a dict mapping
    keyword to file, but with these extra handling included.

    This class is an abstract interface, it is useful for testing to have a simple
    implementation that doesn't depend on the Measurement_ID.asc and strategy tables
    files (e.g., a hardcoded dict with the values).
    '''

    @abc.abstractproperty
    def filter_list(self) -> 'list[str]':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        raise NotImplementedError

    def value_float(self, keyword: str) -> float:
        '''Value, converted from string to float'''
        return float(self.value(keyword))

    def value_int(self, keyword: str) -> int:
        '''Value, converted from string to int'''
        return int(self.value(keyword))

    @abc.abstractmethod
    def value(self, keyword: str) -> str:
        '''Return a value found in the Measurement_ID file, or if not there
        in the strategy table file.'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def filename(self, keyword:str) -> str:
        '''Return a filename found in the Measurement_ID file (handling
        relative paths), or if not there then in the strategy table file.
        '''

class MeasurementIdDict(MeasurementId):
    '''Implementation of MeasurementId that uses a dict'''
    def __init__(self, measurement_dict : dict, filter_list: list):
        self.measurement_dict = measurement_dict
        self._filter_list = filter_list

    @property
    def filter_list(self) -> 'list[str]':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        return self._filter_list

    def value(self, keyword: str) -> str:
        return self.measurement_dict[value]

    def filename(self, keyword:str) -> str:
        # Assume we've already handled any relative paths when filling in measurement_dict
        return self.value(keyword)
        
class MeasurementIdFile(MeasurementId):
    '''Implementation of MeasurementId that uses the Measurement_ID.asc file.'''
    def __init__(self, fname, strategy_table: 'StrategyTable'):
        self.fname = fname
        self._dir_relative_to = os.path.abspath(os.path.dirname(self.fname))
        self._p = mpy.tes_file_get_struct(mpy.read_all_tes(self.fname)[1])["preferences"]
        self._filter_list = strategy_table.filter_list_all()
        self._strategy_table = strategy_table
    
    @property
    def filter_list(self) -> 'list[str]':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        return self._filter_list

    def value(self, keyword: str) -> str:
        '''Return a value found in the Measurement_ID file, or if not there
        in the strategy table file.'''
        if(keyword in self._p):
            return self._p[keyword]
        if(keyword in self._strategy_table.preferences):
            return self._strategy_table.preferences[keyword]
        raise KeyError(keyword)

    def filename(self, keyword:str) -> str:
        '''Return a filename found in the Measurement_ID file (handling
        relative paths), or if not there then in the strategy table file.
        '''
        if(keyword in self._p):
            fname = self._p[keyword]
            if(os.path.isabs(fname)):
                return fname
            return os.path.normpath(os.path.join(self._dir_relative_to, fname))
        if(keyword in self._strategy_table.preferences):
            return self._strategy_table.abs_filename(self._strategy_table.preferences[keyword])
        raise KeyError(keyword)
    
    
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

    def state_element_name_list(self):
        '''List of state element names for this observation'''
        return []

    def spectral_domain_with_bad_sample(self, sensor_index):
        '''The spectral domain for the sensor index,  but including bad samples also'''
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
    
    def notify_update(self, sv):
        super().notify_update(sv)
        self._spec = [None,] * self.num_channels
        
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
        sd = self.spectral_domain_full(sensor_index)
        if(self._force_no_bad_sample):
            return self.spectral_window_with_bad_sample.apply(sd, sensor_index)
        else:
            return self.spectral_window.apply(sd, sensor_index)

    def spectral_domain_with_bad_sample(self, sensor_index):
        sd = self.spectral_domain_full(sensor_index)
        return self.spectral_window_with_bad_sample.apply(sd, sensor_index)
        
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
            self._spec[sensor_index] = self.spectral_window.apply(self.spectrum_full(sensor_index),
                                                                  sensor_index)
        return self._spec[sensor_index]

    def radiance_with_bad_sample(self, sensor_index):
        return self.spectral_window_with_bad_sample.apply(self.spectrum_full(sensor_index),
                                                          sensor_index)

    def spectrum_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        sd = self.spectral_domain_full(sensor_index)
        sr = rf.SpectralRange(self.radiance_full(sensor_index, skip_jacobian=skip_jacobian),
                              rf.Unit("sr^-1"), self.nesr_full(sensor_index))
        return rf.Spectrum(sd, sr)

    def radiance_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def spectral_domain_full(self, sensor_index):
        '''Spectral domain before we have removed bad samples or applied  the microwindows.'''
        # By convention, sample index starts with 1. This was from OCO-2, I'm not
        # sure if that necessarily makes sense here or not. But I think we have code
        # that depends on the 1 base.
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        return rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
        
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

    def notify_update_target(self, measurement_id : MeasurementId):
        # Need to read new data when the target changes
        self.obs = None

    def observation(self, instrument_name : str,
                    current_state : 'CurrentState',
                    spec_win : rf.SpectralWindowRange,
                    fm_sv: rf.StateVector,
                    include_bad_sample=False,
                    rs=None,    # Short term, we pass in RetrievalStategy. This will go
                                # away, but we use this as a transition
                    **kwargs):
        if(instrument_name != self.instrument_name):
            return None
        
        # Short term, create a completely new object. We'll want to update this
        # to handle intelligent cloning because we don't want to read the files each
        # time. But short term set that aside so we can work on other pieces before getting
        # to this.
        self.obs = None
        
        if(self.obs is None):
            self.obs = self.obs_cls.create_from_rs(rs)

        # Update the spectral window
        
        # Always include a spectral window which doesn't remove bad samples. I'm not
        # sure how much sense this makes, but py-retrieve has various output that include
        # bad samples. We might want to remove this in the future, but for now keep the
        # option of handling this
        self.obs.spectral_window_with_bad_sample = spec_win
        swin = copy.deepcopy(spec_win)
        if(not include_bad_sample):
            for i in range(self.obs.num_channels):
                swin.bad_sample_mask(self.obs.bad_sample_mask(i), i)
        self.obs.spectral_window = swin
        # Short term
        self.obs.notify_update_rs(rs)

        # Add to state vector
        current_state.add_fm_state_vector_if_needed(fm_sv, self.obs.state_element_name_list(),
                                                    [self.obs,])
        return self.obs
        
class MusesAirsObservation(MusesObservationImp):
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
        return "MusesAirsObservation"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            mid = rs.measurement_id
            filter_list = mid.filter_list["AIRS"]
            filename = mid.filename('AIRS_filename')
            granule = mid.value('AIRS_Granule')
            xtrack = mid.value_int('AIRS_XTrack_Index')
            atrack = mid.value_int('AIRS_ATrack_Index')
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


class MusesCrisObservation(MusesObservationImp):
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
        return "MusesCrisObservation"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            mid = rs.measurement_id
            filter_list = mid.filter_list["CRIS"]
            filename = mid.filename('CRIS_filename')
            granule = mid.value("CRIS_Granule")
            xtrack = mid.value_int('CRIS_XTrack_Index')
            atrack = mid.value_int('CRIS_ATrack_Index')
            pixel_index = mid.value_int('CRIS_Pixel_Index')
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
    def __init__(self, original_wav, bad_sample_mask, parent_obj, offset_index,
                 slope_index, order):
        '''For convenience, we take the offset and slope as a index into parent.mapped_state.
        This allows us to directly use data from the Observation class without needing to worry
        about routing this.

        Note we had previously just passed a lambda function of offset and slope, but we want
        to be able to pickle this object and we can't pickle lambdas, at least without extra
        work (e.g., using dill or hand coding something)'''
        self.orgwav = original_wav.copy()
        self.parent_obj = parent_obj
        self.offset_index = offset_index
        self.slope_index = slope_index
        self.order = order
        self.orgwav_mean = np.mean(original_wav[bad_sample_mask != True])

    def pixel_grid(self):
        '''Return the pixel grid. This is in "nm", although for convenience we just return
        the data.'''
        if(self.order == 1):
            offset = self.parent_obj.mapped_state[self.offset_index]
            return [rf.AutoDerivativeDouble(float(self.orgwav[i])) -
                    offset for i in range(self.orgwav.shape[0])]
        elif(self.order == 2):
            offset = self.parent_obj.mapped_state[self.offset_index]
            slope = self.parent_obj.mapped_state[self.slope_index]
            return [rf.AutoDerivativeDouble(float(self.orgwav[i])) -
                    (offset+(rf.AutoDerivativeDouble(float(self.orgwav[i]))-
                             self.orgwav_mean) * slope) 
                    for i in range(self.orgwav.shape[0])]
        else:
            raise RuntimeError("order needs to be 1 or 2.")

class LinearInterpolate(rf.LinearInterpolateAutoDerivative):
    '''The refractor LinearInterpolateAutoDerivative is what we want to use for our
    interpolation, but it is pretty low level and is also not something that can be
    pickled. We add a little higher level interface here. This might be generally useful,
    we can elevate this if it turns out to be useful. But right now, this just lives in
    our MusesObservation code.'''
    def __init__(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        x_ad = rf.vector_auto_derivative()
        y_ad = rf.vector_auto_derivative()
        for xv in self.x:
            x_ad.append(rf.AutoDerivativeDouble(float(xv)))
        for yv in self.y:
            y_ad.append(rf.AutoDerivativeDouble(float(yv)))
        super().__init__(x_ad, y_ad)

    def __reduce__(self):
        return (_new_from_init, (self.__class__, self.x, self.y))

class LinearInterpolate2(rf.LinearInterpolateAutoDerivative):
    def __init__(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        x_ad = rf.vector_auto_derivative()
        y_ad = rf.vector_auto_derivative()
        for xv in self.x:
            x_ad.append(rf.AutoDerivativeDouble(float(xv)))
        for yv in self.y:
            y_ad.append(yv)
        super().__init__(x_ad, y_ad)
    
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
                                                   self, 0*len(self.filter_list)+i, None,
                                                   order=1))
            self._norm_rad_wav.append(MusesDispersion(self._freq_data[i], self.bad_sample_mask(i),
                                                      self, 1*len(self.filter_list)+i,
                                                      2*len(self.filter_list)+i,
                                                      order=2))
            # Create a interpolator for the solar model, only using good data.
            solar_data = muses_py_dict['Solar_Radiance']['AdjustedSolarRadiance'][flt_sub]
            orgwav_good = self._freq_data[i][self.bad_sample_mask(i) != True]
            solar_good = solar_data[self.bad_sample_mask(i) != True]
            self._solar_interp.append(LinearInterpolate(orgwav_good, solar_good))

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
        solar_rad = self.solar_radiance(sensor_index)
        norm_rad_good = [rf.AutoDerivativeDouble(self._earth_rad[sensor_index][i]) / solar_rad[i]
                         for i in range(len(self._earth_rad[sensor_index]))
                         if self.bad_sample_mask(sensor_index)[i] != True]
        orgwav_good = self._freq_data[sensor_index][self.bad_sample_mask(sensor_index) != True]
        return LinearInterpolate2(orgwav_good, norm_rad_good)

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
        sd = self.spectral_domain_full(sensor_index)
        return rf.Spectrum(sd, sr)
    

class MusesTropomiObservation(MusesObservationReflectance):
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
        return "MusesTropomiObservation"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            mid = rs.measurement_id
            filter_list = mid.filter_list["TROPOMI"]
            if(mid.value_int('TROPOMI_Rad_calRun_flag') != 1):
                raise RuntimeError("Don't support calibration files yet")
            irr_filename = mid.filename('TROPOMI_IRR_filename')
            cld_filename = mid.filename('TROPOMI_Cloud_filename')
            atrack = mid.value_int('TROPOMI_ATrack_Index')
            utc_time = mid.value('TROPOMI_utcTime')
            filename_list = [mid.filename(f"TROPOMI_filename_{flt}")
                             for flt in filter_list]
            xtrack_list = [mid.value_int(f"TROPOMI_XTrack_Index_{flt}")
                           for flt in filter_list]
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

class MusesOmiObservation(MusesObservationReflectance):
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
        return "MusesOmiObservation"

    @classmethod
    def create_from_rs(cls, rs: 'RetrievalStategy'):
        # Measurement ID may have relative paths, so go ahead and run in that directory
        with rs.chdir_run_dir():
            mid = rs.measurement_id
            filter_list = mid.filter_list["OMI"]
            xtrack_uv1 = mid.value_int("OMI_XTrack_UV1_Index")
            xtrack_uv2 = mid.value_int("OMI_XTrack_UV2_Index")
            atrack = mid.value_int('OMI_ATrack_Index')
            filename = mid.filename("OMI_filename")
            cld_filename = mid.filename('OMI_Cloud_filename')
            utc_time = mid.value('OMI_utcTime')
            calibration_filename = mid.filename("omi_calibrationFilename")
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
    

# We have old code with the TES sounding_desc. This isn't used anywhere, and should
# go into a TES observation class when we get around to incorporating this. But
# keep this code around for reference until we can create full observations.
class Level1bTes:
    '''This is like a Level1b class from framework, although right now we won't
    bother making this actually one those. Instead this pulls stuff out of
    StateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from StateInfo and put this
    over with the Observation.'''
    def __init__(self, state_info):
        self.state_info = state_info

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        info_file = self.state_info.info_file
        return {
            "TES_RUN" : np.int16(info_file['preferences']['TES_run']),
            "TES_SEQUENCE" : np.int16(info_file['preferences']['TES_sequence']),
            "TES_SCAN" : np.int16(info_file['preferences']['TES_scan']),
            "POINTINGANGLE_TES" : self.boresight_angle.convert("deg").value
        }

    @property
    def boresight_angle(self):
        return rf.DoubleWithUnit(self.state_info.state_info_dict["current"]["boresightNadirRadians"], "rad")


ObservationHandleSet.add_default_handle(MusesObservationHandle("AIRS", MusesAirsObservation))
ObservationHandleSet.add_default_handle(MusesObservationHandle("CRIS", MusesCrisObservation))
ObservationHandleSet.add_default_handle(MusesObservationHandle("TROPOMI",
                                                               MusesTropomiObservation))
ObservationHandleSet.add_default_handle(MusesObservationHandle("OMI",
                                                               MusesOmiObservation))

__all__ = ["MusesAirsObservation", "MusesObservation", "MusesObservationHandle",
           "MusesCrisObservation", "MusesObservationReflectance",
           "MusesTropomiObservation", "MusesOmiObservation", "MeasurementId",
           "MeasurementIdDict", "MeasurementIdFile"]
