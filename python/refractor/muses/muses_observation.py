from .misc import osp_setup
from .observation_handle import ObservationHandle, ObservationHandleSet
from .muses_spectral_window import MusesSpectralWindow
from .retrieval_configuration import RetrievalConfiguration
from contextlib import contextmanager
import refractor.muses.muses_py as mpy
import os
import numpy as np
import refractor.framework as rf
import abc
import copy
import logging
import pickle
import subprocess
import itertools
import collections.abc
import re

logger = logging.getLogger("py-retrieve")

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class MeasurementId(collections.abc.Mapping):
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
    def filter_list_dict(self) -> 'dict(str,list[str])':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
        
    def __len__(self):
        raise NotImplementedError
    

class MeasurementIdDict(MeasurementId):
    '''Implementation of MeasurementId that uses a dict'''
    def __init__(self, measurement_dict : dict, filter_list_dict: dict):
        self.measurement_dict = measurement_dict
        self._filter_list_dict = filter_list_dict

    @property
    def filter_list_dict(self) -> 'dict(str,list[str])':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        return self._filter_list_dict

    def __getitem__(self, key):
        return self.measurement_dict[key]

    def __iter__(self):
        return self.measurement_dict.__iter__()
        
    def __len__(self):
        return len(self.measurement_dict)
        
class MeasurementIdFile(MeasurementId):
    '''Implementation of MeasurementId that uses the Measurement_ID.asc file.'''
    def __init__(self, fname, retrieval_config: RetrievalConfiguration,
                 filter_list_dict : 'dict(str, list(str))'):
        self.fname = fname
        self.base_dir = os.path.abspath(os.path.dirname(self.fname))
        self._p = mpy.tes_file_get_struct(mpy.read_all_tes(self.fname)[1])["preferences"]
        self._filter_list_dict = filter_list_dict
        self._retrieval_config = retrieval_config
    
    @property
    def filter_list_dict(self) -> 'list[str]':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        return self._filter_list_dict

    def __getitem__(self, key):
        if(key in self._p):
            return self._abs_dir(self._p[key])
        if(key in self._retrieval_config):
            return self._retrieval_config[key]
        raise KeyError(key)

    def __iter__(self):
        return itertools.chain(self._p, self._retrieval_config)
        
    def __len__(self):
        return len(self._p) + len(self._retrieval_config)

    def _abs_dir(self, v):
        v = copy.copy(v)
        v = os.path.expandvars(os.path.expanduser(v))
        if(re.match(r'^\.\./', v) or re.match(r'^\./', v)):
            v = os.path.normpath(f"{self.base_dir}/{v}")
        return v
    
class MusesObservation(rf.ObservationSvImpBase):
    '''The Observation for MUSES is a standard ReFRACtor Observation, with a few
    extra pieces needed by the MUSES code.

    This class specifies the interface needed. Note like most of the time we use
    standard duck typing, so a MUSES observation doesn't actually need to inherit
    from this class if for whatever reason that isn't convenient. But is still useful
    to know that the interface is.

    The things added are:

    instrument_name - the name of the instrument the MusesObservation is for
    filter_data - metadata about the filters covered the MusesObservation
    sounding_desc - this is a dictionary with the instrument specific way of describing
        what sounding we are using. This is used in the product output files (so stuff
        in RetrievalOutput)
    spectral_window - an rf.Observation needs to supply the radiance/reflectance data to
        match against the forward model. Often this is a subset of the full instrument
        data (e.g., removing bad samples, applying microwindows). The rf.Observation class
        doesn't specify how this is done. The input data might have just already been subsetted,
        or we might apply a SpectralWindow to a larger set of data. For MusesObservation we
        always use a SpectralWindow. 
    state_element_name_list - List of state elements if any that are used to deterimine
        radiance

    Right now, we require the spectral_window to be the more specific MusesSpectralWindow.
    This is so we have handling for including bad samples, extending for RamanSioris, or
    doing a full band. If needed we can probably relax this requirement - we just need a
    way to handle these different cases. MusesSpectralWindow is a pretty general adapter
    around another SpectralWindow, so this probably isn't too much of a constraint requiring
    this.
    
    We have all the normal rf.Observation stuff, plus what is found in this class.
    '''

    @property
    def instrument_name(self) -> str:
        '''Name of instrument observation is for.'''
        raise NotImplementedError()
    
    @property
    def filter_data(self) -> "list[list[str,int]]":
        '''This returns a list of filter names and sizes. This is used
        as metadata in the py-retrieve structure called
        "radianceStep".

        Note this is similar but distinct from the filter_list_dict
        used in MeasurementId. That list corresponds to specific data
        read from a file. Often this is the same as the filter data
        used in "radianceStep", but in some cases py-retrieve wants to
        think of data as different filters even if it is read from one
        structure - so for example CrIS data gets separated into
        'CrIS-fsr-lw', 'CrIS-fsr-mw', 'CrIS-fsr-sw' even though the
        data is read into one array in read_noaa_cris_fsr. Individual
        classes can handle generating this filter_data however they
        like, there is no requirement that the number of filters is
        the same as the number of channels.
        
        This should return a list of pairs as a filter name and length
        of data (using the spectral_window).

        '''
        raise NotImplementedError()

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        raise NotImplementedError()

    @property
    def spectral_window(self) -> MusesSpectralWindow:
        '''SpectralWindow to apply to the observation data.'''
        raise NotImplementedError()

    @spectral_window.setter
    def spectral_window(self, val : MusesSpectralWindow):
        '''Set the SpectralWindow to apply to the observation data.'''
        raise NotImplementedError()

    def state_element_name_list(self):
        '''List of state element names for this observation'''
        return []

    def radiance_all_extended(self, skip_jacobian=True,
                              include_bad_sample=False, full_band=False,
                              do_raman_ext=False):
        '''Convenience function that changes the spectral_window (e.g., turn
        on bad samples), calls radiance_all, and then changes back.

        Normally we want just the radiance data, so the default is to skip the
        jacobian part. You can select that if you like by passing skip_jacobian=False'''
        with self.modify_spectral_window(include_bad_sample=include_bad_sample,
                                         full_band=full_band, do_raman_ext=do_raman_ext):
            return self.radiance_all(skip_jacobian)

    @contextmanager
    def modify_spectral_window(self, include_bad_sample=False, full_band=False,
                               do_raman_ext=False):
        '''Convenience context that changes the spectral_window (e.g., turn
        on bad samples), does something, and then changes back.'''
        t1 = self.spectral_window.include_bad_sample
        t2 = self.spectral_window.full_band
        t3 = self.spectral_window.do_raman_ext
        try:
            self.spectral_window.include_bad_sample=include_bad_sample
            self.spectral_window.full_band=full_band
            self.spectral_window.do_raman_ext=do_raman_ext
            yield
        finally:
            self.spectral_window.include_bad_sample=t1
            self.spectral_window.full_band=t2
            self.spectral_window.do_raman_ext=t3

    
class MusesObservationImp(MusesObservation):
    '''Common behavior for each of the MusesObservation classes we have'''
    def __init__(self, muses_py_dict, sdesc, num_channels=1, coeff=None,mp=None):
        if(coeff is None):
            super().__init__([])
        else:
            if(mp is None):
                raise RuntimeError("Both coeff and mp need to be None or not None")
            super().__init__(coeff, mp)
        self.muses_py_dict = muses_py_dict
        self._spectral_window = MusesSpectralWindow(None, None)
        self._num_channels = num_channels
        self._sounding_desc = sdesc
        self._filter_data_name = []
        self._filter_data_swin = None

    @property
    def filter_data(self) -> "list[str,int]":
        res = []
        sd = self.spectral_domain_all()
        for i, fltname in enumerate(self._filter_data_name):
            sz = self._filter_data_swin.apply(sd,i).data.shape[0]
            if(sz > 0):
                res.append([fltname,sz])
        return res
    
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
    def spectral_window(self):
        return self._spectral_window

    @spectral_window.setter
    def spectral_window(self, val):
        self._spectral_window = val

    def spectral_domain(self, sensor_index):
        sd = self.spectral_domain_full(sensor_index)
        return self.spectral_window.apply(sd, sensor_index)

    def radiance(self, sensor_index, skip_jacobian=False):
        return self.spectral_window.apply(self.spectrum_full(sensor_index), sensor_index)

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


class MusesObservationHandle(ObservationHandle):
    '''A lot of our observation classes just map a name to
    a object of a specific class. This handles this generic construction.'''
    def __init__(self, instrument_name, obs_cls):
        self.instrument_name = instrument_name
        self.obs_cls = obs_cls
        # Keep the same observation around as long as the target doesn't
        # change - we just update the spectral windows.
        self.existing_obs = None
        self.measurement_id = None

    def __getstate__(self):
        # If we pickle, don't include the stashed obs
        attributes = self.__dict__.copy()
        del attributes['existing_obs']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.existing_obs = None

    def notify_update_target(self, measurement_id : MeasurementId):
        # Need to read new data when the target changes
        self.existing_obs = None
        self.measurement_id = measurement_id
        
    def observation(self, instrument_name : str,
                    current_state : 'Optional(CurrentState)',
                    spec_win : "Optional(MusesSpectralWindow)",
                    fm_sv: "Optional(rf.StateVector)",
                    osp_dir=None,
                    **kwargs):
        if(instrument_name != self.instrument_name):
            return None
        
        obs = self.obs_cls.create_from_id(self.measurement_id,
                                          self.existing_obs,
                                          current_state, spec_win, fm_sv,
                                          osp_dir=osp_dir,
                                          **kwargs)
        if(self.existing_obs is None):
            self.existing_obs = obs
        return obs
        
class MusesAirsObservation(MusesObservationImp):
    def __init__(self, o_airs, sdesc, num_channels=1, coeff=None,mp=None):
        '''Note you don't normally create an object of this class with the
        __init__. Instead, call one of the create_xxx class methods.'''
        super().__init__(o_airs, sdesc)
        # Set up stuff for the filter_data metadata
        self._filter_data_name = o_airs["radiance"]["filterNames"]
        mw_range = np.zeros((len(self._filter_data_name),1,2))
        sindex = 0
        for i in range(mw_range.shape[0]):
            eindex = o_airs["radiance"]["filterSizes"][i] + sindex
            freq = o_airs["radiance"]["frequency"][sindex:eindex]
            mw_range[i,0,:] = min(freq),max(freq)
            sindex = eindex
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        self._filter_data_swin = rf.SpectralWindowRange(mw_range)

    @classmethod
    def _read_data(cls, filename, granule, xtrack, atrack, filter_list,
                   osp_dir=None):
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
        return (o_airs, sdesc)

    def desc(self):
        return "MusesAirsObservation"

    @property
    def instrument_name(self):
        return "AIRS"
    
    @classmethod
    def create_from_filename(cls, filename, granule, xtrack, atrack, filter_list,
                             osp_dir=None):
        '''Create from just the filenames. Note that spectral window doesn't get
        set here, but this can be useful if you just want access to the underlying
        data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).'''
        o_airs, sdesc = cls._read_data(filename, granule, xtrack, atrack, filter_list,
                                       osp_dir=osp_dir)
        return cls(o_airs, sdesc)
        

    @classmethod
    def create_from_id(cls, mid : MeasurementId,
                       existing_obs : 'cls',
                       current_state: 'Optional(CurrentState)',
                       spec_win: "Optional(MusesSpectralWindow)",
                       fm_sv: "Optional(rf.StateVector)",
                       osp_dir=None,
                       **kwargs):
        '''Create from a MeasurementId. If this depends on any state information, you can
        pass in the CurrentState. This can be given as None if you just want to use default
        values, e.g. you aren't doing a retrieval. If the CurrentState is supplied, you can
        also pass a StateVector to add this class to as needed.'''
        if(existing_obs is not None):
            # Take data from existing observation
            obs = cls(existing_obs.muses_py_dict, existing_obs.sounding_desc,
                      num_channels=existing_obs.num_channels)
        else:
            # Read the data from disk, because it doesn't already exist.
            filter_list = mid.filter_list_dict["AIRS"]
            filename = mid['AIRS_filename']
            granule = mid['AIRS_Granule']
            xtrack = int(mid['AIRS_XTrack_Index'])
            atrack = int(mid['AIRS_ATrack_Index'])
            o_airs, sdesc = cls._read_data(filename, granule, xtrack, atrack, filter_list,
                                           osp_dir=osp_dir)
            obs = cls(o_airs, sdesc)
        obs.spectral_window = \
            spec_win if spec_win is not None else MusesSpectralWindow(None,None)
        obs.spectral_window.add_bad_sample_mask(obs)
        if(fm_sv is not None):
            if(current_state is None):
                raise RuntimeError("If fm_sv is not None, current_state needs to also be not None")
            current_state.add_fm_state_vector_if_needed(
                fm_sv, obs.state_element_name_list(), [obs,])
        return obs
    
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
    def __init__(self, o_cris, sdesc, num_channels=1, coeff=None,mp=None):
        '''Note you don't normally create an object of this class with the
        __init__. Instead, call one of the create_xxx class methods.'''
        super().__init__(o_cris, sdesc)
        # This is just hardcoded in py-retrieve, see about line 395 in
        # script_retrieval_setup_ms.py
        self._filter_data_name = ["CrIS-fsr-lw", "CrIS-fsr-mw", "CrIS-fsr-sw"]
        mw_range = np.zeros((3,1,2))
        mw_range[0,0,:] = 0.0,1200.00
        mw_range[1,0,:] = 1200.01,2145.00
        mw_range[2,0,:] = 2145.01,9999.00
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        self._filter_data_swin = rf.SpectralWindowRange(mw_range)

    @classmethod
    def _read_data(cls, filename, granule, xtrack, atrack, pixel_index, osp_dir=None):
        i_fileid = {'CRIS_filename' : os.path.abspath(filename),
                    'CRIS_XTrack_Index' : xtrack,
                    'CRIS_ATrack_Index' : atrack,
                    'CRIS_Pixel_Index' : pixel_index,
                    }
        filename = os.path.abspath(filename)
        with(osp_setup(osp_dir)):
            if(cls.l1b_type_from_filename(filename) in ('snpp_fsr', 'noaa_fsr')):
               o_cris = mpy.read_noaa_cris_fsr(i_fileid)
            else:
               o_cris = mpy.read_nasa_cris_fsr(i_fileid)

        # Add in RADIANCESTRUCT. Not sure if this is used, but easy enough to put in
        radiance = o_cris["RADIANCE"]
        frequency = o_cris["FREQUENCY"]
        nesr = o_cris["NESR"]
        filters = np.full((len(nesr),), 'CrIS-fsr-lw')
        filters[frequency > 1200] = 'CrIS-fsr-mw'
        filters[frequency > 2145] = 'CrIS-fsr-sw'
        o_cris["RADIANCESTRUCT"] = mpy.radiance_data(radiance, nesr, [0], frequency, filters,
                                                     'CRIS')
        # We can perhaps clean this up, but for now there is some metadata  written
        # in the output file that depends on getting the l1b_type through o_cris,
        # so set that up
        o_cris['l1bType'] = cls.l1b_type_from_filename(filename)
        sdesc = {
            "CRIS_GRANULE" : np.int16(granule),
            "CRIS_ATRACK_INDEX" : np.int16(atrack),
            "CRIS_XTRACK_INDEX" : np.int16(xtrack),
            "CRIS_PIXEL_INDEX" : np.int16(pixel_index),
            "POINTINGANGLE_CRIS" : abs(o_cris['SCANANG']),
            "CRIS_L1B_TYPE" : np.int16(cls.l1b_type_int_from_filename(filename))
        }
        return(o_cris, sdesc)


    @classmethod
    def l1b_type_int_from_filename(cls, filename):
        '''Enumeration used in output metadata for the l1b_type'''
        return ['suomi_nasa_nsr', 'suomi_nasa_fsr', 'suomi_nasa_nomw',
                'jpss1_nasa_fsr', 'suomi_cspp_fsr','jpss1_cspp_fsr',
                'jpss2_cspp_fsr'].index(cls.l1b_type_from_filename(filename))

    @classmethod
    def l1b_type_from_filename(cls, filename):
        '''There are a number of sources for the CRIS data, and two different file format
        types. This determines the l1b_type by looking at the path/filename. This isn't
        particularly robust, it depends on the specific directory structure. However it
        isn't clear what a better way to handle this would be - this is really needed
        metadata that isn't included in the Measurement_ID file but inferred by where the
        CRIS data comes from.'''
        if 'nasa_nsr' in filename:
            return 'suomi_nasa_nsr'
        elif 'nasa_fsr' in filename:
            return 'suomi_nasa_fsr'
        elif 'jpss_1_fsr' in filename:
            return 'jpss1_nasa_fsr'
        elif 'snpp_fsr' in filename:
            return 'suomi_cspp_fsr'
        elif 'noaa_fsr' in filename:
            return 'suomi_noaa_fsr'
        else:
            raise RuntimeError(f"Don't recognize CRIS file type from path/filename {self.filename}")

    @property
    def l1b_type_int(self):
        return self.l1b_type_int_from_filename(self.filename)
        
    @property
    def l1b_type(self):
        return self.l1b_type_from_filename(self.filename)
    
    def desc(self):
        return "MusesCrisObservation"

    @property
    def instrument_name(self):
        return "CRIS"
    
    @classmethod
    def create_from_filename(cls, filename, granule, xtrack, atrack, pixel_index,
                             osp_dir=None):
        '''Create from just the filenames. Note that spectral window doesn't get
        set here, but this can be useful if you just want access to the underlying
        data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).'''
        o_cris, sdesc = cls._read_data(filename, granule, xtrack, atrack, pixel_index,
                                       osp_dir=osp_dir)
        return cls(o_cris, sdesc)
        
    
    @classmethod
    def create_from_id(cls, mid : MeasurementId,
                       existing_obs : 'cls',
                       current_state: 'Optional(CurrentState)',
                       spec_win: "Optional(MusesSpectralWindow)",
                       fm_sv: "Optional(rf.StateVector)",
                       osp_dir=None,
                       **kwargs):
        '''Create from a MeasurementId. If this depends on any state information, you can
        pass in the CurrentState. This can be given as None if you just want to use default
        values, e.g. you aren't doing a retrieval. If the CurrentState is supplied, you can
        also pass a StateVector to add this class to as needed.'''
        if(existing_obs is not None):
            # Take data from existing observation
            obs = cls(existing_obs.muses_py_dict, existing_obs.sounding_desc,
                      num_channels=existing_obs.num_channels)
        else:
            filter_list = mid.filter_list_dict["CRIS"]
            filename = mid['CRIS_filename']
            granule = mid["CRIS_Granule"]
            xtrack = int(mid['CRIS_XTrack_Index'])
            atrack = int(mid['CRIS_ATrack_Index'])
            pixel_index = int(mid['CRIS_Pixel_Index'])
            o_cris, sdesc = cls._read_data(filename, granule, xtrack, atrack,
                                           pixel_index, osp_dir=osp_dir)
            obs = cls(o_cris, sdesc)
        obs.spectral_window = \
            spec_win if spec_win is not None else MusesSpectralWindow(None,None)
        obs.spectral_window.add_bad_sample_mask(obs)
        if(fm_sv is not None):
            if(current_state is None):
                raise RuntimeError("If fm_sv is not None, current_state needs to also be not None")
            current_state.add_fm_state_vector_if_needed(
                fm_sv, obs.state_element_name_list(), [obs,])
        return obs
    
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
    '''Both omi and tropomi actually use reflectance rather than radiance. In addition,
    both the solar model and the radiance data have state elements that control the
    Dispersion for the data.

    This object captures the common behavior between the two.
    '''
    def __init__(self, muses_py_dict, sdesc, filter_list,
                 existing_obs=None, coeff=None, mp=None):
        self.filter_list = filter_list
        # Placeholder values if not passed in
        if(coeff is None):
            coeff = np.zeros((len(self.filter_list)*3))
            mp = rf.StateMappingLinear()
        super().__init__(muses_py_dict, sdesc, num_channels=len(self.filter_list),
                         coeff=coeff, mp=mp)

        # Grab values from existing_obs if available
        if(existing_obs is not None):
            self._freq_data = existing_obs._freq_data
            self._nesr_data = existing_obs._nesr_data
            self._bsamp = existing_obs._bsamp
            self._solar_interp = existing_obs._solar_interp
            self._earth_rad = existing_obs._earth_rad
            self._nesr = existing_obs._nesr
            self._solar_spectrum = existing_obs._solar_spectrum
        else:
            # Stash some values we use in later calculations. Note that the radiance data
            # is all smooshed together, so we separate this.
            #
            # It isn't clear here if the best indexing is the full
            # instrument (so 8 bands) with only some of the bands filled in, or
            # instead the index number into the passed in filter_list. For now, we
            # are using the index into the filter_list. We can possibly
            # reevaluate this - it wouldn't be huge change in the code we have here.
            self._freq_data = []
            self._nesr_data = []
            self._bsamp = []
            self._solar_interp = []
            self._earth_rad = []
            self._nesr = []
            self._solar_spectrum = []
            erad = muses_py_dict['Earth_Radiance']
            srad = muses_py_dict['Solar_Radiance']
            for i,flt in enumerate(filter_list):
                flt_sub = (erad['EarthWavelength_Filter'] == flt)
                self._freq_data.append(erad['Wavelength'][flt_sub])
                self._nesr_data.append(erad['EarthRadianceNESR'][flt_sub])
                self._bsamp.append(
                    (erad['EarthRadianceNESR'][flt_sub] <= 0.0)  |
                    (srad['AdjustedSolarRadiance'][flt_sub]<=0.0))
                self._earth_rad.append(erad['CalibratedEarthRadiance'][flt_sub])
                self._nesr.append(erad['EarthRadianceNESR'][flt_sub])
                
                # Note this looks wrong (why not use Solar_Radiance Wavelength here?),
                # but is actually correct. The solar data has already been interpolated
                # to the same wavelengths as  the Earth_Radiance, this happens in
                # daily_tropomi_irad for TROPOMI, and similarly for OMI. Not sure
                # why the original wavelengths are left in rad_info['Solar_Radiance'],
                # that is actually misleading.
                
                sol_domain = rf.SpectralDomain(erad['Wavelength'][flt_sub], rf.Unit("nm"))
                sol_range = rf.SpectralRange(srad["AdjustedSolarRadiance"][flt_sub],
                                             rf.Unit("ph / nm / s"))
                self._solar_spectrum.append(rf.Spectrum(sol_domain, sol_range))
                
                # Create a interpolator for the solar model, only using good data.
                solar_data = srad['AdjustedSolarRadiance'][flt_sub]
                orgwav_good = self._freq_data[i][self.bad_sample_mask(i) != True]
                solar_good = solar_data[self.bad_sample_mask(i) != True]
                self._solar_interp.append(LinearInterpolate(orgwav_good, solar_good))

        # Always create a new _solar_wav and _norm_rad_wav because the dispersion
        # will have independent values
        self._solar_wav = []
        self._norm_rad_wav = []
        for i,flt in enumerate(filter_list):
            self._solar_wav.append(MusesDispersion(
                self._freq_data[i], self.bad_sample_mask(i), self,
                0*len(self.filter_list)+i, None, order=1))
            self._norm_rad_wav.append(MusesDispersion(
                self._freq_data[i], self.bad_sample_mask(i), self,
                1*len(self.filter_list)+i, 2*len(self.filter_list)+i, order=2))

    def desc(self):
        return "MusesObservationReflectance"

    @property
    def filter_data(self) -> "list[list[str,int]]":
        self._filter_data_name = self.filter_list
        self._filter_data_swin = self._spectral_window
        return super().filter_data
    
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

    def solar_spectrum(self, sensor_index):
        '''Not sure how much sense it makes, but the RamanSioris gets
        it solar model from the observation. I suppose this sort of
        makes sense, because we already need the solar model.  It
        seems like this should be a separate thing, and perhaps at
        some point we will pull this out. But for now make this
        available here.

        Note this is the original, unaltered solar model - so without
        the TROPOMISOLARSHIFT or OMINRADWAV.
        '''
        return self.spectral_window.apply(self._solar_spectrum[sensor_index], sensor_index)

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
        '''Upper limit for SNR, we adjust uncertainty if we are greater than this.'''
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
    def __init__(self, muses_py_dict, sdesc, filter_list,
                 existing_obs=None, coeff=None, mp=None):
        '''Note you don't normally create an object of this class with the
        __init__. Instead, call one of the create_xxx class methods.'''
        super().__init__(muses_py_dict, sdesc, filter_list, existing_obs=existing_obs,
                         coeff=coeff, mp=mp)

    @classmethod
    def _read_data(cls, filename_list, irr_filename, cld_filename, xtrack_list, atrack,
                 utc_time, filter_list, calibration_filename=None, osp_dir=None):
        # Filter list should be in the same order as filename_list, and should be
        # things like "BAND3"
        if(calibration_filename is not None):
            # The existing py-retrieve code doesn't actually work with the
            # calibration_filename. This needs to get added before we can use this
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
        return (o_tropomi, sdesc)

    def desc(self):
        return "MusesTropomiObservation"

    @property
    def instrument_name(self):
        return "TROPOMI"
    
    @classmethod
    def create_from_filename(cls, filename_list, irr_filename, cld_filename, xtrack_list,
                             atrack, utc_time, filter_list, calibration_filename=None,
                             osp_dir=None):
        '''Create from just the filenames. Note that spectral window doesn't get
        set here, but this can be useful if you just want access to the underlying
        data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).'''
        o_tropomi, sdesc = cls._read_data(
            filename_list, irr_filename, cld_filename, xtrack_list,
            atrack, utc_time, filter_list, calibration_filename=calibration_filename,
            osp_dir=osp_dir)
        return cls(o_tropomi, sdesc, filter_list)
    
    @classmethod
    def create_from_id(cls, mid : MeasurementId,
                       existing_obs : 'cls',
                       current_state: 'Optional(CurrentState)',
                       spec_win: "Optional(MusesSpectralWindow)",
                       fm_sv: "Optional(rf.StateVector)",
                       osp_dir=None,
                       write_tropomi_radiance_pickle=False,
                       **kwargs):
        '''Create from a MeasurementId. If this depends on any state information, you can
        pass in the CurrentState. This can be given as None if you just want to use default
        values, e.g. you aren't doing a retrieval. If the CurrentState is supplied, you can
        also pass a StateVector to add this class to as needed.

        Note that VLIDORT depends on having a pickle file created. This is a bad interface,
        basically this is like a hidden variable. But to support the old code,
        we can optionally generate that pickle file.'''
        coeff = None
        mp = None
        if(existing_obs is not None):
            # Take data from existing observation
            if(current_state is not None):
                coeff, mp = current_state.object_state(existing_obs.state_element_name_list())
            obs = cls(existing_obs.muses_py_dict, existing_obs.sounding_desc,
                      existing_obs.filter_list, existing_obs=existing_obs,
                      coeff=coeff, mp=mp)
        else:
            filter_list = mid.filter_list_dict["TROPOMI"]
            if(current_state is not None):
                coeff,mp=current_state.object_state(cls.state_element_name_list_from_filter(filter_list))
            if(int(mid['TROPOMI_Rad_calRun_flag']) != 1):
                # The current py-retrieve code just silently ignores calibration,
                # see about line 614 of script_retrieval_setup_ms. We duplicate
                # this behavior, but go ahead and warn that we are doing that.
                logger.warning("Don't support calibration files yet. Ignoring TROPOMI_Rad_calRun_flag")
            irr_filename = mid['TROPOMI_IRR_filename']
            cld_filename = mid['TROPOMI_Cloud_filename']
            atrack = int(mid['TROPOMI_ATrack_Index'])
            utc_time = mid['TROPOMI_utcTime']
            filename_list = [mid[f"TROPOMI_filename_{flt}"]
                             for flt in filter_list]
            xtrack_list = [int(mid[f"TROPOMI_XTrack_Index_{flt}"])
                           for flt in filter_list]
            o_tropomi, sdesc = cls._read_data(
                filename_list, irr_filename, cld_filename, xtrack_list, atrack,
                utc_time, filter_list, osp_dir=osp_dir)
            obs = cls(o_tropomi, sdesc,filter_list, coeff=coeff, mp=mp)
            
        if(write_tropomi_radiance_pickle):
            # Save file needed by py-retrieve VLIDORT code
            pfname = os.path.normpath(f"{mid['initialGuessDirectory']}/../Radiance_TROPOMI_.pkl")
            if(not os.path.exists(pfname)):
                subprocess.run(["mkdir", "-p", os.path.dirname(pfname)])
                pickle.dump(obs.muses_py_dict, open(pfname, "wb"))

        obs.spectral_window = \
            spec_win if spec_win is not None else MusesSpectralWindow(None,None)
        obs.spectral_window.add_bad_sample_mask(obs)
        if(fm_sv is not None):
            if(current_state is None):
                raise RuntimeError("If fm_sv is not None, current_state needs to also be not None")
            current_state.add_fm_state_vector_if_needed(
                fm_sv, obs.state_element_name_list(), [obs,])
        return obs
    
    def snr_uplimit(self, sensor_index):
        '''Upper limit for SNR, we adjust uncertainty is we are greater than this.'''
        return 500.0

    @classmethod
    def state_element_name_list_from_filter(cls, filter_list):
        '''List of state element names for this observation'''
        res = []
        for flt in filter_list:
            res.append(f"TROPOMISOLARSHIFT{flt}")
        for flt in filter_list:
            res.append(f"TROPOMIRADIANCESHIFT{flt}")
        for flt in filter_list:
            res.append(f"TROPOMIRADSQUEEZE{flt}")
        return res
        
    
    def state_element_name_list(self):
        '''List of state element names for this observation'''
        return self.state_element_name_list_from_filter(self.filter_list)
    
class MusesOmiObservation(MusesObservationReflectance):
    '''Observation for OMI'''
    def __init__(self, muses_py_dict, sdesc, filter_list,
                 existing_obs=None, coeff=None, mp=None):
        '''Note you don't normally create an object of this class with the
        __init__. Instead, call one of the create_xxx class methods.'''
        super().__init__(muses_py_dict, sdesc, filter_list, existing_obs=existing_obs,
                         coeff=coeff, mp=mp)

    @classmethod
    def _read_data(cls, filename, xtrack_uv1, xtrack_uv2, atrack, utc_time,
                   calibration_filename, cld_filename=None, osp_dir=None):
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
        return (o_omi, sdesc)

    def desc(self):
        return "MusesOmiObservation"

    @property
    def instrument_name(self):
        return "OMI"
    
    @classmethod
    def create_from_filename(cls, filename, xtrack_uv1, xtrack_uv2, atrack, utc_time,
                             calibration_filename, filter_list, cld_filename=None, osp_dir=None):
        '''Create from just the filenames. Note that spectral window doesn't get
        set here, but this can be useful if you just want access to the underlying
        data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).'''
        o_omi, sdesc = cls._read_data(
            filename, xtrack_uv1, xtrack_uv2, atrack, utc_time, calibration_filename,
            cld_filename=cld_filename, osp_dir=osp_dir)
        return cls(o_omi, sdesc, filter_list)
    
    @classmethod
    def create_from_id(cls, mid : MeasurementId,
                       existing_obs : 'cls',
                       current_state: 'Optional(CurrentState)',
                       spec_win: "Optional(MusesSpectralWindow)",
                       fm_sv: "Optional(rf.StateVector)",
                       osp_dir=None,
                       **kwargs):
        '''Create from a MeasurementId. If this depends on any state information, you can
        pass in the CurrentState. This can be given as None if you just want to use default
        values, e.g. you aren't doing a retrieval. If the CurrentState is supplied, you can
        also pass a StateVector to add this class to as needed.'''
        coeff = None
        mp = None
        if(existing_obs is not None):
            if(current_state is not None):
                coeff, mp = current_state.object_state(existing_obs.state_element_name_list())
            obs = cls(existing_obs.muses_py_dict, existing_obs.sounding_desc,
                      existing_obs.filter_list, existing_obs=existing_obs,
                      coeff=coeff, mp=mp)
        else:
            filter_list = mid.filter_list_dict["OMI"]
            if(current_state is not None):
                coeff,mp=current_state.object_state(
                    cls.state_element_name_list_from_filter(filter_list))
            xtrack_uv1 = int(mid["OMI_XTrack_UV1_Index"])
            xtrack_uv2 = int(mid["OMI_XTrack_UV2_Index"])
            atrack = int(mid['OMI_ATrack_Index'])
            filename = mid["OMI_filename"]
            cld_filename = mid['OMI_Cloud_filename']
            utc_time = mid['OMI_utcTime']
            if(int(mid['OMI_Rad_calRun_flag']) != 1):
                calibration_filename = mid["omi_calibrationFilename"]
            else:
                logger.info("Calibration run. Disabling EOF application.")
                calibration_filename = None
            o_omi, sdesc = cls._read_data(
                filename, xtrack_uv1, xtrack_uv2, atrack, utc_time, calibration_filename,
                cld_filename=cld_filename, osp_dir=osp_dir)
            obs = cls(o_omi, sdesc,filter_list, coeff=coeff, mp=mp)
        obs.spectral_window = \
            spec_win if spec_win is not None else MusesSpectralWindow(None,None)
        obs.spectral_window.add_bad_sample_mask(obs)
        if(fm_sv is not None):
            if(current_state is None):
                raise RuntimeError("If fm_sv is not None, current_state needs to also be not None")
            current_state.add_fm_state_vector_if_needed(
                fm_sv, obs.state_element_name_list(), [obs,])
        return obs
        

    def snr_uplimit(self, sensor_index):
        '''Upper limit for SNR, we adjust uncertainty is we are greater than this.'''
        if(self.filter_list[sensor_index] == "UV2"):
            return 800.0
        return 500.0
        
    @classmethod
    def state_element_name_list_from_filter(cls, filter_list):
        '''List of state element names for this observation'''
        res = []
        for flt in filter_list:
            res.append(f"OMINRADWAV{flt}")
        for flt in filter_list:
            res.append(f"OMIODWAV{flt}")
        for flt in filter_list:
            res.append(f"OMIODWAVSLOPE{flt}")
        return res
    
    def state_element_name_list(self):
        '''List of state element names for this observation'''
        return self.state_element_name_list_from_filter(self.filter_list)

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
