from .misc import osp_setup
from .priority_handle_set import PriorityHandleSet
from .cost_function_creator import ObservationHandle, ObservationHandleSet
import refractor.muses.muses_py as mpy
import os
import numpy as np
import refractor.framework as rf
import abc

    
class MusesObservation(rf.ObservationSvImpBase):
    '''It isn't clear what exactly we want in this class. The Observation
    in py-retrieve also include other things, like the SpectralDomain,
    bad samples, and various observation metadata like the solar and viewing
    geometry.

    For now, we wrap this up into a class. At least for now, we'll also keep
    the various dictionaries py-retrieve has like o_airs etc.

    We wrap up the existing py-retrieve calls for reading this data.

    We may modify this over time, but this is at least a good place to start.
    '''
    def __init__(self, muses_py_dict, sdesc, num_channels=1):
        super().__init__([])
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
        # I think all the radiance data acts the same way, we throw out data with a negative
        # nesr
        return np.array(self.nesr_full(sensor_index) < 0)

class MusesObservationHandle(ObservationHandle):
    '''A lot of our observation classes just map a name to
    a object of a specific class. This handles this generic construction.'''
    def __init__(self, instrument_name, obs_cls):
        self.instrument_name = instrument_name
        self.obs_cls = obs_cls
        # Keep the same observation around as long as the target doesn't
        # change - we just update the spectral windows.
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
            swin.bad_sample_mask(self.obs.bad_sample_mask(0), 0)
        self.obs.spectral_window = swin
        return self.obs
        

# muses_forward_model has an older observation class named MusesAirsObservation.
# Short term we add a "New" here. We should sort that out -
# we are somewhat rewriting MusesObservationBase to not use a UIP. This will
# probably get married into one clases, but we aren't ready to do that yet.

class MusesAirsObservationNew(MusesObservation):
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


class MusesCrisObservationNew(MusesObservation):
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

# We'll probably pull some of this out, omi and tropomi are similar    
class MusesTropomiObservationNew(MusesObservation):
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
        self.filter_list = filter_list
        super().__init__(o_tropomi, sdesc)

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
        for i,flt in enumerate(filter_list):
            flt_sub = (o_tropomi['Earth_Radiance']['EarthWavelength_Filter'] == flt)
            self._freq_data.append(o_tropomi['Earth_Radiance']['Wavelength'][flt_sub])
            self._nesr_data.append(o_tropomi['Earth_Radiance']['EarthRadianceNESR'][flt_sub])
            self._bsamp.append(
                (o_tropomi['Earth_Radiance']['EarthRadianceNESR'][flt_sub] <= 0.0)  |
                 (o_tropomi['Solar_Radiance']['AdjustedSolarRadiance'][flt_sub]<=0.0))

    def spectral_domain(self, sensor_index):
        # Since self.radiance involves more calculation, give a optimized version of
        # the spectral_domain function for when we just want this.
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        sd = rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
        return self.spectral_window.apply(sd, sensor_index)
        
    def radiance_full(self, sensor_index, skip_jacobian=False):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        if(sensor_index < 0 or sensor_index >= self.num_channels):
            raise RuntimeError("sensor_index out of range")
        raise NotImplementedError

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
    
ObservationHandleSet.add_default_handle(MusesObservationHandle("AIRS", MusesAirsObservationNew))
ObservationHandleSet.add_default_handle(MusesObservationHandle("CRIS", MusesCrisObservationNew))

__all__ = ["MusesAirsObservationNew", "MusesObservation", "MusesObservationHandle",
           "MusesCrisObservationNew", "MusesTropomiObservationNew",]
