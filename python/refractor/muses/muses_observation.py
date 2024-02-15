from .misc import osp_setup
from .priority_handle_set import PriorityHandleSet
import refractor.muses.muses_py as mpy
import os
import numpy as np
import refractor.framework as rf

class MusesObservation:
    '''It isn't clear what exactly we want in this class. The Observation
    in py-retrieve also include other things, like the SpectralDomain,
    bad samples, and various observation metadata like the solar and viewing
    geometry.

    For now, we wrap this up into a class. At least for now, we'll also keep
    the various dictionaries py-retrieve has like o_airs etc.

    We wrap up the existing py-retrieve calls for reading this data.

    We may modify this over time, but this is at least a good place to start.
    '''
    def __init__(self, muses_py_dict, channel_list):
        self.muses_py_dict = muses_py_dict
        self.channel_list = channel_list
        self._spectral_window = None
        self._sd = [None,]
        self._spec = [None,]
        
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
        self._sd = [None,]
        self._spec = [None,]

    def spectral_domain(self, spec_index):
        return self.spectrum(spec_index).spectral_domain

    def radiance(self, sensor_index, skip_jacobian=False):
        return self.spectrum(spec_index).spectral_range

    def spectrum(self, spec_index):
        # Not sure how to work with spec_index, we'll need need to sort that out
        if(spec_index != 0):
            raise RuntimeError("Not sure how to handle spec_index yet")
        if(self._spec[spec_index] is None):
            # By convention, sample index starts with 1. This was from OCO-2, I'm not
            # sure if that necessarily makes sense here or not. But I think we have code
            # that depends on the 1 base.
            freq = self.frequency_full()
            sindex = np.array(list(range(len(freq)))) + 1
            sd = rf.SpectralDomain(freq, sindex, rf.Unit("nm"))
            # More stuff goes here
            sr = rf.SpectralRange(self.radiance_full(), rf.Unit("sr^-1"), self.nesr_full())
            sp = rf.Spectrum(sd, sr)
            self._spec[spec_index] = self.spectral_window.apply(sp, spec_index)
        return self._spec[spec_index]
            
    def radiance_full(self):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def frequency_full(self):
        '''The full list of frequency, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def nesr_full(self):
        '''The full list of NESR, before we have removed bad samples or applied the
        microwindows.'''
        raise NotImplementedError

    def bad_sample_mask(self, sensor_index):
        # Not sure how to work with spec_index, we'll need need to sort that out
        if(spec_index != 0):
            raise RuntimeError("Not sure how to handle spec_index yet")
        # I think all the radiance data acts the same way, we throw out data with a negative
        # nesr
        return np.array(self.nesr_full() < 0)
        

# muses_forward_model has an older observation class named MusesAirsObservation.
# Short term we add a "New" here. We should sort that out -
# we are somewhat rewriting MusesObservationBase to not use a UIP. This will
# probably get married into one clases, but we aren't ready to do that yet.

class MusesAirsObservationNew(MusesObservation):
    def __init__(self, filename, xtrack, atrack, channel_list, osp_dir=None):
        i_fileid = {}
        i_fileid['preferences'] = {'AIRS_filename' : os.path.abspath(filename),
                                   'AIRS_XTrack_Index' : xtrack,
                                   'AIRS_ATrack_Index' : atrack}
        i_window = []
        for cname in channel_list:
            i_window.append({'filter': cname})
        with(osp_setup(osp_dir)):
            o_airs = mpy.read_airs(i_fileid, i_window)
        super().__init__(o_airs, channel_list)

    def radiance_full(self):
        '''The full list of radiance, before we have removed bad samples or applied the
        microwindows.'''
        return self.muses_py_dict['radiance']['radiance']

    def frequency_full(self):
        '''The full list of frequency, before we have removed bad samples or applied the
        microwindows.'''
        return self.muses_py_dict['radiance']['frequency']

    def nesr_full(self):
        '''The full list of NESR, before we have removed bad samples or applied the
        microwindows.'''
        return self.muses_py_dict['radiance']['NESR']
        
