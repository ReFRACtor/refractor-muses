from . import muses_py as mpy
from .refractor_uip import RefractorUip
from .fm_obs_creator import (InstrumentHandle, StateVectorHandle,
                             InstrumentHandleSet, StateVectorHandleSet)
from .osswrapper import osswrapper
from .refractor_capture_directory import muses_py_call
import refractor.framework as rf
import os
import numpy as np

# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel, and wrapper for Observation. This is used by
# FmObsCreator to use the using muses-py code for different
# instruments rather than ReFRACtor.

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class MusesObservationBase(rf.ObservationSvImpBase):
    # Note the handling of include_bad_sample is important here. muses-py
    # expects to get all the samples in the forward model run in the routine
    # run_forward_model/fm_wrapper. I'm not sure what it does with the bad
    # data, but we need to have the ability to include it.
    # run_retrieval/residual_fm_jacobian on the other hand does the normal
    # filtering of bad samples. We handle this by toggling the behavior of
    # bad_sample_mask, either masking bad samples or having a empty mask that
    # lets everything pass through.
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 obs_rad, meas_err, include_bad_sample=False, **kwargs):
        super().__init__([])
        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        self.obs_rad = obs_rad
        self.meas_err = meas_err
        self.include_bad_sample = include_bad_sample
        
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index, inc_bad_sample=False):
        gmask = self.bad_sample_mask(sensor_index) != True
        if(inc_bad_sample):
            gmask[:] = True
        return rf.SpectralDomain(self.rf_uip.frequency_list(self.instrument_name)[gmask], rf.Unit("nm"))

    def bad_sample_mask(self, sensor_index):
        subset = [t == self.instrument_name for t in self.rf_uip.instrument_list]
        uncer = self.meas_err[subset]
        bmask = np.array(uncer < 0)
        if(self.include_bad_sample):
            bmask[:] = False
        return bmask
        
    def radiance_all_with_bad_sample(self):
        return self.radiance(0, skip_jacobian=True, inc_bad_sample=True)
    
    def radiance(self, sensor_index, skip_jacobian = False,
                 inc_bad_sample=False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        gmask = self.bad_sample_mask(sensor_index) != True
        if(inc_bad_sample):
            gmask[:] = True
        sd = self.spectral_domain(sensor_index, inc_bad_sample)
        subset = [t == self.instrument_name for t in self.rf_uip.instrument_list]
        r = self.obs_rad[subset][gmask]
        uncer = self.meas_err[subset][gmask]
        sr = rf.SpectralRange(r, rf.Unit("sr^-1"), uncer)
        if(sr.data.shape != sd.data.shape):
            raise RuntimeError("sd and sr are different lengths")
        return rf.Spectrum(sd, sr)

# There are a number of things in common with the different forward models,
# so we capture these in these base classes.

class MusesForwardModelBase(rf.ForwardModel):
    '''Common behavior for the different MUSES forward models'''
    # Note the handling of include_bad_sample is important here. muses-py
    # expects to get all the samples in the forward model run in the routine
    # run_forward_model/fm_wrapper. I'm not sure what it does with the bad
    # data, but we need to have the ability to include it.
    # run_retrieval/residual_fm_jacobian on the other hand does the normal
    # filtering of bad samples. We handle this by toggling the behavior of
    # bad_sample_mask, either masking bad samples or having a empty mask that
    # lets everything pass through.
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 obs, use_full_state_vector=True, include_bad_sample=False,
                 **kwargs):
        super().__init__()
        self.instrument_name = instrument_name
        self.rf_uip = rf_uip
        self.use_full_state_vector = use_full_state_vector
        self.obs = obs
        self.include_bad_sample = include_bad_sample
        self.kwargs = kwargs

    def bad_sample_mask(self, sensor_index):
        bmask = self.obs.bad_sample_mask(sensor_index)
        if(self.include_bad_sample):
            bmask[:] = False
        return bmask
    
    def setup_grid(self):
        # Nothing that we need to do for this
        pass
    
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index):
        gmask = self.bad_sample_mask(sensor_index) != True
        return rf.SpectralDomain(self.rf_uip.frequency_list(self.instrument_name)[gmask], rf.Unit("nm"))

# Wrapper so we can get timing at a top level of ReFRACtor relative to the rest of the code
# using something like --profile-svg in pytest
class RefractorForwardModel(rf.ForwardModel):
    def __init__(self, fm):
        super().__init__()
        self.fm = fm

    def setup_grid(self):
        self.fm.setup_grid()
    
    def _v_num_channels(self):
        return self.fm.num_channels

    def spectral_domain(self, sensor_index):
        return self.fm.spectral_domain(sensor_index)

    def radiance(self, sensor_index, skip_jacobian = False):
        print("hi, in radiance")
        return self.fm.radiance(sensor_index, skip_jacobian)
        
class MusesOssForwardModelBase(MusesForwardModelBase):
    '''Common behavior for the OSS based forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name, obs,
                 **kwargs):
        super().__init__(rf_uip, instrument_name, obs, **kwargs)
        
    def radiance(self, sensor_index, skip_jacobian = False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        with osswrapper(self.rf_uip.uip):
            rad, jac = mpy.fm_oss_stack(self.rf_uip.uip_all(self.instrument_name))
        gmask = self.bad_sample_mask(sensor_index) != True
        sd = self.spectral_domain(sensor_index)
        if(skip_jacobian):
            sr = rf.SpectralRange(rad[gmask], rf.Unit("sr^-1"))
        else:
            # jacobian is 1) on the forward model grid and
            # 2) tranposed from the ReFRACtor convention of the
            # column being the state vector variables. So
            # translate the oss jac to what we want from ReFRACtor

            sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name, use_full_state_vector=self.use_full_state_vector)
            if(jac is not None and jac.ndim > 0):
                if(self.rf_uip.is_bt_retrieval):
                    # Only one column has data, although oss returns a larger
                    # jacobian. Note that fm_wrapper just "knows" this, it
                    # would be nice if this wasn't sort of magic knowledge.
                    jac = jac.transpose()[:,0:1]
                else:
                    jac = np.matmul(sub_basis_matrix, jac).transpose()
                a = rf.ArrayAd_double_1(rad[gmask], jac[gmask,:])
            else:
                a = rf.ArrayAd_double_1(rad[gmask])
            sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)

class MusesTropomiOrOmiForwardModelBase(MusesForwardModelBase,
                                        rf.CacheInvalidatedObserver):
    '''Common behavior for the omi/tropomi based forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
                 vlidort_nstokes=2,
                 vlidort_nstreams=4,
                 include_bad_sample=False,
                 **kwargs):
        MusesForwardModelBase.__init__(self, rf_uip, instrument_name, obs=None,
                                       **kwargs)
        rf.CacheInvalidatedObserver.__init__(self)
        
        self.vlidort_nstreams = vlidort_nstreams
        self.vlidort_nstokes = vlidort_nstokes
        self.vlidort_cli = vlidort_cli
        self.include_bad_sample = include_bad_sample
        # For performance reasons, we get both the radiance and obs
        # in one step - holding this in a cache.
        self.obs_rad = None
        self.rad_spec = None

    def bad_sample_mask(self, sensor_index):
        self._fill_in_cache()
        bmask = self.obs_rad["measured_nesr"] < 0
        if(self.include_bad_sample):
            bmask[:] = False
        return bmask

    def _fill_in_cache(self):
        if(self.cache_valid_flag):
            return
        with muses_py_call(self.rf_uip.run_dir, vlidort_cli=self.vlidort_cli,
                           vlidort_nstokes=self.vlidort_nstokes,
                           vlidort_nstreams=self.vlidort_nstreams):
            if(self.instrument_name == "TROPOMI"):
                jac, rad, self.obs_rad, success_flag = mpy.tropomi_fm(self.rf_uip.uip_all(self.instrument_name))
            elif(self.instrument_name == "OMI"):
                jac, rad, self.obs_rad, success_flag = mpy.omi_fm(self.rf_uip.uip_all(self.instrument_name))
            else:
                raise RuntimeError(f"Unrecognized instrument name {self.instrument_name}")
        # Haven't filled everything in yet, but mark as cache full.
        # otherwise bad_sample_mask and spectral_domain will enter an
        # infinite loop
        self.cache_valid_flag = True
        gmask = self.bad_sample_mask(0) != True
        sd = self.spectral_domain(0)
        # jacobian is 1) on the forward model grid and
        # 2) transposed from the ReFRACtor convention of the
        # column being the state vector variables. So
        # translate the oss jac to what we want from ReFRACtor
        # The logic in pack_omi_jacobian and pack_tropomi_jacobian
        # over counts the size of atmosphere jacobians by 1 for each
        # species. This is harmless,
        # it gives an extra row of zeros that then gets trimmed before leaving
        # fm_wrapper. But because we are calling the lower level function
        # ourselves we need to trim this.
        sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name, use_full_state_vector=self.use_full_state_vector)
        if(jac is not None and jac.shape[0] > 0 and sub_basis_matrix.shape[1] > 0):
            jac = np.matmul(sub_basis_matrix, jac[:sub_basis_matrix.shape[1],:]).transpose()
            a = rf.ArrayAd_double_1(rad[gmask], jac[gmask,:])
        else:
            a = rf.ArrayAd_double_1(rad[gmask])
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        self.rad_spec = rf.Spectrum(sd, sr)
        self.cache_valid_flag = True
        
    def radiance(self, sensor_index, skip_jacobian = False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        self._fill_in_cache()
        return self.rad_spec

class MusesTropomiOrOmiObservation(rf.ObservationSvImpBase):
    def __init__(self, fm : MusesTropomiOrOmiForwardModelBase,
                 include_bad_sample=False, **kwargs):
        super().__init__([])
        self.fm = fm
        self.include_bad_sample = include_bad_sample

    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.fm, self.include_bad_sample))
    
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index, inc_bad_sample=False):
        gmask = self.bad_sample_mask(sensor_index) != True
        if(inc_bad_sample):
            gmask[:] = True
        return rf.SpectralDomain(self.fm.rf_uip.frequency_list(self.fm.instrument_name)[gmask], rf.Unit("nm"))

    def bad_sample_mask(self, sensor_index):
        self.fm._fill_in_cache()
        uncer = self.fm.obs_rad["measured_nesr"]
        bsamp = np.array(uncer < 0)
        if(self.include_bad_sample):
            bsamp[:] = False
        return bsamp

    def radiance_all_with_bad_sample(self):
        return self.radiance(0, skip_jacobian=True, inc_bad_sample=True)
    
    def radiance(self, sensor_index, skip_jacobian = False,
                 inc_bad_sample=False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        self.fm._fill_in_cache()
        gmask = self.bad_sample_mask(sensor_index) != True
        if(inc_bad_sample):
            gmask[:] = True
        sd = self.spectral_domain(sensor_index, inc_bad_sample)
        r = self.fm.obs_rad["measured_radiance_field"][gmask]
        uncer = self.fm.obs_rad["measured_nesr"][gmask]
        sr = rf.SpectralRange(r, rf.Unit("sr^-1"), uncer)
        if(sr.data.shape != sd.data.shape):
            raise RuntimeError("sd and sr are different lengths")
        return rf.Spectrum(sd, sr)

class MusesTropomiObservation(MusesTropomiOrOmiObservation):
    pass

class MusesOmiObservation(MusesTropomiOrOmiObservation):
    pass

class MusesStateVectorObserverHandle(StateVectorHandle):
    def __init__(self, fm : MusesTropomiOrOmiForwardModelBase):
        super().__init__()
        self.fm = fm
        
    def add_sv(self, sv, species_name, pstart, plen, **kwargs):
        # Always pass the handling of this on, but for the start of
        # the state vector add fm as a cache observer
        if(pstart == 0):
            sv.add_cache_invalidated_observer(self.fm)
        return False
    
class MusesTropomiForwardModel(MusesTropomiOrOmiForwardModelBase):
    def __init__(self, rf_uip : RefractorUip, **kwargs):
        super().__init__(rf_uip, "TROPOMI", **kwargs)
        
    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.rf_uip, self.kwargs))

class MusesOmiForwardModel(MusesTropomiOrOmiForwardModelBase):
    def __init__(self, rf_uip : RefractorUip, **kwargs):
        super().__init__(rf_uip, "OMI", **kwargs)
    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.rf_uip, self.kwargs))
        
class MusesCrisForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for CRiS instrument'''
    def __init__(self, rf_uip : RefractorUip, obs, use_full_state_vector=True, **kwargs):
        super().__init__(rf_uip, "CRIS", obs,
                         use_full_state_vector=use_full_state_vector, **kwargs)
    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.rf_uip, self.obs, self.kwargs))

class MusesCrisObservation(MusesObservationBase):
    '''Wrapper that just returns the passed in measured radiance
    and uncertainty for CRIS'''
    def __init__(self, rf_uip : RefractorUip, obs_rad, meas_err, **kwargs):
        super().__init__(rf_uip, "CRIS", obs_rad, meas_err, **kwargs)
        
    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.rf_uip, self.obs_rad, self.meas_err))

class MusesAirsForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for Airs instrument'''
    def __init__(self, rf_uip : RefractorUip, obs, use_full_state_vector=True, **kwargs):
        super().__init__(rf_uip, "AIRS",
                         obs,
                         use_full_state_vector=use_full_state_vector,
                         **kwargs)
        
    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.rf_uip, self.obs, self.kwargs))

class MusesAirsObservation(MusesObservationBase):
    '''Wrapper that just returns the passed in measured radiance
    and uncertainty for AIRS'''
    def __init__(self, rf_uip : RefractorUip, obs_rad, meas_err, **kwargs):
        super().__init__(rf_uip, "AIRS", obs_rad, meas_err, **kwargs)
        
    def __reduce__(self):
        return (_new_from_init,
                (self.__class__, self.rf_uip, self.obs_rad, self.meas_err))
        
class StateVectorPlaceHolder(rf.StateVectorObserver):
    '''Place holder for parts of the StateVector that ReFRACtor objects
    don't need. Just gives the right name in the state vector, and
    act as a placeholder for any future stuff.'''
    def __init__(self, pstart, plen, species_name):
        super().__init__()
        self.pstart = pstart
        self.plen = plen
        self.species_name = species_name
        self.coeff = None

    def notify_update(self, sv):
        self.coeff = sv.state[self.pstart:(self.pstart+self.plen)]

    def state_vector_name(self, sv, sv_namev):
        svnm = ["",] * len(sv.state)
        for i in range(self.plen):
            svnm[i+self.pstart] = f"{self.species_name} coefficient {i+1}"
        self.sv_name = svnm
        super().state_vector_name(sv,sv_namev)

class MusesStateVectorHandle(StateVectorHandle):
    def add_sv(self, sv, species_name, pstart, plen, **kwargs):
        svh = StateVectorPlaceHolder(pstart, plen, species_name)
        sv.add_observer_and_keep_reference(svh)
        return True
    
class MusesCrisInstrumentHandle(InstrumentHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, svhandle,
                   use_full_state_vector=True,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "CRIS"):
            return (None, None)
        # This has already been handled below, by adding to the
        # default handle list
        #svhandle.add_handle(MusesStateVectorHandle(),
        #                    priority_order=-1)
        obs = MusesCrisObservation(rf_uip, obs_rad, meas_err, **kwargs)
        return (MusesCrisForwardModel(rf_uip,obs, use_full_state_vector=use_full_state_vector, **kwargs), obs)

class MusesAirsInstrumentHandle(InstrumentHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, svhandle,
                   use_full_state_vector=True,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "AIRS"):
            return (None, None)
        # This has already been handled below, by adding to the
        # default handle list
        #svhandle.add_handle(MusesStateVectorHandle(),
        #                    priority_order=-1)
        obs = MusesAirsObservation(rf_uip, obs_rad, meas_err, **kwargs)
        return (MusesAirsForwardModel(rf_uip,obs,use_full_state_vector=use_full_state_vector, **kwargs), obs)

class MusesTropomiInstrumentHandle(InstrumentHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, svhandle,
                   use_full_state_vector=True,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "TROPOMI"):
            return (None, None)
        fm = MusesTropomiForwardModel(rf_uip,use_full_state_vector=use_full_state_vector, **kwargs)
        # We don't actually attach anything to the state vector, but
        # we want to make sure that the forward model gets attached
        # as a CacheInvalidatedObserver.
        svhandle.add_handle(MusesStateVectorObserverHandle(fm),
                            priority_order=1000)
        return (fm, MusesTropomiObservation(fm, **kwargs))

class MusesOmiInstrumentHandle(InstrumentHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, svhandle,
                   use_full_state_vector=True,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "OMI"):
            return (None, None)
        fm = MusesOmiForwardModel(rf_uip,use_full_state_vector=use_full_state_vector, **kwargs)
        # We don't actually attach anything to the state vector, but
        # we want to make sure that the forward model gets attached
        # as a CacheInvalidatedObserver.
        svhandle.add_handle(MusesStateVectorObserverHandle(fm),
                            priority_order=1000)
        return (fm, MusesOmiObservation(fm, **kwargs))

# The Muses code is the fallback, so add with the lowest priority
StateVectorHandleSet.add_default_handle(MusesStateVectorHandle(),
                                        priority_order=-1)
InstrumentHandleSet.add_default_handle(MusesCrisInstrumentHandle(),
                                       priority_order=-1)
InstrumentHandleSet.add_default_handle(MusesAirsInstrumentHandle(),
                                       priority_order=-1)
InstrumentHandleSet.add_default_handle(MusesTropomiInstrumentHandle(),
                                       priority_order=-1)
InstrumentHandleSet.add_default_handle(MusesOmiInstrumentHandle(),
                                       priority_order=-1)

__all__ = [ "StateVectorPlaceHolder",
            "MusesCrisForwardModel", "MusesCrisObservation", 
           "MusesAirsForwardModel", "MusesAirsObservation",
            "MusesTropomiForwardModel", "MusesTropomiObservation",
            "MusesOmiForwardModel", "MusesOmiObservation",
            "MusesStateVectorObserverHandle",
           ]

