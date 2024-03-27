from . import muses_py as mpy
from .refractor_uip import RefractorUip
from .cost_function_creator import (ForwardModelHandle, StateVectorHandle,
                                    ForwardModelHandleSet, StateVectorHandleSet)
from .osswrapper import osswrapper
from .refractor_capture_directory import muses_py_call
import refractor.framework as rf
import os
import numpy as np

# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel

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
                 obs, include_bad_sample=False,
                 **kwargs):
        super().__init__()
        self.instrument_name = instrument_name
        self.rf_uip = rf_uip
        self.obs = obs
        self.include_bad_sample = include_bad_sample
        self.kwargs = kwargs

    def bad_sample_mask(self, sensor_index):
        bmask = self.obs.bad_sample_mask(sensor_index)
        if(self.include_bad_sample):
            bmask[:] = False
        # This is the full bad sample mask, for all the indices. But here we only
        # want the portion that fits in the spectral window
        gindex = self.obs.spectral_window_with_bad_sample.grid_indexes(self.obs.spectral_domain_full(sensor_index), sensor_index)
        return bmask[list(gindex)]
    
    def setup_grid(self):
        # Nothing that we need to do for this
        pass
    
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index):
        if(sensor_index > 0):
            raise RuntimeError("sensor_index out of range")
        sd = np.concatenate([self.obs.spectral_domain(i).data for i in range(self.obs.num_channels)])
        return rf.SpectralDomain(sd, rf.Unit("nm"))

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
        # This is for the full set            
        gmask = self.bad_sample_mask(sensor_index) != True
        sd = self.spectral_domain(sensor_index)
        if(skip_jacobian):
            sr = rf.SpectralRange(rad[gmask], rf.Unit("sr^-1"))
        else:
            # jacobian is 1) on the forward model grid and
            # 2) tranposed from the ReFRACtor convention of the
            # column being the state vector variables. So
            # translate the oss jac to what we want from ReFRACtor

            sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name)
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
    def __init__(self, rf_uip : RefractorUip, instrument_name, obs,
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
        self.obs = obs

    def _fill_in_cache(self):
        if(self.cache_valid_flag):
            return
        with muses_py_call(self.rf_uip.run_dir, vlidort_cli=self.vlidort_cli,
                           vlidort_nstokes=self.vlidort_nstokes,
                           vlidort_nstreams=self.vlidort_nstreams):
            if(self.instrument_name == "TROPOMI"):
                jac, rad, _, success_flag = mpy.tropomi_fm(self.rf_uip.uip_all(self.instrument_name))
            elif(self.instrument_name == "OMI"):
                jac, rad, _, success_flag = mpy.omi_fm(self.rf_uip.uip_all(self.instrument_name))
            else:
                raise RuntimeError(f"Unrecognized instrument name {self.instrument_name}")
        # Haven't filled everything in yet, but mark as cache full.
        # otherwise bad_sample_mask and spectral_domain will enter an
        # infinite loop
        self.cache_valid_flag = True
        gmask =  np.concatenate([self.bad_sample_mask(i) != True for i in range(self.obs.num_channels)])
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
        sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name)
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
    def __init__(self, rf_uip : RefractorUip, obs, **kwargs):
        super().__init__(rf_uip, "TROPOMI", obs, **kwargs)

class MusesOmiForwardModel(MusesTropomiOrOmiForwardModelBase):
    def __init__(self, rf_uip : RefractorUip, obs, **kwargs):
        super().__init__(rf_uip, "OMI", obs, **kwargs)
        
class MusesCrisForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for CRiS instrument'''
    def __init__(self, rf_uip : RefractorUip, obs, **kwargs):
        super().__init__(rf_uip, "CRIS", obs, **kwargs)

class MusesAirsForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for Airs instrument'''
    def __init__(self, rf_uip : RefractorUip, obs, **kwargs):
        super().__init__(rf_uip, "AIRS", obs, **kwargs)
        
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
    
class MusesCrisForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, obs, svhandle,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "CRIS"):
            return (None, None)
        # This has already been handled below, by adding to the
        # default handle list
        #svhandle.add_handle(MusesStateVectorHandle(),
        #                    priority_order=-1)
        return (MusesCrisForwardModel(rf_uip,obs, **kwargs), obs)

class MusesAirsForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, obs, svhandle,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "AIRS"):
            return (None, None)
        # This has already been handled below, by adding to the
        # default handle list
        #svhandle.add_handle(MusesStateVectorHandle(),
        #                    priority_order=-1)
        #obs = MusesAirsObservation(rf_uip, obs_rad, meas_err, **kwargs)
        return (MusesAirsForwardModel(rf_uip,obs, **kwargs), obs)

class MusesTropomiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, obs, svhandle,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "TROPOMI"):
            return (None, None)
        fm = MusesTropomiForwardModel(rf_uip, obs, **kwargs)
        # We don't actually attach anything to the state vector, but
        # we want to make sure that the forward model gets attached
        # as a CacheInvalidatedObserver.
        svhandle.add_handle(MusesStateVectorObserverHandle(fm),
                            priority_order=1000)
        return (fm, obs)

class MusesOmiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, obs, svhandle,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "OMI"):
            return (None, None)
        fm = MusesOmiForwardModel(rf_uip,obs, **kwargs)
        # We don't actually attach anything to the state vector, but
        # we want to make sure that the forward model gets attached
        # as a CacheInvalidatedObserver.
        svhandle.add_handle(MusesStateVectorObserverHandle(fm),
                            priority_order=1000)
        return (fm, obs)

# The Muses code is the fallback, so add with the lowest priority
StateVectorHandleSet.add_default_handle(MusesStateVectorHandle(),
                                        priority_order=-1)
ForwardModelHandleSet.add_default_handle(MusesCrisForwardModelHandle(),
                                       priority_order=-1)
ForwardModelHandleSet.add_default_handle(MusesAirsForwardModelHandle(),
                                       priority_order=-1)
ForwardModelHandleSet.add_default_handle(MusesTropomiForwardModelHandle(),
                                       priority_order=-1)
ForwardModelHandleSet.add_default_handle(MusesOmiForwardModelHandle(),
                                       priority_order=-1)

__all__ = [ "StateVectorPlaceHolder",
            "MusesCrisForwardModel", 
            "MusesAirsForwardModel", 
            "MusesTropomiForwardModel", 
            "MusesOmiForwardModel", 
            "MusesStateVectorObserverHandle",
           ]

