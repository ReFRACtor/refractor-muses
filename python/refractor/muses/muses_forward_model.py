from . import muses_py as mpy
from .refractor_uip import RefractorUip
from .cost_func_creator import (InstrumentHandle, StateVectorHandle,
                                InstrumentHandleSet, StateVectorHandleSet)
from .osswrapper import osswrapper
import refractor.framework as rf
import os
import numpy as np

# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel, and wrapper for Observation. This is used by
# CostFuncCreator to use the using muses-py code for different
# instruments rather than ReFRACtor.

class MusesObservationBase(rf.ObservationSvImpBase):
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 obs_rad, meas_err):
        super().__init__([])
        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        self.obs_rad = obs_rad
        self.meas_err = meas_err
        
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index, include_bad_sample=False):
        return rf.SpectralDomain(self.rf_uip.frequency_list(self.instrument_name), rf.Unit("nm"))

    def radiance(self, sensor_index, skip_jacobian = False,
                 include_bad_sample=False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        sd = self.spectral_domain(sensor_index)
        subset = [t == self.instrument_name for t in self.rf_uip.instrument_list()]
        if(self.obs_rad is not None):
            r = self.obs_rad[subset]
            uncer = self.meas_err[subset]
        else:
            r = np.zeros(sd.data.shape)
            uncer = np.ones(r.shape)
        sr = rf.SpectralRange(r, rf.Unit("sr^-1"), uncer)
        if(sr.data.shape != sd.data.shape):
            raise RuntimeError("sd and sr are different lengths")
        return rf.Spectrum(sd, sr)

# There are a number of things in common with the different forward models,
# so we capture these in these base classes.

class MusesForwardModelBase(rf.ForwardModel):
    '''Common behavior for the different MUSES forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 use_full_state_vector=False):
        super().__init__()
        self.instrument_name = instrument_name
        self.rf_uip = rf_uip
        self.use_full_state_vector = use_full_state_vector
        
    def setup_grid(self):
        # Nothing that we need to do for this
        pass
    
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index):
        return rf.SpectralDomain(self.rf_uip.frequency_list(self.instrument_name), rf.Unit("nm"))

class MusesOssForwardModelBase(MusesForwardModelBase):
    '''Common behavior for the OSS based forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 **kwargs):
        super().__init__(rf_uip, instrument_name, **kwargs)
        
    def radiance(self, sensor_index, skip_jacobian = False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        with osswrapper(self.rf_uip.uip):
            rad, jac = mpy.fm_oss_stack(self.rf_uip.uip_all(self.instrument_name))
        sd = self.spectral_domain(sensor_index)
        if(skip_jacobian):
            sr = rf.SpectralRange(rad, rf.Unit("sr^-1"))
        else:
            # jacobian is 1) on the forward model grid and
            # 2) tranposed from the ReFRACtor convention of the
            # column being the state vector variables. So
            # translate the oss jac to what we want from ReFRACtor

            if(self.use_full_state_vector):
                jac = jac.transpose()
            else:
                sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name)
                jac = np.matmul(sub_basis_matrix, jac).transpose()
            if(self.rf_uip.is_bt_retrieval()):
                # Only one column has data, although oss returns a larger
                # jacobian. Note that fm_wrapper just "knows" this, it
                # would be nice if this wasn't sort of magic knowledge.
                jac = jac[:,0:1]
            a = rf.ArrayAd_double_1(rad, jac)
            sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)
    
class MusesCrisForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for CRiS instrument'''
    def __init__(self, rf_uip : RefractorUip, use_full_state_vector=False):
        super().__init__(rf_uip, "CRIS",
                         use_full_state_vector=use_full_state_vector)

class MusesCrisObservation(MusesObservationBase):
    '''Wrapper that just returns the passed in measured radiance
    and uncertainty for CRIS'''
    def __init__(self, rf_uip : RefractorUip, obs_rad, meas_err):
        super().__init__(rf_uip, "CRIS", obs_rad, meas_err)

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
    def add_sv(self, sv, species_name, pstart, plen):
        svh = StateVectorPlaceHolder(pstart, plen, species_name)
        sv.add_observer_and_keep_reference(svh)
        return True
    
class MusesCrisInstrumentHandle(InstrumentHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        
    def fm_and_obs(self, instrument_name, rf_uip, svhandle,
                   use_full_state_vector=False,
                   obs_rad=None, meas_err=None, **kwargs):
        if(instrument_name != "CRIS"):
            return (None, None)
        # This has already been handled below, by adding to the
        # default handle list
        #svhandle.add_handle(MusesStateVectorHandle(),
        #                    priority_order=-1)
        return (MusesCrisForwardModel(rf_uip,use_full_state_vector=use_full_state_vector),
                MusesCrisObservation(rf_uip, obs_rad, meas_err))
        


# The Muses code is the fallback, so add with the lowest priority
StateVectorHandleSet.add_default_handle(MusesStateVectorHandle(),
                                        priority_order=-1)
InstrumentHandleSet.add_default_handle(MusesCrisInstrumentHandle(),
                                       priority_order=-1)

__all__ = ["MusesObservationBase", "MusesForwardModelBase", 
           "MusesOssForwardModelBase", "MusesCrisForwardModel",
           "MusesCrisObservation", "StateVectorPlaceHolder",
           "MusesCrisInstrumentHandle"]

