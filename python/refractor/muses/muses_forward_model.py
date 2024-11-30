from . import muses_py as mpy
from .refractor_uip import RefractorUip
from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
from .osswrapper import osswrapper
from .refractor_capture_directory import muses_py_call
import refractor.framework as rf
from loguru import logger
import os
import numpy as np

# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel

# There are a number of things in common with the different forward models,
# so we capture these in these base classes.

class MusesForwardModelBase(rf.ForwardModel):
    '''Common behavior for the different MUSES forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name,
                 obs, **kwargs):
        super().__init__()
        self.instrument_name = instrument_name
        self.rf_uip = rf_uip
        self.obs = obs
        self.kwargs = kwargs

    def bad_sample_mask(self, sensor_index):
        bmask = self.obs.bad_sample_mask(sensor_index)
        if(self.obs.spectral_window.include_bad_sample):
            bmask[:] = False
        # This is the full bad sample mask, for all the indices. But here we only
        # want the portion that fits in the spectral window
        with self.obs.modify_spectral_window(include_bad_sample=True):
            sd = self.obs.spectral_domain_full(sensor_index)
            gindex = self.obs.spectral_window.grid_indexes(sd, sensor_index)
        return bmask[list(gindex)]
    
    def setup_grid(self):
        # Nothing that we need to do for this
        pass
    
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index):
        if(sensor_index > 0):
            raise RuntimeError("sensor_index out of range")
        sdlist = []
        sd = np.concatenate([self.obs.spectral_domain(i).data
                             for i in range(self.obs.num_channels)])
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
        # We ran into this issue because the fm_oss_stack uses the UIP for the
        # frequency grid, and self.obs uses MusesSpectralWindow. These are
        # generally the same, but we might get odd situations where they aren't.
        # Give a clearer error message.
        if(gmask.shape[0] != rad.shape[0]):
            raise RuntimeError(f"gmask and rad don't match in size. gmask size is {gmask.shape[0]} and rad size if {rad.shape[0]}")
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

class MusesTropomiOrOmiForwardModelBase(MusesForwardModelBase):
    '''Common behavior for the omi/tropomi based forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name, obs,
                 vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
                 vlidort_nstokes=2,
                 vlidort_nstreams=4,
                 **kwargs):
        MusesForwardModelBase.__init__(self, rf_uip, instrument_name, obs=None,
                                       **kwargs)
        self.vlidort_nstreams = vlidort_nstreams
        self.vlidort_nstokes = vlidort_nstokes
        self.vlidort_cli = vlidort_cli
        self.obs = obs

    def radiance(self, sensor_index, skip_jacobian = False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
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
        gmask =  np.concatenate([self.bad_sample_mask(i) != True
                                 for i in range(self.obs.num_channels)])
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
        return rf.Spectrum(sd, sr)
    
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

class MusesTesForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for TES instrument'''
    def __init__(self, rf_uip : RefractorUip, obs, **kwargs):
        super().__init__(rf_uip, "TES", obs, **kwargs)
        
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
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.coeff = sv.state[self.pstart:(self.pstart+self.plen)]

    def state_vector_name(self, sv, sv_namev):
        svnm = ["",] * len(sv.state)
        for i in range(self.plen):
            svnm[i+self.pstart] = f"{self.species_name} coefficient {i+1}"
        self.sv_name = svnm
        super().state_vector_name(sv,sv_namev)

class MusesForwardModelHandle(ForwardModelHandle):
    def __init__(self, instrument_name, cls, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        self.instrument_name = instrument_name
        self.cls = cls
        
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      **kwargs):
        if(instrument_name != self.instrument_name):
            return None
        logger.debug(f"Creating forward model {self.cls.__name__}")
        return self.cls(rf_uip_func(instrument=instrument_name), obs, **kwargs)

# The Muses code is the fallback, so add with the lowest priority
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle("CRIS", MusesCrisForwardModel), priority_order=-1)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle("AIRS", MusesAirsForwardModel), priority_order=-1)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle("TES", MusesTesForwardModel), priority_order=-1)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle("TROPOMI", MusesTropomiForwardModel), priority_order=-1)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle("OMI", MusesOmiForwardModel), priority_order=-1)

__all__ = [ "MusesCrisForwardModel", 
            "MusesAirsForwardModel", 
            "MusesTesForwardModel", 
            "MusesTropomiForwardModel", 
            "MusesOmiForwardModel", 
           ]

