from . import muses_py as mpy
from .refractor_uip import RefractorUip
import refractor.framework as rf
import os
import numpy as np

# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel.
# Also set up Observation

# There are a number of things in common with the different forward models,
# so we capture these in these base classes.

class MusesForwardModelBase(rf.ForwardModel):
    '''Common behavior for the different MUSES forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name):
        super().__init__()
        self.instrument_name = instrument_name
        self.rf_uip = rf_uip
        # The jacobians from muses forward model routines only contains
        # the subset of
        # the columns that are listed in uip_all["jacobians"]
        self.sub_basis_matrix = rf_uip.ret_info["basis_matrix"][:,[t in list(self.uip_all["jacobians"]) for t in rf_uip.uip["speciesListFM"]]]

    def setup_grid(self):
        # Nothing that we need to do for this
        pass
    
    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index):
        return rf.SpectralDomain(self.uip_all["frequencyList"], rf.Unit("nm"))

    @property
    def uip_all(self):
        return mpy.struct_combine(self.rf_uip.uip, self.rf_uip.uip[f"uip_{self.instrument_name}"])

class MusesOssForwardModelBase(MusesForwardModelBase):
    '''Common behavior for the OSS based forward models'''
    def __init__(self, rf_uip : RefractorUip, instrument_name):
        super().__init__(rf_uip, instrument_name)
        
    def radiance(self, sensor_index, skip_jacobian = False):
        if(sensor_index !=0):
            raise ValueError("sensor_index must be 0")
        try:
            os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
            uall = self.uip_all
            uall,_,_ = mpy.fm_oss_init(mpy.ObjectView(uall), self.instrument_name)
            uall = mpy.fm_oss_windows(mpy.ObjectView(uall))
            rad, jac = mpy.fm_oss_stack(self.uip_all)
        finally:
            mpy.fm_oss_delete()            
        sd = self.spectral_domain(sensor_index)
        if(skip_jacobian):
            sr = rf.SpectralRange(rad, rf.Unit("sr^-1"))
        else:
            # jacobian is 1) on the forward model grid and
            # 2) tranposed from the ReFRACtor convention of the
            # column being the state vector variables. So
            # translate the oss jac to what we want from ReFRACtor
            jac = np.matmul(self.sub_basis_matrix, jac).transpose()
            a = rf.ArrayAd_double_1(rad, jac)
            sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)
    
class MusesCrisForwardModel(MusesOssForwardModelBase):
    '''Wrapper around fm_oss_stack call for CRiS instrument'''
    def __init__(self, rf_uip : RefractorUip):
        super().__init__(rf_uip, "CRIS")


    
        
