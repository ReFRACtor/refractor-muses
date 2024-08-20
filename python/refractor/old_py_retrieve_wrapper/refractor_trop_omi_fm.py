from .tropomi_radiance import TropomiRadiancePyRetrieve
from .muses_py_forward_model import (RefractorTropOrOmiFmMusesPy, RefractorTropOrOmiFm)
from refractor.muses import CurrentStateUip, MeasurementIdDict
from refractor.tropomi import TropomiFmObjectCreator
import refractor.muses.muses_py as mpy
import numpy as np
import pandas as pd
import os
import refractor.framework as rf
import pickle
import logging
import copy

logger = logging.getLogger('py-retrieve')

#============================================================================
# This set of classes replace the lower level call to tropomi_fm in
# muses-py. This was used when initially comparing ReFRACtor and muses-py.
# This has been replaced with RefractorResidualFmJacobian which is higher
# in the call chain and has a cleaner interface.
# We'll leave these classes here for now, since it can be useful to do
# lower level comparisons. But these should largely be considered deprecated
#============================================================================

class RefractorTropOmiFmMusesPy(RefractorTropOrOmiFmMusesPy):
    def __init__(self, **kwargs):
        super().__init__(func_name="tropomi_fm", **kwargs)

    @property
    def observation(self):
        return TropomiRadiancePyRetrieve(self.rf_uip, skip_jac=True)        
    
class RefractorTropOmiFm(RefractorTropOrOmiFm):
    '''
    NOTE - this is deprecated

    Use a ReFRACtor ForwardModel as a replacement for tropomi_fm.'''

    def __init__(self, obs, **kwargs):
        super().__init__(func_name="tropomi_fm", **kwargs)
        self._obs = obs

    @property
    def observation(self):
        return self._obs

    @property
    def have_obj_creator(self):
        return "tropomi_fm_object_creator" in self.rf_uip.refractor_cache
    
    @property
    def obj_creator(self):
        '''Object creator using to generate forward model. You can use
        this to get various pieces we use to create the forward model.'''
        if("tropomi_fm_object_creator" not in self.rf_uip.refractor_cache):
            # Don't have an easy way to get this, so just pass an empty one. At
            # some point this may break, and perhaps we can just drop this old
            # class
            mid = MeasurementIdDict({},{})
            self.rf_uip.refractor_cache["tropomi_fm_object_creator"] = \
                TropomiFmObjectCreator(CurrentStateUip(self.rf_uip), mid, self._obs,
                                   rf_uip=self.rf_uip, **self.obj_creator_args)
        return self.rf_uip.refractor_cache["tropomi_fm_object_creator"]

    
__all__ = ["RefractorTropOmiFm", "RefractorTropOmiFmMusesPy"]
