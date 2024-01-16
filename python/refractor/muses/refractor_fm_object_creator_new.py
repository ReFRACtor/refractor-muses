try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from .muses_optical_depth_file import MusesOpticalDepthFile
from .muses_altitude import MusesAltitude
from .muses_spectrum_sampling import MusesSpectrumSampling
from .muses_raman import MusesRaman
from .state_info import StateInfo
import refractor.framework as rf
import os
from pathlib import Path
import logging
import numpy as np
import glob
import abc

from typing import Sequence

logger = logging.getLogger("py-retrieve")

class RefractorFmObjectCreatorNew(object, metaclass=abc.ABCMeta):
    '''There are a lot of interrelated object needed to be created to
    make a ForwardModel.

    This class provides a framework for a lot of the common pieces. It
    is intended that derived classes override certain functionality (e.g.,
    have a difference Pressure object). The changes then ripple through
    the creation - so anything that needs a Pressure object to create itself
    will use whatever the derived class supplies.

    Note that this class is a convenience - for things like CostFuncCreator
    we just need *a* ForwardModel. There is no requirement that you use this
    particular class to create the ForwardModel. But because most of the
    time the steps are pretty similar this can be a useful class to start with.

    Take a look at TropomiFmObjectCreator and OmiObjectCreator for examples
    of modifying things.

    A note for developers, the various @cached_property decorators are 
    important, this isn't just for performance (which is pretty minor
    for most of the objects). It is important that if two different
    piece of code access for example the pressure object, it gets the
    *same* pressure object, not just two objects that have the same 
    pressure levels. Because these objects get updated by for example the 
    rf.StateVector, we need to have only one instance of them.
    '''

    def __init__(self, state_info : StateInfo,
                 i_windows,
                 instrument_name: str, 
                 # Short term, so we can flip between pca vs lidort
                 use_pca=True, use_lrad=False, lrad_second_order=False,
                 use_raman=True,
                 use_full_state_vector=True,
                 include_bad_sample=True,
                 ):
        '''Constructor. This takes a RefractorUip (so *not* the
        muses-py dictionary, but rather a RefractorUip created from
        that).

        For most purposes, you want use_full_state_vector=True
        This uses the "Full State Vector".
        See "Tropospheric Emission Spectrometer: Retrieval Method and Error
        Analysis" (IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,
        VOL. 44, NO. 5, MAY 2006) section III.A.1 for a discussion of this.
        the CostFunction handles mapping the retrieval vector to
        the full state vector. However for stand alone uses of the the
        forward model, it can sometimes be useful to use the retrieval
        vector in the StateVector, so you can specify
        use_full_state_vector=False if desired.
        
        The input directory can be given, this is used to read the 
        solar model data (omisol_v003_avg_nshi_backup.h5). If not supplied,
        we use the default directory path.
        '''


        self.use_pca = use_pca
        self.use_lrad = use_lrad
        self.use_raman = use_raman
        self.instrument_name = instrument_name
        self.lrad_second_order = lrad_second_order
        self.use_full_state_vector = use_full_state_vector
        self.include_bad_sample = include_bad_sample

        self.state_info = state_info
        self.i_windows = i_windows

        #self.num_channel = len(self.channel_list())

        #self.sza = np.array([float(self.rf_uip.solar_zenith(i))
        #                     for i in self.channel_list() ])
        #self.oza = np.array([float(self.rf_uip.observation_zenith(i))
        #                     for i in self.channel_list() ])
        # For TROPOMI view azimuth angle isn't available. Not sure if
        # that matters, I don't think this gets used for anything (only
        # relative azimuth is used). But go ahead and fill this in if
        # we aren't working with TROPOMI.
        #if self.instrument_name != "TROPOMI":
        #    self.oaz = np.array([float(self.rf_uip.observation_azimuth(i))
        #                         for i in self.channel_list() ])
        #self.raz = np.array([float(self.rf_uip.relative_azimuth(i))
        #                     for i in self.channel_list() ])

        #self.sza_with_unit = rf.ArrayWithUnit(self.sza, "deg")
        #self.oza_with_unit = rf.ArrayWithUnit(self.oza, "deg")
        #self.raz_with_unit = rf.ArrayWithUnit(self.raz, "deg")
        #self.filter_name = [self.rf_uip.filter_name(i) for i in self.channel_list()]

    @cached_property
    def spec_win(self):
        t = np.vstack([np.array([[[float(r['start']), float(r['endd'])]]]) for r in self.i_windows
                       if r['instrument'] == self.instrument_name])
        swin= rf.SpectralWindowRange(rf.ArrayWithUnit(t, "nm"))
        if(not self.include_bad_sample):
            for i in range(swin.number_spectrometer):
                raise NotImplementedError()
                #swin.bad_sample_mask(self.observation.bad_sample_mask_full(i), i)
        return swin
        

