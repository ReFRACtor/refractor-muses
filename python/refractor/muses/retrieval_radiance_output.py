from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy
from .retrieval_output import RetrievalOutput
import numpy as np

logger = logging.getLogger("py-retrieve")

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class RetrievalRadianceOutput(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_Radiance files.'''
    def __init__(self):
        self.myobsrad = None
        
    def __reduce__(self):
        return (_new_from_init, (self.__class__,))
    
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "retrieval step"):
            return
        if len(glob(f"{self.out_fname}*")) == 0:
            # First argument isn't actually used in write_products_one_jacobian.
            # It is special_name, which doesn't actually apply to the jacobian file.
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # This is an odd interface, but it is how radiance gets the input data.
            # We should perhaps just rewrite write_products_one_radiance, but for
            # now just conform with what it wants.

            # Note, I think this logic is actually wrong. If a previous step had
            # OMI or TROPOMI, it looks like this get left in rather than using
            # CRIS or AIRS radiance. But leave this for now, so we duplicate what
            # was done previously
            if(self.myobsrad is None):
                self.myobsrad = self.radiance_full
            for inst in ("OMI", "TROPOMI"):
                if(inst in self.radianceStep.instrumentNames):
                    i = self.radianceStep.instrumentNames.index(inst)
                    istart = sum(self.radianceStep.instrumentSizes[:i])
                    iend = istart + self.radianceStep.instrumentSizes[i]
                    r = range(istart, iend)
                    self.myobsrad = {"instrumentNames" : [inst],
                                "frequency" : self.radianceStep.frequency[r],
                                "radiance" : self.radianceStep.radiance[r],
                                "NESR" : self.radianceStep.NESR[r]}

                    
            self.write_radiance()
        else:
            logger.info(f"Found a radiance product file: {self.out_fname}")

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Radiance-{self.species_tag}{self.special_tag}.nc"

    def write_radiance(self):
        if len(self.myobsrad['instrumentNames']) == 1 or self.myobsrad['instrumentNames'][0] == self.myobsrad['instrumentNames'][1]:
            num_trueFreq = self.myobsrad['frequency']
            fullRadiance = self.myobsrad['radiance']
            fullNESR = self.myobsrad['NESR']
        else:
            instruIndex = self.myobsrad['instrumentNames'].index(self.instruments[0])
            if instruIndex == 0:
                r = range(0, self.myobsrad['instrumentSizes'][0])
            elif instruIndex == 1:
                r = range(self.myobsrad['instrumentSizes'][0], self.myobsrad['frequency'].shape[0])
            num_trueFreq = self.myobsrad['frequency'][r]
            fullRadiance = self.myobsrad['radiance'][r]
            fullNESR = self.myobsrad['NESR'][r]


        my_data = dict.fromkeys([
            'radianceFit',
            'radianceFitInitial',
            'radianceObserved',
            'nesr',
            'frequency',
            'radianceFullBand',
            'frequencyFullBand',
            'nesrFullBand',
            'soundingID',
            'latitude',
            'longitude',
            'surfaceAltitudeMeters',
            'radianceResidualMean',
            'radianceResidualRMS',
            'land',
            'quality',
            'cloudOpticalDepth',
            'cloudTopPressure',
            'surfaceTemperature',
            'scanDirection',
            'time',
            'emis',
            'emisFreq',
            'cloud',
            'cloudFreq',
        ])

        my_data = mpy.ObjectView(my_data)

        my_data.radianceFullBand = fullRadiance
        my_data.frequencyFullBand = num_trueFreq
        my_data.nesrFullBand = fullNESR

        my_data.radianceFit = self.results.radiance[0, :].astype(np.float32)
        my_data.radianceFitInitial = self.results.radianceInitial[0, :].astype(np.float32)
        my_data.radianceObserved = self.radianceStep.radiance.astype(np.float32)
        my_data.nesr = self.radianceStep.NESR.astype(np.float32)
        my_data.frequency = self.results.frequency.astype(np.float32)

        smeta = self.state_info.sounding_metadata()
        my_data.time = np.float64(smeta.tai_time)
        my_data.soundingID = np.frombuffer(smeta.sounding_id.encode('utf8'), dtype=np.dtype('b'))[np.newaxis,:]
        my_data.latitude = np.float32(smeta.latitude.convert("deg").value)
        my_data.longitude = np.float32(smeta.longitude.convert("deg").value)
        my_data.surfaceAltitudeMeters = np.float32(smeta.surface_altitude.convert("m").value)
        my_data.land = np.int16(1 if smeta.is_land else 0)
        my_data.scanDirection = np.int16(0)

        sstate = self.state_info.state_element("emissivity")
        my_data.emis = sstate.value.astype(np.float32)
        my_data.emisFreq = sstate.wavelength.astype(np.float32)
        
        sstate = self.state_info.state_element("cloudEffExt")
        my_data.cloud = sstate.value.astype(np.float32)[0,:]
        my_data.cloudFreq = sstate.wavelength.astype(np.float32)

        my_data.quality = np.int16(self.results.masterQuality)
        my_data.radianceResidualMean = np.float32(self.results.radianceResidualMean[0])
        my_data.radianceResidualRMS = np.float32(self.results.radianceResidualRMS[0])
        my_data.cloudTopPressure = np.float32(self.state_info.state_element("PCLOUD").value[0])
        my_data.cloudOpticalDepth = np.float32(self.results.cloudODAve)
        my_data.surfaceTemperature = np.float32(self.state_info.state_element('TSUR').value[0])
        # Write out, use units as dummy: "()"
        my_data = my_data.__dict__
        mpy.cdf_write(my_data, self.out_fname, [{"UNITS" : "()"},] * len(my_data))
        

__all__ = ["RetrievalRadianceOutput", ] 
