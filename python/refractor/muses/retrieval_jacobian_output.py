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

class RetrievalJacobianOutput(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_Jacobian files.'''
    def __reduce__(self):
        return (_new_from_init, (self.__class__,))
    
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "retrieval step"):
            return
        if len(glob(f"{self.out_fname}*")) == 0:
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            self.write_jacobian()
        else:
            logger.info(f"Found a jacobian product file: {self.out_fname}")

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Jacobian-{self.species_tag}{self.special_tag}.nc"

    def write_jacobian(self):

        # this section is to make all pressure grids have a standard size, 
        # like 65 levels

        speciesAll = self.retrievalInfo.species_list_fm
        # Python idiom for getting a unique list
        species = list(dict.fromkeys(speciesAll))

        pressureAll = self.retrievalInfo.pressure_list_fm
        jacobianAll = self.results.jacobian[0, :, :]
        nf = jacobianAll.shape[1]

        mypressure = []
        myspecies = []
        myjacobian = []
        for spc in species:
            ind = [s == spc for s in speciesAll]
            nn = np.count_nonzero(ind)
            species_type = mpy.specie_type(spc)

            nlevel = 65
            if species_type == 'ATMOSPHERIC':
                my_list = ["none",] * nlevel
                my_list[nlevel-nn:] = [spc,] * nn
                pressure = np.zeros(shape=(nlevel), dtype=np.float32)
                jacobian = np.zeros(shape=(nlevel, nf), dtype=np.float64)
                pressure[nlevel-nn:] = pressureAll[ind]
                jacobian[nlevel-nn:, :] = jacobianAll[ind, :]
            else:
                my_list = [spc,] * nn
                pressure = pressureAll[ind]
                jacobian = jacobianAll[ind, :]
            mypressure.append(pressure)
            myspecies.append(my_list)
            myjacobian.append(jacobian)
        # end for ii in range(0, len(species)):
        mypressure = np.concatenate(mypressure)
        myspecies = np.concatenate(myspecies)
        myjacobian = np.concatenate(myjacobian, axis=0)
        my_data = {
            'jacobian': np.transpose(myjacobian), # We transpose to match IDL shape.
            'frequency': None,
            'species': ','.join(myspecies), 
            'pressure': mypressure,
            'soundingID': None,
            'latitude': None,
            'longitude': None,
            'surfaceAltitudeMeters': None,
            'radianceResidualMean': None,
            'radianceResidualRMS': None,
            'land': None,
            'quality': None,
            'cloudOpticalDepth': None,
            'cloudTopPressure': None,
            'surfaceTemperature': None,
        }

        my_data = mpy.ObjectView(my_data)


        my_data.frequency = self.results.frequency.astype(np.float32)

        smeta = self.state_info.sounding_metadata()
        my_data.soundingID = smeta.sounding_id
        my_data.latitude = np.float32(smeta.latitude.convert("deg").value)
        my_data.longitude = np.float32(smeta.longitude.convert("deg").value)
        my_data.surfaceAltitudeMeters = np.float32(smeta.surface_altitude.convert("m").value)
        my_data.land = np.int16(1 if smeta.is_land else 0)
        
        my_data.quality = np.int16(self.results.masterQuality)
        my_data.radianceResidualMean = np.float32(self.results.radianceResidualMean[0])
        my_data.radianceResidualRMS = np.float32(self.results.radianceResidualRMS[0])
        my_data.cloudTopPressure = np.float32(self.state_info.state_element("PCLOUD").value[0])
        my_data.cloudOpticalDepth = np.float32(self.results.cloudODAve)
        my_data.surfaceTemperature = np.float32(self.state_info.state_element('TSUR').value[0])

        # Write out, use units as dummy: "()"
        my_data = my_data.__dict__
        mpy.cdf_write(my_data, self.out_fname, [{"UNITS" : "()"},] * len(my_data))


__all__ = ["RetrievalJacobianOutput", ] 
