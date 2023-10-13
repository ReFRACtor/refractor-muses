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
            # First argument isn't actually used in write_products_one_jacobian.
            # It is special_name, which doesn't actually apply to the jacobian file.
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # Code assumes we are in rundir
            with self.retrieval_strategy.chdir_run_dir():
                self.write_jacobian()
        else:
            logger.info(f"Found a jacobian product file: {self.out_fname}")

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Jacobian-{self.species_tag}{self.special_tag}.nc"

    def write_jacobian(self):
        # AT_LINE 8 write_products_one_jacobian.pro
        ndet = 1 
        nitems = len(self.results.error)
        ff = len(self.results.frequency)

        # this section is to make all pressure grids have a standard size, 
        # like 65 levels
        # put fill in for line species
        speciesAll = self.retrievalInfo.speciesListFM[0:self.retrievalInfo.n_totalParametersFM]
        # Python idiom for getting a unique list
        species = list(dict.fromkeys(speciesAll))

        pressureAll = self.retrievalInfo.pressureListFM[0:self.retrievalInfo.n_totalParametersFM]
        jacobianAll = self.results.jacobian[0, :, :]
        nf = jacobianAll.shape[1]

        for ii in range(0, len(species)):
            ind = [s == species[ii] for s in speciesAll]
            nn = np.count_nonzero(ind)
            species_type = mpy.specie_type(species[ii])

            if species_type == 'ATMOSPHERIC':
                my_list = np.asarray(['none' for xx in range(0, 65)])
                my_list[65-nn:65] = species[ii]
                pressure = np.zeros(shape=(65), dtype=np.float32)
                jacobian = np.zeros(shape=(65, nf), dtype=np.float64)
                pressure[65-nn:65] = pressureAll[ind]
                jacobian[65-nn:65, :] = jacobianAll[ind, :]
            else:
                my_list = [species[ii] for xx in range(0, nn)]# STRARR(nn) + species[ii]
                pressure = pressureAll[ind]
                jacobian = jacobianAll[ind, :]
            # end part of if specie_type(species[ii]) == 'ATMOSPHERIC':

            if ii == 0:
                mypressure = pressure
                myspecies = my_list
                myjacobian = jacobian
            else:
                mypressure = np.append(mypressure, pressure)
                myspecies = np.append(myspecies, my_list)
                myjacobian = np.append(myjacobian, jacobian, axis=0)
        # end for ii in range(0, len(species)):

        my_data = {
            'jacobian': np.transpose(myjacobian), # We transpose to match IDL shape.
            'frequency': np.zeros(shape=(ff), dtype=np.float32),  #
            'species': ','.join(myspecies), #  has to be a list.  can't have string matrix
            'pressure': mypressure,
            'soundingID': '',
            'latitude': np.float32(0.0),
            'longitude': np.float32(0.0),
            'surfaceAltitudeMeters': np.float32(0.0),
            'radianceResidualMean': np.float32(0.0),
            'radianceResidualRMS': np.float32(0.0),
            'land': np.int16(0),
            'quality': np.int16(0),
            'cloudOpticalDepth': np.float32(0.0),
            'cloudTopPressure': np.float32(0.0),
            'surfaceTemperature': np.float32(0.0),
        }

        my_data = mpy.ObjectView(my_data)


        my_data.frequency[:] = self.results.frequency[:]

        filename = './Measurement_ID.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)
        infoFile = mpy.tes_file_get_struct(fileID)  # infoFile OBJECT_TYPE dict

        my_data.soundingID = infoFile['preferences']['key']
        my_data.latitude = np.float32(self.stateInfo.state_info_obj.current['latitude'])
        my_data.longitude = np.float32(self.stateInfo.state_info_obj.current['longitude'])
        my_data.surfaceAltitudeMeters = np.float32(self.stateInfo.state_info_obj.current['tsa']['surfaceAltitudeKm']*1000)

        if self.stateInfo.state_info_obj.current['surfaceType'].upper() == 'OCEAN':
            my_data.land = np.int16(0)
        else:
            my_data.land = np.int16(1)

        my_data.quality = np.int16(self.results.masterQuality)
        my_data.radianceResidualMean = np.float32(self.results.radianceResidualMean[0])
        my_data.radianceResidualRMS = np.float32(self.results.radianceResidualRMS[0])
        my_data.cloudTopPressure = np.float32(self.stateInfo.state_info_obj.current['PCLOUD'][0])

        my_data.cloudOpticalDepth = np.float32(self.results.cloudODAve)
        my_data.surfaceTemperature = np.float32(self.stateInfo.state_info_obj.current['TSUR'])

        # Create a dictionary of units.  In our case, the units are dummy: "()"
        mydata_as_dict = my_data.__dict__
        my_keys = list(mydata_as_dict.keys())
        structUnits = []
        for xx in range(0, len(my_keys)):
            structUnits.append({'UNITS':"()"})

        # print(function_name, "Writing: ", i_filenameOut + '.nc')
        mpy.cdf_write(my_data.__dict__, self.out_fname,
                      [{"UNITS" : "()"},] * len(my_keys))


__all__ = ["RetrievalJacobianOutput", ] 
