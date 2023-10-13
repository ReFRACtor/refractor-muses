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

                    
            # Code assumes we are in rundir
            with self.retrieval_strategy.chdir_run_dir():
                self.write_radiance()
        else:
            logger.info(f"Found a radiance product file: {self.out_fname}")

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Radiance-{self.species_tag}{self.special_tag}.nc"

    def write_radiance(self):
        ndet = 1
        nitems = len(self.results.error)
        ff = len(self.results.frequency)
        num_emis_points = self.stateInfo.state_info_obj.emisPars['num_frequencies']
        num_cloud_points = self.stateInfo.state_info_obj.cloudPars['num_frequencies']

        if len(self.myobsrad['instrumentNames']) == 1 or self.myobsrad['instrumentNames'][0] == self.myobsrad['instrumentNames'][1]:
            fullFrequency = len(self.myobsrad['frequency'])
            num_trueFreq = self.myobsrad['frequency']
            fullRadiance = self.myobsrad['radiance']
            fullNESR = self.myobsrad['NESR']
        else:
            instruIndex = self.myobsrad['instrumentNames'].index(self.instruments[0])
            if instruIndex == 0:
                num_trueFreq = self.myobsrad['frequency'][0:self.myobsrad['instrumentSizes'][0]]
                fullRadiance = self.myobsrad['radiance'][0:self.myobsrad['instrumentSizes'][0]]
                fullNESR = self.myobsrad['NESR'][0:self.myobsrad['instrumentSizes'][0]]
            elif instruIndex == 1:
                num_trueFreq = self.myobsrad['frequency'][self.myobsrad['instrumentSizes'][0]:]
                fullRadiance = self.myobsrad['radiance'][self.myobsrad['instrumentSizes'][0]:]
                fullNESR = self.myobsrad['NESR'][self.myobsrad['instrumentSizes'][0]:]

            fullFrequency = len(num_trueFreq)


        filename = './Measurement_ID.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)
        infoFile = mpy.tes_file_get_struct(fileID)  # infoFile OBJECT_TYPE dict

        strlength = len(infoFile['preferences']['key'])

        # AT_LINE 12 write_products_one_radiance.pro
        my_data = {
            'radianceFit': np.zeros(shape=(ff), dtype=np.float32),
            'radianceFitInitial': np.zeros(shape=(ff), dtype=np.float32),
            'radianceObserved': np.zeros(shape=(ff), dtype=np.float32),
            'nesr': np.zeros(shape=(ff), dtype=np.float32),
            'frequency': np.zeros(shape=(ff), dtype=np.float32),
            'radianceFullBand': np.zeros(shape=(fullFrequency), dtype=np.float32),
            'frequencyFullBand': np.zeros(shape=(fullFrequency), dtype=np.float32),
            'nesrFullBand': np.zeros(shape=(fullFrequency), dtype=np.float32),
            'soundingID': np.zeros(shape=(1, strlength), dtype=np.dtype('b')),
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
            'scanDirection': np.int16(0),
            'time': np.float64(0.0),
            'emis': np.zeros(shape=(num_emis_points), dtype=np.float32) - 999,
            'emisFreq': np.zeros(shape=(num_emis_points), dtype=np.float32) - 999,
            'cloud': np.zeros(shape=(num_cloud_points), dtype=np.float32) - 999,
            'cloudFreq': np.zeros(shape=(num_cloud_points), dtype=np.float32) - 999,
        }

        my_data = mpy.ObjectView(my_data)

        # AT_LINE 36 write_products_one_radiance.pro
        # EM NOTE - Adding the full wavelengths of the instrument, for example CrIS only shows radiance at window frequencies, this shows the whole band
        my_data.radianceFullBand = fullRadiance
        my_data.frequencyFullBand = num_trueFreq
        my_data.nesrFullBand = fullNESR

        # SETTING_1:
        if self.results.radiance.size < len(my_data.radianceFit):
            if len(self.results.radiance.shape) > 1:
                # Check to see that the first dimension is 1.
                if self.results.radiance.shape[0] == 1:
                    rhs_length = self.results.radiance.shape[1]
                    my_data.radianceFit[0:rhs_length] = self.results.radiance[0, 0:rhs_length]
                else:
                    raise RuntimeError("Unexpected dimension")
            else:
                rhs_length = self.results.radiance.shape[0]
                my_data.radianceFit[0:rhs_length] = self.results.radiance[0:rhs_length]
        else:
            my_data.radianceFit[:] = self.results.radiance[:]

        # AT_LINE 37 write_products_one_radiance.pro

        # SETTING_2:
        if self.results.radianceInitial.size < len(my_data.radianceFitInitial):
            if len(self.results.radianceInitial.shape) > 1:
                # Check to see that the first dimension is 1.
                if self.results.radianceInitial.shape[0] == 1:
                    rhs_length = self.results.radianceInitial.shape[1]
                    my_data.radianceFitInitial[0:rhs_length] = self.results.radianceInitial[0, 0:rhs_length]
                else:
                    raise RuntimeError("Unexpected dimension")
            else:
                rhs_length = self.results.radianceInitial.shape[0]
                my_data.radianceFitInitial[0:rhs_length] = self.results.radianceInitial[0:rhs_length]
        else:
            my_data.radianceFitInitial[:] = self.results.radianceInitial[:]

        # AT_LINE 38 write_products_one_radiance.pro

        # SETTING_3:
        if self.radianceStep.radiance.size < len(my_data.radianceObserved):
            if len(self.radianceStep.radiance.shape) > 1:
                rhs_length = self.radianceStep.radiance.shape[1]

                # Check to see that the first dimension is 1.
                if self.radianceStep.radiance.shape[0] == 1:
                    my_data.radianceObserved[0:rhs_length] = self.radianceStep.radiance[0, 0:rhs_length]
                else:
                    raise RuntimeError("Unexpected dimension")
            else:
                rhs_length = self.radianceStep.radiance.shape[0]
                my_data.radianceObserved[0:rhs_length] = self.radianceStep.radiance[0:rhs_length]
        else:
            my_data.radianceObserved[:] = self.radianceStep.radiance[:]

        # AT_LINE 39 write_products_one_radiance.pro

        # SETTING_4:
        if self.radianceStep.NESR.size < len(my_data.nesr):
            if len(self.radianceStep.NESR.shape) > 1:
                # Check to see that the first dimension is 1.
                if self.radianceStep.NESR.shape[0] == 1:
                    rhs_length = self.radianceStep.NESR.shape[1]
                    my_data.nesr[0:rhs_length] = self.radianceStep.NESR[0, 0:rhs_length] # Note that NESR in self.radianceStep is uppercase here.
                else:
                    raise RuntimeError("Unexpected dimension")
            else:
                rhs_length = self.radianceStep.NESR.shape[0]
                my_data.nesr[0:rhs_length] = self.radianceStep.NESR[0:rhs_length] # Note that NESR in self.radianceStep is uppercase here.
        else:
            my_data.nesr[:] = self.radianceStep.NESR[:] # Note that NESR in self.radianceStep is uppercase here.


        filenamex = './Step00*/ELANORInput/Radiance*_metadata.bin'
        file_list = glob(filenamex)
        if len(file_list) == 0:
            logger.info(f"No files found for glob expression {filenamex}")

        scanDirection = 0 # Set to default to zero otherwise Python will complain.
        if len(file_list) > 0:
            logger.info(f"file_list, len(file_list) {file_list} {len(file_list)}")
            (o_read_status, o_dataOut) = mpy.read_all_tes(file_list[0])
            scanDirection = mpy.tes_file_get_preference(o_dataOut, 'scanDirection')
            if scanDirection != '':
                my_data.scanDirection = np.int16(scanDirection)

        filename = './DateTime.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)

        taitime = mpy.tes_file_get_preference(fileID, "TAI_Time_of_ZPD")
        my_data.time = np.float64(taitime)  # Because taitime is a string '6.4668561156126893E+08' we have to convert it to double.

        my_data.emis[:] = self.stateInfo.state_info_obj.current['emissivity'][0:self.stateInfo.state_info_obj.emisPars['num_frequencies']]
        my_data.emisFreq[:] = self.stateInfo.state_info_obj.emisPars['frequency'][0:self.stateInfo.state_info_obj.emisPars['num_frequencies']]

        if self.stateInfo.state_info_obj.cloudPars['num_frequencies'] > 0:
            my_data.cloud[:] = self.stateInfo.state_info_obj.current['cloudEffExt'][0, 0:self.stateInfo.state_info_obj.cloudPars['num_frequencies']]
            my_data.cloudFreq[:] = self.stateInfo.state_info_obj.cloudPars['frequency'][0:self.stateInfo.state_info_obj.cloudPars['num_frequencies']]

        my_data.frequency[:] = self.results.frequency[:]

        filename = './Measurement_ID.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)
        infoFile = mpy.tes_file_get_struct(fileID)  # infoFile OBJECT_TYPE dict

        my_data.soundingID[:] = bytearray(infoFile['preferences']['key'], 'utf8')
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

        mpy.cdf_write(my_data.__dict__, self.out_fname,
                      [{"UNITS" : "()"},] * len(my_keys))
        

__all__ = ["RetrievalRadianceOutput", ] 
