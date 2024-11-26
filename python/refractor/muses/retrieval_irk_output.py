from glob import glob
from loguru import logger
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy
from .retrieval_output import RetrievalOutput, CdfWriteTes
import numpy as np
import math

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class RetrievalIrkOutput(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_IRK files.'''
    
    def __reduce__(self):
        return (_new_from_init, (self.__class__,))

    @property
    def retrieval_info(self) -> 'RetrievalInfo':
        return self.retrieval_strategy.retrieval_info

    @property
    def propagated_qa(self) -> 'PropagatedQa':
        return self.retrieval_strategy.propagated_qa

    @property
    def results_irk(self) -> 'ObjectView':
        return mpy.ObjectView(self.retrieval_strategy_step.results_irk)
    
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "IRK step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        logger.info("fake output for IRK")
        self.out_fname = f"{self.output_directory}/Products/Products_IRK.nc"
        os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
        self.write_irk()

    def write_irk(self):
        # Copy of write_products_irk_one, so we can try cleaning this up a bit
        function_name = "write_products_irk_one: "
        o_write_status = 0
    
        utilGeneral = mpy.UtilGeneral()
    
        # AT_LINE 7 write_products_irk_one.pro
        nobs = 1 
        num_points = 67   # 'np' is teh numpy alias, we change the variable here to num_points.

        nfreqBandFM = len(self.results_irk.fluxSegments)
        nfreqBand = len(self.results_irk.TSUR['irfk_segs'][0, :])
        nfreqEmis = len(self.results_irk.EMIS['irfk'])
        nfreqCloud = len(self.results_irk.CLOUDOD['irfk'])

        # AT_LINE 14 write_products_irk_one.pro
        irk_data = {
            'fmBandFlux': np.zeros(shape=(nobs), dtype=np.float32)-999,
            'l1BBandFlux': np.zeros(shape=(nobs), dtype=np.float32)-999,
            'fmFluxSegs': np.zeros(shape=(nfreqBandFM, nobs), dtype=np.float32)-999,
            'fmFluxSegsCenterFreq': np.zeros(shape=(nfreqBandFM, nobs), dtype=np.float32)-999,
            'l1BFluxSegs': np.zeros(shape=(nfreqBandFM, nobs), dtype=np.float32)-999,
            'o3LIRK': np.zeros(shape=(num_points, nobs), dtype=np.float32)-999,
            'o3IRKSegs': np.zeros(shape=(num_points, nfreqBand, nobs), dtype=np.float32)-999,
            'IRKSegsCenterFreq': np.zeros(shape=(nfreqBand, nobs), dtype=np.float32)-999,
            'h2oLIRK': np.zeros(shape=(num_points, nobs), dtype=np.float32)-999,
            'tatmIRK': np.zeros(shape=(num_points, nobs), dtype=np.float32)-999,
            'tsurIRK': np.zeros(shape=(nobs), dtype=np.float32)-999,
            'CloudEffectiveOpticalDepthLIRK': np.zeros(shape=(nfreqCloud, nobs), dtype=np.float32),
            'CloudTopPressureIRK': np.zeros(shape=(nobs), dtype=np.float32)-999,
            'emisIRK': np.zeros(shape=(nfreqEmis, nobs), dtype=np.float32)-999,
            'h2o': np.zeros(shape=(num_points, nobs), dtype=np.float32),
            'co2': np.zeros(shape=(num_points, nobs), dtype=np.float32),
            'n2o': np.zeros(shape=(num_points, nobs), dtype=np.float32),
            'o3': np.zeros(shape=(num_points, nobs), dtype=np.float32),
            'tatm': np.zeros(shape=(num_points, nobs), dtype=np.float32),
            'cloudod': np.zeros(shape=(nfreqCloud, nobs), dtype=np.float32),
            'emis': np.zeros(shape=(nfreqEmis, nobs), dtype=np.float32),
            'tsur': np.zeros(shape=(nobs), dtype=np.float32),
            'tatm_QA': np.zeros(shape=(nobs), dtype=np.int16),
            'o3_QA': np.zeros(shape=(nobs), dtype=np.int16),
            'utctime': ['  ' for i in range(nobs)],
            'daynightFlag': np.zeros(shape=(nobs), dtype=np.int16)-999,
            'dominantSurfaceType': ['  ' for i in range(nobs)],
            'LATITUDE': np.zeros(shape=(nobs), dtype=np.float64)-999,
            'LONGITUDE': np.zeros(shape=(nobs), dtype=np.float64)-999,
            'TIME': np.zeros(shape=(nobs), dtype=np.int32)-999,
            'BoresightNadirAngle': np.zeros(shape=(nobs), dtype=np.float32)-999,
            'surfaceTypeFootprint': np.zeros(shape=(nobs), dtype=np.int32)-999,
            'soundingID': ['  ' for i in range(nobs)],
            'omi_sza_uv2': np.zeros(shape=(nobs), dtype=np.float32),
            'omi_raz_uv2': np.zeros(shape=(nobs), dtype=np.float32),
            'omi_vza_uv2': np.zeros(shape=(nobs), dtype=np.float32),
            'omi_sca_uv2': np.zeros(shape=(nobs), dtype=np.float32),
        }
    
        # Convert to ObjectView so we can use the dot '.' notation to access the fields.
        irk_data = mpy.ObjectView(irk_data)
    
        # AT_LINE 56 write_products_irk_one.pro
        nn = num_points-len(self.results_irk.H2O['vmr'])
    
        # PYTHON_NOTE: When assigning the right hand side to the left hand side, we have to be explicit of the indices.
        #              IDL is more forgiving if the sizes on both sides of equation are different, Python is not.
        #              So to make Python happy, we have to specify the slice indices on the left hand sides and the right hand sides.
        irk_data.fmBandFlux[:] = self.results_irk.flux
        irk_data.l1BBandFlux[:] = self.results_irk.flux_l1b
        irk_data.fmFluxSegs[:, 0] = self.results_irk.fluxSegments[:]
        irk_data.fmFluxSegsCenterFreq[:, 0] = self.results_irk.freqSegments[:]
        irk_data.l1BFluxSegs[:, 0] = self.results_irk.fluxSegments_l1b[:]
        
        # AT_LINE 75 src_ms-2018-12-10/write_products_irk_one.pro
        irk_data.o3LIRK[nn:, 0] = self.results_irk.O3['lirfk'][:]
    
        # The shape of self.results_irk.O3['irfk_segs'] is (64, 33)
        # irk_data.o3IRKSegs[nn:, :, 0] = self.results_irk.O3['irfk_segs'][:, :]
    
        # Build array of indices both for the left hand side and the right hand side.
        lhs_first_indices = [ii for ii in range(nn, num_points)]                          # [3, 67) is 3 through 67 not inclusive 67
        lhs_second_indices = [jj for jj in range(0, self.results_irk.O3['irfk_segs'].shape[1])]  # [0, 33)
        rhs_first_indices = [ii for ii in range(0, self.results_irk.O3['irfk_segs'].shape[0])]   # [0, 64)
        rhs_second_indices = [jj for jj in range(0, self.results_irk.O3['irfk_segs'].shape[1])]  # [0, 33)
        
        # Use slow method to set o3IRKSegs field because the fast method does not work.
        irk_data.o3IRKSegs = utilGeneral.ManualArraySetsWithLHSRHSIndices(irk_data.o3IRKSegs, self.results_irk.O3['irfk_segs'], lhs_first_indices, lhs_second_indices, rhs_first_indices, rhs_second_indices)
    
        irk_data.IRKSegsCenterFreq[:, 0] = self.results_irk.freqSegments_irk[:]
    
        # AT_LINE 78 src_ms-2018-12-10/write_products_irk_one.pro
        irk_data.h2oLIRK[nn:, 0] = self.results_irk.H2O['lirfk'][:]
    
        # AT_LINE 79 src_ms-2018-12-10/write_products_irk_one.pro
        irk_data.tatmIRK[nn:, 0] = self.results_irk.TATM['irfk'][:]
        irk_data.tsurIRK[:] = self.results_irk.TSUR['irfk'][:]
        irk_data.CloudEffectiveOpticalDepthLIRK[:, 0] = self.results_irk.CLOUDOD['lirfk'][:]
        irk_data.CloudTopPressureIRK[:] = self.results_irk.PCLOUD['irfk'][:]
        irk_data.emisIRK[:, 0] = self.results_irk.EMIS['irfk'][:]
        irk_data.h2o[nn:, 0] = self.results_irk.H2O['vmr'][:]
        irk_data.co2[nn:, 0] = self.state_info.state_info_obj.current['values'][self.state_info.state_info_obj.species.index('CO2'), :]*1e6
        irk_data.n2o[nn:, 0] = self.state_info.state_info_obj.current['values'][self.state_info.state_info_obj.species.index('N2O'), :]*1e6
        irk_data.o3[nn:, 0] = self.results_irk.O3['vmr'][:]
        irk_data.tatm[nn:, 0] = self.results_irk.TATM['vmr'][:]
        irk_data.cloudod[:, 0] = self.results_irk.CLOUDOD['vmr'][:]
        irk_data.emis[:, 0] = self.results_irk.EMIS['vmr'][:]
        irk_data.tsur[:] = self.results_irk.TSUR['vmr'][:]
        irk_data.tatm_QA[:] = self.propagated_qa.tatm_qa
        irk_data.o3_QA[:] = self.propagated_qa.o3_qa
    
        # AT_LINE 83 write_products_irk_one.pro
        filename = './DateTime.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)
    
        irk_data.utctime = mpy.tes_file_get_preference(fileID, "UTC_Time")
        timestruct = mpy.utc_from_string(irk_data.utctime)
    
        # AT_LINE 90 write_products_irk_one.pro
        irk_data.omi_sza_uv2 = np.float32(self.state_info.state_info_obj.current['omi']['sza_uv2']) # The value from 'omi' key is double.  Convert to float32
        irk_data.omi_raz_uv2 = np.float32(self.state_info.state_info_obj.current['omi']['raz_uv2']) # The value from 'omi' key is double.  Convert to float32
        irk_data.omi_vza_uv2 = np.float32(self.state_info.state_info_obj.current['omi']['vza_uv2']) # The value from 'omi' key is double.  Convert to float32
        irk_data.omi_sca_uv2 = np.float32(self.state_info.state_info_obj.current['omi']['sca_uv2']) # The value from 'omi' key is double.  Convert to float32

        # AT_LINE 95 write_products_irk_one.pro
        if 'OMI' in self.retrieval_info.retrieval_info_obj.speciesList:
            irk_data.BoresightNadirAngle[:] = self.state_info.state_info_obj.current['omi']['vza_uv1'][:].astype(np.float32) # degrees
      
        #irk_data.time = self.m_utilTime.tai(timestruct, True).astype(np.int32)
    
        # AT_LINE 102 write_products_irk_one.pro
        # get target ID
        filename = './Measurement_ID.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)
    
        irk_data.soundingID = fileID['preferences']['key']

        if 'AIRS_ATrack_Index' in fileID['preferences']:
            irk_data.airs_granule = np.int16(fileID['preferences']['AIRS_Granule'])
            irk_data.airs_atrack_index = np.int16(fileID['preferences']['AIRS_ATrack_Index'])
            irk_data.airs_xtrack_index = np.int16(fileID['preferences']['AIRS_XTrack_Index'])
    
        if 'tes_run' in fileID['preferences']:
            # Not sure if these attributes below exist in fileID['preferences']
            irk_data.tes_run = fileID['preferences']['tes_run']
            irk_data.tes_sequence = fileID['preferences']['tes_sequence']
            irk_data.tes_scan = fileID['preferences']['tes_scan']
    
        if 'omi_xtrack_index' in fileID['preferences']:
            # Not sure if these attributes below exist in fileID['preferences']
            irk_data.omi_atrack_index = fileID['preferences']['omi_atrack_index']
            irk_data.omi_xtrack_index = fileID['preferences']['omi_xtrack_index']
    
        # AT_LINE 126 write_products_irk_one.pro
        # should do fresh water too
        irk_data.dominantSurfaceType = self.state_info.state_info_obj.current['surfaceType']
        if irk_data.dominantSurfaceType.upper() == 'OCEAN':
            irk_data.surfaceTypeFootprint = np.int32(2)
        else:
            irk_data.surfaceTypeFootprint = np.int32(3)
    
        # AT_LINE 134 write_products_irk_one.pro
        # get surface type using hres database
        if self.retrieval_info.retrieval_info_obj.surfaceType.upper() == 'OCEAN':
            irk_data.surfaceTypeFootprint = np.int32(2)
            if np.amin(np.abs(self.state_info.state_info_obj.current['heightKm'])) > 0.1:
                irk_data.surfaceTypeFootprint = np.int32(1)
    
        irk_data.LATITUDE = self.state_info.state_info_obj.current['latitude']
        irk_data.LONGITUDE = self.state_info.state_info_obj.current['longitude']
        irk_data.BoresightNadirAngle[:] = self.state_info.state_info_obj.current['tes']['boresightNadirRadians'] * 180 / math.pi # convert to degrees
        if 'OMI' in self.retrieval_info.retrieval_info_obj.speciesList:
            # Not sure if these attributes below exist in i_state.current['omi'] and what type it is.
            irk_data.BoresightNadirAngle[:] = self.state_info.state_info_obj.current['omi']['vza_uv1'][:] # degrees
    
        # AT_LINE 148 write_products_irk_one.pro
        # discriminate day or night
        # approximate time using longitude to adjust
        timestruct = mpy.utc(irk_data.utctime)
        hour = timestruct['hour'] + irk_data.LONGITUDE / 180. * 12
    
        if hour >= 8 and hour <= 22: # day
            irk_data.daynightFlag[0] = 1
    
        if hour <= 5 or hour >= 22:   # night
            irk_data.daynightFlag[0] = 0
    
        # Create a dictionary of units.  In our case, the units are dummy: "()"
        irk_as_dict = irk_data.__dict__
        irk_keys = list(irk_as_dict.keys())
        structUnits = [] 
        for xx in range(0, len(irk_keys)):
            structUnits.append({'UNITS':"()"})
    
        logger.info(f"Writing: {self.out_fname}")
        mpy.cdf_write(irk_data.__dict__, self.out_fname, structUnits)

        

__all__ = ["RetrievalIrkOutput", ] 
        
