from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy
from .retrieval_output import RetrievalOutput, CdfWriteTes
import numpy as np

logger = logging.getLogger("py-retrieve")

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class RetrievalL2Output(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_L2 files.'''
    
    def __reduce__(self):
        return (_new_from_init, (self.__class__,))
    
    @property
    def retrieval_info(self):
        return self.retrieval_strategy.retrieval_info
    
    @property
    def species_count(self):
        '''Dictionary that gives the index we should use for product file names.
        This is 0 if the species doesn't get retrieved in a following step, and
        the count of other times the species is retrieved. So for example if
        O3 is retrieved 4 times, the first time we retrieve it the file has
        a "O3-3" in the name, followed by "O3-2", "O3-1" and "O3-0"'''
        if(self._species_count is None):
            self._species_count = defaultdict(lambda: 0)
            tstep = self.table_step
            for i in range(self.table_step+1, self.number_table_step):
                for spc in self.retrieval_strategy.retrieval_elements(i):
                    self._species_count[spc] += 1
        return self._species_count

    @property
    def species_list(self):
        '''List of species, partially ordered so TATM comes before H2O, H2O before HDO,
        and N2O before CH4.

        The ordering is because TATM, H2O and N2O are used in making the lite files
        of CH4, HDO and H2O lite files, so we need to data from these before we get
        to the lite files.'''
        if(self._species_list is None):
            self._species_list = self.retrieval_info.species_names
            for spc in ('N2O', 'H2O', 'TATM'):
                if(spc in self._species_list):
                    self._species_list.remove(spc)
                    self._species_list.insert(0, spc)
        return self._species_list
            
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        # Save these, used in later lite files. Note these actually get
        # saved between steps, so we initialize these for the first step but
        # then leave them alone
        if(location == "retrieval step" and "dataTATM" not in self.__dict__):
            self.dataTATM = None
            self.dataH2O = None
            self.dataN2O = None
        if(location != "retrieval step"):
            return
        # Regenerate this for the current step
        self._species_count = None
        self._species_list = None
        for self.spcname in self.species_list:
            if(self.retrieval_info.species_list_fm.count(self.spcname) <= 1 or
               self.spcname in ('CLOUDEXT', 'EMIS') or 
               self.spcname.startswith('OMI') or
               self.spcname.startswith('NIR')):
                continue
            self.out_fname = f"{self.retrieval_strategy.output_directory}/Products/Products_L2-{self.spcname}-{self.species_count[self.spcname]}.nc"
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # Not sure about the logic here, but this is what script_retrieval_ms does
            if(not os.path.exists(self.out_fname) or self.spcname in ('TATM', 'H2O', 'N2O')):
                dataInfo = self.write_l2()
                if(self.spcname == "TATM"):
                    self.dataTATM = dataInfo
                elif(self.spcname == "H2O"):
                    self.dataH2O = dataInfo
                elif(self.spcname == "N2O"):
                    self.dataN2O = dataInfo
                self.lite_file(dataInfo)

    def lite_file(self, dataInfo):
        '''Create lite file.'''
        if(self.spcname == "CH4"):
            if self.dataN2O is not None:
                data2 = self.dataN2O
            else:
                # Fake the data
                logger.warn("code has not been tested for species_name CH4 and dataN2O is None")
                data2 = copy.deepcopy(dataInfo)
                value = self.state_info.state_element("N2O", step="initial").value
                data2['SPECIES'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['INITIAL'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['CONSTRAINTVECTOR'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['AVERAGINGKERNEL'].fill(0.0)
                data2['OBSERVATIONERRORCOVARIANCE'].fill(0.0)
        elif(self.spcname == "HDO"):
            data2 = self.dataH2O
        else:
            data2 = None

        state_element_out = [t for t in self.state_info.state_element_list() if t.should_write_to_l2_product(self.instruments)]
            
        if(self.spcname == "H2O" and self.dataTATM is not None):
            self.out_fname = f"{self.retrieval_strategy.output_directory}/Products/Lite_Products_L2-RH-{self.species_count[self.spcname]}.nc"
            if("OCO2" not in self.instruments):
                liteDirectory = f"{self.retrieval_strategy.run_dir}/../OSP/Lite/"
                # Code assumes we are in rundir
                t = CdfWriteTes()
                t.write_lite(
                    self.table_step,
                    self.out_fname, self.quality_name, self.instruments,
                    liteDirectory, dataInfo, self.dataTATM, "RH",
                    step=self.species_count[self.spcname],
                    times_species_retrieved=self.species_count[self.spcname],
                    state_element_out=state_element_out)
                
        self.out_fname = f"{self.retrieval_strategy.output_directory}/Products/Lite_Products_L2-{self.spcname}-{self.species_count[self.spcname]}.nc"
        if 'OCO2' not in self.instruments:
            liteDirectory = f"{self.retrieval_strategy.run_dir}/../OSP/Lite/"
            t = CdfWriteTes()
            data2 = t.write_lite(
                self.table_step, self.out_fname, self.quality_name,
                self.instruments, liteDirectory, dataInfo, data2, self.spcname,
                step=self.species_count[self.spcname],
                times_species_retrieved=self.species_count[self.spcname],
                state_element_out=state_element_out)

    def generate_geo_data(self, species_data):
        '''Generate the geo_data, pulled out just to keep write_l2 from getting
        too long.'''
        nobs = 1
        geo_data = {
            'DAYNIGHTFLAG': None,
            'landFlag'.upper(): np.zeros(shape=(nobs), dtype=np.int32) - 999,
            'LATITUDE': None,
            'LONGITUDE': None,
            'TIME': None,
            'surfaceTypeFootprint'.upper(): np.zeros(shape=(nobs), dtype=np.int64) - 999,
            'SOUNDINGID': None
        }

        geo_data = mpy.ObjectView(geo_data)
        smeta = self.state_info.sounding_metadata()
        geo_data.TIME = np.int32(smeta.wrong_tai_time)
        geo_data.LATITUDE = smeta.latitude.convert("deg").value
        geo_data.LONGITUDE = smeta.longitude.convert("deg").value
        geo_data.SOUNDINGID = smeta.sounding_id
        geo_data.LANDFLAG = np.int32(0 if smeta.is_ocean else 1)
        geo_data.SURFACETYPEFOOTPRINT = np.int32(2 if smeta.is_ocean else 3)        

        for i, inst in enumerate(self.instruments):
            geo_data.__dict__.update(self.obs_list[i].sounding_desc)
        
        # get surface type using hres database
        if self.retrieval_info.is_ocean:
            geo_data.LANDFLAG = np.int32(0)
            geo_data.SURFACETYPEFOOTPRINT = np.int32(2)

            if np.amin(np.abs(smeta.height.convert("km").value)) > 0.1:
                geo_data.SURFACETYPEFOOTPRINT = 1

        hour = smeta.local_hour
        if (hour >= 8 and hour <= 22):
            geo_data.DAYNIGHTFLAG = np.int16(1)
        elif (hour <= 5 or hour >= 22):
            geo_data.DAYNIGHTFLAG = np.int16(0)
        else:
            geo_data.DAYNIGHTFLAG = np.int16(-999)

        # if sza defined, then use this for daynight, update 12/2017, DF, SSK
        if 'omi_sza_uv2' in species_data:
            geo_data.DAYNIGHTFLAG = 0
            if species_data["OMI_SZA_UV2"] < 85:
                geo_data.DAYNIGHTFLAG = np.int16(1)

        geo_data = geo_data.__dict__

        for k, v in geo_data.items():
            if isinstance(v, list):
                geo_data[k] = np.asarray(v)
        return geo_data

    def write_l2(self):
        '''Create L2 product file'''
        runtime_attributes = dict()

        # AT_LINE 7 write_products_one.pro
        nobs = 1 # number of observations
        # num_pressures varies based on surface pressure.  We set it to max here.
        num_pressures = 67  
        nfreqEmis = 121

        if self.state_info.has_true_values:
            # TODO If we get sample data, we can put this back in
            logger.warning("There is a block of code in the muses-py for reporting true values. We don't have that code, because we don't have any test data for this. So skipping.")
        if("OCO-2" in self.instruments):
            # TODO If we get sample data, we can put this back in
            logger.warning("There is a block of code in the muses-py for reporting OCO-2 values. We don't have that code, because we don't have any test data for this. So skipping.")

        nfilter = len(self.results.filter_index)-1

        species_data = {
            'SPECIES'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999, 
            'PRIORCOVARIANCE'.upper(): np.zeros(shape=(num_pressures, num_pressures), dtype=np.float32) - 999, 
            'cloudTopPressureDOF'.upper(): 0.0,
            'PRECISION'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'airDensity'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'altitude'.upper()  : np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'AVERAGECLOUDEFFOPTICALDEPTH'.upper(): 0.0,
            'AVERAGINGKERNEL'.upper(): np.zeros(shape=(num_pressures, num_pressures), dtype=np.float32) - 999,
            'AVERAGINGKERNELDIAGONAL'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'CLOUDEFFECTIVEOPTICALDEPTH'.upper(): np.zeros(shape=(28), dtype=np.float32) - 999,
            'CLOUDEFFECTIVEOPTICALDEPTHERROR'.upper(): np.zeros(shape=(28), dtype=np.float32) - 999,
            'cloudTopPressure'.upper(): 0.0,
            'cloudTopPressureError'.upper(): 0.0,
            'CLOUDVARIABILITY_QA'.upper(): 0.0,
            'CONSTRAINTVECTOR'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'DOFS'.upper(): 0.0,
            'deviation_QA'.upper(): -999,
            'num_deviations_QA'.upper(): -999,
            'DeviationBad_QA'.upper(): -999,
            'H2O_H2O_corr_QA'.upper(): 0.0,
            'INITIAL'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'KDotDL_QA'.upper(): 0.0,
            'KDotDLSys_QA'.upper(): 0.0,
            'LDotDL_QA'.upper(): 0.0,
            'MEASUREMENTERRORCOVARIANCE'.upper(): np.zeros(shape=(num_pressures, num_pressures), dtype=np.float32) - 999,
            'OBSERVATIONERRORCOVARIANCE'.upper(): np.zeros(shape=(num_pressures, num_pressures), dtype=np.float32) - 999,
            'PRESSURE'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,

            'propagated_H2O_QA'.upper(): 0,
            'propagated_O3_QA'.upper(): 0,
            'propagated_TATM_QA'.upper(): 0,

            'radianceResidualRMS'.upper(): 0.0,
            'radianceResidualMean'.upper(): 0.0,
            'radiance_residual_stdev_change'.upper(): 0.0,
            'FILTER_INDEX':np.zeros(shape=(nfilter), dtype=np.int16),
            'radianceResidualRMS_filter'.upper(): np.zeros(shape=(nfilter), dtype=np.float32),
            'radianceResidualMean_filter'.upper(): np.zeros(shape=(nfilter), dtype=np.float32),
            'radianceResidualRMSRelativeContinuum_filter'.upper(): np.zeros(shape=(nfilter), dtype=np.float32),
            'radiance_continuum_filter'.upper() :np.zeros(shape=(nfilter), dtype=np.float32),
            'radianceSNR_filter'.upper(): np.zeros(shape=(nfilter), dtype=np.float32),                             
            'radianceResidualSlope_filter'.upper(): np.zeros(shape=(nfilter), dtype=np.float32),
            'radianceResidualQuadratic_filter'.upper(): np.zeros(shape=(nfilter), dtype=np.float32),


            'residualNormFinal'.upper(): 0.0,
            'residualNormInitial'.upper(): 0.0,
            'retrieveInLog'.upper(): -999,
            'Quality'.upper(): np.int16(-999),

            'Desert_Emiss_QA'.upper(): 0.0,
            'EMISSIVITY_CONSTRAINT'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'EMISSIVITY_ERROR'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'EMISSIVITY_INITIAL'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'EMISSIVITY'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'emissivity_Wavenumber'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'SURFACEEMISSMEAN_QA'.upper(): 0.0,
            'SURFACEEMISSIONLAYER_QA'.upper(): 0.0,
            'SURFACETEMPCONSTRAINT'.upper(): 0.0,
            'SURFACETEMPDEGREESOFFREEDOM'.upper(): 0.0,
            'SURFACETEMPERROR'.upper(): 0.0,
            'SURFACETEMPINITIAL'.upper(): 0.0,
            'SURFACETEMPOBSERVATIONERROR'.upper(): 0.0,
            'SURFACETEMPPRECISION'.upper(): 0.0,
            'SURFACETEMPVSAPRIORI_QA'.upper(): 0.0,
            'SURFACETEMPVSATMTEMP_QA'.upper(): 0.0,
            'SURFACETEMPERATURE'.upper(): 0.0,
            'column'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_air'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_DOFS'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_error'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_initial'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_PressureMax'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_PressureMin'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'column_Units'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'COLUMN_PRIOR'.upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            'TOTALERROR'.upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            'TOTALERRORCOVARIANCE'.upper(): np.zeros(shape=(num_pressures, num_pressures), dtype=np.float32) - 999,
            'CLOUDFREQUENCY'.upper(): np.zeros(shape=(28), dtype=np.float32) - 999,
            'tropopausePressure'.upper(): 0.0,
            'lmresults_delta'.upper(): self.results.LMResults_delta[self.results.bestIteration]
        }

        emis_state = self.state_info.state_element("emissivity")
        if emis_state.wavelength.shape[0] == 0:
            del species_data['EMISSIVITY_CONSTRAINT']
            del species_data['EMISSIVITY_ERROR']
            del species_data['EMISSIVITY_INITIAL']
            del species_data['EMISSIVITY']
            del species_data['EMISSIVITY_WAVENUMBER']


        species_data = mpy.ObjectView(species_data) 

        # AT_LINE 121 write_products_one.pro
        species_data.TROPOPAUSEPRESSURE = self.state_info.gmao_tropopause_pressure() if self.state_info.gmao_tropopause_pressure() > 0 else self.results.tropopausePressure

        # AT_LINE 126 write_products_one.pro
        species_data.DESERT_EMISS_QA = self.results.Desert_Emiss_QA
        species_data.PROPAGATED_TATM_QA = np.int32(self.results.propagatedTATMQA)
        species_data.PROPAGATED_O3_QA = np.int32(self.results.propagatedO3QA)
        species_data.PROPAGATED_H2O_QA = np.int32(self.results.propagatedH2OQA)
        species_data.RADIANCEMAXIMUMSNR = self.results.radianceMaximumSNR
        species_data.RESIDUALNORMFINAL = self.results.residualNormFinal
        species_data.RESIDUALNORMINITIAL = self.results.residualNormInitial

        smeta = self.state_info.sounding_metadata()

        # Determine subset of the max num_pressures that we actually have
        # data for
        num_actual_pressures = self.state_info.state_element(self.state_info.state_element_on_levels[0]).value.shape[0]
        # And get the range of data we use to fill in our fields
        pslice = slice(num_pressures-num_actual_pressures,num_pressures)
        # get column / altitude / air density / trop column stuff
        altitudeResult, _ = mpy.compute_altitude_pge(
            self.state_info.pressure, 
            self.state_info.state_element("TATM").value, 
            self.state_info.state_element("H2O").value, 
            smeta.surface_altitude.convert("m").value, 
            smeta.latitude.convert("deg").value, 
            i_waterType=None, i_pge=True)

        altitudeResult = mpy.ObjectView(altitudeResult) 
        species_data.AIRDENSITY[pslice] = (altitudeResult.airDensity*1e6)[:] #convert molec/cm3 -> molec/m3

        species_data.ALTITUDE[pslice] = altitudeResult.altitude[:]


        # AT_LINE 169 write_products_one.pro
        if self.spcname == 'O3':
            species_data.O3_CCURVE_QA = np.int32(self.results.ozoneCcurve)
            species_data.O3_SLOPE_QA = self.results.ozone_slope_QA

            species_data.O3_COLUMNERRORDU = self.results.O3_columnErrorDU
            species_data.O3_TROPO_CONSISTENCY_QA = self.results.O3_tropo_consistency

        if self.spcname in self.results.columnSpecies:
            indcol = self.results.columnSpecies.index(self.spcname)

            species_data.COLUMN = copy.deepcopy(self.results.column[:, indcol])
            species_data.COLUMN_AIR = copy.deepcopy(self.results.columnAir[:])
            species_data.COLUMN_DOFS = copy.deepcopy(self.results.columnDOFS[:, indcol])
            species_data.COLUMN_ERROR = copy.deepcopy(self.results.columnError[:, indcol])
            species_data.COLUMN_INITIAL = copy.deepcopy(self.results.columnInitial[:, indcol])
            species_data.COLUMN_PRESSUREMAX = copy.deepcopy(self.results.columnPressureMax[:])
            species_data.COLUMN_PRESSUREMIN = copy.deepcopy(self.results.columnPressureMin[:])
            species_data.COLUMN_PRIOR = copy.deepcopy(self.results.columnPrior[:, indcol])
        species_data.RADIANCERESIDUALRMS_FILTER = self.results.radianceResidualRMS[1:]
        species_data.RADIANCERESIDUALMEAN_FILTER = self.results.radianceResidualMean[1:]
        species_data.radianceResidualRMSRelativeContinuum_FILTER = self.results.radianceResidualRMSRelativeContinuum[1:]
        species_data.RADIANCE_CONTINUUM_FILTER = self.results.radianceContinuum[1:]
        species_data.RADIANCESNR_FILTER = self.results.radianceSNR[1:]
        species_data.FILTER_INDEX = self.results.filter_index[1:]
        species_data.RADIANCERESIDUALSLOPE_FILTER = self.results.residualSlope[1:]
        species_data.RADIANCERESIDUALQUADRATIC_FILTER = self.results.residualQuadratic[1:]
        species_data.RADIANCERESIDUALRMS = self.results.radianceResidualRMS[0]
        species_data.RADIANCERESIDUALMEAN = self.results.radianceResidualMean[0]
        species_data.RADIANCE_RESIDUAL_STDEV_CHANGE = self.results.radianceResidualRMSInitial[0] - self.results.radianceResidualRMS[0]
        
        # ==============> Cleanup to this point
        if 'OMI' in self.instruments:
            # Make all names uppercased to make life easier.
            species_data.OMI_SZA_UV2 = self.state_info.state_info_obj.current['omi']['sza_uv2']
            species_data.OMI_RAZ_UV2 = self.state_info.state_info_obj.current['omi']['raz_uv2']
            species_data.OMI_VZA_UV2 = self.state_info.state_info_obj.current['omi']['vza_uv2']
            species_data.OMI_SCA_UV2 = self.state_info.state_info_obj.current['omi']['sca_uv2']

            species_data.OMI_SZA_UV1 = self.state_info.state_info_obj.current['omi']['sza_uv1']
            species_data.OMI_RAZ_UV1 = self.state_info.state_info_obj.current['omi']['raz_uv1']
            species_data.OMI_VZA_UV1 = self.state_info.state_info_obj.current['omi']['vza_uv1']
            species_data.OMI_SCA_UV1 = self.state_info.state_info_obj.current['omi']['sca_uv1']

            # could get these from state.current.omipars
            # note I added this to the new_state_structures and Make_UIP_OMI.pro
            species_data.OMI_CLOUDFRACTION = self.state_info.state_info_obj.current['omi']['cloud_fraction']
            species_data.OMI_CLOUDFRACTIONCONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['omi']['cloud_fraction']
            species_data.OMI_CLOUDTOPPRESSURE = self.state_info.state_info_obj.constraint['omi']['cloud_pressure']

            species_data.OMI_SURFACEALBEDOUV1 = self.state_info.state_info_obj.current['omi']['surface_albedo_uv1']
            species_data.OMI_SURFACEALBEDOUV1CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['omi']['surface_albedo_uv1']

            species_data.OMI_SURFACEALBEDOUV2 = self.state_info.state_info_obj.current['omi']['surface_albedo_uv2']
            species_data.OMI_SURFACEALBEDOUV2CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['omi']['surface_albedo_uv2']

            species_data.OMI_SURFACEALBEDOSLOPEUV2 = self.state_info.state_info_obj.current['omi']['surface_albedo_slope_uv2']
            species_data.OMI_SURFACEALBEDOSLOPEUV2CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['omi']['surface_albedo_slope_uv2']

            species_data.OMI_NRADWAVUV1 = self.state_info.state_info_obj.current['omi']['nradwav_uv1']
            species_data.OMI_NRADWAVUV2 = self.state_info.state_info_obj.current['omi']['nradwav_uv2']

            species_data.OMI_ODWAVUV1 = self.state_info.state_info_obj.current['omi']['odwav_uv1']
            species_data.OMI_ODWAVUV2 = self.state_info.state_info_obj.current['omi']['odwav_uv2']

            species_data.OMI_RINGSFUV1 = self.state_info.state_info_obj.current['omi']['ring_sf_uv1']
            species_data.OMI_RINGSFUV2 = self.state_info.state_info_obj.current['omi']['ring_sf_uv2']
        if 'TROPOMI' in self.instruments:
            # As with OMI, make all names uppercased to make life easier.
            # EM NOTE - This will have to be expanded if additional tropomi bands are used
            species_data.TROPOMI_SZA_BAND1 = self.state_info.state_info_obj.current['tropomi']['sza_BAND1']
            species_data.TROPOMI_RAZ_BAND1 = self.state_info.state_info_obj.current['tropomi']['raz_BAND1']
            species_data.TROPOMI_VZA_BAND1 = self.state_info.state_info_obj.current['tropomi']['vza_BAND1']
            species_data.TROPOMI_SCA_BAND1 = self.state_info.state_info_obj.current['tropomi']['sca_BAND1']

            species_data.TROPOMI_SZA_BAND2 = self.state_info.state_info_obj.current['tropomi']['sza_BAND2']
            species_data.TROPOMI_RAZ_BAND2 = self.state_info.state_info_obj.current['tropomi']['raz_BAND2']
            species_data.TROPOMI_VZA_BAND2 = self.state_info.state_info_obj.current['tropomi']['vza_BAND2']
            species_data.TROPOMI_SCA_BAND2 = self.state_info.state_info_obj.current['tropomi']['sca_BAND2']

            species_data.TROPOMI_SZA_BAND3 = self.state_info.state_info_obj.current['tropomi']['sza_BAND3']
            species_data.TROPOMI_RAZ_BAND3 = self.state_info.state_info_obj.current['tropomi']['raz_BAND3']
            species_data.TROPOMI_VZA_BAND3 = self.state_info.state_info_obj.current['tropomi']['vza_BAND3']
            species_data.TROPOMI_SCA_BAND3 = self.state_info.state_info_obj.current['tropomi']['sca_BAND3']
            
            # could get these from state.current.tropomipars
            species_data.TROPOMI_CLOUDFRACTION = self.state_info.state_info_obj.current['tropomi']['cloud_fraction']
            species_data.TROPOMI_CLOUDFRACTIONCONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['cloud_fraction']
            species_data.TROPOMI_CLOUDTOPPRESSURE = self.state_info.state_info_obj.constraint['tropomi']['cloud_pressure']

            species_data.TROPOMI_SURFACEALBEDOBAND1 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_BAND1']
            species_data.TROPOMI_SURFACEALBEDOBAND1CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_BAND1']

            species_data.TROPOMI_SURFACEALBEDOBAND2 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_BAND2']
            species_data.TROPOMI_SURFACEALBEDOBAND2CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_BAND2']

            species_data.TROPOMI_SURFACEALBEDOBAND3 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_BAND3']
            species_data.TROPOMI_SURFACEALBEDOBAND3CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_BAND3']

            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND2 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_slope_BAND2']
            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND2CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND2']

            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND3 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_slope_BAND3']
            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND3CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND3']

            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND2 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_slope_order2_BAND2']
            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND2CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND2']

            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3 = self.state_info.state_info_obj.current['tropomi']['surface_albedo_slope_order2_BAND3']
            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3CONSTRAINTVECTOR = self.state_info.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND3']

            species_data.TROPOMI_SOLARSHIFTBAND1 = self.state_info.state_info_obj.current['tropomi']['solarshift_BAND1']
            species_data.TROPOMI_SOLARSHIFTBAND2 = self.state_info.state_info_obj.current['tropomi']['solarshift_BAND2']
            species_data.TROPOMI_SOLARSHIFTBAND3 = self.state_info.state_info_obj.current['tropomi']['solarshift_BAND3']

            species_data.TROPOMI_RADIANCESHIFTBAND1 = self.state_info.state_info_obj.current['tropomi']['radianceshift_BAND1']
            species_data.TROPOMI_RADIANCESHIFTBAND2 = self.state_info.state_info_obj.current['tropomi']['radianceshift_BAND2']
            species_data.TROPOMI_RADIANCESHIFTBAND3 = self.state_info.state_info_obj.current['tropomi']['radianceshift_BAND3']

            species_data.TROPOMI_RADSQUEEZEBAND1 = self.state_info.state_info_obj.current['tropomi']['radsqueeze_BAND1']
            species_data.TROPOMI_RADSQUEEZEBAND2 = self.state_info.state_info_obj.current['tropomi']['radsqueeze_BAND2']
            species_data.TROPOMI_RADSQUEEZEBAND3 = self.state_info.state_info_obj.current['tropomi']['radsqueeze_BAND3']

            species_data.TROPOMI_RINGSFBAND1 = self.state_info.state_info_obj.current['tropomi']['ring_sf_BAND1']
            species_data.TROPOMI_RINGSFBAND2 = self.state_info.state_info_obj.current['tropomi']['ring_sf_BAND2']
            species_data.TROPOMI_RINGSFBAND3 = self.state_info.state_info_obj.current['tropomi']['ring_sf_BAND3']

            species_data.TROPOMI_TEMPSHIFTBAND3 = self.state_info.state_info_obj.current['tropomi']['temp_shift_BAND3']


        #species_data.TROPOMI_EOF1 = 1.0
        species_data.SPECIES[pslice] = self.retrieval_info.species_results(self.results, self.spcname)
        species_data.INITIAL[pslice] = self.retrieval_info.species_initial(self.spcname)
        species_data.CONSTRAINTVECTOR[pslice] = self.retrieval_info.species_constraint(self.spcname)
        species_data.PRESSURE[pslice] = self.state_info.pressure
        species_data.CLOUDTOPPRESSURE = self.state_info.state_element("PCLOUD").value[0]
        
        utilList = mpy.UtilList()
        indx = utilList.WhereEqualIndices(self.retrieval_info.species_list_fm, 'PCLOUD')
        if len(indx) > 0:
            indx = indx[0]
            species_data.CLOUDTOPPRESSUREDOF = self.results.A[indx, indx]
            species_data.CLOUDTOPPRESSUREERROR = self.results.errorFM[indx]

        # AT_LINE 288 write_products_one.pro
        species_data.AVERAGECLOUDEFFOPTICALDEPTH = np.nan_to_num(self.results.cloudODAve)
        species_data.CLOUDVARIABILITY_QA = np.int32(np.nan_to_num(self.results.cloudODVar))
        species_data.H2O_H2O_CORR_QA = self.results.H2O_H2OQuality
        species_data.KDOTDL_QA = self.results.KDotDL
        species_data.KDOTDLSYS_QA = self.results.maxKDotDLSys
        species_data.LDOTDL_QA = self.results.LDotDL
        species_data.QUALITY = np.int16(self.results.masterQuality)
        species_data.SURFACEEMISSMEAN_QA = self.results.emisDev
        species_data.SURFACEEMISSIONLAYER_QA = self.results.emissionLayer
        species_data.SURFACETEMPVSAPRIORI_QA = self.results.tsur_minus_prior
        species_data.SURFACETEMPVSATMTEMP_QA = self.results.tsur_minus_tatm0

        # AT_LINE 300 write_products_one.pro
        species_data.SURFACETEMPERATURE = self.state_info.state_info_obj.current['TSUR']
        unique_speciesListFM = utilList.GetUniqueValues(self.retrieval_info.species_list_fm)

        indx = utilList.WhereEqualIndices(unique_speciesListFM, 'TSUR')
        if len(indx) > 0:
            indx = indx[0]

            indxRet = utilList.WhereEqualIndices(self.retrieval_info.species_list, 'TSUR')[0]
            species_data.SURFACETEMPCONSTRAINT = self.retrieval_info.constraint_vector[indxRet]

            # AT_LINE 306 src_ms-2018-12-10/write_products_one.pro
            indy = utilList.WhereEqualIndices(self.retrieval_info.species_names, 'TSUR')[0]
            species_data.SURFACETEMPDEGREESOFFREEDOM = self.results.degreesOfFreedomForSignal[indy]

            species_data.SURFACETEMPERROR = self.results.errorFM[indx]
            species_data.SURFACETEMPINITIAL = self.retrieval_info.initial_guess_list[indxRet]

        # AT_LINE 324 write_products_one.pro
        ispecie = utilList.WhereEqualIndices(self.retrieval_info.species_names, self.spcname)
        ispecie = ispecie[0]  # We just need one from the list so we can index into various variables.

        species_data.DEVIATION_QA = self.results.deviation_QA[ispecie]
        species_data.NUM_DEVIATIONS_QA = self.results.num_deviations_QA[ispecie]
        species_data.DEVIATIONBAD_QA = self.results.DeviationBad_QA[ispecie]

        ind1 = self.retrieval_info.retrieval_info_obj.parameterStart[ispecie]
        ind2 = self.retrieval_info.retrieval_info_obj.parameterEnd[ispecie]
        ind1FM = self.retrieval_info.retrieval_info_obj.parameterStartFM[ispecie]
        ind2FM = self.retrieval_info.retrieval_info_obj.parameterEndFM[ispecie]

        if self.retrieval_info.retrieval_info_obj.mapType[ispecie].lower() == 'linear':
            species_data.RETRIEVEINLOG = np.int32(0)
        elif self.retrieval_info.retrieval_info_obj.mapType[ispecie].lower() == 'linearpca':
            species_data.RETRIEVEINLOG = np.int32(0)
        else:
            species_data.RETRIEVEINLOG = np.int32(1)

        # AT_LINE 342 write_products_one.pro
        species_data.DOFS = np.sum(mpy.get_diagonal(self.results.A[ind1FM: ind2FM+1, ind1FM: ind2FM+1]))
        species_data.PRECISION[pslice] = np.sqrt(mpy.get_diagonal(self.results.Sx_rand[ind1FM: ind2FM+1, ind1FM: ind2FM+1]))

        # Build a 3D array so we can use it to access the below assignments.
        #third_index = np.asarray([0 for ii in range(1)])  # Set the 3rd index all to 0.
        #array_3d_indices = np.ix_(indConv, indConv, third_index)  # (64, 64, 1)

        # Generate an array of indices for the right hand side based on the slice information.
        #rhs_range_index = np.asarray([ii for ii in range(ind1FM, ind2FM + 1)])

        # Using slow method because fast method (using Python awesome list of locations as arrays for indices) is not working.
        species_data.AVERAGINGKERNEL[pslice, pslice] = self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
        species_data.MEASUREMENTERRORCOVARIANCE[pslice, pslice] = self.results.Sx_rand[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
        species_data.TOTALERRORCOVARIANCE[pslice, pslice] = self.results.Sx[ind1FM:ind2FM+1, ind1FM:ind2FM+1]

        # We pass in for rhs_start_index because sum_Sx_Sx_sys_Sx_crossState already contain the correct shape.
        sum_Sx_Sx_sys_Sx_crossState = self.results.Sx_rand[ind1FM:ind2FM+1, ind1FM:ind2FM+1] + \
                                      self.results.Sx_sys[ind1FM:ind2FM+1, ind1FM:ind2FM+1]  + \
                                      self.results.Sx_crossState[ind1FM:ind2FM+1, ind1FM:ind2FM+1]

        species_data.OBSERVATIONERRORCOVARIANCE[pslice, pslice] = sum_Sx_Sx_sys_Sx_crossState

        #
        # Not sure if the right hand side indices are correct for Python.
        #
        #

        species_data.PRIORCOVARIANCE[pslice, pslice] = self.results.Sa[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
        species_data.AVERAGINGKERNELDIAGONAL[pslice] = mpy.get_diagonal(self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]) ## utilGeneral.ManualArraySets(species_data.AVERAGINGKERNELDIAGONAL, get_diagonal(self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]), indConv, rhs_start_index=0)
        species_data.TOTALERROR[pslice] = np.sqrt(mpy.get_diagonal(self.results.Sx[ind1FM:ind2FM+1, ind1FM:ind2FM+1]))

        # AT_LINE 355 write_products_one.pro
        if self.state_info.state_info_obj.cloudPars['num_frequencies'] > 0:
            species_data.CLOUDFREQUENCY = [600, 650, 700, 750, 800, 850, 900, 950, 975, 100, 1025, 1050, 1075, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1900, 2000, 2040, 2060, 2080, 2100, 2200, 2250]

            # AT_LINE 365 src_ms-2018-12-10/write_products_one.pro
            factor = mpy.compute_cloud_factor(
                self.state_info.state_info_obj.current['pressure'], 
                self.state_info.state_info_obj.current['values'][self.state_info.state_info_obj.species.index('TATM'), :], 
                self.state_info.state_info_obj.current['values'][self.state_info.state_info_obj.species.index('H2O'), :], 
                self.state_info.state_info_obj.current['PCLOUD'][0], 
                self.state_info.state_info_obj.current['scalePressure'], 
                self.state_info.state_info_obj.current['tsa']['surfaceAltitudeKm'] * 1000, 
                self.state_info.state_info_obj.current['latitude'])

            self.state_info.state_info_obj.current['convertToOD'] = factor

            # AT_LINE 363 write_products_one.pro
            # AT_LINE 374 src_ms-2018-12-10/write_products_one.pro
            species_data.CLOUDEFFECTIVEOPTICALDEPTH = self.state_info.state_info_obj.current['cloudEffExt'][0, 0:self.state_info.state_info_obj.cloudPars['num_frequencies']] * self.state_info.state_info_obj.current['convertToOD']

            indf = utilList.WhereEqualIndices(self.retrieval_info.species_list_fm, 'CLOUDEXT')
            if len(indf) > 0:
                species_data.CLOUDEFFECTIVEOPTICALDEPTHERROR = self.results.errorFM[indf] * self.state_info.state_info_obj.current['convertToOD']
        # end if self.state_info.state_info_obj.cloudPars['num_frequencies'] > 0:

        # add special fields for HDO
        # AT_LINE 383 src_ms-2018-12-10/write_products_one.pro
        if self.spcname == 'HDO':
            indfh = utilList.WhereEqualIndices(self.retrieval_info.species_list_fm, 'H2O')
            indfd = utilList.WhereEqualIndices(self.retrieval_info.species_list_fm, 'HDO')

            indp = np.where(species_data.SPECIES > 0)[0]

            matrix = species_data.AVERAGINGKERNEL * 0 - 999
            species_data.HDO_H2OAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.HDO_H2OAVERAGINGKERNEL[pslice, pslice] = self.results.A[indfh, indfd]

            species_data.H2O_HDOAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.H2O_HDOAVERAGINGKERNEL[pslice, pslice] = self.results.A[indfd, indfh]

            # AT_LINE 407 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.H2O_H2OAVERAGINGKERNEL[pslice, pslice] = self.results.A[indfh, indfh]

            species_data.HDO_H2OMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OMEASUREMENTERRORCOVARIANCE[pslice, pslice] = self.results.Sx_rand[indfh, indfd]

            # AT_LINE 396 write_products_one.pro
            species_data.H2O_HDOMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOMEASUREMENTERRORCOVARIANCE[pslice, pslice] = self.results.Sx_rand[indfd, indfh]

            # AT_LINE 417 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OMEASUREMENTERRORCOVARIANCE[pslice, pslice] = self.results.Sx_rand[indfh, indfh]

            # AT_LINE 400 write_products_one.pro
            error = self.results.Sx_rand + self.results.Sx_crossState + self.results.Sx_sys

            species_data.HDO_H2OOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OOBSERVATIONERRORCOVARIANCE[pslice, pslice] = error[indfh, indfd]

            species_data.H2O_HDOOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOOBSERVATIONERRORCOVARIANCE[pslice, pslice] = error[indfd, indfh]

            # AT_LINE 434 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OOBSERVATIONERRORCOVARIANCE[pslice, pslice] = error[indfh, indfh]

            # AT_LINE 408 write_products_one.pro
            error = self.results.Sx

            species_data.HDO_H2OTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OTOTALERRORCOVARIANCE[pslice, pslice] = error[indfh, indfd]

            species_data.H2O_HDOTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOTOTALERRORCOVARIANCE[pslice, pslice] = error[indfd, indfh]

            # AT_LINE 445 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OTOTALERRORCOVARIANCE[pslice, pslice] = error[indfh, indfh]

            # AT_LINE 448 src_ms-2018-12-10/write_products_one.pro
            vector_of_fills = np.ndarray(shape=(num_pressures), dtype=np.float32)
            vector_of_fills.fill(-999.0)

            # add H2O constraint and result
            species_data.H2O_CONSTRAINTVECTOR = copy.deepcopy(vector_of_fills)
            species_data.H2O_CONSTRAINTVECTOR[indp] = self.retrieval_info.species_constraint("H2O")

            species_data.H2O_SPECIES = copy.deepcopy(vector_of_fills)
            species_data.H2O_SPECIES[indp] = self.retrieval_info.species_results(self.results, "H2O")

            species_data.H2O_INITIAL = copy.deepcopy(vector_of_fills)

            # TODO
            # This looks wrong to me. Although this is marked initial, it is getting
            # this from the results. It is possible this is correct, perhaps this
            # the value used for the HDO? We should double check this. But this
            # is what the current muses-py code does
            species_data.H2O_INITIAL[indp] = self.retrieval_info.species_results(self.results, "H2O", INITIAL_Flag=True)
        # end if self.spcname == 'HDO':


        ind = utilList.WhereEqualIndices(self.retrieval_info.species_list_fm, 'EMIS')
        if len(ind) > 0:
            # Create an array of indices so we can access i_results.Sx matrix.
            array_2d_indices = np.ix_(ind, ind)  # (64, 64)
            species_data.EMISSIVITY_ERROR = np.sqrt(mpy.get_diagonal(self.results.Sx[array_2d_indices]))

        nx = self.state_info.state_info_obj.emisPars['num_frequencies']
        if nx > 0:
            species_data.EMISSIVITY_CONSTRAINT = self.state_info.state_info_obj.constraint['emissivity'][0:nx]
            species_data.EMISSIVITY_INITIAL = self.state_info.state_info_obj.initialInitial['emissivity'][0:nx]
            species_data.EMISSIVITY = self.state_info.state_info_obj.current['emissivity'][0:nx]
            species_data.EMISSIVITY_WAVENUMBER = self.state_info.state_info_obj.emisPars['frequency'][0:nx]
            if 'native_emissivity' in self.state_info.state_info_obj.initialInitial:
                species_data.NATIVE_HSR_EMISSIVITY_INITIAL = self.state_info.state_info_obj.initialInitial['native_emissivity']
                species_data.NATIVE_HSR_EMIS_WAVENUMBER = self.state_info.state_info_obj.initialInitial['native_emis_wavenumber']
            if 'camel_distance' in self.state_info.state_info_obj.emisPars:
                species_data.EMISSIVITY_OFFSET_DISTANCE = np.array([self.state_info.state_info_obj.emisPars['camel_distance']])
            if 'emissivity_prior_source' in self.state_info.state_info_obj.emisPars:
                runtime_attributes.setdefault('EMISSIVITY_INITIAL', dict())
                runtime_attributes['EMISSIVITY_INITIAL']['database'] = self.state_info.state_info_obj.emisPars['emissivity_prior_source']


        # AT_LINE 631 write_products_one.pro
        # for CH4 add in N2O results, constraint vector, calculate
        # n2o-corrected, save original_species
        if self.spcname == 'CH4' and 'CH4' in self.retrieval_info.species_names:
            species_data.N2O_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.N2O_CONSTRAINTVECTOR = np.zeros(shape=(num_pressures), dtype=np.float32)

            # AT_LINE 676 src_ms-2018-12-10/write_products_one.pro
            species_data.N2O_DOFS = 0.0

            ispecieN2O = -1
            if 'N2O' in self.retrieval_info.species_names:
                ispecieN2O = self.retrieval_info.species_names.index('N2O')
                ind1FMN2O = self.retrieval_info.retrieval_info_obj.parameterStartFM[ispecieN2O]
                ind2FMN2O = self.retrieval_info.retrieval_info_obj.parameterEndFM[ispecieN2O]

            # AT_LINE 683 write_products_one.pro
            if ispecieN2O >= 0:
                # AT_LINE 642 write_products_one.pro
                species_data.N2O_DOFS = 0.0
                species_data.N2O_DOFS = np.sum(mpy.get_diagonal(self.results.A[ind1FMN2O:ind2FMN2O+1, ind1FMN2O:ind2FMN2O+1]))

                species_data.N2O_SPECIES[pslice] = self.retrieval_info.species_results(self.results, 'N2O')
                species_data.N2O_CONSTRAINTVECTOR[pslice] = self.retrieval_info.species_constraint('N2O')
            else:
                # N2O not retrieved... use values from initial guess
                logger.warning("code has not been tested for N2O not retrieved.")
                indn2o = utilList.WhereEqualIndices(self.state_info.state_info_obj.species, 'N2O')
                value = self.state_info.state_info_obj.initial['values'][indn2o, :]
                species_data.N2O_SPECIES[pslice] = copy.deepcopy(value)
                species_data.N2O_CONSTRAINTVECTOR[pslice] = copy.deepcopy(value)

            # correct ch4 from n2o
            species_data.ORIGINAL_SPECIES = copy.deepcopy(species_data.SPECIES)

            # AT_LINE 649 write_products_one.pro
            # AT_LINE 699 src_ms-2018-12-10/write_products_one.pro
            n2o = species_data.N2O_SPECIES[pslice]
            n2o_xa = species_data.N2O_CONSTRAINTVECTOR[pslice]
            ch4 = species_data.SPECIES[pslice]

            species_data.SPECIES[pslice] = np.exp(np.log(ch4) + np.log(n2o_xa) - np.log(n2o))

            # track ev's used
            # AT_LINE 657 write_products_one.pro
            species_data.CH4_EVS = np.zeros(shape=(10), dtype=np.float32)
            species_data.CH4_EVS[:] = self.results.ch4_evs[:]
        # end if self.spcname == 'CH4' and 'CH4' in self.retrieval_info.retrieval_info_obj.species:

        # for CH4 if jointly retrieved with TATM add TATM
        if self.spcname == 'CH4' and 'TATM' in self.retrieval_info.species_names:
            species_data.TATM_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.TATM_CONSTRAINTVECTOR = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.TATM_DEVIATION = 0.0

            ispecieTATM = self.retrieval_info.species_names.index('TATM')
            ind1FMTATM = self.retrieval_info.retrieval_info_obj.parameterStartFM[ispecieTATM]
            ind2FMTATM = self.retrieval_info.retrieval_info_obj.parameterEndFM[ispecieTATM]

            species_data.TATM_SPECIES[pslice] = self.retrieval_info.species_results(self.results, "TATM")
            species_data.TATM_CONSTRAINTVECTOR[pslice] = self.retrieval_info.species_constraint("TATM")

            # AT_LINE 725 src_ms-2018-12-10/write_products_one.pro

            # add in H2O results
            species_data.H2O_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.H2O_CONSTRAINTVECTOR = np.zeros(shape=(num_pressures), dtype=np.float32)

            ispecieH2O = self.retrieval_info.species_names.index('H2O')
            ind1FMH2O = self.retrieval_info.retrieval_info_obj.parameterStartFM[ispecieH2O]
            ind2FMH2O = self.retrieval_info.retrieval_info_obj.parameterEndFM[ispecieH2O]

            species_data.H2O_SPECIES[pslice] = self.retrieval_info.species_results(self.results, "H2O")
            species_data.H2O_CONSTRAINTVECTOR[pslice] = self.retrieval_info.species_constraint("H2O")

            indp = np.where(species_data.TATM_SPECIES > 0)[0]
            maxx = np.amax(np.abs(species_data.TATM_SPECIES[indp] - species_data.TATM_CONSTRAINTVECTOR[indp]))
            species_data.TATM_DEVIATION = maxx # maximum deviation from prior
        # end if species_name == 'CH4' and 'TATM' in self.retrieval_info.retrieval_info_obj.species:

        species_data = species_data.__dict__
        for k, v in species_data.items():
            if isinstance(v, list):
                species_data[k] = np.asarray(v)

        state_element_out = [t for t in self.state_info.state_element_list() if t.should_write_to_l2_product(self.instruments)]
                
        #######
        # write with lite format using cdf_write_tes

        o_data = species_data
        o_data.update(self.generate_geo_data(species_data))
        t = CdfWriteTes()
        t.write(o_data, self.out_fname, runtimeAttributes=runtime_attributes,
                state_element_out=state_element_out)
        
        return o_data

__all__ = ["RetrievalL2Output", ] 
    
