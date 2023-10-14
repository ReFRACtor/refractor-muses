from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy
from .retrieval_output import RetrievalOutput
import numpy as np

logger = logging.getLogger("py-retrieve")

class RetrievalL2Output(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_L2 files.'''
    
    def __reduce__(self):
        return (_new_from_init, (self.__class__,))
    
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
                for spc in self.strategy_table.table_entry('retrievalElements', i).split(","):
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
            self._species_list = self.retrievalInfo.species_names
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
        if(location == "retrieval step" and self.table_step == 0):
            self.dataTATM = None
            self.dataH2O = None
            self.dataN2O = None
        if(location != "retrieval step"):
            return
        # Regenerate this for the current step
        self._species_count = None
        self._species_list = None
        for self.spcname in self.species_list:
            if(self.retrievalInfo.species_list_fm.count(self.spcname) <= 1 or
               self.spcname in ('CLOUDEXT', 'EMIS') or 
               self.spcname.startswith('OMI') or
               self.spcname.startswith('NIR')):
                continue
            self.out_fname = f"{self.retrieval_strategy.output_directory}/Products/Products_L2-{self.spcname}-{self.species_count[self.spcname]}.nc"
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # Not sure about the logic here, but this is what script_retrieval_ms does
            if(not os.path.exists(self.out_fname) or self.spcname in ('TATM', 'H2O', 'N2O')):
                # Code assumes we are in rundir
                with self.retrieval_strategy.chdir_run_dir():
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
                indn = self.stateInfo.state_info_obj.species.index('N2O')
                value = self.stateInfo.state_info_obj.initial['values'][indn, :]
                data2['SPECIES'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['INITIAL'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['CONSTRAINTVECTOR'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['AVERAGINGKERNEL'].fill(0.0)
                data2['OBSERVATIONERRORCOVARIANCE'].fill(0.0)
        elif(self.spcname == "HDO"):
            data2 = self.dataH2O
        else:
            data2 = None

        if(self.spcname == "H2O" and self.dataTATM is not None):
            self.out_fname = f"{self.retrieval_strategy.output_directory}/Products/Lite_Products_L2-RH-{self.species_count[self.spcname]}.nc"
            if("OCO2" not in self.instruments):
                liteDirectory = '../OSP/Lite/'
                # Code assumes we are in rundir
                with self.retrieval_strategy.chdir_run_dir():
                    mpy.make_lite_casper_script_retrieval(self.table_step,
                                  self.out_fname, self.quality_name, self.instruments,
                                  liteDirectory, dataInfo, self.dataTATM, "RH",
                                  step=self.species_count[self.spcname],
                                  times_species_retrieved=self.species_count[self.spcname])
                
        self.out_fname = f"{self.retrieval_strategy.output_directory}/Products/Lite_Products_L2-{self.spcname}-{self.species_count[self.spcname]}.nc"
        if 'OCO2' not in self.instruments:
            liteDirectory = '../OSP/Lite/'
            # Code assumes we are in rundir
            with self.retrieval_strategy.chdir_run_dir():
                data2 = mpy.make_lite_casper_script_retrieval(self.table_step,
                                  self.out_fname, self.quality_name, self.instruments,
                                  liteDirectory, dataInfo, data2, self.spcname,
                                  step=self.species_count[self.spcname],
                                  times_species_retrieved=self.species_count[self.spcname])

    def write_l2(self):
        '''Create L2 product file'''
        runtime_attributes = dict()

        # AT_LINE 7 write_products_one.pro
        nobs = 1 # number of observations
        # num_pressures varies based on surface pressure.  We set it to max here.
        num_pressures = 67  # 'np' is the numpy alias so we change to num_pressures.
        nfreqEmis = 121

        if len(self.stateInfo.state_info_obj.current['pressure']) == 20:
            num_pressures = 20  # OCO-2 has 20 levels (sigma)

        if np.max(self.stateInfo.state_info_obj.true['values']) > 0:
            have_true = True
        else:
            have_true = False



        # 0.0 = np.ndarray(shape=(nobs), dtype=np.float32)
        # 0.0.fill(-999)
        # if 0.0.size == 1:
        #     0.0 = -999

        # -999 = np.ndarray(shape=(nobs), dtype=np.int32)
        # -999.fill(-999)
        # if -999.size == 1:
        #     -999 = -999

        niter = len(self.results.LMResults_costThresh) # set this so that uniform size
        nfilter = len(self.results.filter_index)-1

        # AT_LINE 16 write_products_one.pro
        # PYTHON_NOTE: To make life easier, we will store all keys in this dictionary as uppercase.  All keys coming from other objects will be treated accordingly.

        # TODO: Replace with Python data class. Use snake_case for member variables
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

        nx = self.stateInfo.state_info_obj.emisPars['num_frequencies']
        if nx == 0:
            del species_data['EMISSIVITY_CONSTRAINT']
            del species_data['EMISSIVITY_ERROR']
            del species_data['EMISSIVITY_INITIAL']
            del species_data['EMISSIVITY']
            del species_data['EMISSIVITY_WAVENUMBER']


        species_data = mpy.ObjectView(species_data)  # Convert to ObjectView so we can use the '.' notation to access the fields.

        # AT_LINE 92 write_products_one.pro
        # PYTHON_NOTE: To make life easier, we will store all keys in this class as uppercase.  All keys coming from other objects will be treated accordingly.
        geo_data = {
            'DayNightFlag'.upper(): -999,                            
            'landFlag'.upper(): np.zeros(shape=(nobs), dtype=np.int32) - 999,
            'LATITUDE'.upper(): np.zeros(shape=(nobs), dtype=np.float64) - 999,
            'LONGITUDE'.upper(): np.zeros(shape=(nobs), dtype=np.float64) - 999,
            'TIME'.upper(): np.zeros(shape=(nobs), dtype=np.int64) - 999,
            'surfaceTypeFootprint'.upper(): np.zeros(shape=(nobs), dtype=np.int64) - 999,
            'soundingID'.upper(): ['' for ii in range(nobs)]
        }

        geo_data = mpy.ObjectView(geo_data)  # Convert to ObjectView so we can use the '.' notation to access the fields.

        # AT_LINE 103 write_products_one.pro
        # PYTHON_NOTE: To make life easier, we will store all keys in this class as uppercase.  All keys coming from other objects will be treated accordingly.
        ancillary_data = {
            'FILTER_POSITION_1A'.upper(): -999,
            'FILTER_POSITION_1B'.upper(): -999,
            'FILTER_POSITION_2A'.upper(): -999,
            'FILTER_POSITION_2B'.upper(): -999,
            'ORBITASCENDINGFLAG'.upper(): -999,
            'PIXELSUSEDFLAG'.upper(): np.zeros(shape=(64), dtype=np.int32) - 999,
            'SOLARAZIMUTHANGLE'.upper(): np.zeros(shape=(64), dtype=np.float32) - 999,
            'SPACECRAFTALTITUDE'.upper(): -999,
            'SPACECRAFTLATITUDE'.upper(): 0.0,
            'SPACECRAFTLONGITUDE'.upper(): 0.0,
            'SURFACEEMISSCONSTRAINT'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'SURFACEEMISSERRORS'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'SURFACEEMISSINITIAL'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'SURFACEEMISSIVITY'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            'emissivity_Wavenumber'.upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
        }

        ancillary_data = mpy.ObjectView(ancillary_data) # Convert to ObjectView so we can use the '.' notation to access the fields.

        # AT_LINE 121 write_products_one.pro
        species_data.TROPOPAUSEPRESSURE = self.results.tropopausePressure
        if self.stateInfo.state_info_obj.gmaoTropopausePressure > 0:
            species_data.TROPOPAUSEPRESSURE = self.stateInfo.state_info_obj.gmaoTropopausePressure

        # AT_LINE 126 write_products_one.pro
        species_data.DESERT_EMISS_QA = self.results.Desert_Emiss_QA
        species_data.PROPAGATED_TATM_QA = np.int32(self.results.propagatedTATMQA)
        species_data.PROPAGATED_O3_QA = np.int32(self.results.propagatedO3QA)
        species_data.PROPAGATED_H2O_QA = np.int32(self.results.propagatedH2OQA)
        species_data.RADIANCEMAXIMUMSNR = self.results.radianceMaximumSNR
        species_data.RESIDUALNORMFINAL = self.results.residualNormFinal
        species_data.RESIDUALNORMINITIAL = self.results.residualNormInitial

        # AT_LINE 149 write_products_one.pro
        # get column / altitude / air density / trop column stuff
        stateOne = mpy.ObjectView(self.stateInfo.state_info_obj.current) # Convert to ObjectView so we can use the dot '.' notation to access the fields.
        waterType = None
        pge_flag = True

        utilList = mpy.UtilList()
        indt = np.where(np.array(self.stateInfo.state_info_obj.species) == 'TATM')[0][0]
        indh = np.where(np.array(self.stateInfo.state_info_obj.species) == 'H2O')[0][0]
        (altitudeResult, x) = mpy.compute_altitude_pge(
            self.stateInfo.state_info_obj.current['pressure'], 
            stateOne.values[indt, :], 
            stateOne.values[indh, :], 
            stateOne.tsa['surfaceAltitudeKm'] * 1000, 
            stateOne.latitude, 
            waterType, pge_flag)

        # altitudeResult OBJECT_TYPE is dict
        altitudeResult = mpy.ObjectView(altitudeResult) # Convert to ObjectView so we can use the dot '.' notation to access the fields.

        # AT_LINE 169 write_products_one.pro
        if self.spcname == 'O3':
            species_data.O3_CCURVE_QA = np.int32(self.results.ozoneCcurve)
            species_data.O3_SLOPE_QA = self.results.ozone_slope_QA

            species_data.O3_COLUMNERRORDU = self.results.O3_columnErrorDU
            species_data.O3_TROPO_CONSISTENCY_QA = self.results.O3_tropo_consistency

        # AT_LINE 191 write_products_one.pro
        indcol = utilList.WhereEqualIndices(self.results.columnSpecies, self.spcname)

        # 0 = total
        # 1 is tropospheric
        # 2 is upper trop
        # 3 is lower trop
        # 4 is strato

        # Because 'column' fields have different shapes, care must be taken to use the appropriate indices on the right hand side.
        # AT_LINE 196 write_products_one.pro
        if len(indcol) > 0:
            indcol = indcol[0]

            species_data.COLUMN = copy.deepcopy(self.results.column[:, indcol])
            species_data.COLUMN_AIR = copy.deepcopy(self.results.columnAir[:])
            species_data.COLUMN_DOFS = copy.deepcopy(self.results.columnDOFS[:, indcol])
            species_data.COLUMN_ERROR = copy.deepcopy(self.results.columnError[:, indcol])
            species_data.COLUMN_INITIAL = copy.deepcopy(self.results.columnInitial[:, indcol])
            species_data.COLUMN_PRESSUREMAX = copy.deepcopy(self.results.columnPressureMax[:])
            species_data.COLUMN_PRESSUREMIN = copy.deepcopy(self.results.columnPressureMin[:])
            species_data.COLUMN_PRIOR = copy.deepcopy(self.results.columnPrior[:, indcol])

        # AT_LINE 211 write_products_one.pro
        # move 'ALL' to it's own variable.  So set [1:*] to [0:*-1]
        #filters = ['UV1', 'UV2', 'VIS', '2B1', '1B2', '2A1', '1A1']
        #index to filter names
        species_data.RADIANCERESIDUALRMS_FILTER = self.results.radianceResidualRMS[1:]
        species_data.RADIANCERESIDUALMEAN_FILTER = self.results.radianceResidualMean[1:]
        species_data.radianceResidualRMSRelativeContinuum_FILTER = self.results.radianceResidualRMSRelativeContinuum[1:]
        species_data.RADIANCE_CONTINUUM_FILTER = self.results.radianceContinuum[1:]
        species_data.RADIANCESNR_FILTER = self.results.radianceSNR[1:]
        species_data.FILTER_INDEX = self.results.filter_index[1:]
        species_data.RADIANCERESIDUALSLOPE_FILTER = self.results.residualSlope[1:]
        species_data.RADIANCERESIDUALQUADRATIC_FILTER = self.results.residualQuadratic[1:]



        # AT_LINE 211 write_products_one.pro
        species_data.RADIANCERESIDUALRMS = self.results.radianceResidualRMS[0]
        species_data.RADIANCERESIDUALMEAN = self.results.radianceResidualMean[0]
        species_data.RADIANCE_RESIDUAL_STDEV_CHANGE = self.results.radianceResidualRMSInitial[0] - self.results.radianceResidualRMS[0]


        if 'OMI' in self.instruments:
            # Make all names uppercased to make life easier.
            species_data.OMI_SZA_UV2 = self.stateInfo.state_info_obj.current['omi']['sza_uv2']
            species_data.OMI_RAZ_UV2 = self.stateInfo.state_info_obj.current['omi']['raz_uv2']
            species_data.OMI_VZA_UV2 = self.stateInfo.state_info_obj.current['omi']['vza_uv2']
            species_data.OMI_SCA_UV2 = self.stateInfo.state_info_obj.current['omi']['sca_uv2']

            species_data.OMI_SZA_UV1 = self.stateInfo.state_info_obj.current['omi']['sza_uv1']
            species_data.OMI_RAZ_UV1 = self.stateInfo.state_info_obj.current['omi']['raz_uv1']
            species_data.OMI_VZA_UV1 = self.stateInfo.state_info_obj.current['omi']['vza_uv1']
            species_data.OMI_SCA_UV1 = self.stateInfo.state_info_obj.current['omi']['sca_uv1']

            # could get these from state.current.omipars
            # note I added this to the new_state_structures and Make_UIP_OMI.pro
            species_data.OMI_CLOUDFRACTION = self.stateInfo.state_info_obj.current['omi']['cloud_fraction']
            species_data.OMI_CLOUDFRACTIONCONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['omi']['cloud_fraction']
            species_data.OMI_CLOUDTOPPRESSURE = self.stateInfo.state_info_obj.constraint['omi']['cloud_pressure']

            species_data.OMI_SURFACEALBEDOUV1 = self.stateInfo.state_info_obj.current['omi']['surface_albedo_uv1']
            species_data.OMI_SURFACEALBEDOUV1CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['omi']['surface_albedo_uv1']

            species_data.OMI_SURFACEALBEDOUV2 = self.stateInfo.state_info_obj.current['omi']['surface_albedo_uv2']
            species_data.OMI_SURFACEALBEDOUV2CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['omi']['surface_albedo_uv2']

            species_data.OMI_SURFACEALBEDOSLOPEUV2 = self.stateInfo.state_info_obj.current['omi']['surface_albedo_slope_uv2']
            species_data.OMI_SURFACEALBEDOSLOPEUV2CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['omi']['surface_albedo_slope_uv2']

            species_data.OMI_NRADWAVUV1 = self.stateInfo.state_info_obj.current['omi']['nradwav_uv1']
            species_data.OMI_NRADWAVUV2 = self.stateInfo.state_info_obj.current['omi']['nradwav_uv2']

            species_data.OMI_ODWAVUV1 = self.stateInfo.state_info_obj.current['omi']['odwav_uv1']
            species_data.OMI_ODWAVUV2 = self.stateInfo.state_info_obj.current['omi']['odwav_uv2']

            species_data.OMI_RINGSFUV1 = self.stateInfo.state_info_obj.current['omi']['ring_sf_uv1']
            species_data.OMI_RINGSFUV2 = self.stateInfo.state_info_obj.current['omi']['ring_sf_uv2']

        if 'TROPOMI' in self.instruments:
            # As with OMI, make all names uppercased to make life easier.
            # EM NOTE - This will have to be expanded if additional tropomi bands are used
            species_data.TROPOMI_SZA_BAND1 = self.stateInfo.state_info_obj.current['tropomi']['sza_BAND1']
            species_data.TROPOMI_RAZ_BAND1 = self.stateInfo.state_info_obj.current['tropomi']['raz_BAND1']
            species_data.TROPOMI_VZA_BAND1 = self.stateInfo.state_info_obj.current['tropomi']['vza_BAND1']
            species_data.TROPOMI_SCA_BAND1 = self.stateInfo.state_info_obj.current['tropomi']['sca_BAND1']

            species_data.TROPOMI_SZA_BAND2 = self.stateInfo.state_info_obj.current['tropomi']['sza_BAND2']
            species_data.TROPOMI_RAZ_BAND2 = self.stateInfo.state_info_obj.current['tropomi']['raz_BAND2']
            species_data.TROPOMI_VZA_BAND2 = self.stateInfo.state_info_obj.current['tropomi']['vza_BAND2']
            species_data.TROPOMI_SCA_BAND2 = self.stateInfo.state_info_obj.current['tropomi']['sca_BAND2']

            species_data.TROPOMI_SZA_BAND3 = self.stateInfo.state_info_obj.current['tropomi']['sza_BAND3']
            species_data.TROPOMI_RAZ_BAND3 = self.stateInfo.state_info_obj.current['tropomi']['raz_BAND3']
            species_data.TROPOMI_VZA_BAND3 = self.stateInfo.state_info_obj.current['tropomi']['vza_BAND3']
            species_data.TROPOMI_SCA_BAND3 = self.stateInfo.state_info_obj.current['tropomi']['sca_BAND3']

            species_data.TROPOMI_SZA_BAND7 = self.stateInfo.state_info_obj.current['tropomi']['sza_BAND7']
            species_data.TROPOMI_RAZ_BAND7 = self.stateInfo.state_info_obj.current['tropomi']['raz_BAND7']
            species_data.TROPOMI_VZA_BAND7 = self.stateInfo.state_info_obj.current['tropomi']['vza_BAND7']
            species_data.TROPOMI_SCA_BAND7 = self.stateInfo.state_info_obj.current['tropomi']['sca_BAND7']

            # could get these from state.current.tropomipars
            species_data.TROPOMI_CLOUDFRACTION = self.stateInfo.state_info_obj.current['tropomi']['cloud_fraction']
            species_data.TROPOMI_CLOUDFRACTIONCONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['cloud_fraction']
            species_data.TROPOMI_CLOUDTOPPRESSURE = self.stateInfo.state_info_obj.constraint['tropomi']['cloud_pressure']

            species_data.TROPOMI_SURFACEALBEDOBAND1 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_BAND1']
            species_data.TROPOMI_SURFACEALBEDOBAND1CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_BAND1']

            species_data.TROPOMI_SURFACEALBEDOBAND2 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_BAND2']
            species_data.TROPOMI_SURFACEALBEDOBAND2CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_BAND2']

            species_data.TROPOMI_SURFACEALBEDOBAND3 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_BAND3']
            species_data.TROPOMI_SURFACEALBEDOBAND3CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_BAND3']

            species_data.TROPOMI_SURFACEALBEDOBAND7 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_BAND7']
            species_data.TROPOMI_SURFACEALBEDOBAND7CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_BAND7']

            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND2 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_slope_BAND2']
            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND2CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND2']

            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND3 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_slope_BAND3']
            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND3CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND3']

            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND7 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_slope_BAND7']
            species_data.TROPOMI_SURFACEALBEDOSLOPEBAND7CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND7']

            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND2 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_slope_order2_BAND2']
            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND2CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND2']

            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_slope_order2_BAND3']
            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND3']

            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND7 = self.stateInfo.state_info_obj.current['tropomi']['surface_albedo_slope_order2_BAND7']
            species_data.TROPOMI_SURFACEALBEDOSLOPEORDER2BAND7CONSTRAINTVECTOR = self.stateInfo.state_info_obj.constraint['tropomi']['surface_albedo_slope_BAND7']

            species_data.TROPOMI_SOLARSHIFTBAND1 = self.stateInfo.state_info_obj.current['tropomi']['solarshift_BAND1']
            species_data.TROPOMI_SOLARSHIFTBAND2 = self.stateInfo.state_info_obj.current['tropomi']['solarshift_BAND2']
            species_data.TROPOMI_SOLARSHIFTBAND3 = self.stateInfo.state_info_obj.current['tropomi']['solarshift_BAND3']
            species_data.TROPOMI_SOLARSHIFTBAND7 = self.stateInfo.state_info_obj.current['tropomi']['solarshift_BAND7']

            species_data.TROPOMI_RADIANCESHIFTBAND1 = self.stateInfo.state_info_obj.current['tropomi']['radianceshift_BAND1']
            species_data.TROPOMI_RADIANCESHIFTBAND2 = self.stateInfo.state_info_obj.current['tropomi']['radianceshift_BAND2']
            species_data.TROPOMI_RADIANCESHIFTBAND3 = self.stateInfo.state_info_obj.current['tropomi']['radianceshift_BAND3']
            species_data.TROPOMI_RADIANCESHIFTBAND7 = self.stateInfo.state_info_obj.current['tropomi']['radianceshift_BAND7']

            species_data.TROPOMI_RADSQUEEZEBAND1 = self.stateInfo.state_info_obj.current['tropomi']['radsqueeze_BAND1']
            species_data.TROPOMI_RADSQUEEZEBAND2 = self.stateInfo.state_info_obj.current['tropomi']['radsqueeze_BAND2']
            species_data.TROPOMI_RADSQUEEZEBAND3 = self.stateInfo.state_info_obj.current['tropomi']['radsqueeze_BAND3']
            species_data.TROPOMI_RADSQUEEZEBAND7 = self.stateInfo.state_info_obj.current['tropomi']['radsqueeze_BAND7']

            species_data.TROPOMI_RINGSFBAND1 = self.stateInfo.state_info_obj.current['tropomi']['ring_sf_BAND1']
            species_data.TROPOMI_RINGSFBAND2 = self.stateInfo.state_info_obj.current['tropomi']['ring_sf_BAND2']
            species_data.TROPOMI_RINGSFBAND3 = self.stateInfo.state_info_obj.current['tropomi']['ring_sf_BAND3']
            species_data.TROPOMI_RINGSFBAND7 = self.stateInfo.state_info_obj.current['tropomi']['ring_sf_BAND7']

            species_data.TROPOMI_TEMPSHIFTBAND3 = self.stateInfo.state_info_obj.current['tropomi']['temp_shift_BAND3']
            species_data.TROPOMI_TEMPSHIFTBAND7 = self.stateInfo.state_info_obj.current['tropomi']['temp_shift_BAND7']

        if 'OCO2' in self.instruments:
            # info from pre-processor or instrument information used in quality flags
            species_data.OCO2_CO2_RATIO_IDP = self.stateInfo.state_info_obj.current['oco2']['co2_ratio_idp']
            species_data.OCO2_H2O_RATIO_IDP = self.stateInfo.state_info_obj.current['oco2']['h2o_ratio_idp']
            species_data.OCO2_DP_ABP = self.stateInfo.state_info_obj.current['oco2']['dp_abp']
            species_data.OCO2_ALTITUDE_STDDEV = self.stateInfo.state_info_obj.current['oco2']['altitude_stddev']
            species_data.OCO2_MAX_DECLOCKING_FACTOR_WCO2 = self.stateInfo.state_info_obj.current['oco2']['max_declocking_factor_wco2']
            species_data.OCO2_MAX_DECLOCKING_FACTOR_SCO2 = self.stateInfo.state_info_obj.current['oco2']['max_declocking_factor_sco2']

            # get albedo polynomial
            from .nir.make_albedo_maps import make_albedo_maps
            map_to_pars, map_to_state = make_albedo_maps(3, self.stateInfo.state_info_obj.current['nir']['albplwave']) 
            albedo_poly = map_to_pars @ self.stateInfo.state_info_obj.current['nir']['albpl']
            albedo_full = self.stateInfo.state_info_obj.current['nir']['albpl']

            # convert to Lambertian
            if self.stateInfo.state_info_obj.current['nir']['albtype'] == 2:
                albedo_poly = albedo_poly * 0.07
                albedo_full = albedo_full * 0.07
            species_data.NIR_ALBEDO_POLY2 = albedo_poly
            species_data.NIR_ALBEDO = albedo_full

            # info from retrieval used in quality flags
            if 'NIR1' in self.results.filter_list: # O2A is generically labeled NIR1
                ind = np.where(np.array(self.results.filter_list) == 'NIR1')[0][0]
                species_data.NIR_FLUOR_REL = self.stateInfo.state_info_obj.current['nir']['fluor'][0] / self.results.radianceContinuum[ind]
            else:
                species_data.NIR_FLUOR_REL = self.stateInfo.state_info_obj.current['nir']['fluor'][0]/self.results.radianceContinuum[1]*0 - 999 # keep same type
            species_data.DELTA_P = self.stateInfo.state_info_obj.current['pressure'][0] - self.stateInfo.state_info_obj.constraint['pressure'][0]
            indco2 = np.where(np.array(self.stateInfo.state_info_obj.species) == 'CO2')
            indtatm = np.where(np.array(self.stateInfo.state_info_obj.species) == 'TATM')
            species_data.CO2_GRAD_DEL = (self.stateInfo.state_info_obj.current['values'][indco2,0] - self.stateInfo.state_info_obj.constraint['values'][indco2,0] - self.stateInfo.state_info_obj.current['values'][indco2,7] + self.stateInfo.state_info_obj.constraint['values'][indco2,7])*1e6
            ind = np.argmin(np.abs(self.stateInfo.state_info_obj.current['pressure'] - 750))
            species_data.DELTA_T = np.mean(self.stateInfo.state_info_obj.current['values'][indtatm,ind]) - np.mean(self.stateInfo.state_info_obj.constraint['values'][indtatm,ind])
            species_data.NIR_WINDSPEED = self.stateInfo.state_info_obj.current['nir']['wind']

            # aod info, make od's have consistent ordering with 0 = total, 1 = ice_cloud..., 2 = wc_008, ...
            mylist = [b'total', b'ice_cloud_MODIS6_deltaM_1000',b'wc_008',b'DU',b'SO',b'strat',b'oc', b'SS',b'BC']
            naer = len(mylist)
            aerod = np.zeros((naer),dtype=np.float32) - 999
            aerp = np.zeros((naer),dtype=np.float32) - 999
            for ii in range(0,naer):
                ind = np.where(np.array(self.stateInfo.state_info_obj.current['nir']['aertype']) == mylist[ii])[0]
                if len(ind) > 0:
                    aerod[ii] = self.stateInfo.state_info_obj.current['nir']['aerod'][ind[0]]
                    aerp[ii] = self.stateInfo.state_info_obj.current['nir']['aerp'][ind[0]]

            # get total OD
            aerod[0]=np.sum(self.stateInfo.state_info_obj.current['nir']['aerod'])
            # get aerosol mean pressure by weighting with OD
            aerp[0]=np.sum(self.stateInfo.state_info_obj.current['nir']['aerod'] * self.stateInfo.state_info_obj.current['nir']['aerp'])/np.sum(self.stateInfo.state_info_obj.current['nir']['aerod'])
            species_data.NIR_AEROD = aerod
            species_data.NIR_AERP = aerp

            # cloud3d
            species_data.NIR_CLOUD3D_SLOPE = self.stateInfo.state_info_obj.current['nir']['cloud3dslope']
            species_data.NIR_CLOUD3D_OFFSET = self.stateInfo.state_info_obj.current['nir']['cloud3doffset']

            if have_true:
                # true values for nir quantities
                # species_data.NIR_ALBEDO_POLY2_TRUE
                # species_data.NIR_FLUOR_REL_TRUE
                # species_data.DELTA_P_TRUE
                # species_data.CO2_GRAD_DEL_TRUE
                # species_data.DELTA_T_TRUE
                # species_data.NIR_WINDSPEED_TRUE
                # species_data.NIR_AEROD_TRUE
                # species_data.NIR_AERP_TRUE
                # species_data.NIR_CLOUD3D_SLOPE_TRUE

                # get albedo polynomial
                albedo_poly = map_to_pars @ self.stateInfo.state_info_obj.true['nir']['albpl']
                albedo_full = self.stateInfo.state_info_obj.true['nir']['albpl']
                albedo_poly_full = map_to_state @ map_to_pars @ self.stateInfo.state_info_obj.true['nir']['albpl']

                # convert to Lambertian
                if self.stateInfo.state_info_obj.true['nir']['albtype'] == 2:
                    albedo_poly = albedo_poly * 0.07
                    albedo_full = albedo_full * 0.07
                    albedo_poly_full = albedo_poly_full * 0.07
                species_data.NIR_ALBEDO_POLY2_TRUE = albedo_poly
                species_data.NIR_ALBEDO_TRUE = albedo_full

                # how much different 2nd order polynomial is from true albedo
                species_data.NIR_ALBEDO_POLY2_ERROR_TRUE = albedo_full - albedo_poly_full

                # info from retrieval used in quality flags
                if 'NIR1' in self.results.filter_list: # O2A is generically labeled NIR1
                    ind = np.where(np.array(self.results.filter_list) == 'NIR1')[0][0]
                    species_data.NIR_FLUOR_REL_TRUE = self.stateInfo.state_info_obj.true['nir']['fluor'][0] / self.results.radianceContinuum[ind]
                else:
                    species_data.NIR_FLUOR_REL_TRUE = self.stateInfo.state_info_obj.true['nir']['fluor'][0]/self.results.radianceContinuum[1]*0 - 999 # keep same type

                species_data.DELTA_P_TRUE = self.stateInfo.state_info_obj.true['pressure'][0] - self.stateInfo.state_info_obj.constraint['pressure'][0]
                indco2 = np.where(np.array(self.stateInfo.state_info_obj.species) == 'CO2')
                indtatm = np.where(np.array(self.stateInfo.state_info_obj.species) == 'TATM')
                species_data.CO2_GRAD_DEL_TRUE = (self.stateInfo.state_info_obj.true['values'][indco2,0] - self.stateInfo.state_info_obj.constraint['values'][indco2,0] - self.stateInfo.state_info_obj.true['values'][indco2,7] + self.stateInfo.state_info_obj.constraint['values'][indco2,7])*1e6
                ind = np.argmin(np.abs(self.stateInfo.state_info_obj.true['pressure'] - 750))
                species_data.DELTA_T_TRUE = np.mean(self.stateInfo.state_info_obj.true['values'][indtatm,ind]) - np.mean(self.stateInfo.state_info_obj.constraint['values'][indtatm,ind])
                species_data.NIR_WINDSPEED_TRUE = self.stateInfo.state_info_obj.true['nir']['wind']

                # aod info, make od's have consistent ordering with 0 = total, 1 = ice_cloud..., 2 = wc_008, ...
                mylist = [b'total', b'ice_cloud_MODIS6_deltaM_1000',b'wc_008',b'DU',b'SO',b'strat',b'oc', b'SS',b'BC']
                naer = len(mylist)
                aerod = np.zeros((naer),dtype=np.float32) - 999
                aerp = np.zeros((naer),dtype=np.float32) - 999
                for ii in range(0,naer):
                    ind = np.where(np.array(self.stateInfo.state_info_obj.true['nir']['aertype']) == mylist[ii])[0]
                    if len(ind) > 0:
                        aerod[ii] = self.stateInfo.state_info_obj.true['nir']['aerod'][ind[0]]
                        aerp[ii] = self.stateInfo.state_info_obj.true['nir']['aerp'][ind[0]]

                # get total OD
                aerod[0]=np.sum(self.stateInfo.state_info_obj.true['nir']['aerod'])
                # get aerosol mean pressure by weighting with OD
                aerp[0]=np.sum(self.stateInfo.state_info_obj.true['nir']['aerod'] * self.stateInfo.state_info_obj.true['nir']['aerp'])/np.sum(self.stateInfo.state_info_obj.true['nir']['aerod'])
                species_data.NIR_AEROD_TRUE = aerod
                species_data.NIR_AERP_TRUE = aerp

                # cloud3d
                species_data.NIR_CLOUD3D_SLOPE_TRUE = self.stateInfo.state_info_obj.true['nir']['cloud3dslope']
                species_data.NIR_CLOUD3D_OFFSET_TRUE = self.stateInfo.state_info_obj.true['nir']['cloud3doffset']


        # AT_LINE 268 write_products_one.pro
        # get results... first how many pressures?
        myP = len(self.stateInfo.state_info_obj.current['values'][0, :])
        indConv = np.asarray([ii for ii in range(myP)])
        indConv = indConv + (num_pressures - myP)
        indf1 = num_pressures - myP
        indf2 = num_pressures # index + 1

        # get species results
        FM_Flag = True

        # IDL_NOTE: FLTARR(67, 1) is the same as FLTARR(67)
        # PYTHON_NOTE: Because in Python, the 2nd dimension of 1 is explicit, we have to use it to refer on the left hand side as [indConv, 0].
        species_data.SPECIES[indConv] = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, self.spcname, FM_Flag)[:]

        if have_true:
            species_data.TRUE = np.zeros(shape=(num_pressures), dtype=np.float32) - 999
            utilList = mpy.UtilList()
            indfs = np.where(np.array(self.retrievalInfo.retrieval_info_obj.speciesListFM) == self.spcname)[0]
            species_data.TRUE[indConv] = self.retrievalInfo.retrieval_info_obj.trueParameterListFM[indfs]

            species_data.TRUE_AK = np.zeros(shape=(num_pressures), dtype=np.float32) - 999
            true = self.retrievalInfo.retrieval_info_obj.trueParameterListFM[indfs]
            xa = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, self.spcname, 1, i_CONSTRAINT_Flag=True)[:]
            ispecie = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.species, self.spcname)
            ispecie = ispecie[0]  # We just need one from the list so we can index into various variables.
            ind1FM = self.retrievalInfo.retrieval_info_obj.parameterStartFM[ispecie]
            ind2FM = self.retrievalInfo.retrieval_info_obj.parameterEndFM[ispecie]
            ak = self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
            species_data.TRUE_AK[indConv] = xa + ak.T @ (true - xa)



        FM_Flag = True
        INITIAL_Flag = True
        species_data.INITIAL[indConv] = mpy.get_vector(self.retrievalInfo.retrieval_info_obj.initialGuessList, self.retrievalInfo.retrieval_info_obj, self.spcname, FM_Flag, INITIAL_Flag)[:]

        FM_Flag = True
        INITIAL_Flag = True
        species_data.CONSTRAINTVECTOR[indConv] = mpy.get_vector(self.retrievalInfo.retrieval_info_obj.constraintVector, self.retrievalInfo.retrieval_info_obj, self.spcname, FM_Flag, INITIAL_Flag)[:]

        species_data.PRESSURE[indConv] = self.stateInfo.state_info_obj.current['pressure'][:]

        # altitude /air density
        # PYTHON_NOTE: Something is weird here.  We need to be smart about converting from from molec/cm3 to molec/m3
        # As of 12/19/2018, doing the division make the output much smaller than the IDL code in column_integrate() function.
        # For now, we will only do the division of the largest value is larger than 1e25.
        # AT_LINE 278 write_products_one.pro
        species_data.AIRDENSITY[indConv] = (altitudeResult.airDensity*1e6)[:] #convert molec/cm3 -> molec/m3

        species_data.ALTITUDE[indConv] = altitudeResult.altitude[:]

        # AT_LINE 281 write_products_one.pro
        species_data.CLOUDTOPPRESSURE = self.stateInfo.state_info_obj.current['PCLOUD'][0]
        indx = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.speciesListFM, 'PCLOUD')
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
        species_data.SURFACETEMPERATURE = self.stateInfo.state_info_obj.current['TSUR']
        unique_speciesListFM = utilList.GetUniqueValues(self.retrievalInfo.retrieval_info_obj.speciesListFM)

        indx = utilList.WhereEqualIndices(unique_speciesListFM, 'TSUR')
        if len(indx) > 0:
            indx = indx[0]

            indxRet = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.speciesList, 'TSUR')[0]
            species_data.SURFACETEMPCONSTRAINT = self.retrievalInfo.retrieval_info_obj.constraintVector[indxRet]

            # AT_LINE 306 src_ms-2018-12-10/write_products_one.pro
            indy = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.species, 'TSUR')[0]
            species_data.SURFACETEMPDEGREESOFFREEDOM = self.results.degreesOfFreedomForSignal[indy]

            species_data.SURFACETEMPERROR = self.results.errorFM[indx]
            species_data.SURFACETEMPINITIAL = self.retrievalInfo.retrieval_info_obj.initialGuessList[indxRet]

        # AT_LINE 324 write_products_one.pro
        ispecie = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.species, self.spcname)
        ispecie = ispecie[0]  # We just need one from the list so we can index into various variables.

        species_data.DEVIATION_QA = self.results.deviation_QA[ispecie]
        species_data.NUM_DEVIATIONS_QA = self.results.num_deviations_QA[ispecie]
        species_data.DEVIATIONBAD_QA = self.results.DeviationBad_QA[ispecie]

        ind1 = self.retrievalInfo.retrieval_info_obj.parameterStart[ispecie]
        ind2 = self.retrievalInfo.retrieval_info_obj.parameterEnd[ispecie]
        ind1FM = self.retrievalInfo.retrieval_info_obj.parameterStartFM[ispecie]
        ind2FM = self.retrievalInfo.retrieval_info_obj.parameterEndFM[ispecie]

        if self.retrievalInfo.retrieval_info_obj.mapType[ispecie].lower() == 'linear':
            species_data.RETRIEVEINLOG = np.int32(0)
        elif self.retrievalInfo.retrieval_info_obj.mapType[ispecie].lower() == 'linearpca':
            species_data.RETRIEVEINLOG = np.int32(0)
        else:
            species_data.RETRIEVEINLOG = np.int32(1)

        utilMath = mpy.UtilMath()

        # AT_LINE 342 write_products_one.pro
        species_data.DOFS = np.sum(mpy.get_diagonal(self.results.A[ind1FM: ind2FM+1, ind1FM: ind2FM+1]))
        species_data.PRECISION[indConv] = np.sqrt(mpy.get_diagonal(self.results.Sx_rand[ind1FM: ind2FM+1, ind1FM: ind2FM+1]))

        # Build a 3D array so we can use it to access the below assignments.
        #third_index = np.asarray([0 for ii in range(1)])  # Set the 3rd index all to 0.
        #array_3d_indices = np.ix_(indConv, indConv, third_index)  # (64, 64, 1)

        # Generate an array of indices for the right hand side based on the slice information.
        #rhs_range_index = np.asarray([ii for ii in range(ind1FM, ind2FM + 1)])

        # Using slow method because fast method (using Python awesome list of locations as arrays for indices) is not working.
        species_data.AVERAGINGKERNEL[indf1:indf2, indf1:indf2] = self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
        species_data.MEASUREMENTERRORCOVARIANCE[indf1:indf2, indf1:indf2] = self.results.Sx_rand[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
        species_data.TOTALERRORCOVARIANCE[indf1:indf2, indf1:indf2] = self.results.Sx[ind1FM:ind2FM+1, ind1FM:ind2FM+1]

        # We pass in for rhs_start_index because sum_Sx_Sx_sys_Sx_crossState already contain the correct shape.
        sum_Sx_Sx_sys_Sx_crossState = self.results.Sx_rand[ind1FM:ind2FM+1, ind1FM:ind2FM+1] + \
                                      self.results.Sx_sys[ind1FM:ind2FM+1, ind1FM:ind2FM+1]  + \
                                      self.results.Sx_crossState[ind1FM:ind2FM+1, ind1FM:ind2FM+1]

        species_data.OBSERVATIONERRORCOVARIANCE[indf1:indf2, indf1:indf2] = sum_Sx_Sx_sys_Sx_crossState

        #
        # Not sure if the right hand side indices are correct for Python.
        #
        #

        species_data.PRIORCOVARIANCE[indf1:indf2, indf1:indf2] = self.results.Sa[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
        species_data.AVERAGINGKERNELDIAGONAL[indConv] = mpy.get_diagonal(self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]) ## utilGeneral.ManualArraySets(species_data.AVERAGINGKERNELDIAGONAL, get_diagonal(self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]), indConv, rhs_start_index=0)
        species_data.TOTALERROR[indConv] = np.sqrt(mpy.get_diagonal(self.results.Sx[ind1FM:ind2FM+1, ind1FM:ind2FM+1]))

        # AT_LINE 355 write_products_one.pro
        if self.stateInfo.state_info_obj.cloudPars['num_frequencies'] > 0:
            species_data.CLOUDFREQUENCY = [600, 650, 700, 750, 800, 850, 900, 950, 975, 100, 1025, 1050, 1075, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1900, 2000, 2040, 2060, 2080, 2100, 2200, 2250]

            # AT_LINE 365 src_ms-2018-12-10/write_products_one.pro
            factor = mpy.compute_cloud_factor(
                self.stateInfo.state_info_obj.current['pressure'], 
                self.stateInfo.state_info_obj.current['values'][self.stateInfo.state_info_obj.species.index('TATM'), :], 
                self.stateInfo.state_info_obj.current['values'][self.stateInfo.state_info_obj.species.index('H2O'), :], 
                self.stateInfo.state_info_obj.current['PCLOUD'][0], 
                self.stateInfo.state_info_obj.current['scalePressure'], 
                self.stateInfo.state_info_obj.current['tsa']['surfaceAltitudeKm'] * 1000, 
                self.stateInfo.state_info_obj.current['latitude'])

            self.stateInfo.state_info_obj.current['convertToOD'] = factor

            # AT_LINE 363 write_products_one.pro
            # AT_LINE 374 src_ms-2018-12-10/write_products_one.pro
            species_data.CLOUDEFFECTIVEOPTICALDEPTH = self.stateInfo.state_info_obj.current['cloudEffExt'][0, 0:self.stateInfo.state_info_obj.cloudPars['num_frequencies']] * self.stateInfo.state_info_obj.current['convertToOD']

            indf = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.speciesListFM, 'CLOUDEXT')
            if len(indf) > 0:
                species_data.CLOUDEFFECTIVEOPTICALDEPTHERROR = self.results.errorFM[indf] * self.stateInfo.state_info_obj.current['convertToOD']
        # end if self.stateInfo.state_info_obj.cloudPars['num_frequencies'] > 0:

        # add special fields for HDO
        # AT_LINE 383 src_ms-2018-12-10/write_products_one.pro
        if self.spcname == 'HDO':
            indfh = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.speciesListFM, 'H2O')
            indfd = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.speciesListFM, 'HDO')

            indp = np.where(species_data.SPECIES > 0)[0]

            matrix = species_data.AVERAGINGKERNEL * 0 - 999
            species_data.HDO_H2OAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.HDO_H2OAVERAGINGKERNEL[indf1:indf2, indf1:indf2] = self.results.A[indfh, indfd]

            species_data.H2O_HDOAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.H2O_HDOAVERAGINGKERNEL[indf1:indf2, indf1:indf2] = self.results.A[indfd, indfh]

            # AT_LINE 407 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.H2O_H2OAVERAGINGKERNEL[indf1:indf2, indf1:indf2] = self.results.A[indfh, indfh]

            species_data.HDO_H2OMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OMEASUREMENTERRORCOVARIANCE[indf1:indf2, indf1:indf2] = self.results.Sx_rand[indfh, indfd]

            # AT_LINE 396 write_products_one.pro
            species_data.H2O_HDOMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOMEASUREMENTERRORCOVARIANCE[indf1:indf2, indf1:indf2] = self.results.Sx_rand[indfd, indfh]

            # AT_LINE 417 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OMEASUREMENTERRORCOVARIANCE[indf1:indf2, indf1:indf2] = self.results.Sx_rand[indfh, indfh]

            # AT_LINE 400 write_products_one.pro
            error = self.results.Sx_rand + self.results.Sx_crossState + self.results.Sx_sys

            species_data.HDO_H2OOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OOBSERVATIONERRORCOVARIANCE[indf1:indf2, indf1:indf2] = error[indfh, indfd]

            species_data.H2O_HDOOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOOBSERVATIONERRORCOVARIANCE[indf1:indf2, indf1:indf2] = error[indfd, indfh]

            # AT_LINE 434 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OOBSERVATIONERRORCOVARIANCE[indf1:indf2, indf1:indf2] = error[indfh, indfh]

            # AT_LINE 408 write_products_one.pro
            error = self.results.Sx

            species_data.HDO_H2OTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OTOTALERRORCOVARIANCE[indf1:indf2, indf1:indf2] = error[indfh, indfd]

            species_data.H2O_HDOTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOTOTALERRORCOVARIANCE[indf1:indf2, indf1:indf2] = error[indfd, indfh]

            # AT_LINE 445 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OTOTALERRORCOVARIANCE[indf1:indf2, indf1:indf2] = error[indfh, indfh]

            # AT_LINE 448 src_ms-2018-12-10/write_products_one.pro
            vector_of_fills = np.ndarray(shape=(num_pressures), dtype=np.float32)
            vector_of_fills.fill(-999.0)

            # add H2O constraint and result
            species_data.H2O_CONSTRAINTVECTOR = copy.deepcopy(vector_of_fills)
            species_data.H2O_CONSTRAINTVECTOR[indp] = mpy.get_vector(self.retrievalInfo.retrieval_info_obj.constraintVector, self.retrievalInfo.retrieval_info_obj, 'H2O', FM_Flag)

            species_data.H2O_SPECIES = copy.deepcopy(vector_of_fills)
            species_data.H2O_SPECIES[indp] = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, 'H2O', FM_Flag)

            species_data.H2O_INITIAL = copy.deepcopy(vector_of_fills)

            INITIAL_Flag = True
            species_data.H2O_INITIAL[indp] = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, 'H2O', FM_Flag, INITIAL_Flag)
        # end if self.spcname == 'HDO':


        # AT_LINE 432 write_products_one.pro
        filename = './DateTime.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)

        utctime = mpy.tes_file_get_preference(fileID, "UTC_Time")
        timestruct = mpy.utc(utctime)
        geo_data.TIME = mpy.tai(timestruct, True).astype(np.int32)

        # get target ID
        # AT_LINE 439 write_products_one.pro
        filename = './Measurement_ID.asc'
        (read_status, fileID) = mpy.read_all_tes(filename)
        infoFile = mpy.tes_file_get_struct(fileID)

        geo_data.SOUNDINGID = infoFile['preferences']['key']

        # AT_LINE 483 src_ms-2019-05-29/write_products_one.pro
        if 'AIRS' in self.instruments:
            geo_data.AIRS_GRANULE = np.int16(infoFile['preferences']['AIRS_Granule'])       
            geo_data.AIRS_ATRACK_INDEX = np.int16(infoFile['preferences']['AIRS_ATrack_Index'])
            geo_data.AIRS_XTRACK_INDEX = np.int16(infoFile['preferences']['AIRS_XTrack_Index'])    
            geo_data.POINTINGANGLE_AIRS = abs(stateOne.airs['scanAng'])

        # AT_LINE 490 src_ms-2019-05-29/write_products_one.pro
        if 'TES' in self.instruments:
            geo_data.TES_RUN = np.int16(infoFile['preferences']['TES_run'])
            geo_data.TES_SEQUENCE = np.int16(infoFile['preferences']['TES_sequence'])
            geo_data.TES_SCAN = np.int16(infoFile['preferences']['TES_scan'])
            geo_data.POINTINGANGLE_TES = self.stateInfo.state_info_obj.current['tes']['boresightNadirRadians']*180/np.pi

        # AT_LINE 497 src_ms-2019-05-29/write_products_one.pro
        if 'OMI' in self.instruments:
            geo_data.OMI_ATRACK_INDEX = np.int16(infoFile['preferences']['OMI_ATrack_Index'])
            geo_data.OMI_XTRACK_INDEX_UV1 = np.int16(infoFile['preferences']['OMI_XTrack_UV1_Index'])
            geo_data.OMI_XTRACK_INDEX_UV2 = np.int16(infoFile['preferences']['OMI_XTrack_UV2_Index'])
            geo_data.POINTINGANGLE_OMI = np.abs(self.stateInfo.state_info_obj.current['omi']['vza_uv2'])


        if 'TROPOMI' in self.instruments:
            geo_data.TROPOMI_ATRACK_INDEX = np.int16(infoFile['preferences']['TROPOMI_ATrack_Index'])
            # EM NOTE - Because variable numbers of bands may be used in retrievals, will have to try each one
            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND1 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND1'])
                geo_data.POINTINGANGLE_TROPOMI_BAND1 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND1'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND1 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND1 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND2 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND2'])
                geo_data.POINTINGANGLE_TROPOMI_BAND2 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND2'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND2 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND2 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND3 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND3'])
                geo_data.POINTINGANGLE_TROPOMI_BAND3 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND3'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND3 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND3 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND7 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND7'])
                geo_data.POINTINGANGLE_TROPOMI_BAND7 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND7'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND7 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND7 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND4 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND4'])
                geo_data.POINTINGANGLE_TROPOMI_BAND4 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND4'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND4 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND4 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND5 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND5'])
                geo_data.POINTINGANGLE_TROPOMI_BAND5 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND5'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND5 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND5 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND6 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND6'])
                geo_data.POINTINGANGLE_TROPOMI_BAND6 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND6'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND6 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND6 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND7 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND7'])
                geo_data.POINTINGANGLE_TROPOMI_BAND7 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND7'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND7 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND7 = -999.0

            try:
                geo_data.TROPOMI_XTRACK_INDEX_BAND8 = np.int16(infoFile['preferences']['TROPOMI_XTrack_Index_BAND8'])
                geo_data.POINTINGANGLE_TROPOMI_BAND8 = np.abs(self.stateInfo.state_info_obj.current['tropomi']['vza_BAND8'])
            except:
                geo_data.TROPOMI_XTRACK_INDEX_BAND8 = np.int16(-999)
                geo_data.POINTINGANGLE_TROPOMI_BAND8 = -999.0


        # AT_LINE 504 src_ms-2018-12-10/write_products_one.pro
        if 'CRIS' in self.instruments:
            geo_data.CRIS_GRANULE = np.int16(infoFile['preferences']['CRIS_Granule'])
            geo_data.CRIS_ATRACK_INDEX = np.int16(infoFile['preferences']['CRIS_ATrack_Index'])
            geo_data.CRIS_XTRACK_INDEX = np.int16(infoFile['preferences']['CRIS_XTrack_Index'])
            geo_data.CRIS_PIXEL_INDEX = np.int16(infoFile['preferences']['CRIS_Pixel_Index'])
            geo_data.POINTINGANGLE_CRIS = abs(stateOne.cris['scanAng'])

            # track instrument and resolution for cris
            mydict = {'suomi_nasa_nsr':0, 'suomi_nasa_fsr':1, 'suomi_nasa_nomw':2,
                'jpss1_nasa_fsr':3, 'suomi_cspp_fsr':4, 'jpss1_cspp_fsr':5, 'jpss2_cspp_fsr':6}
            geo_data.CRIS_L1B_TYPE = np.int16(mydict[self.stateInfo.state_info_obj.current['cris']['l1bType']])

        # AT_LINE 554 write_products_one.pro (from idl-retrieve 1.12, not 1.3)

        # should do fresh water too
        if self.stateInfo.state_info_obj.current['surfaceType'].upper() == 'OCEAN':
            geo_data.SURFACETYPEFOOTPRINT = np.int32(2)
            geo_data.LANDFLAG = np.int32(0)
        else:
            geo_data.SURFACETYPEFOOTPRINT = np.int32(3)
            geo_data.LANDFLAG = np.int32(1)

        # get surface type using hres database
        if self.retrievalInfo.retrieval_info_obj.surfaceType == 'OCEAN':
            geo_data.LANDFLAG = np.int32(0)
            geo_data.SURFACETYPEFOOTPRINT = np.int32(2)

            # Not sure how to translate this: IF min(ABS(state.current.heightKm)) GT 0.1 THEN geo_data.surfaceTypeFootprint = 1
            if np.amin(np.abs(self.stateInfo.state_info_obj.current['heightKm'])) > 0.1:
                geo_data.SURFACETYPEFOOTPRINT = 1
        # end if self.retrievalInfo.retrieval_info_obj.surfaceType == 'OCEAN':

        geo_data.LATITUDE = self.stateInfo.state_info_obj.current['latitude']
        geo_data.LONGITUDE = self.stateInfo.state_info_obj.current['longitude']

        ancillary_data.ORBITASCENDINGFLAG = self.stateInfo.state_info_obj.current['tes']['orbitAscending']

        # AT_LINE 499 write_products_one.pro
        # discriminate day or night
        # approximate time using longitude to adjust

        # there was a domain error near the dateline.  SSK 6/2023
        hour = timestruct['hour'] + geo_data.LONGITUDE / 180. * 12
        if hour < 0:
            hour += 24
        if hour > 24:
            hour -= 24

        if (hour >= 8 and hour <= 22):
            geo_data.DAYNIGHTFLAG = np.int16(1)

        if (hour <= 5 or hour >= 22):
            geo_data.DAYNIGHTFLAG = np.int16(0)

        # if sza defined, then use this for daynight, update 12/2017, DF, SSK
        if 'omi_sza_uv2' in species_data.__dict__:
            geo_data.DAYNIGHTFLAG = 0
            if species_data.OMI_SZA_UV2 < 85:
                geo_data.DAYNIGHTFLAG = np.int16(1)

        # AT_LINE 536 write_products_one.pro
        ind = utilList.WhereEqualIndices(self.retrievalInfo.retrieval_info_obj.speciesListFM, 'EMIS')
        if len(ind) > 0:
            # Create an array of indices so we can access i_results.Sx matrix.
            array_2d_indices = np.ix_(ind, ind)  # (64, 64)
            ancillary_data.SURFACEEMISSERRORS = np.sqrt(mpy.get_diagonal(self.results.Sx[array_2d_indices]))

        nx = self.stateInfo.state_info_obj.emisPars['num_frequencies']
        if nx > 0:
            ancillary_data.SURFACEEMISSINITIAL = self.stateInfo.state_info_obj.initialInitial['emissivity'][0:nx]
            ancillary_data.SURFACEEMISSIVITY = self.stateInfo.state_info_obj.current['emissivity'][0:nx]
            ancillary_data.EMISSIVITY_WAVENUMBER = self.stateInfo.state_info_obj.emisPars['frequency'][0:nx]

            # add emissivity to products files
            species_data.EMISSIVITY_CONSTRAINT = self.stateInfo.state_info_obj.constraint['emissivity'][0:nx]
            species_data.EMISSIVITY_ERROR = ancillary_data.SURFACEEMISSERRORS
            species_data.EMISSIVITY_INITIAL = ancillary_data.SURFACEEMISSINITIAL
            species_data.EMISSIVITY = ancillary_data.SURFACEEMISSIVITY
            species_data.EMISSIVITY_WAVENUMBER = ancillary_data.EMISSIVITY_WAVENUMBER
            if 'native_emissivity' in self.stateInfo.state_info_obj.initialInitial:
                species_data.NATIVE_HSR_EMISSIVITY_INITIAL = self.stateInfo.state_info_obj.initialInitial['native_emissivity']
                species_data.NATIVE_HSR_EMIS_WAVENUMBER = self.stateInfo.state_info_obj.initialInitial['native_emis_wavenumber']
            if 'camel_distance' in self.stateInfo.state_info_obj.emisPars:
                species_data.EMISSIVITY_OFFSET_DISTANCE = np.array([self.stateInfo.state_info_obj.emisPars['camel_distance']])
            if 'emissivity_prior_source' in self.stateInfo.state_info_obj.emisPars:
                runtime_attributes.setdefault('EMISSIVITY_INITIAL', dict())
                runtime_attributes['EMISSIVITY_INITIAL']['database'] = self.stateInfo.state_info_obj.emisPars['emissivity_prior_source']


        # AT_LINE 631 write_products_one.pro
        # for CH4 add in N2O results, constraint vector, calculate
        # n2o-corrected, save original_species
        if self.spcname == 'CH4' and 'CH4' in self.retrievalInfo.retrieval_info_obj.species:
            species_data.N2O_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.N2O_CONSTRAINTVECTOR = np.zeros(shape=(num_pressures), dtype=np.float32)

            # AT_LINE 676 src_ms-2018-12-10/write_products_one.pro
            species_data.N2O_DOFS = 0.0

            ispecieN2O = -1
            if 'N2O' in self.retrievalInfo.retrieval_info_obj.species:
                ispecieN2O = self.retrievalInfo.retrieval_info_obj.species.index('N2O')
                ind1FMN2O = self.retrievalInfo.retrieval_info_obj.parameterStartFM[ispecieN2O]
                ind2FMN2O = self.retrievalInfo.retrieval_info_obj.parameterEndFM[ispecieN2O]

            # AT_LINE 683 write_products_one.pro
            if ispecieN2O >= 0:
                # AT_LINE 642 write_products_one.pro
                species_data.N2O_DOFS = 0.0
                species_data.N2O_DOFS = np.sum(mpy.get_diagonal(self.results.A[ind1FMN2O:ind2FMN2O+1, ind1FMN2O:ind2FMN2O+1]))

                FM_Flag = True
                species_data.N2O_SPECIES[indConv] = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, 'N2O', FM_Flag)

                INITIAL_Flag = True
                species_data.N2O_CONSTRAINTVECTOR[indConv] = mpy.get_vector(self.retrievalInfo.retrieval_info_obj.constraintVector, self.retrievalInfo.retrieval_info_obj, 'N2O', FM_Flag, INITIAL_Flag)
            else:
                # N2O not retrieved... use values from initial guess
                logger.warning("code has not been tested for N2O not retrieved.")
                indn2o = utilList.WhereEqualIndices(self.stateInfo.state_info_obj.species, 'N2O')
                value = self.stateInfo.state_info_obj.initial['values'][indn2o, :]
                species_data.N2O_SPECIES[indConv] = copy.deepcopy(value)
                species_data.N2O_CONSTRAINTVECTOR[indConv] = copy.deepcopy(value)

            # correct ch4 from n2o
            species_data.ORIGINAL_SPECIES = copy.deepcopy(species_data.SPECIES)

            # AT_LINE 649 write_products_one.pro
            # AT_LINE 699 src_ms-2018-12-10/write_products_one.pro
            n2o = species_data.N2O_SPECIES[indConv]
            n2o_xa = species_data.N2O_CONSTRAINTVECTOR[indConv]
            ch4 = species_data.SPECIES[indConv]

            species_data.SPECIES[indConv] = np.exp(np.log(ch4) + np.log(n2o_xa) - np.log(n2o))

            # track ev's used
            # AT_LINE 657 write_products_one.pro
            species_data.CH4_EVS = np.zeros(shape=(10), dtype=np.float32)
            species_data.CH4_EVS[:] = self.results.ch4_evs[:]
        # end if self.spcname == 'CH4' and 'CH4' in self.retrievalInfo.retrieval_info_obj.species:

        # for CH4 if jointly retrieved with TATM add TATM
        if self.spcname == 'CH4' and 'TATM' in self.retrievalInfo.retrieval_info_obj.species:
            species_data.TATM_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.TATM_CONSTRAINTVECTOR = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.TATM_DEVIATION = 0.0

            ispecieTATM = self.retrievalInfo.retrieval_info_obj.species.index('TATM')
            ind1FMTATM = self.retrievalInfo.retrieval_info_obj.parameterStartFM[ispecieTATM]
            ind2FMTATM = self.retrievalInfo.retrieval_info_obj.parameterEndFM[ispecieTATM]

            FM_Flag = True
            species_data.TATM_SPECIES[indConv] = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, 'TATM', FM_Flag)

            INITIAL_Flag = True
            species_data.TATM_CONSTRAINTVECTOR[indConv] = mpy.get_vector(self.retrievalInfo.retrieval_info_obj.constraintVector, self.retrievalInfo.retrieval_info_obj, 'TATM', FM_Flag, INITIAL_Flag)

            # AT_LINE 725 src_ms-2018-12-10/write_products_one.pro

            # add in H2O results
            species_data.H2O_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.H2O_CONSTRAINTVECTOR = np.zeros(shape=(num_pressures), dtype=np.float32)

            ispecieH2O = self.retrievalInfo.retrieval_info_obj.species.index('H2O')
            ind1FMH2O = self.retrievalInfo.retrieval_info_obj.parameterStartFM[ispecieH2O]
            ind2FMH2O = self.retrievalInfo.retrieval_info_obj.parameterEndFM[ispecieH2O]

            species_data.H2O_SPECIES[indConv] = mpy.get_vector(self.results.resultsList, self.retrievalInfo.retrieval_info_obj, 'H2O', FM_Flag)
            species_data.H2O_CONSTRAINTVECTOR[indConv] = mpy.get_vector(self.retrievalInfo.retrieval_info_obj.constraintVector, self.retrievalInfo.retrieval_info_obj, 'H2O', FM_Flag, INITIAL_Flag)

            indp = np.where(species_data.TATM_SPECIES > 0)[0]
            maxx = np.amax(np.abs(species_data.TATM_SPECIES[indp] - species_data.TATM_CONSTRAINTVECTOR[indp]))
            species_data.TATM_DEVIATION = maxx # maximum deviation from prior
        # end if species_name == 'CH4' and 'TATM' in self.retrievalInfo.retrieval_info_obj.species:

        # AT_LINE 682 write_products_one.pro

        # convert species_data lists to arrays
        mydata_as_dict = species_data.__dict__
        my_keys = list(mydata_as_dict.keys())

        for xx in range(0, len(my_keys)):
            if isinstance(mydata_as_dict[my_keys[xx]], list):
                mydata_as_dict[my_keys[xx]] = np.asarray(mydata_as_dict[my_keys[xx]])

        # convert geo_data lists to arrays
        mydata_as_dict = geo_data.__dict__
        my_keys = list(mydata_as_dict.keys())

        for xx in range(0, len(my_keys)):
            if isinstance(mydata_as_dict[my_keys[xx]], list):
                mydata_as_dict[my_keys[xx]] = np.asarray(mydata_as_dict[my_keys[xx]])

        # print(function_name, "Writing: ", netcdf_filename)

        #######
        # write with lite format using cdf_write_tes

        # need to combine species_data and geo_data into one dictionary
        # NOTE: No ancillary_data. Do we need that?
        # o_data = struct_combine(species_data.__dict__, geo_data.__dict__)

        # same as struct_combine
        o_data = vars(species_data)
        data2 = vars(geo_data)
        o_data.update(data2)

        # then can immediately use cdf_write_tes
        mpy.cdf_write_tes(o_data, self.out_fname, runtimeAttributes=runtime_attributes)

        # AT_LINE 684 write_products_one.pro

        return o_data

__all__ = ["RetrievalL2Output", ] 
    
