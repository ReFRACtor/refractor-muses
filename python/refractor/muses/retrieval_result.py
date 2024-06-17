import refractor.muses.muses_py as mpy
from .observation_handle import mpy_radiance_from_observation_list
import numpy as np
import os
import logging
logger = logging.getLogger("py-retrieve")

class PropagatedQA:
    '''There are a few parameters that get propagated from one step to the next. Not
    sure exactly what this gets looked for, it look just like flags copied from one
    step to the next. But pull this together into one place so we can track this.
    '''
    def __init__(self):
        self.propagated_qa = {'TATM' : 1, 'H2O' : 1, 'O3' : 1}

    @property
    def tatm_qa(self):
        return self.propagated_qa['TATM']
    
    @property
    def h2o_qa(self):
        return self.propagated_qa['H2O']

    @property
    def o3_qa(self):
        return self.propagated_qa['O3']

    def update(self, retrieval_state_element : 'list[str]', qa_flag : int):
        '''Update the QA flags for items that we retrieved.'''
        for state_element_name in retrieval_state_element:
            if(state_element_name in self.propagated_qa):
                self.propagated_qa[state_element_name] = qa_flag
               
class RetrievalResult:
    '''There are a few top level functions that work with a structure called
    retrieval_results. Pull all this together into an object so we can clearly
    see the interface and possibly change things.

    Unlike a number of things that we want to elevate to a class, this really does
    look like just a structure of various calculated things that then get reported in
    the output files - so I think this is probably little more than wrapping up stuff in
    one place.'''
    def __init__(self, ret_res : dict, strategy_table : 'StrategyTable',
                 retrieval_info : 'RetrievalInfo', state_info : 'StateInfo',
                 obs_list : 'list(MusesObservation)',
                 radiance_full : 'dict',
                 propagated_qa : PropagatedQA):
        '''ret_res is what we get returned from MusesLevmarSolver'''
        self.rstep = mpy.ObjectView(mpy_radiance_from_observation_list(obs_list, include_bad_sample=True))
        self.radiance_full = radiance_full
        self.obs_list = obs_list
        self.instruments = [obs.instrument_name for obs in self.obs_list]
        self.retrieval_info = retrieval_info
        self.state_info = state_info
        self.sounding_metadata = state_info.sounding_metadata()
        self.strategy_table = strategy_table
        self.ret_res = mpy.ObjectView(ret_res)
        # Get old retrieval results structure, and merge in with this object
        d = self.set_retrieval_results()
        d = mpy.set_retrieval_results_derived(
            d, self.rstep, propagated_qa.tatm_qa,
            propagated_qa.o3_qa, propagated_qa.h2o_qa)
        self.__dict__.update(d.__dict__)

    def update_jacobian_sys(self, cfunc_sys : 'CostFunction'):
        '''Run the forward model in cfunc to get the jacobian_sys set.'''
        self.jacobianSys = \
            cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[np.newaxis,:,:]

    def update_error_analysis(self, error_analysis : 'ErrorAnalysis'):
        '''Run the error analysis and calculate various summary statistics for retrieval'''
        res = error_analysis.error_analysis(self.rstep.__dict__, self.retrieval_info,
                                            self.state_info, self)
        res = mpy.write_retrieval_summary(
            self.strategy_table.analysis_directory,
            self.retrieval_info.retrieval_info_obj,
            self.state_info.state_info_obj,
            None,
            res,
            {},
            None,
            self.quality_name, 
            None,
            error_analysis.error_current, 
            writeOutputFlag=False, 
            errorInitial=error_analysis.error_initial
        )
        # This is already done in place
        #self.__dict__.update(res)
                              
    @property
    def species_list_fm(self) -> 'list(str)':
        '''This is the length of the forward model state vector, with a
        retrieval_element name for each location.'''
        return self.retrieval_info.species_list_fm

    @property
    def species_list_retrieval(self) -> 'list(str)':
        '''This is the length of the retrieval state vector, with a
        retrieval_element name for each location.'''
        return self.retrieval_info.species_list
    
    @property
    def pressure_list_fm(self) -> 'list(float)':
        return self.retrieval_info.pressure_list_fm
    
    @property
    def best_iteration(self):
        return self.bestIteration

    @property
    def results_list(self):
        return self.resultsList

    @property
    def master_quality(self):
        return self.masterQuality

    @property
    def jacobian_sys(self):
        return self.jacobianSys

    @property
    def press_list(self):
        return [float(self.strategy_table.preferences["plotMaximumPressure"]),
                float(self.strategy_table.preferences["plotMinimumPressure"])]

    @property
    def quality_name(self):
        with self.strategy_table.chdir_run_dir():
            res = os.path.basename(self.strategy_table.spectral_filename)
            res = res.replace("Microwindows_", "QualityFlag_Spec_")
            res = res.replace("Windows_", "QualityFlag_Spec_")
            res = self.strategy_table.preferences["QualityFlagDirectory"] + res
            
            # if this does not exist use generic nadir / limb quality flag
            if not os.path.isfile(res):
                logger.warning(f'Could not find quality flag file: {res}')
                viewingMode = self.strategy_table.preferences["viewingMode"]
                viewMode = viewingMode.lower().capitalize()

                res = f"{os.path.dirname(res)}/QualityFlag_Spec_{viewMode}.asc"
                logger.warning(f"Using generic quality flag file: {res}")
                # One last check.
                if not os.path.isfile(res):
                    raise RuntimeError(f"Quality flag filename not found: {res}")
            return os.path.abspath(res)

    def set_retrieval_results(self):
        '''This is our own copy of mpy.set_retrieval_results, so we can start making changes
        to clean up the coupling of this.'''
        # Convert any dict to ObjectView so we can have a consistent way of referring to our input.
        num_species = self.retrieval_info.n_species
        nfreqs = len(self.rstep.frequency)
    
        niter = len(self.ret_res.resdiag[:, 0])
        # to standardize the size of maxIter, set it to maxIter from strategy table
        maxIter = self.strategy_table.max_num_iterations
        maxIter = int(maxIter) + 1  # Add 1 to account for initial.


        detectorsUsed = 0
        num_detectors = 1

        # get filters start and end
        # have standard list of filters
        # We need a better way of instantiating this
        ff0 = []
        for i in range(len(self.rstep.filterNames)):
            ff0.append([self.rstep.filterNames[i],] * self.rstep.filterSizes[i])
        ff0 = np.concatenate(ff0)
        # TODO Replace this logic
        filtersName = ['ALL', 'UV1', 'UV2', 'VIS', 'CrIS-fsr-lw', 'CrIS-fsr-mw', 'CrIS-fsr-sw', '2B1',  '1B2',  '2A1',  '1A1',  'BAND1', 'BAND2', 'BAND3', 'BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'O2A',  'WCO2',  'SCO2',  'CH4']
        filtersMap =  ['ALL', 'UV1', 'UV2', 'VIS', 'TIR1',        'TIR3',        'TIR4',        'TIR1', 'TIR2', 'TIR3', 'TIR4', 'UV1',   'UV2',   'UVIS',  'VIS',   'NIR1',  'NIR2',  'SWIR3', 'SWIR4', 'NIR1', 'SWIR1', 'SWIR2', 'SWIR3']
        filters = ['ALL', 'UV1', 'UV2', 'VIS', 'UVIS', 'NIR1','NIR2','SWIR1','SWIR2','SWIR3','SWIR4','TIR1','TIR2','TIR3','TIR4']
        num_filters = len(filters)
        filterStart = [0]
        filterEnd = [len(ff0)-1]
        filter_index = [0]
        filter_list = ['ALL']

        for jj in range(1, len(filtersName)):   # Start jj with 1 since we are leaving the first value alone.
            ind1 = np.where(ff0 == filtersName[jj])[0]
            ind2 = np.where(np.array(filters) == filtersMap[jj])[0][0]
            if len(ind1) > 0:
                filter_index.append(ind2)
                filter_list.append(filters[ind2])
                filterStart.append(np.amin(ind1))
                filterEnd.append(np.amax(ind1))


        num_filters = len(filter_index)
        if num_filters == 1:
            raise RuntimeError("Update set_retrieval_results.  Filter not found in list: ", ff0)

        # get the total number of frequency points in all microwindows for the 
        # gain matrix
        rows = self.retrieval_info.n_totalParameters
        rowsSys = self.retrieval_info.n_totalParametersSys
        rowsFM = self.retrieval_info.n_totalParametersFM
        if rowsSys == 0:
            rowsSys = 1

        o_results = {
            'retrieval': '',
            'is_ocean': self.retrieval_info.is_ocean,
            'badRetrieval': -999,                               
            'retIteration': self.ret_res.xretIterations,    
            'bestIteration': self.ret_res.bestIteration,     
            'num_iterations': self.ret_res.num_iterations,    
            'stopCode': self.ret_res.stopCode,    
            'filter_index':filter_index, 
            'radianceResidualMean':np.zeros(shape=(num_filters), dtype=np.float32),    
            'radianceResidualMeanInitial':np.zeros(shape=(num_filters), dtype=np.float32),    
            'radianceResidualRMS' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'radianceResidualRMSInitial' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'radianceResidualRMSMean' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'radianceResidualRMSRelativeContinuum' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'residualSlope' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'residualQuadratic' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'radianceContinuum' :np.zeros(shape=(num_filters), dtype=np.float32), 
            'radianceSNR': np.zeros(shape=(num_filters), dtype=np.float32),                             
            'residualNormInitial': 0.0,                               
            'residualNormFinal': 0.0,                            
            'detectorsUsed': detectorsUsed,                  
            'radianceResidualMeanDet': np.zeros(shape=(num_detectors), dtype=np.float64), 
            'radianceResidualRMSDet': np.zeros(shape=(num_detectors), dtype=np.float64), 
            'radianceResidualRMSSys': 0.0,                               
            'freqShift': 0.0,                          
            'chiApriori': 0.0,                          
            'radianceResidualMax': 0.0,                          
            'chiLM': np.copy(self.ret_res.residualRMS[self.ret_res.bestIteration]),   
            'num_radiance ': 0, 
            'resultsList': np.zeros(shape=(rows), dtype=np.float64), 
            'resultsListFM': np.zeros(shape=(rowsFM), dtype=np.float64), 
            'error': np.zeros(shape=(rows), dtype=np.float64), 
            'errorFM': np.zeros(shape=(rowsFM), dtype=np.float64), 
            'precision': np.zeros(shape=(rowsFM), dtype=np.float64), 
            'resolution': np.zeros(shape=(rowsFM), dtype=np.float64), 
            # jacobians - for last outputStep
            'GdL': np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),                
            'jacobian': np.zeros(shape=(num_detectors, rowsFM, nfreqs), dtype=np.float64),  
            'jacobianSys': np.zeros(shape=(num_detectors, rowsSys, nfreqs), dtype=np.float64), 
            #  radiances - for all steps + IG, true
            'frequency': np.zeros(shape=(num_detectors, nfreqs), dtype=np.float64),         
            'LMResults_costThresh': np.full((maxIter), -999, dtype=np.float32), 
            'LMResults_iterList': np.full((maxIter, rows), -999, dtype=np.float32), 
            'LMResults_resNorm': np.full((maxIter), -999, dtype=np.float32), 
            'LMResults_resNormNext': np.full((maxIter), -999, dtype=np.float32), 
            'LMResults_jacresNorm': np.full((maxIter), -999, dtype=np.float32), 
            'LMResults_jacResNormNext': np.full((maxIter), -999, dtype=np.float32), 
            'LMResults_pnorm': np.full((maxIter), -999, dtype=np.float32), 
            'LMResults_delta': np.full((maxIter,), -999, dtype=np.float64),
            'radianceObserved': np.zeros(shape=(num_detectors, nfreqs), dtype=np.float32),  # fit
            'radiance': np.zeros(shape=(num_detectors, nfreqs), dtype=np.float32), # fit
            'radianceInitial': np.zeros(shape=(num_detectors, nfreqs), dtype=np.float32), # fit initial  
            # error stuff follows - calc later
            'A': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float32), 
            'A_ret': np.zeros(shape=(rows, rows), dtype=np.float32), 
            'KtSyK': np.zeros(shape=(rows, rows), dtype=np.float32),     
            'Sx': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            'Sa_ret': np.zeros(shape=(rows, rows), dtype=np.float64),
            'Sx_ret_smooth': np.zeros(shape=(rows, rows), dtype=np.float64),
            'Sx_ret_crossState': np.zeros(shape=(rows, rows), dtype=np.float64),
            'Sx_ret_rand': np.zeros(shape=(rows, rows), dtype=np.float64),
            'Sx_ret_sys': np.zeros(shape=(rows, rows), dtype=np.float64),
            'Sx_ret_mapping': np.zeros(shape=(rows, rows), dtype=np.float64),         
            'Sa': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'Sb': np.zeros(shape=(rowsSys, rowsSys), dtype=np.float64), 
            'Sx_smooth_self': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'Sx_smooth': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'Sx_crossState': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'Sx_sys': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'Sx_rand': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'Sx_mapping': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            'SxActual': np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64), 
            'GMatrix': np.zeros(shape=(nfreqs, rows), dtype=np.float64),   
            'GMatrixFM': np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64), 
            # by species
            'informationContentSpecies': np.zeros(shape=(num_species), dtype=np.float64), 
            'degreesOfFreedomNoise': np.zeros(shape=(num_species), dtype=np.float64), 
            'degreesOfFreedomForSignal': np.zeros(shape=(num_species), dtype=np.float64), 
            'degreesOfFreedomForSignalTrop':np.zeros(shape=(num_species), dtype=np.float64), 
            'bestDegreesOfFreedomList' :['' for x in range(num_species)],                   
            'bestDegreesOfFreedomTotal':['' for x in range(num_species)],                   
            'verticalResolution': np.zeros(shape=(num_species), dtype=np.float64), 
            'deviationVsError': 0.0,                                            
            'deviationVsRetrievalCovariance': 0.0,                                       
            'deviationVsAprioriCovariance': 0.0,                                       
            'deviationVsErrorSpecies': np.zeros(shape=(num_species), dtype=np.float64), 
            'deviationVsRetrievalCovarianceSpecies': np.zeros(shape=(num_species), dtype=np.float64), 
            'deviationVsAprioriCovarianceSpecies': np.zeros(shape=(num_species), dtype=np.float64), 
            # quality and general
            'KDotDL': 0.0,                                     
            'KDotDL_list': np.zeros(shape=(rows), dtype=np.float32),  
            'KDotDL_byspecies': np.zeros(shape=(num_species), dtype=np.float32), 
            'KDotDL_species': ['' for x in range(num_species)],     
            'KDotDL_byfilter': np.zeros(shape=(num_filters), dtype=np.float32), 
            'filter_list': filter_list, #  Note, we changed 'filter' to 'filter_list' since 'filter' is a Python keyword.
            'filterStart': filterStart,           
            'filterEnd': filterEnd,             
            'maxKDotDLSys': 0.0 
        }                 

        struct2 = {
            'LDotDL': 0.0, 
            'LDotDL_byfilter': np.zeros(shape=(num_filters), dtype=np.float32), 
            'cloudODAve': 0.0, 
            'cloudODAveError': 0.0, 
            'emisDev': 0.0, 
            'cloudODVar': 0.0, 
            'calscaleMean': 0.0, 
            'Desert_Emiss_QA': 0.0, 
            'H2O_H2OQuality': 0.0, 
            'emissionLayer': 0.0, 
            'ozoneCcurve': 0.0, 
            'ozone_slope_QA': -999.0, 
            'propagatedTATMQA': 0.0, 
            'propagatedO3QA': 0.0, 
            'propagatedH2OQA': 0.0, 
            'masterQuality': -999.0, 
            'tropopausePressure': -999.0, 
            'columnAir': np.full((5), -999, dtype=np.float64),
            'column': np.full((5, 20), -999, dtype=np.float64), # DBLARR(4, 20)-999.0 
            'columnError': np.full((5, 20), -999, dtype=np.float64), # DBLARR(4, 20)-999.0 
            'columnPriorError': np.full((5, 20), -999, dtype=np.float64), #  DBLARR(4, 20)-999.0 
            'columnInitialInitial': np.full((5, 20), -999, dtype=np.float64), # DBLARR(4, 20)-999.0 
            'columnInitial': np.full((5, 20), -999, dtype=np.float64), # DBLARR(4, 20)-999.0 
            'columnPrior': np.full((5, 20), -999, dtype=np.float64), # DBLARR(4, 20)-999.0 
            'columnTrue': np.full((5, 20), -999, dtype=np.float64), # DBLARR(4, 20)-999.0 
            'columnSpecies': ['' for x in range(20)],   # STRARR(20) 
            # EM NOTE - Modified to increase vector size to allow for stratosphere capture
            'columnPressureMax': np.zeros(shape=(5), dtype=np.float32), # FLTARR(4) 
            'columnPressureMin': np.zeros(shape=(5), dtype=np.float32), # FLTARR(4) 
            'columnDOFS': np.zeros(shape=(5, 20), dtype=np.float32), # FLTARR(4, 20) 
            'tsur_minus_tatm0': -999.0, 
            'tsur_minus_prior':-999.0, 
            'deviation_QA': np.full((num_species), -999, dtype=np.float32), # FLTARR(num_species)-999 
            'num_deviations_QA': np.full((num_species), -999, dtype=np.int32),   # INTARR(num_species)-999 
            'DeviationBad_QA': np.full((num_species), -999, dtype=np.int32),   # INTARR(num_species)-999 
            'omi_cloudfraction': 0.0,    # cloud fraction in UV, if used
            'tropomi_cloudfraction': 0.0,   # cloud fraction in UV for tropomi, possibly worth combining the omi and tropomi values 
            'O3_columnErrorDU' : 0.0,  # total colummn error 
            'O3_tropo_consistency': 0.0, # tropospheric column change from initial 
            'ch4_evs': np.zeros(shape=(10), dtype=np.float32), # FLTARR(10) 
            'NESR': np.zeros(shape=(num_detectors, nfreqs), dtype=np.float32) # FLTARR(num_detectors, nfreqs) $ ; nesr
        }
        o_results.update(struct2)

        # Convert to ObjectView from now on.
        o_results = mpy.ObjectView(o_results)

        niter = len(self.ret_res.resdiag[:, 0])
        if niter > maxIter:
            raise RuntimeError(
f'''ERROR: Increase # iterations in Set_Retrieval_Result.  It is too small right now.
niter: {nitr}
len(o_results.LMResults_costThresh): {len(o_results.LMResults_costThresh)}
            ''')

        o_results.LMResults_costThresh[0:niter] = self.ret_res.stopCriteria[:, 0]  # Note the spelling of stopCriteria to match the actual key.
        o_results.LMResults_resNorm[0:niter] = self.ret_res.resdiag[:, 0]  # Note the spelling of resdiag to match the actual key.
        o_results.LMResults_resNormNext[0:niter] = self.ret_res.resdiag[:, 1]  # Note the spelling of resdiag to match the actual key.
        o_results.LMResults_jacresNorm[0:niter] = self.ret_res.resdiag[:, 2]  # Note the spelling of resdiag to match the actual key.
        o_results.LMResults_jacResNormNext[0:niter] = self.ret_res.resdiag[:, 3]  # Note the spelling of resdiag to match the actual key.
        o_results.LMResults_pnorm[0:niter] = self.ret_res.resdiag[:, 4]  # Note the spelling of resdiag to match the actual key.
        o_results.LMResults_delta[0:niter] = self.ret_res.delta
        
        # get retrieval vector result (for all species) for best iteration
        ii = o_results.bestIteration
        if ii == 0:
            result = self.retrieval_info.initial_guess_list[0:self.retrieval_info.n_totalParameters]
        else:
            result = self.ret_res.xretIterations[o_results.bestIteration, :]

        o_results.resultsList[:] = result[:]
        o_results.resultsListFM[:] = self.ret_res.xretFM

        # get retrieval vector result (for all species) for all iterations

        for iq in range(self.ret_res.num_iterations+1):
            if iq == 0:
                o_results.LMResults_iterList[iq, :] = self.retrieval_info.initial_guess_list[0:self.retrieval_info.n_totalParameters]
            else:
                o_results.LMResults_iterList[iq, :] = self.ret_res.xretIterations[iq, :]

        # AT_LINE 328 Set_Retrieval_Results.pro Set_Retrieval_Results
        o_results.frequency = self.rstep.frequency

        o_results.jacobian[:] = self.ret_res.jacobian['jacobian_data']  # Note that we have chosen 'jacobian_data' as the key in jacobian dict.
        o_results.radiance[:] = self.ret_res.radiance['radiance']
        o_results.radianceObserved[:] = self.rstep.radiance

        o_results.retrieval = 'true'
        if o_results.num_iterations == 0:
            logger.error(f'Retrieval files not found.  Value of o_results.num_iterations {o_results.num_iterations}')
            o_results.retrieval = 'false'
            o_results.radianceInitial[:, :] = self.ret_res.radiance['radiance'][:, :]
        else:
            o_results.radianceInitial[:, :] = self.ret_res.radianceIterations[0, :, :]

        o_results.NESR = self.rstep.NESR    

        x = np.amin(np.abs(self.state_info.state_info_obj.emisPars['frequency'] - 1025))
        ind = np.argmin(np.abs(self.state_info.state_info_obj.emisPars['frequency'] - 1025))
        o_results.Desert_Emiss_QA = self.state_info.state_info_obj.current['emissivity'][ind]

        o_results.omi_cloudfraction = self.state_info.state_info_obj.current['omi']['cloud_fraction']
        o_results.tropomi_cloudfraction = self.state_info.state_info_obj.current['tropomi']['cloud_fraction']

        return o_results
        

__all__ = ["PropagatedQA", "RetrievalResult"]    
