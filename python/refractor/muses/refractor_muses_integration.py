# This will likely migrate to refractor.muses, but it will start here as
# we initially focus on tropomi.

import refractor.muses.muses_py as mpy
import refractor.framework as rf
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .refractor_uip import RefractorUip
from .retrieval_info import RetrievalInfo
from .fm_obs_creator import FmObsCreator
from .cost_function import CostFunction
from .muses_retrieval_step import MusesRetrievalStep
from .muses_forward_model_step import MusesForwardModelStep
import numpy as np
import copy
import os
import pickle
import sys
import logging
logger = logging.getLogger('py-retrieve')

class RefractorMusesIntegration(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''This handles the Refractor/Muses integration.

    Note that this should be largely replaced with RetrievalStrategy, which
    largely replaces Muses-py. But we leave this integration in place for
    doing more details testing at a lower level.

    We do this by replacing two top level functions, run_retrieval
    and run_forward_model. Both of these are changed to use our CostFunction,
    and applies the various plumbing to give Muses-py what it expects from
    these two functions.

    The CostFunction can turn around and use MusesForwardModel classes to
    just call the existing muses-py code - the intention is to use ReFRACtor
    plumbing but the existing functionality. We can then selectively replace
    pieces with ReFRACtor code, e.g. use the muses-py AIRS forward model with
    the ReFRACtor OMI forward model.

    Note because I keep needing to look this up, the call tree for a
    muses-py is:

      cli - top level entry point
      script_retrieval_ms- Handles all the strategy steps
      run_retrieval - Solves a single step of the strategy
      levmar_nllsq_elanor - Solver
      residual_fm_jacobian - cost function
      fm_wrapper - Forward model. Note this handles combining instruments
                   (e.g., AIRS+OMI)
      omi_fm (for OMI) Forward model
      rtf_omi - lower level of omi forward model

    '''
    
    def __init__(self, save_debug_data=False, **kwargs):
        '''This take the keywords that we pass to FmObsCreator to create
        the forward model and state vector.

        If save_debug_data is True, then we save MusesRetrievalStep and
        MusesForwardModelStep data each time run_retrieval or run_forward_model
        is called. This can then be used to debug any issues with a
        particular step of the processing.
        '''
        self.fm_obs_creator = FmObsCreator()
        self.instrument_handle_set = self.fm_obs_creator.instrument_handle_set
        self.save_debug_data = save_debug_data
        self.kwargs = kwargs
        
    def register_with_muses_py(self):
        '''Register this object and the helper objects with muses-py,
        to replace a call to omi_fm.
        '''
        mpy.register_replacement_function("run_retrieval", self)
        mpy.register_replacement_function("run_forward_model", self)
        
    def should_replace_function(self, func_name, parms):
        return True

    def replace_function(self, func_name, parms):
        if(func_name == "run_retrieval"):
            return self.run_retrieval(**parms)
        elif(func_name == "run_forward_model"):
            return self.run_forward_model(**parms)

    def run_retrieval_zero_iterations(self, i_stateInfo, i_tableStruct,
                      i_windows, i_retrievalInfo, i_radianceInfo, 
                      i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2, 
                      mytimingFlag, writeoutputFlag, rf_uip):
        '''run_retrieval when maxIter is 0, pulled out just to simplify
        the code'''
        if(isinstance(i_retrievalInfo, RetrievalInfo)):
            jacobian_speciesNames = i_retrievalInfo.species_names
            jacobian_speciesList = i_retrievalInfo.species_list_fm
        else:
            jacobian_speciesNames = i_retrievalInfo.species[0:i_retrievalInfo.n_species]
            jacobian_speciesList = i_retrievalInfo.speciesListFM[0:i_retrievalInfo.n_totalParametersFM]
        (uip, radianceOut, jacobianOut) = self.run_forward_model(
            i_tableStruct, i_stateInfo, i_windows, i_retrievalInfo, 
            jacobian_speciesNames, jacobian_speciesList, 
            i_radianceInfo, 
            i_airs, i_cris, i_tes, i_omi, i_tropomi, i_oco2,
            mytimingFlag, 
            writeoutputFlag)
        if jacobian_speciesList[0] != '':
            if(isinstance(i_retrievalInfo, RetrievalInfo)):
                xret = i_retrievalInfo.apriori
            else:
                xret = i_retrievalInfo.constraintVector[0:i_retrievalInfo.n_totalParameters]
        else:
            jacobianOut = 0
            xret = 0
        o_retrievalResults = {
            'bestIteration': 0,                                  
            'xretIterations': xret,                               
            'num_iterations': 0,                                  
            'residualRMS': np.asarray([0]),                    
            'stopCode': -1,                                 
            'stopCriteria': np.zeros(shape=(1, 3), dtype=np.int),
            'resdiag': np.zeros(shape=(1, 5), dtype=np.int),
            'xret': xret,
            'radiance': radianceOut,
            'jacobian': jacobianOut,
            'delta': 0,
            'rho': 0,
            'lambda': 0            
        }
        o_uip = rf_uip.uip
        rayInfo = None
        ret_info = None
        windowsF = copy.deepcopy(i_windows)
        success_flag = 1 
        return (o_retrievalResults, o_uip, rayInfo, ret_info, windowsF, success_flag)
        

    def run_retrieval(self, i_stateInfo, i_tableStruct, i_windows,     
                      i_retrievalInfo, i_radianceInfo, 
                      i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2, 
                      mytimingFlag=False, writeoutputFlag=False):
        '''run_retrieval

        Note despite the name i_radianceInfo get updated with any radiance
        changes from tropomi/omi.'''
        if self.save_debug_data:
            # The magic incantation below grabs the parameters passed
            # to this func
            params=sys._getframe(0).f_locals
            # Don't include self in this
            del params['self']
            sve = MusesRetrievalStep(params=params)
            sve.capture_directory.save_directory(".", vlidort_input=None)
            pickle.dump(sve,
                  open(f"run_retrieval_step_{i_tableStruct['step']}.pkl", "wb"))

        rf_uip = RefractorUip.create_uip(i_stateInfo, i_tableStruct, i_windows,
                                         i_retrievalInfo, i_airs, i_tes,
                                         i_cris, i_omi, i_tropomi, i_oco2)
        maxIter = int(mpy.table_get_entry(i_tableStruct, i_tableStruct["step"], "maxNumIterations"))
        if maxIter == 0:
            return self.run_retrieval_zero_iterations(
                i_stateInfo, i_tableStruct, i_windows,i_retrievalInfo,
                i_radianceInfo, i_airs, i_tes, i_cris, i_omi, i_tropomi,
                i_oco2, mytimingFlag, writeoutputFlag, rf_uip)
        
        
        # Observation data
        tweaked_array_radiance = i_radianceInfo["radiance"]
        tweaked_array_nesr = i_radianceInfo["NESR"]
        if len(tweaked_array_radiance.shape) == 1:
            tweaked_array_radiance = np.reshape(tweaked_array_radiance, (1, tweaked_array_radiance.shape[0]))
        if len(tweaked_array_nesr.shape) == 1:
            tweaked_array_nesr = np.reshape(tweaked_array_nesr, (1, tweaked_array_nesr.shape[0]))

        obs_rad = mpy.glom(tweaked_array_radiance, 0, 1)
        meas_err = mpy.glom(tweaked_array_nesr, 0, 1)

        # apriori and sqrt_constraint
        if(isinstance(i_retrievalInfo, RetrievalInfo)):
            constraint = i_retrievalInfo.apriori_cov
            xa = i_retrievalInfo.apriori
        else:
            constraint = i_retrievalInfo.Constraint[0:i_retrievalInfo.n_totalParameters, 0:i_retrievalInfo.n_totalParameters]
            xa = i_retrievalInfo.constraintVector[0:i_retrievalInfo.n_totalParameters]
        if constraint.size > 1:
            sqrt_constraint = (mpy.sqrt_matrix(constraint)).transpose()
        else:
            sqrt_constraint = np.sqrt(constraint)

        # ret_info structue
        ret_info = { 
            'obs_rad': obs_rad,             
            'meas_err': meas_err,            
            'CostFunction': 'MAX_APOSTERIORI',   
            'basis_matrix': rf_uip.basis_matrix,   
            'sqrt_constraint': sqrt_constraint,  
            'const_vec': xa,
            'minimumList':i_retrievalInfo.minimumList[0:i_retrievalInfo.n_totalParameters],
            'maximumList':i_retrievalInfo.maximumList[0:i_retrievalInfo.n_totalParameters],
            'maximumChangeList':i_retrievalInfo.maximumChangeList[0:i_retrievalInfo.n_totalParameters]
        }

        # Various thresholds from the input table
        ConvTolerance_CostThresh = np.float32(mpy.table_get_pref(i_tableStruct, "ConvTolerance_CostThresh"))
        ConvTolerance_pThresh = np.float32(mpy.table_get_pref(i_tableStruct, "ConvTolerance_pThresh"))     
        ConvTolerance_JacThresh = np.float32(mpy.table_get_pref(i_tableStruct, "ConvTolerance_JacThresh"))
        Chi2Tolerance = 2.0 / meas_err.size # theoretical value for tolerance
        retrievalType = mpy.table_get_entry(i_tableStruct, i_tableStruct["step"], "retrievalType").lower()
        if retrievalType == "bt_ig_refine":
            ConvTolerance_CostThresh = 0.00001
            ConvTolerance_pThresh = 0.00001
            ConvTolerance_JacThresh = 0.00001
            Chi2Tolerance = 0.00001
        ConvTolerance = [ConvTolerance_CostThresh, ConvTolerance_pThresh, ConvTolerance_JacThresh]
        delta_str = mpy.table_get_pref(i_tableStruct, 'LMDelta') # 100 // original LM step size
        delta_value = int(delta_str.split()[0])  # We only need the first token sinc
            
        # Create a cost function, and use to implement residual_fm_jacobian
        # when we call levmar_nllsq_elanor
        cfunc = CostFunction(*self.fm_obs_creator.fm_and_obs(rf_uip, ret_info,
                                                             **self.kwargs))
        
        outputDir = i_tableStruct['dirStep']
        dir_tokens = outputDir.split("/")  # e.g. ..//00_AIRS_LAND-20130629_189_000_09/Step01_TATM, H2O, HDO, N2O, CH4, TSUR, CLOUDEXT, EMIS/
        if dir_tokens[-1] == '':
            stepName = dir_tokens[-2]
        else:
            stepName = dir_tokens[-1]
        logfile = os.path.join(outputDir, f'Logfile-{stepName}.log')
        
        # TODO Make this look like a ReFRACtor solver
        with register_replacement_function_in_block("residual_fm_jacobian",
                                                    cfunc):
            (xret, diag_lambda_rho_delta, stopcrit, resdiag, 
             x_iter, res_iter, radiance_fm, radiance_iter, jacobian_fm, 
             iterNum, stopCode, success_flag) =  mpy.levmar_nllsq_elanor(  
                    rf_uip.current_state_x, 
                    i_tableStruct["step"], 
                    rf_uip.uip, 
                    ret_info, 
                    maxIter, 
                    verbose=False, 
                    delta_value=delta_value, 
                    ConvTolerance=ConvTolerance,   
                    Chi2Tolerance=Chi2Tolerance,
                    logWrite=writeoutputFlag,
                    logFile=logfile
                    )

        # Update radiance data, based on what we got from the forward
        # model. This is really pretty sneaky, and a bad design - the "i_"
        # in the name seems to indicate this is input only. But this
        # matches what muses-py does.
        if(maxIter > 0 and len(ret_info) > 0):
            i_radianceInfo['radiance'][:] = ret_info['obs_rad'][:]
            i_radianceInfo['NESR'][:] = ret_info['meas_err'][:]
        
        # Find iteration used, only keep the best iteration
        rms = np.array([np.sqrt(np.sum(res_iter[i,:]*res_iter[i,:])/
                       res_iter.shape[1]) for i in range(iterNum+1)])
        bestIter = np.argmin(rms)
        residualRMS = rms
        
        # Take results and put into the expected output structures.
        
        radianceOut2 = copy.deepcopy(i_radianceInfo)
        radianceOut2['NESR'][:] = 0
        radianceOut2['radiance'][:] = radiance_fm

        detectors = [0]
        if(isinstance(i_retrievalInfo, RetrievalInfo)):
            speciesFM = i_retrievalInfo.species_list_fm
        else:
            speciesFM = i_retrievalInfo.speciesListFM[0:i_retrievalInfo.n_totalParametersFM]
        jacobianOut2 = mpy.jacobian_data(jacobian_fm, detectors,
                                         i_radianceInfo['frequency'], speciesFM)
        radianceOutIter = radiance_iter[:,np.newaxis,:]
        
        o_retrievalResults = {
            'bestIteration'  : int(bestIter), 
            'num_iterations' : iterNum, 
            'stopCode'       : stopCode, 
            'xret'           : xret, 
            'xretFM': rf_uip.current_state_x_fm,
            'radiance'       : radianceOut2, 
            'jacobian'       : jacobianOut2, 
            'radianceIterations': radianceOutIter, 
            'xretIterations' : x_iter, 
            'stopCriteria'   : np.copy(stopcrit), 
            'resdiag'        : np.copy(resdiag), 
            'residualRMS'    : residualRMS,
            'delta': diag_lambda_rho_delta[:, 2],
            'rho': diag_lambda_rho_delta[:, 1],
            'lambda': diag_lambda_rho_delta[:, 0],        
        }
        rayInfo = None
        for k in ("AIRS", "OMI", "TES", "CRIS", "TROPOMI", "OCO2"):
            if(f"uip_{k}" in rf_uip.uip):
                rayInfo = rf_uip.ray_info(k, set_pointing_angle_zero=False)
        windowsF = rf_uip.uip["microwindows_all"]
        # Note not all of these return elements are actually used
        # in script_retrieval_ms. rayInfo, ret_info are just ignored.
        # The other elements seem to be used to write out data
        return (o_retrievalResults, rf_uip.uip, rayInfo, ret_info, windowsF, success_flag)

    def run_forward_model(self, i_table, i_stateInfo, i_windows,
                          i_retrievalInfo, jacobian_speciesIn,
                          jacobian_speciesListIn,
                          uip, airs, cris, tes, omi, tropomi, oco2,
                          mytiming, writeOutputFlag=False, trueFlag=False,
                          RJFlag=False, rayTracingFlag=False):
        if self.save_debug_data:
            # The magic incantation below grabs the parameters passed
            # to this func
            params=sys._getframe(0).f_locals
            # Don't include self in this
            del params['self']
            sve = MusesForwardModelStep(params=params)
            sve.capture_directory.save_directory(".", vlidort_input=None)
            pickle.dump(sve,
                  open(f"run_forward_model_step_{i_table['step']}.pkl", "wb"))
        rf_uip = RefractorUip.create_uip(i_stateInfo, i_table, i_windows,
                                         i_retrievalInfo, airs, tes, cris,
                                         omi, tropomi, oco2,
                                         jacobian_speciesIn=jacobian_speciesIn)
        # As a convenience, we use the CostFunction to stitch stuff together.
        # We don't have the Observation for this, but we don't actually use
        # it in fm_wrapper. So we just fake it so we have the proper fields
        # for CostFunction.
        cfunc = CostFunction(*self.fm_obs_creator.fm_and_fake_obs(rf_uip,
                             **self.kwargs, use_full_state_vector=True,
                             include_bad_sample=True))
        (o_radiance, jac_fm, _, _, _) = cfunc.fm_wrapper(rf_uip.uip, None, {})
        o_jacobian = mpy.jacobian_data(jac_fm, o_radiance['detectors'],
                                       o_radiance['frequency'],
                                       rf_uip.uip['speciesListFM'])
        return (rf_uip.uip, o_radiance, o_jacobian)

__all__ = ["RefractorMusesIntegration",]
    
