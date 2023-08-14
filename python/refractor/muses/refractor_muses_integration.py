# This will likely migrate to refractor.muses, but it will start here as
# we initially focus on tropomi.

import refractor.muses.muses_py as mpy
import refractor.framework as rf
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .refractor_uip import RefractorUip
from .fm_obs_creator import FmObsCreator
from .cost_function import CostFunction
import numpy as np

class RefractorMusesIntegration(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''This handles the Refractor/Muses integration.

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
    
    def __init__(self, **kwargs):
        '''This interface probably needs work. But for now we take the keywords associated
        with RefractorObjectCreator. What we will probably want is someway of registering
        how to handle each instrument type, but for a start do this'''
        self.fm_obs_creator = FmObsCreator()
        self.instrument_handle_set = self.fm_obs_creator.instrument_handle_set
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
            return self.run_forward_model(parms)

    def run_retrieval(self, i_stateInfo, i_tableStruct, i_windows,     
                      i_retrievalInfo, i_radianceInfo, 
                      i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2, 
                      mytimingFlag=False, writeoutputFlag=False):
        rf_uip = RefractorUip.create_uip(i_stateInfo, i_tableStruct, i_windows,
                                         i_retrievalInfo, i_airs, i_tes,
                                         i_cris, i_omi, i_tropomi, i_oco2)
        
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
        maxIter = int(mpy.table_get_entry(i_tableStruct, i_tableStruct["step"], "maxNumIterations"))
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
                    Chi2Tolerance=Chi2Tolerance
                    )

        # Take results and put into the expected output structures.
        o_retrievalResults = None
        rayInfo = None
        windowsF = None
        return (o_retrievalResults, rf_uip.uip, rayInfo, ret_info, windowsF, success_flag)

    def run_forward_model(self, i_table, i_stateInfo, i_windows,
                          i_retrievalInfo, jacobian_speciesIn,
                          jacobian_speciesListIn,
                          uip, airs, cris, tes, omi, tropomi, oco2,
                          mytiming, writeOutputFlag=False, trueFlag=False,
                          RJFlag=False, rayTracingFlag=False):
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
    
