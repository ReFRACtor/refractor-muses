# This will likely migrate to refractor.muses, but it will start here as
# we initially focus on tropomi.

import refractor.muses.muses_py as mpy
import refractor.framework as rf
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .refractor_uip import RefractorUip
from .cost_func_creator import CostFuncCreator
from .muses_residual_fm_jacobian import MusesResidualFmJacobian
import numpy as np

class RefractorResidualFmJacobian(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''This replaces residual_fm_jacobian. This is pretty much the ReFRACtor cost function,
    plus some extra stuff calculated.

    Note because I keep needing to look this up, the call tree for a
    py-retrieve is:

      cli - top level entry point
      script_retrieval_ms- Handles all the strategy steps
      run_retrieval - Solves a single step of the strategy
      levmar_nllsq_elanor - Solver
    ->  residual_fm_jacobian - cost function
      fm_wrapper - Forward model. Note this handles combining instruments
                   (e.g., AIRS+OMI)
      omi_fm (for OMI) Forward model
      rtf_omi - lower level of omi forward model

    We implement residual_fm_jacobian here as a set of ForwardModel and Observation
    objects in a NLLSProblem. We wrap up the existing py-retrieve code into a
    ForwardModel and Observation so we can put this into the same framework.
    '''
    
    def __init__(self, **kwargs):
        '''This interface probably needs work. But for now we take the keywords associated
        with RefractorObjectCreator. What we will probably want is someway of registering
        how to handle each instrument type, but for a start do this'''
        self.cost_func_creator = CostFuncCreator(**kwargs)
        self.instrument_handle_set = self.cost_func_creator.instrument_handle_set

    def register_with_muses_py(self):
        '''Register this object and the helper objects with muses-py,
        to replace a call to omi_fm.
        '''
        mpy.register_replacement_function("residual_fm_jacobian", self)
        
    def should_replace_function(self, func_name, parms):
        return True

    def replace_function(self, func_name, parms):
        return self.residual_fm_jacobian(**parms)
    
    def residual_fm_jacobian(self, uip, ret_info, retrieval_vec, iterNum,
                             oco_info = {}):
        # In addition to the returned items, the uip gets updated (and
        # returned). I think it is just the retrieval_vec that updates
        # the uip.
        #
        # In additon, ret_info has obs_rad and meas_err
        # updated for OMI and TROPOMI. This seems kind of bad to me,
        # but the values get used in run_retrieval of py-retrieval, so
        # we need to update this.
        uip.iteration = iterNum
        rf_uip = RefractorUip(uip, ret_info=ret_info,
                              retrieval_vec=retrieval_vec)
        if(not "cost_func" in rf_uip.refractor_cache):
            rf_uip.refractor_cache["cost_func"] = \
                self.cost_func_creator.create_cost_func(rf_uip)
                
        cfunc = rf_uip.refractor_cache["cost_func"]
        # Temp, we should get this into the state vector
        rf_uip.update_uip(retrieval_vec)

        cfunc.parameters = retrieval_vec
        ret_info["obs_rad"] = cfunc.max_a_posteriori.measurement
        ret_info["meas_err"] = np.sqrt(cfunc.max_a_posteriori.measurement_error_cov)
        residual = cfunc.residual
        jac_residual = cfunc.jacobian.transpose()
        radiance_fm = cfunc.max_a_posteriori.model
        # We calculate the jacobian on the retrieval grid, but
        # this function is expecting this on the forward model grid.
        # We don't actually have this available here, but calculate
        # something similar so basis_matrix * jacobian_fm_placholder = jac_ret
        jac_retrieval_grid = \
            cfunc.max_a_posteriori.model_measure_diff_jacobian.transpose()
        jac_fm_placeholder, _, _, _ = np.linalg.lstsq(ret_info["basis_matrix"],
                                                      jac_retrieval_grid)
        stop_flag = 0
        return (uip, residual, jac_residual, radiance_fm,
                jac_fm_placeholder, stop_flag)

    def run_pickle_file(self, pfile, path=".", change_to_dir = False,
                        osp_dir=None, gmao_dir=None):
        '''This is a convenience function that loads a MusesResidualFmJacobian
        pickle file and runs residual_fm_jacobian. Mostly useful for
        testing.'''
        self.muses_residual_fm_jac = MusesResidualFmJacobian.load_residual_fm_jacobian(pfile, path=path, change_to_dir=change_to_dir, osp_dir=osp_dir, gmao_dir=gmao_dir)
        with register_replacement_function_in_block("residual_fm_jacobian",
                                                    self):
            return self.muses_residual_fm_jac.residual_fm_jacobian()
        

__all__ = ["RefractorResidualFmJacobian",]
    
