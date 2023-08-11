# This will likely migrate to refractor.muses, but it will start here as
# we initially focus on tropomi.

import refractor.muses.muses_py as mpy
import refractor.framework as rf
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .refractor_uip import RefractorUip
from .fm_obs_creator import CostFuncCreator
from .muses_residual_fm_jacobian import MusesResidualFmJacobian
import numpy as np

class RefractorMusesIntegration(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''This handles the Refractor/Muses integration.

    We do this by replacing two top level functions, residual_fm_jacobian
    and run_forward_model. Both of these are changed to use our CostFunction,
    and applies the various plumbing to give Muses-py what it expects from
    these two functions.

    We would like to replace run_retrieval instead of residual_fm_jacobian,
    it is the more natural function. We'll do that at some point in the
    future.

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
    ->  residual_fm_jacobian - cost function
      fm_wrapper - Forward model. Note this handles combining instruments
                   (e.g., AIRS+OMI)
      omi_fm (for OMI) Forward model
      rtf_omi - lower level of omi forward model

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
        mpy.register_replacement_function("run_forward_model", self)
        
    def should_replace_function(self, func_name, parms):
        return True

    def replace_function(self, func_name, parms):
        if(func_name == "residual_fm_jacobian"):
            return self.residual_fm_jacobian(**parms)
        elif(func_name == "run_forward_model"):
            return self.run_forward_model(parms)
        elif(func_name == "fm_wrapper"):
            return self.fm_wrapper(**parms)

    def run_forward_model(self, parms):
        # We need to calculate the uip from all the parms. Rather than
        # duplicating what muses-py does, just call it back and intercept
        # the fm_wrapper it does after it has generated the uip
        with suppress_replacement("run_forward_model"):
            with register_replacement_function_in_block("fm_wrapper", self):
                return mpy.run_forward_model(**parms)

    def fm_wrapper(self, i_uip, i_windows, oco_info):
        # Some of the forward model wrappers might call muses-py code
        # that in turn calls fm_wrapper. We don't want to intercept
        # these lower level function calls
        with suppress_replacement("fm_wrapper"):
            rf_uip = RefractorUip(i_uip)
            if(not "cost_func" in rf_uip.refractor_cache):
                rf_uip.refractor_cache["cost_func"] = \
                    self.cost_func_creator.create_cost_func(rf_uip,
                                ret_info=None, use_full_state_vector=True)
                
            cfunc = rf_uip.refractor_cache["cost_func"]
            cfunc.parameters = rf_uip.uip["currentGuessListFM"]
            radiance_fm = cfunc.max_a_posteriori.model
            jac_fm = \
                cfunc.max_a_posteriori.model_measure_diff_jacobian.transpose()
            bad_flag = 0
            freq_fm = np.concatenate([fm.spectral_domain_all().data
                 for fm in cfunc.max_a_posteriori.forward_model])
            # This duplicates what mpy.fm_wrapper does. It looks like
            # a number of these are placeholders, but the struct returned
            # by mpy.radiance_data looks like something that is just dumped
            # to a file, so I guess the placeholders make sense in an output
            # file where we don't have these values.

            # Seems to by indexed by detector, of which we only have one
            # dummy one
            detectors=[-1]
            radiance_fm = np.array([radiance_fm])
            nesr_fm = np.zeros(radiance_fm.shape)
            # Oddly frequency isn't indexed by detectors
            # freq_fm = np.array([freq_fm])
            # Not sure what filters is, but fm_wrapper just supplies this
            # as a empty array
            filters = []
            instrument = ''
            o_radiance = mpy.radiance_data(radiance_fm,  nesr_fm, detectors,
                                           freq_fm, filters, instrument)
            # We can fill these in if needed, but run_forward_model doesn't
            # actually use these values so we don't bother.
            o_measured_radiance_omi = None
            o_measured_radiance_tropomi = None
            return (o_radiance, jac_fm, bad_flag,
                    o_measured_radiance_omi, o_measured_radiance_tropomi)
            
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
        rf_uip = RefractorUip(uip, basis_matrix=ret_info['basis_matrix'])
        if(not "cost_func" in rf_uip.refractor_cache):
            rf_uip.refractor_cache["cost_func"] = \
                self.cost_func_creator.create_cost_func(rf_uip,
                                                        ret_info=ret_info)
                
        cfunc = rf_uip.refractor_cache["cost_func"]
        # Temp, we should get this into the state vector
        rf_uip.update_uip(retrieval_vec)

        cfunc.parameters = retrieval_vec
        # obs_rad and meas_err includes bad samples, so we can't use
        # cfunc.max_a_posteriori.measurement here which filters out
        # bad samples. Instead we access the observation list we stashed 
        # when we created the cost function.
        d = []
        for obs in cfunc.obs_list:
            if(hasattr(obs, "radiance_all_with_bad_sample")):
                d.append(obs.radiance_all_with_bad_sample())
            else:
                d.append(obs.radiance_all(True).spectral_range.data)
        ret_info["obs_rad"] = np.concatenate(d)
        # Covariance for bad pixels get set to sqr(-999), so meas_err is
        # 999 rather than -999 here. Work around this by only updating the
        # good pixels.
        gpt = ret_info["meas_err"] >= 0
        ret_info["meas_err"][gpt] = np.sqrt(cfunc.max_a_posteriori.measurement_error_cov)
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
        

__all__ = ["RefractorMusesIntegration",]
    
