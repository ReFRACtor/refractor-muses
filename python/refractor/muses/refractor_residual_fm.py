from . import muses_py as mpy
from .replace_function_helper import suppress_replacement
from .refractor_uip import RefractorUip
import logging
import numpy as np
import os
import pickle

logger = logging.getLogger('py-retrieve')

class RefractorResidualFmJac(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''Place holder for replacing residual_fm_jacobian. This is pretty
    much the ReFRACtor cost function, plus some extra stuff calculated.

    Right now this just gives a place for us to grab the output to
    look at differences between muses-py and ReFRACtor.

    Note because I keep needing to look this up, the call tree for a
    retrieval is:

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

    def register_with_muses_py(self):
        '''Register this object with muses-py, to replace a call 
        to residual_fm_jacobian.
        '''
        mpy.register_replacement_function("residual_fm_jacobian", self)

    def should_replace_function(self, func_name, parms):
        return True

    def replace_function(self, func_name, parms):
        return self.residual_fm_jacobian(**parms)

    def residual_fm_jacobian(self, uip, ret_info, retrieval_vec, iterNum):
        with suppress_replacement("residual_fm_jacobian"):
            (uip, o_residual, o_jacobian_ret, radiance_out, o_jacobianOut,
             o_stop_flag) = mpy.residual_fm_jacobian(uip, ret_info,
                                                     retrieval_vec, iterNum)
        if False:
            print('o_residual: ', o_residual.shape, "\n", o_residual)
            print('obs_rad: ', ret_info['obs_rad'])
            print('meas_err: ', ret_info['meas_err'])
            print('radiance_out: ', radiance_out)
            print('CostFunction: ', ret_info['CostFunction'])
            print('Basis matrix: ', ret_info['basis_matrix'])
        print('chisq: ', np.sum(np.square(o_residual)))
        return (uip, o_residual, o_jacobian_ret, radiance_out,
                o_jacobianOut, o_stop_flag)


class RefractorResidualFmJac2Call(RefractorResidualFmJac):
    '''Do two calls, one to refractor and one to muses-py so we can
    directly compare.  call_obj2 drives the actual retrieval.

    The objects passed in should have a register_with_muses_py class.
    For example, in the omi repository this could be a RefractorOmiFm and
    RefractorOmiFmMusesPy

    In addition to printing out log messages, we also write out a
    residual_fm_jacobian_%d.pkl file for each iteration, if we need to
    examine this output in detail.
    '''

    def __init__(self, call_obj1, call_obj2):
        self.call_obj1 = call_obj1
        self.call_obj2 = call_obj2
        self.iter_count = 0

    def residual_fm_jacobian(self, uip, ret_info, retrieval_vec, iterNum,
                             **kwargs):
        self.iter_count += 1
        self.uip_arr = [ ]
        o_residual_arr = [ ]
        o_jacobian_ret_arr = [ ]
        obs_rad_arr = []
        meas_err_arr = []
        radiance_out_arr = [ ]
        o_jacobian_out_arr = [ ]
        o_stop_flag_arr = [ ]
        uip_arr = []
        for i in range(2):
            if(i == 0):
                self.call_obj1.register_with_muses_py()
            else:
                self.call_obj2.register_with_muses_py()
            with suppress_replacement("residual_fm_jacobian"):
                (uip, o_residual, o_jacobian_ret, radiance_out,
                 o_jacobianOut, o_stop_flag) = \
                     mpy.residual_fm_jacobian(uip, ret_info,
                                              retrieval_vec, iterNum)
            uip_arr.append(uip)
            o_residual_arr.append(o_residual)
            o_jacobian_ret_arr.append(o_jacobian_ret)
            radiance_out_arr.append(radiance_out)
            o_jacobian_out_arr.append(o_jacobianOut)
            o_stop_flag_arr.append(o_stop_flag)
            if(i == 0):
                print("------------------- object 1 ---------------------")
            else:
                print("------------------- object 2 ---------------------")
            if False:
                print('o_residual: ', o_residual.shape, "\n", o_residual)
                print('obs_rad: ', ret_info['obs_rad'])
                print('meas_err: ', ret_info['meas_err'])
                print('radiance_out: ', radiance_out)
                print('CostFunction: ', ret_info['CostFunction'])
                print('Basis matrix: ', ret_info['basis_matrix'])
            print('chisq: ', np.sum(np.square(o_residual)))

        # Capture input, so we can run steps outside of a retrieval
        rf_uip = RefractorUip(uip_arr[1],
                              strategy_table=os.environ.get("MUSES_DEFAULT_RUN_DIR") + "/Table.asc")
        vlidort_input = None
        if("uip_OMI" in rf_uip.uip):
            vlidort_input = rf_uip.uip['uip_OMI']["vlidort_input"]
        if("uip_TROPOMI" in rf_uip.uip):
            vlidort_input = res.uip['uip_TROPOMI']["vlidort_input"]
            res.capture_directory.save_directory(os.path.dirname(strategy_table), vlidort_input)
        rf_uip.tar_directory()
        pickle.dump({ "rf_uip" : rf_uip,
                      "ret_info" : ret_info,
                      "retrieval_vec" : retrieval_vec,
                      "o_residual_arr" : o_residual_arr,
                      "o_jacobian_ret_arr" : o_jacobian_ret_arr,
                      "radiance_out_arr" : radiance_out_arr,
                      "o_jacobian_out_arr" : o_jacobian_out_arr,
                      "o_stop_flag_arr" : o_stop_flag_arr,
                     },
                    open("residual_fm_jacobian_%d.pkl" % self.iter_count,
                          "wb"))
        return (uip_arr[1], o_residual_arr[1], o_jacobian_ret_arr[1],
                radiance_out_arr[1],
                o_jacobian_out_arr[1], o_stop_flag_arr[1])


__all__ = ['RefractorResidualFmJac', 'RefractorResidualFmJac2Call']
