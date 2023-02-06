from . import muses_py as mpy
from .replace_function_helper import suppress_replacement
import logging

logger = logging.getLogger('py-retrieve')

class RefractorRunRetrieval(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''This is a place holder for replacing run_retrieval in py-retrieve.
    This is pretty much the results of running a solver on a forward
    model based cost function. We need to handle the plumbing of mapping
    from ReFRACtor to what py-retrieve wants.

    This runs a single step of a strategy.

    This class just turns around and runs the existing retrieval code in
    py-retrieve. Derived classes might do something else. As is, this base
    class is useful for checking that we have the plumbing in place.

    Note because I keep needing to look this up, the call tree for a
    retrieval is:

      cli - top level entry point
      script_retrieval_ms- Handles all the strategy steps
    -> run_retrieval - Solves a single step of the strategy
      levmar_nllsq_elanor - Solver
      residual_fm_jacobian - cost function
      fm_wrapper - Forward model. Note this handles combining instruments
                   (e.g., AIRS+OMI)
      omi_fm (for OMI) Forward model
      rtf_omi - lower level of omi forward model
    '''
    def register_with_muses_py(self):
        '''Register this object with muses-py, to replace a call 
        to residual_fm_jacobian.
        '''
        mpy.register_replacement_function("run_retrieval", self)
    
    def should_replace_function(self, func_name, parms):
        return True

    def replace_function(self, func_name, parms):
        return self.run_retrieval(**parms)

    def run_retrieval(self, i_stateInfo, i_tableStruct, i_windows,     
                      i_retrievalInfo, i_radianceInfo, i_airs, i_tes,
                      i_cris, i_omi, i_tropomi, i_oco2,
                      mytimingFlag=False, writeoutputFlag=False):
        # Just turn around and call mpy
        logger.info("In RefractorRunRetrieval.run_retrieval")
        with suppress_replacement("run_retrieval"):
            (o_retrievalResults, o_uip, rayInfo, ret_info, windowsF,
             success_flag) = mpy.run_retrieval(i_stateInfo, i_tableStruct,
                      i_windows, i_retrievalInfo, i_radianceInfo, i_airs,
                      i_tes, i_cris, i_omi, i_tropomi, i_oco2,
                      mytimingFlag, writeoutputFlag)
        # Note that although rayInfo, ret_info are returned,
        # they aren't actually used. Also success_flag is only used if it
        # is 0, in which case the program is exited. o_retrievalResults is
        # the primary thing returned, windowsF get used if we are writing
        # debug information, so this should get populated. We'll go ahead
        # a set everything that we aren't actually needing to fill to None
        # to make it clear that we don't need to fill this in once we
        # have a ReFRACtor replacement for this.
        rayInfo = None
        ret_info = None
        if(success_flag == 0):
            raise RuntimeError("success_flag is 0")
        success_flag = 1
        return (o_retrievalResults, o_uip, rayInfo, ret_info, windowsF,
                success_flag)
    
__all__ = ["RefractorRunRetrieval",]
