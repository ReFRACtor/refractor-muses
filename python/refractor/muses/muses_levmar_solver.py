from .cost_function import CostFunction
import pickle
import numpy as np
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
import refractor.muses.muses_py as mpy

class MusesLevmarSolver:
    '''This is a wrapper around levmar_nllsq_elanor that makes it look like
    a NLLSSolver. Right now we don't actually derive from that, we can perhaps
    put that in place if useful. But for now, just provide a "solve" function.
    '''
    def __init__(self, cfunc: CostFunction, 
                 max_iter: int, delta_value: float, conv_tolerance: float,
                 chi2_tolerance: float):
        self.cfunc = cfunc
        self.max_iter = max_iter
        self.delta_value = delta_value
        self.conv_tolerance = conv_tolerance
        self.chi2_tolerance = chi2_tolerance
        # Defaults, so if we skip solve we have what is needed for output
        self.success_flag = 1
        self.bestIter = 0
        self.residualRMS = np.asarray([0])
        self.diag_lambda_rho_delta = np.zeros((1,3))
        self.stopcrit = np.zeros(shape=(1, 3), dtype=int)
        self.resdiag = np.zeros(shape=(1, 5), dtype=int)
        self.radiance_iter = np.zeros((1,1))
        self.iterNum = 0
        self.stopCode = -1

    def retrieval_results(self):
        '''Return the retrieval results dict. Hopefully this can go away, this
        is just used in mpy.set_retrieval_results (another function we would like
        to remove). It would probably be better for things
        to get this directly from this solver and the cost function. But for
        now we have this.

        Note that this works even in solve() hasn't been called - this returns
        what is expected if max_iter is 0.'''
        gpt = self.cfunc.good_point()
        
        radiance_fm = np.full(gpt.shape, -999.0)
        radiance_fm[gpt] = self.cfunc.max_a_posteriori.model
        jac_fm_gpt = self.cfunc.max_a_posteriori.model_measure_diff_jacobian_fm.transpose()
        jacobian_fm = np.full((jac_fm_gpt.shape[0], gpt.shape[0]),-999.0)
        jacobian_fm[:, gpt] = jac_fm_gpt
        
        radianceOut2 = {'radiance' : radiance_fm }
        jacobianOut2 = {"jacobian_data" : jacobian_fm }
        # Oddly, set_retrieval_results expects a different shape for num_iterations
        # = 0. Probably should change the code there, but for now just work around
        # this
        if(self.iterNum == 0):
            radianceOut2['radiance'] = radiance_fm[np.newaxis, :]
       
        return {
            'bestIteration'  : int(self.bestIter), 
            'num_iterations' : self.iterNum, 
            'stopCode'       : self.stopCode, 
            'xret'           : self.cfunc.parameters,
            'xretFM'         : self.cfunc.parameters_fm(),
            'radiance'       : radianceOut2, 
            'jacobian'       : jacobianOut2, 
            'radianceIterations': self.radiance_iter[:,np.newaxis,:], 
            'xretIterations' : self.cfunc.parameters if self.iterNum == 0 else self.x_iter, 
            'stopCriteria'   : np.copy(self.stopcrit), 
            'resdiag'        : np.copy(self.resdiag), 
            'residualRMS'    : self.residualRMS,
            'delta': self.diag_lambda_rho_delta[:, 2],
            'rho': self.diag_lambda_rho_delta[:, 1],
            'lambda': self.diag_lambda_rho_delta[:, 0],        
        }
        

    def solve(self):
        with register_replacement_function_in_block("update_uip",
                                                    self.cfunc):
            with register_replacement_function_in_block("residual_fm_jacobian",
                                                        self.cfunc):
            # We want some of these to go away
                (xret, self.diag_lambda_rho_delta, self.stopcrit, self.resdiag, 
                 self.x_iter, res_iter, radiance_fm, self.radiance_iter,
                 jacobian_fm, self.iterNum, self.stopCode, self.success_flag) =  \
                     mpy.levmar_nllsq_elanor(  
                         self.cfunc.parameters, 
                         None, 
                         None, 
                         {},
                         self.max_iter, 
                         verbose=False, 
                         delta_value=self.delta_value, 
                         ConvTolerance=self.conv_tolerance,
                         Chi2Tolerance=self.chi2_tolerance
                     )
            # Since xret is the best iteration, which might not be the last,
            # set the cost function to this. Note the cost function does internal
            # caching, so if this is the last one then we don't recalculate
            # residual and jacobian.
            self.cfunc.parameters = xret
            # Find iteration used, only keep the best iteration
            rms = np.array([np.sqrt(np.sum(res_iter[i,:]*res_iter[i,:])/
                                    res_iter.shape[1]) for i in range(self.iterNum+1)])
            self.bestIter = np.argmin(rms)
            self.residualRMS = rms

        
        
                    

        
        
                 
