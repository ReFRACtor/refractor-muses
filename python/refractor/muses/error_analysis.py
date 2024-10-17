from .state_info import StateInfo
import refractor.muses.muses_py as mpy
import copy
import numpy as np
from scipy.linalg import block_diag

class ErrorAnalysis:
    '''This just groups together some of the error analysis stuff together,
    to put this together. Nothing more than a shuffling around of stuff already
    in muses-py
    '''
    def __init__(self, current_strategy_step: 'CurrentStrategyStep',
                 swin_dict : 'dict(str, MusesSpectralWindow)',
                 state_info: StateInfo,
                 covariance_state_element_name : 'list(str)'):
        self.initialize_error_initial(current_strategy_step,
                                      swin_dict,
                                      state_info,
                                      covariance_state_element_name)
        self.error_current = copy.deepcopy(self.error_initial)
        # Code seems to assume these are object view.
        self.error_initial = mpy.ObjectView(self.error_initial)
        self.error_current = mpy.ObjectView(self.error_current)

    def initialize_error_initial(self,
                                 current_strategy_step: 'CurrentStrategyStep',
                                 swin_dict : 'dict(str, MusesSpectralWindow)',
                                 state_info: StateInfo,
                                 covariance_state_element_name : 'list(str)'):
        '''covariance_state_element_name should be the list of state
        elements we need covariance from. This is all the elements we
        will retrieve, plus any interferents that get added in. This
        list is unique elements, sorted by the order_species sorting'''
        smeta = state_info.sounding_metadata()
        selem_list = []
        # TODO
        # Note clear why, but we get slightly different results if we update
        # the original state_info. May want to track this
        # down, but as a work around we just copy this. This is just needed
        # to get the mapping type, I don't think anything else is needed. We
        # should be able to pull that out from the full initial guess update at
        # some point, so we don't need to do the full initial guess
        sinfo = copy.deepcopy(state_info)
        for sname in covariance_state_element_name:
            selem = sinfo.state_element(sname)
            selem.update_initial_guess(current_strategy_step, swin_dict)
            selem_list.append(selem)
            
        # Note the odd seeming "capitalize" here. This is because get_prior_error
        # uses the map type to look up files, and rather than "linear" or "log" it
        # needs "Linear" or "Log"

        pressure_list = []
        species_list = []
        map_list = []

        # Make block diagonal covariance.
        matrix_list = []
        # AT_LINE 25 get_prior_error.pro 
        for selem in selem_list:
            matrix, pressureSa = selem.sa_covariance()
            pressure_list.extend(pressureSa)
            species_list.extend([selem.name] * matrix.shape[0])
            matrix_list.append(matrix)
            map_list.extend([selem.mapType] * matrix.shape[0])
        
        initial = block_diag(*matrix_list)
        # Off diagonal blocks for covariance.
        for i, selem1 in enumerate(selem_list):
            for selem2 in selem_list[i+1:]:
                matrix = selem1.sa_cross_covariance(selem2)
                if matrix is not None:
                    initial[np.array(species_list) == selem1.name, :][:,np.array(species_list) == selem2.name] = matrix
                    initial[np.array(species_list) == selem2.name, :][:,np.array(species_list) == selem1.name] = np.transpose(matrix)
        self.error_initial = mpy.constraint_data(initial, pressure_list,
                                                 species_list, map_list)

        

    def error_analysis(self, radiance_step: dict,
                       retrieval_info : 'RetrievalInfo',
                       state_info : 'StateInfo',
                       retrieval_results : 'RetrievalResult'):

        '''Update results and error_current'''
        # Doesn't seem to be used for anything, but we need to pass in. I think
        # this might have been something that was used in the past?
        radiance_noise = {"radiance" : np.zeros_like(radiance_step["radiance"]) }
        (results, self.error_current) = mpy.error_analysis_wrapper(
            None,
            None,
            radiance_step,
            radiance_noise,
            retrieval_info.retrieval_info_obj,
            state_info.state_info_obj,
            self.error_initial,
            self.error_current,
            None,
            retrieval_results
            )
        return results

        
