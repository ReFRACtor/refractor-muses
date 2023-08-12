import refractor.framework as rf
from . import muses_py as mpy
from typing import List
import numpy as np

class CostFunction(rf.NLLSMaxAPosteriori):
    '''This is the cost function we use to interface between ReFRACtor
    and muses-py. This is just a standard rf.NLLSMaxAPosteriori with
    some extra convenience functions.'''
    def __init__(self, fm_list: List[rf.ForwardModel],
                 obs_list: List[rf.Observation],
                 sv: rf.StateVector,
                 sv_apriori: np.array,
                 sv_sqrt_constraint: np.array):
        self.obs_list = obs_list
        # Conversion to the std::vector needed by C++ is pretty hinky,
        # and often results in core dumps. Explicitly create this, that
        # tend to work better.
        fm_vec = rf.Vector_ForwardModel()
        obs_vec = rf.Vector_Observation()
        for fm in fm_list:
            fm_vec.push_back(fm)
        for obs in obs_list:
            obs_vec.push_back(obs)
        mstand = rf.MaxAPosterioriSqrtConstraint(fm_vec, obs_vec, sv,
                                                 sv_apriori, sv_sqrt_constraint)
        super().__init__(mstand)

    def fm_wrapper(self, i_uip, i_windows, oco_info):
        '''This uses the CostFunction to calculate the same things that
        muses-py does with it fm_wrapper function. We provide the same
        interface here to 1) provide something that can be used as a
        replacement for mpy.fm_wrapper but possibly using ReFRACtor objects
        and 2) make a more direct comparison between ReFRACtor and muses-py
        (e.g., for testing)'''
        if(hasattr(i_uip, 'currentGuessListFM')):
            p = i_uip.currentGuessListFM
        else:
            p = i_uip["currentGuessListFM"]
        if(self.expected_parameter_size != len(p)):
            raise RuntimeError("We aren't expecting parameters the size of currentGuessListFM. Did you forget use_full_state_vector=True when creating the ForwardModels?")
        self.parameters = p
        radiance_fm = self.max_a_posteriori.model
        jac_fm = self.max_a_posteriori.model_measure_diff_jacobian.transpose()
        bad_flag = 0
        freq_fm = np.concatenate([fm.spectral_domain_all().data
                                  for fm in self.max_a_posteriori.forward_model])
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
        if(self.expected_parameter_size != len(retrieval_vec)):
            raise RuntimeError("We aren't expecting parameters the size of retrieval_vec. Did you forget use_full_state_vector=False when creating the ForwardModels?")
        self.parameters = retrieval_vec
        # obs_rad and meas_err includes bad samples, so we can't use
        # cfunc.max_a_posteriori.measurement here which filters out
        # bad samples. Instead we access the observation list we stashed 
        # when we created the cost function.
        d = []
        for obs in self.obs_list:
            if(hasattr(obs, "radiance_all_with_bad_sample")):
                d.append(obs.radiance_all_with_bad_sample())
            else:
                d.append(obs.radiance_all(True).spectral_range.data)
        ret_info["obs_rad"] = np.concatenate(d)
        # Covariance for bad pixels get set to sqr(-999), so meas_err is
        # 999 rather than -999 here. Work around this by only updating the
        # good pixels.
        gpt = ret_info["meas_err"] >= 0
        ret_info["meas_err"][gpt] = np.sqrt(self.max_a_posteriori.measurement_error_cov)
        residual = self.residual
        jac_residual = self.jacobian.transpose()
        radiance_fm = self.max_a_posteriori.model
        # TODO Rework this, we actually need the jacobian on the FM grid
        
        # We calculate the jacobian on the retrieval grid, but
        # this function is expecting this on the forward model grid.
        # We don't actually have this available here, but calculate
        # something similar so basis_matrix * jacobian_fm_placholder = jac_ret
        jac_retrieval_grid = \
            self.max_a_posteriori.model_measure_diff_jacobian.transpose()
        jac_fm_placeholder, _, _, _ = np.linalg.lstsq(ret_info["basis_matrix"],
                                                      jac_retrieval_grid)
        stop_flag = 0
        return (uip, residual, jac_residual, radiance_fm,
                jac_fm_placeholder, stop_flag)
        
        
