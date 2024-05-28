from .refractor_uip import RefractorUip
from .cost_function import CostFunction
from .uip_updater import (StateVectorUpdateUip, MaxAPosterioriSqrtConstraintUpdateUip)
from .current_state import CurrentState, CurrentStateUip, CurrentStateDict
from .forward_model_handle import ForwardModelHandleSet
from .observation_handle import ObservationHandleSet
import refractor.framework as rf
import abc
import copy
import numpy as np
import refractor.muses.muses_py as mpy
import logging
import os
import pickle
from typing import Optional

logger = logging.getLogger("py-retrieve")

class CostFunctionCreator:
    '''This creates the set of ForwardModel and Observation and then uses those to
    create the CostFunction.

    The default ForwardModel that does the actual calculations wrap
    the existing py-retrieve forward model functions. But this object is
    designed to be modified by updating the forward_model_handle_set and
    observation_handle_set.

    The design of the PriorityHandleSet is a bit overkill for this
    class, we could probably get away with a simple dictionary mapping
    instrument name to functions that handle it. However the
    PriorityHandleSet was already available from another library with
    a much more complicated set of handlers where a dictionary isn't
    sufficient (see https://github.jpl.nasa.gov/Cartography/pynitf and
    https://cartography-jpl.github.io/pynitf/design.html#priority-handle-set).
    The added flexibility can be nice here, and since the code was already
    written we make use of it.

    In practice you create a simple class that just creates the
    ForwardModel and Observation, and register with the ForwardModelHandleSet and
    ObservationHandleSet.
    Take a look at the existing examples (e.g. the unit tests) - the design
    seems complicated but is actually pretty simple to use.
    '''
    def __init__(self, rs : 'Optional(RetrievalStategy)' = None):
        self.forward_model_handle_set = copy.deepcopy(ForwardModelHandleSet.default_handle_set())
        self.observation_handle_set = copy.deepcopy(ObservationHandleSet.default_handle_set())
        self.measurement_id = None

    def update_target(self, measurement_id : 'MeasurementId'):
        '''Set up for processing a target.

        Note we separate this out from the cost_function creator
        because we want to allow internal caching based on the
        sounding - e.g., read the input file only once. Ignoring
        performance, functionally this is just like an extra argument
        passed to cost_function.

        We take measure_id, which is a MeasurementId.
        '''
        self.measurement_id = measurement_id
        self.forward_model_handle_set.notify_update_target(self.measurement_id)
        self.observation_handle_set.notify_update_target(self.measurement_id)

    def _rf_uip_func_wrap(self):
        if(self._rf_uip is None):
            self._rf_uip = self._rf_uip_func()
        return self._rf_uip
        
    def cost_function(self,
                      instrument_name_list : "list[str]",
                      current_state : CurrentState,
                      spec_win : "list[rf.SpectralWindowRange]",
                      rf_uip_func,
                      include_bad_sample=False,
                      do_systematic=False,
                      jacobian_speciesIn=None,
                      obs_list=None,
                      fix_apriori_size=False,
                      **kwargs):
        '''Return cost function for the RetrievalStrategy.

        This takes the list of instrument names that make up this particular retrieval
        step, the current state information (see CurrentState class for description), and
        a list (one per instrument) of SpectralWindowRanges to apply the microwindows.
        Note that the SpectralWindows should *not* exclude bad pixels at this point. We
        generally need the Observation that we create as part of this function to determine
        bad pixels, so we add that to the passed in SpectralWindows.

        The muses-py versions of the ForwardModel depend on a RefractorUip structure. To
        support these older ForwardModel, and function is passed in that can be called to
        return the RefractorUip to use. We have this as a function, so we can avoid creating
        the RefractorUip if we don't need it. The RefractorUip shouldn't be used otherwise,
        it only has information that was needed by the old ForwardModel, and it is a
        convoluted way to pass the information around. Basically the uip + muses-py code
        is a old structural programming way to have a ForwardModel object.  We also don't
        just generate the UIP internally when needed because it depends on other muses-py
        structures that we don't have access to (by design, to reduce coupling in our code).
        If you know you aren't using a muses-py ForwardModel, it is fine to just pass this as
        None. It is possible that this argument will go away if we move far enough away from
        the old muses-py code - however for the near future we want to be able maintain the
        ability to run the old code to test against and diagnose any issues with ReFRACtor.

        It can also be useful in some testing scenarios to have the Observation created by
        some other method, so you can optionally pass in the obs_list to use in place of
        what the class would normally create. This isn't something you would normally use
        for "real", this is just to support testing.

        Similarly, you may sometimes not have an easy way to create the apriori and
        sqrt_constraint. In that case, you can pass in fix_apriori_size=True and we
        create dummy data of the right size rather than getting this from current_state.
        Again, this isn't something you would do for "real", this is more to support testing.
        '''
        # Keep track of this, in case we create one so we know to attach this to the state vector
        self._rf_uip = None
        self._rf_uip_func = rf_uip_func
        args = self._forward_model(
            instrument_name_list, current_state, spec_win,
            self._rf_uip_func_wrap, include_bad_sample=include_bad_sample,
            do_systematic=do_systematic, jacobian_speciesIn=jacobian_speciesIn,
            obs_list=obs_list, fix_apriori_size=fix_apriori_size,
            **kwargs)
        cfunc = CostFunction(*args)
        # If we have an UIP, then update this when the parameters get updated.
        # Note the rf_uip.basis_matrix is None handles the degenerate case of when we
        # have no parameters, for example for RetrievalStrategyStepBT. Any time we
        # have parameters, the basis_matrix shouldn't be None.
        if(self._rf_uip is not None and self._rf_uip.basis_matrix is not None):
            cfunc.max_a_posteriori.add_observer_and_keep_reference(MaxAPosterioriSqrtConstraintUpdateUip(self._rf_uip))
        # TODO, we want to get the parameters from CurrentState object, but we
        # don't have that in place yet. Fall back to the RefractorUip, which we should
        # remove in the future
        cfunc.parameters = self._rf_uip_func_wrap().current_state_x
        return cfunc

    def _forward_model(self,
                       instrument_name_list : "list[str]",
                       current_state : CurrentState,
                       spec_win_dict : "Optional(dict[str, rf.SpectralWindowRange])",
                       rf_uip_func,
                       include_bad_sample=False,
                       obs_list=None,
                       fix_apriori_size=False,
                       **kwargs):
        ret_info = { 
            'sqrt_constraint': current_state.sqrt_constraint,
            'const_vec': current_state.apriori,
        }
        self.obs_list = []
        fm_sv = rf.StateVector()
        if(obs_list is not None):
            self.obs_list = obs_list
        else:
            for instrument_name in instrument_name_list:
                obs = self.observation_handle_set.observation(
                    instrument_name, current_state, spec_win_dict[instrument_name], fm_sv,
                    **kwargs)
                # TODO Would probably be good to remove
                # include_bad_sample, it isn't clear that we ever want
                # to run the forward model for bad samples. But right
                # now the existing py-retrieve code requires this is a
                # few places.a
                if(include_bad_sample):
                    obs.spectral_window.include_bad_sample=include_bad_sample
                self.obs_list.append(obs)
                
        self.fm_list = []
        for i, instrument_name in enumerate(instrument_name_list):
            fm =  self.forward_model_handle_set.forward_model(
                instrument_name, current_state, self.obs_list[i].spectral_window, self.obs_list[i],
                fm_sv, rf_uip_func, **kwargs)
            self.fm_list.append(fm)
        fm_sv.observer_claimed_size = current_state.fm_state_vector_size
        # TODO Get a way to have the basis_matrix that doesn't requier a UIP
        rf_uip = rf_uip_func()
        bmatrix = rf_uip.basis_matrix
        if(not fix_apriori_size):
            # Normally, we get the apriori and constraint from our current state
            retrieval_sv_apriori =  current_state.apriori
            retrieval_sv_sqrt_constraint = current_state.sqrt_constraint.transpose()
        else:
            # This handles when we call with a RefractorUip but without a
            # ret_info, or otherwise don't have a apriori and sqrt_constraint. We
            # create dummy data of the right size.
            # This isn't something we encounter in our normal processing,
            # this is more to support old testing
            if(bmatrix is not None):
                retrieval_sv_apriori = np.zeros((bmatrix.shape[0],))
                retrieval_sv_sqrt_constraint=np.eye(bmatrix.shape[0])
            else:
                # No bmatrix then retrieval_sv and fm_sv are the same
                retrieval_sv_apriori = np.zeros((fm_sv.observer_claimed_size,))
                retrieval_sv_sqrt_constraint=np.eye(fm_sv.observer_claimed_size)

        return (instrument_name_list,
                self.fm_list, self.obs_list, fm_sv, retrieval_sv_apriori,
                retrieval_sv_sqrt_constraint, bmatrix)

    def cost_function_from_uip(self, rf_uip : RefractorUip,
                               obs_list,
                               ret_info : dict,
                               **kwargs):
        '''Create a cost function from a RefractorUip and a
        ret_info. Note that this is really just for backwards testing,
        we are trying to get away from using the RefractorUip because
        it ties stuff too tightly together.

        As a convenience, ret_info can be passed as None. It is useful
        to use our CostFunction to calculate the
        fm_wrapper/run_forward_model function because it has all the
        logic in place for stitching the different ForwardModel
        together. However we don't actually have all the data we need
        to calculate the Observation, nor do we have access to the
        apriori and sqrt_constraint.  However we don't actually need
        that to just calculate the ForwardModel data. So it can be
        useful to create all the pieces and just have dummy data for
        the missing parts.

        This is entirely a matter of convenience, we could instead just
        duplicate the stitching together part of our CostFunction and skip
        this. But for now this seems like the easiest thing thing to do. We
        can revisit this decision in the future if needed - it is never
        great to have fake data but in this case seemed the easiest path
        forward. Since this function is only used for backwards testing, the slightly
        klunky design doesn't seem like much of a problem.
        '''
        # Fake the input for the normal cost_function function
        def uip_func():
            return rf_uip
        cstate = CurrentStateUip(rf_uip)
        if(ret_info):
            fix_apriori_size=False
            cstate.sqrt_constraint = ret_info["sqrt_constraint"]
            cstate.apriori = ret_info["const_vec"]
        else:
            fix_apriori_size=True
            cstate.sqrt_constraint = np.eye(1)
            cstate.apriori = np.zeros((1,))
        return self.cost_function(rf_uip.instrument, cstate, None, uip_func,
                                  obs_list=obs_list,
                                  fix_apriori_size=fix_apriori_size, **kwargs)
    
__all__ = ["CostFunctionCreator"]
        
        
        
