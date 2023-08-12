from .refractor_uip import RefractorUip
from .priority_handle_set import PriorityHandleSet
import refractor.framework as rf
import abc
import copy
import numpy as np

class StateVectorHandle(object, metaclass=abc.ABCMeta):
    '''Base class for StateVectorHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.'''

    def add_sv_once(self, sv : rf.StateVector,
                    obj : rf.SubStateVectorObserver):
        '''Add the given object only once to the state vector.'''
        # If needed we can come up with more complicated logic, but
        # for now just check if the obj is already attached to a state
        # vector. Could get some odd error where obj is attached to
        # *a* state vector, but not sv - but we won't worry about that now,
        if(obj.state_vector_start_index < 0):
            sv.add_observer(obj)

    @abc.abstractmethod
    def add_sv(self, sv: rf.StateVector, species_name : str, 
               pstart : int, plen : int, **kwargs):
        '''Handle the given state vector species_name. Return True if
        we processed this, False otherwise.

        The StateVector sv is modified by adding any observers to it'''
        raise NotImplementedError()

class StateVectorHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and RefractorUip, and
    creates a FowardModel and Observation for that instrument.'''
    def create_state_vector(self, rf_uip : RefractorUip,
                            use_full_state_vector=False, **kwargs):
        '''Create the full StateVector for all the species in rf_uip.

        For the retrieval, we use the "Retrieval State Vector".
        However, for testing it can be useful to use the "Full State Vector".
        See "Tropospheric Emission Spectrometer: Retrieval Method and Error
        Analysis" (IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,
        VOL. 44, NO. 5, MAY 2006) section III.A.1 for a discussion of this.
        Lower level muses-py functions work with the "Full State Vector", so
        it is useful to have the option of supporting this. Set
        use_full_state_vector to True to use the full state vector.
        '''
        sv = rf.StateVector()
        for species_name in rf_uip.jacobian_all:
            pstart, plen = rf_uip.state_vector_species_index(species_name,
                          use_full_state_vector=use_full_state_vector)
            self.add_sv(sv, species_name, pstart, plen, **kwargs)
        return sv
    
    def add_sv(self, sv: rf.StateVector, species_name : str, 
               pstart : int, plen : int, **kwargs):
        '''Attach whatever we need to the state vector for the given
        species.

        The StateVector sv is modified by adding any observers to it'''
        sv.observer_claimed_size = pstart
        res = self.handle(sv, species_name, pstart, plen, **kwargs)
        sv.observer_claimed_size = pstart + plen
        return res
    
    def handle_h(self, h : StateVectorHandle, sv: rf.StateVector,
                 species_name : str, pstart : int, plen : int, **kwargs):
        '''Process a registered function'''
        handled = h.add_sv(sv, species_name, pstart, plen, **kwargs)
        return (handled, None)

class InstrumentHandle(object, metaclass=abc.ABCMeta):
    '''Base class for InstrumentHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.'''
    @abc.abstractmethod
    def fm_and_obs(instrument_name : str, rf_uip : RefractorUip,
                   svhandle: StateVectorHandleSet,
                   use_full_state_vector=False,
                   obs_rad=None, meas_err=None, **kwargs):
        '''Return ForwardModel and Observation if we can process the given
        instrument_name, or (None, None) if we can't. Add any StateVectorHandle
        to the passed in set.

        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''
        raise NotImplementedError()
    
class InstrumentHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and RefractorUip, and
    creates a FowardModel and Observation for that instrument.'''
    def fm_and_obs(self, instrument_name : str, rf_uip : RefractorUip,
                   svhandle : StateVectorHandleSet,
                   use_full_state_vector=False,
                   obs_rad=None, meas_err=None,
                   **kwargs):
        '''Create a ForwardModel and Observation for the given instrument.
        
        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''

        return self.handle(instrument_name, rf_uip, svhandle,
                           use_full_state_vector=use_full_state_vector,
                           obs_rad=obs_rad, meas_err=meas_err,
                           **kwargs)
    
    def handle_h(self, h : InstrumentHandle, instrument_name : str,
                 rf_uip : RefractorUip,
                 svhandle : StateVectorHandleSet,
                 use_full_state_vector=False,
                 obs_rad=None, meas_err=None,
                 **kwargs):
        '''Process a registered function'''
        fm, obs = h.fm_and_obs(instrument_name, rf_uip, svhandle,
                               use_full_state_vector=use_full_state_vector,
                               obs_rad=obs_rad,meas_err=meas_err,**kwargs)
        if(fm is None):
            return (False, None)
        return (True, (fm, obs))
                 
class FmObsCreator:
    '''This creates the set of ForwardModel and Observation that goes with
    a RefractorUip.

    The default ForwardModel that does the actual calculations wrap
    the existing py-retrieve forward model functions. But this object is
    designed to be modified by updating the instrument_handle_set and
    state_vector_handle_set.

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
    ForwardModel and Observation, and register with the StateVectorHandleSet.
    Take a look at the existing examples (e.g. the unit tests of
    RefractorResidualFmJacobian) - the design seems complicated but is
    actually pretty simple to use.
    '''
    def __init__(self):
        self.instrument_handle_set = copy.deepcopy(InstrumentHandleSet.default_handle_set())

    def fm_and_obs(self, rf_uip : RefractorUip,
                   ret_info : dict,
                   use_full_state_vector=False,
                   **kwargs):
        '''This returns a list of ForwardModel and Observation that goes
        with the supplied rf_uip. We also return a StateVector that has
        all the pieces of the ForwardModel and Observation objects
        attached. We also return the apriori for the StateVector and what
        muses-py calls the sqrt_constraint (note that despite the name,
        sqrt_constraint *isn't* actually the sqrt of the constraint matrix - 
        see the description of MaxAPosterioriSqrtConstraint in
        refractor.framework for explanation of this).
        '''
        fm_list = []
        obs_list = []
        state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        obs_rad = ret_info["obs_rad"]
        meas_err = ret_info["meas_err"]
        sv_apriori = ret_info["const_vec"]
        sv_sqrt_constraint = ret_info["sqrt_constraint"].transpose()
        for instrument_name in rf_uip.instrument:
            fm, obs =  self.instrument_handle_set.fm_and_obs(instrument_name,
                                  rf_uip, state_vector_handle_set,
                                  use_full_state_vector=use_full_state_vector,
                                  obs_rad=obs_rad, meas_err=meas_err,**kwargs)
            fm_list.append(fm)
            obs_list.append(obs)
        sv = state_vector_handle_set.create_state_vector(rf_uip,
                               use_full_state_vector=use_full_state_vector,
                               **kwargs)
        return (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint)
    
    def fm_and_fake_obs(self, rf_uip: RefractorUip,
                        use_full_state_vector=False,
                        **kwargs):
        '''It is useful to use our CostFunction to calculate the
        fm_wrapper/run_forward_model function because it has all the logic
        in place for stitching the different ForwardModel together. However
        we don't actually have all the data we need to calculate the
        Observation, nor do we have access to the apriori and sqrt_constraint.
        However we don't actually need that to just calculate the ForwardModel
        data. So it can be useful to create all the pieces and just have dummy
        data for the missing parts.

        This is entirely a matter of convenience, we could instead just
        duplicate the stitching together part of our CostFunction and skip
        this. But for now this seems like the easiest thing thing to do. We
        can revisit this decision in the future if needed - it is never
        great to have fake data but in this case seemed the easiest path
        forward.
        '''
        fake_ret_info = {}
        fake_ret_info["obs_rad"] = np.zeros((len(rf_uip.instrument_list),))
        fake_ret_info["meas_err"] = np.ones(fake_ret_info["obs_rad"].shape)
        fake_ret_info["const_vec"] = np.zeros((1,))
        fake_ret_info["sqrt_constraint"] = np.diag(np.ones((1,)))
        (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint) = \
            self.fm_and_obs(rf_uip, fake_ret_info,
                            use_full_state_vector=use_full_state_vector,
                            **kwargs)
        sv_apriori = np.zeros((sv.observer_claimed_size,))
        sv_sqrt_constraint=np.diag(np.ones((sv.observer_claimed_size,)))
        return (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint)
    
class CostFuncCreator:
    '''This object creates a CostFunc that can be used with py-retrieve.
    This includes handling joint retrievals with multiple instruments.

    The object created is a NLLSMaxAPosteriori that wraps a
    MaxAPosterioriSqrtConstraint (a slight variation on the
    MaxAPosterioriStandard that we typically use in ReFRACtor).

    The default ForwardModel that does the actual calculations wrap
    the existing py-retrieve forward model functions. But this object is
    designed to be modified by updating the instrument_handle_set and
    state_vector_handle_set.

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
    ForwardModel and Observation, and register with the StateVectorHandleSet.
    Take a look at the existing examples (e.g. the unit tests of
    RefractorResidualFmJacobian) - the design seems complicated but is
    actually pretty simple to use.
    '''
    def __init__(self):
        self.instrument_handle_set = copy.deepcopy(InstrumentHandleSet.default_handle_set())

    def create_cost_func(self, rf_uip : RefractorUip,
                         ret_info = None,
                         use_full_state_vector=False):
        fm_list = rf.Vector_ForwardModel()
        obs_list = rf.Vector_Observation()
        # Stash observation, so we have a copy that includes extra
        # python functions. This is just needed to get the measurements
        # with bad samples
        obs_python_list = []
        state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        if(ret_info is not None):
            obs_rad = ret_info["obs_rad"]
            meas_err = ret_info["meas_err"]
        else:
            # TODO Change logic here to generate 0 and 1's
            obs_rad = None
            meas_err = None
        for instrument_name in rf_uip.instrument:
            fm, obs =  self.instrument_handle_set.fm_and_obs(instrument_name,
                                  rf_uip, state_vector_handle_set,
                                  use_full_state_vector=use_full_state_vector,
                                  obs_rad=obs_rad, meas_err=meas_err)
            fm_list.push_back(fm)
            obs_list.push_back(obs)
            obs_python_list.append(obs)
        sv = state_vector_handle_set.create_state_vector(rf_uip,
                                use_full_state_vector=use_full_state_vector)
        # TODO Fix up this logic. Perhaps pass in a flag indicating that
        # we are faking certain parts for ease
        if(ret_info is not None):
            mstand = rf.MaxAPosterioriSqrtConstraint(fm_list, obs_list, sv,
               ret_info["const_vec"], ret_info["sqrt_constraint"].transpose())
        else:
            mstand = rf.MaxAPosterioriSqrtConstraint(fm_list, obs_list, sv,
                               np.zeros(sv.observer_claimed_size),
                               np.identity(sv.observer_claimed_size))
        mprob = rf.NLLSMaxAPosteriori(mstand)
        mprob.obs_list = obs_python_list
        return mprob

__all__ = ["CostFuncCreator",  "StateVectorHandle", "StateVectorHandleSet",
           "InstrumentHandle", "InstrumentHandleSet",
           "FmObsCreator"]
        
        
        
