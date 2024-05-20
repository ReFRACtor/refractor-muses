from .priority_handle_set import PriorityHandleSet
import refractor.framework as rf
import abc
import logging

logger = logging.getLogger("py-retrieve")

class ForwardModelHandle(object, metaclass=abc.ABCMeta):
    '''Base class for ForwardModelHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.

    Note ForwardModelHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.

    notify_update_target will also be called before the first time the
    objects are created - basically it makes sense to separate the
    arguments for notify_update_target and forward_model because they
    have different scopes (notify_update_target for the full
    retrieval, forward_model for a retrieval step).

    However, when forward_model is called a "newish" object should be
    created. Specifically we want to be able to attach each object to
    a separate StateVector and have different SpectralWindowRange set
    - we want to be able to have more than one CostFunction active at
    one time and we don't want updates in one CostFunction to affect
    the others. So this can be thought of as a shallow copy, if that
    make sense for the object. Things that don't depend on the
    StateVector can be shared (e.g., data read from a file), but state
    related parts should be independent.
    '''

    def notify_update_target(self, measurement_id : 'MeasurementId'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass
        
    @abc.abstractmethod
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      spec_win : rf.SpectralWindowRange,
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      **kwargs):
        '''Return ForwardModel if we can process the given
        instrument_name, or None if we can't.

        The forward model state vector is passed in, in case we want to
        attach anything to it as an observer (so object state gets updated as
        we update for forward model state vector).

        Because we sometimes need the metadata we also pass in the MusesObservation
        that goes with the given instrument name.

        The wrapped py-retrieve ForwardModel need the UIP structure. We don't normally
        create this, but pass in a function that can be used to get the UIP if needed.
        This should only be used for py-retrieve code, the UIP doesn't have all the
        information in our CurrentState object - only what was in py-retrieve.
        '''
        raise NotImplementedError()
    
class ForwardModelHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and RefractorUip, and
    creates a ForwardModel and Observation for that instrument.

    Note ForwardModelHandle can assume that they are called for the
    same target, until notify_update_target is called. So if it makes
    sense, these objects can do internal caching for things that don't
    change when the target being retrieved is the same from one call
    to the next.

    notify_update_target will also be called before the first time the
    objects are created - basically it makes sense to separate the
    arguments for notify_update_target and forward_model because they
    have different scopes (notify_update_target for the full
    retrieval, forward_model for a retrieval step).

    However, when forward_model is called a "newish" object should be
    created. Specifically we want to be able to attach each object to
    a separate StateVector and have different SpectralWindowRange set
    - we want to be able to have more than one CostFunction active at
    one time and we don't want updates in one CostFunction to affect
    the others. So this can be thought of as a shallow copy, if that
    make sense for the object. Things that don't depend on the
    StateVector can be shared (e.g., data read from a file), but state
    related parts should be independent.

    '''
    def notify_update_target(self, measurement_id : 'MeasurementId'):
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(measurement_id)
        
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      spec_win : rf.SpectralWindowRange,
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      include_bad_sample=False,
                      **kwargs):
        '''Create a ForwardModel for the given instrument.
        
        The forward model state vector is passed in, in case we want to
        attach anything to it as an observer (so object state gets updated as
        we update for forward model state vector).

        Because we sometimes need the metadata we also pass in the MusesObservation
        that goes with the given instrument name.

        The wrapped py-retrieve ForwardModel need the UIP structure. We don't normally
        create this, but pass in a function that can be used to get the UIP if needed.
        This should only be used for py-retrieve code, the UIP doesn't have all the
        information in our CurrentState object - only what was in py-retrieve.
        '''
        
        return self.handle(instrument_name, current_state, spec_win, obs, fm_sv,
                           rf_uip_func, include_bad_sample=include_bad_sample, **kwargs)
    
    def handle_h(self, h : ForwardModelHandle, instrument_name : str,
                 current_state : 'CurrentState',
                 spec_win : rf.SpectralWindowRange,
                 obs : 'MusesObservation',
                 fm_sv: rf.StateVector,
                 rf_uip_func,
                 include_bad_sample=False,
                 **kwargs):
        '''Process a registered function'''
        fm = h.forward_model(instrument_name, current_state, spec_win, obs, fm_sv, rf_uip_func,
                             include_bad_sample=include_bad_sample, **kwargs)
        if(fm is None):
            return (False, None)
        return (True, fm)

__all__ = ["ForwardModelHandle", "ForwardModelHandleSet",]
