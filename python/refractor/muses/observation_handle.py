from .priority_handle_set import PriorityHandleSet
from .current_state import CurrentState, CurrentStateUip, CurrentStateDict
import refractor.framework as rf
import abc
import logging
import numpy as np

logger = logging.getLogger("py-retrieve")

class ObservationHandle(object, metaclass=abc.ABCMeta):
    '''Base class for ObservationHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.

    Note ObservationHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.

    notify_update_target will also be called before the first time the
    objects are created - basically it makes sense to separate the
    arguments for notify_update_target and observation because they
    have different scopes (notify_update_target for the full
    retrieval, observation for a retrieval step).

    However, when observation is called a "newish" object should be
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
    def observation(self, instrument_name : str,
                    current_state : CurrentState,
                    spec_win : rf.SpectralWindowRange,
                    fm_sv: rf.StateVector,
                    include_bad_sample=False,
                    **kwargs):
        '''Return Observation if we can process the given instrument_name, or
        None if we can't. Add and StateVectorHandle needed to the passed in set.
        '''
        raise NotImplementedError()

class ObservationHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and a RetrievalStategy, and
    creates an Observation for that instrument.

    Note ObservationHandle can assume that they are called for the
    same target, until notify_update_target is called. So if it makes
    sense, these objects can do internal caching for things that don't
    change when the target being retrieved is the same from one call
    to the next.

    notify_update_target will also be called before the first time the
    objects are created - basically it makes sense to separate the
    arguments for notify_update_target and observation because they
    have different scopes (notify_update_target for the full
    retrieval, observation for a retrieval step).

    However, when observation is called a "newish" object should be
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

    def mpy_radiance_full_band(self, current_state : 'Optional[CurrentState]',
                               strategy_table: 'StrategyTable'):
        '''There are a few places where the old py-retrieve 'radiance' structure
        is needed. This includes all the instruments in strategy_table.instrument_name_all,
        smooshed together into a dict structure, without bad samples or windowing applied.
        This function handles this. Note
        we have a chicken and the egg problem with one of the places where we need the
        radiance. The creation of the initial StateInfo needs the radiance data (it uses
        for example in the call to supplier_nh3_type_cris). So the current state is optional,
        if this is passed in as None we use initial values for the various state elements
        needed by the Observation classes.'''
        cs = current_state
        if(cs is None):
            cs = CurrentStateDict({ "TROPOMISOLARSHIFTBAND1" : 0,
                                    "TROPOMIRADIANCESHIFTBAND1" : 0,
                                    "TROPOMIRADSQUEEZEBAND1" : 0,
                                    "TROPOMISOLARSHIFTBAND2" : 0,
                                    "TROPOMIRADIANCESHIFTBAND2" : 0,
                                    "TROPOMIRADSQUEEZEBAND2" : 0,
                                    "TROPOMISOLARSHIFTBAND3" : 0,
                                    "TROPOMIRADIANCESHIFTBAND3" : 0,
                                    "TROPOMIRADSQUEEZEBAND3" : 0,                                
                                    "TROPOMISOLARSHIFTBAND4" : 0,
                                    "TROPOMIRADIANCESHIFTBAND4" : 0,
                                    "TROPOMIRADSQUEEZEBAND4" : 0,
                                    "TROPOMISOLARSHIFTBAND5" : 0,
                                    "TROPOMIRADIANCESHIFTBAND5" : 0,
                                    "TROPOMIRADSQUEEZEBAND5" : 0,
                                    "TROPOMISOLARSHIFTBAND6" : 0,
                                    "TROPOMIRADIANCESHIFTBAND6" : 0,
                                    "TROPOMIRADSQUEEZEBAND6" : 0,
                                    "TROPOMISOLARSHIFTBAND7" : 0,
                                    "TROPOMIRADIANCESHIFTBAND7" : 0,
                                    "TROPOMIRADSQUEEZEBAND7" : 0,
                                    "TROPOMISOLARSHIFTBAND8" : 0,
                                    "TROPOMIRADIANCESHIFTBAND8" : 0,
                                    "TROPOMIRADSQUEEZEBAND8" : 0,
                                    "OMINRADWAVUV1" : 0,
                                    "OMIODWAVUV1" : 0,
                                    "OMIODWAVSLOPEUV1" : 0,
                                    "OMINRADWAVUV2" : 0,
                                    "OMIODWAVUV2" : 0,
                                    "OMIODWAVSLOPEUV2" : 0,
                                   },{})
        spec_win_dict = strategy_table.spectral_window_all(all_step=True)
        res = { "instrumentNames" : [],
                "frequency" : [],
                "radiance" : [],
                "NESR" : [],
                "instrumentSizes" : [] }
        f = []
        r = []
        u = []
        for inst in strategy_table.instrument_name(all_step=True):
            sv = rf.StateVector()
            obs = self.observation(inst, cs, spec_win_dict[inst], sv)
            if(hasattr(obs, "radiance_all_with_bad_sample")):
                s = obs.radiance_all_with_bad_sample(full_band=True)
            else:
                # True skips the jacobian calculation, which we don't
                # need here
                s = obs.radiance_all(True)
            res["instrumentNames"].append(inst)
            f.append(s.spectral_domain.data)
            r.append(s.spectral_range.data)
            u.append(s.spectral_range.uncertainty)
            res["instrumentSizes"].append(s.spectral_domain.data.shape[0])
        res["frequency"] = np.concatenate(f)
        res["radiance"] = np.concatenate(r)
        res["NESR"] = np.concatenate(u)
        return res
        
    def observation(self, instrument_name : str,
                    current_state : CurrentState,
                    spec_win : rf.SpectralWindowRange,
                    fm_sv: rf.StateVector,
                    include_bad_sample=False,
                    **kwargs):
        '''Create an Observation for the given instrument.'''
        return self.handle(instrument_name, current_state, spec_win, fm_sv,
                           include_bad_sample=include_bad_sample, **kwargs)
    
    def handle_h(self, h : ObservationHandle, instrument_name : str,
                 current_state : CurrentState,
                 spec_win : rf.SpectralWindowRange,
                 fm_sv: rf.StateVector,
                 include_bad_sample=False,
                 **kwargs):
        '''Process a registered function'''
        obs = h.observation(instrument_name, current_state, spec_win, fm_sv,
                            include_bad_sample=include_bad_sample,
                            **kwargs)
        if(obs is None):
            return (False, None)
        return (True, obs)
                 
__all__ = ["ObservationHandleSet", "ObservationHandle"]
