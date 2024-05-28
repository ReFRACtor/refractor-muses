from .priority_handle_set import PriorityHandleSet
from .current_state import CurrentState, CurrentStateUip, CurrentStateDict
import refractor.framework as rf
import abc
import logging
import numpy as np

logger = logging.getLogger("py-retrieve")

def mpy_radiance_from_observation_list(obs_list : 'list(MusesObservation)',
                                       include_bad_sample=False, full_band=False):
    '''There are various places where py-retrieve needs the radiance data in
    a particular structure (e.g., its 'radianceStep' calculations.

    This takes a list of MusesObservation and creates this structure.

    Note that in some cases we want to be able to include bad samples or
    do a full band. Right now, we restrict ourselves to MusesObservation. We
    could probably extend this to a general Observation if that proves necessary -
    we just need a way to specify in we are including bad samples or full band.
    Right now we depend on the observation having the function 'modify_spectral_window',
    but if we can do the same functionality in another way.
    '''
    f = []
    d = []
    u = []
    fname = []
    fsize = []
    iname = []
    isize = []
    for obs in obs_list:
        with obs.modify_spectral_window(include_bad_sample=include_bad_sample,
                                        full_band=full_band):
            s = obs.radiance_all(True)
            f.append(s.spectral_domain.data)
            d.append(s.spectral_range.data)
            u.append(s.spectral_range.uncertainty)
            iname.append(obs.instrument_name)
            isize.append(s.spectral_range.data.shape[0])
            for fn, fs in obs.filter_data:
                fname.append(fn)
                fsize.append(fs)
                
        return {"radiance" : np.concatenate(d),
                "NESR" : np.concatenate(u),
                "frequency" : np.concatenate(f),
                "filterNames" : fname,
                "filterSizes" : fsize,
                "instrumentNames" : iname,
                "instrumentSizes" : isize,
                }        
    
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
                    current_state : "Optional(CurrentState)",
                    spec_win : rf.SpectralWindowRange,
                    fm_sv: "Optional(rf.StateVector)",
                    **kwargs):
        '''Return Observation if we can process the given instrument_name, or
        None if we can't. Add to fm_sv. If you don't need a StateVector (e.g., you are
        just accessing the data, not doing a retrieval), you can pass this as None and
        adding the state vector is skipped.

        If you don't need fm_sv, you can also optional set current_state to None, in which
        case default values are used for any coefficients the Observation depends on.
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
            obs = self.observation(inst, cs, spec_win_dict[inst], None)
            s = obs.radiance_all_extended(full_band=True)
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
                    current_state : "Optional(CurrentState)",
                    spec_win : rf.SpectralWindowRange,
                    fm_sv: "Optional(rf.StateVector)",
                    **kwargs):
        '''Create an Observation for the given instrument.'''
        return self.handle(instrument_name, current_state, spec_win, fm_sv, **kwargs)
    
    def handle_h(self, h : ObservationHandle, instrument_name : str,
                 current_state : "Optional(CurrentState)",
                 spec_win : rf.SpectralWindowRange,
                 fm_sv: "Optional(rf.StateVector)",
                 **kwargs):
        '''Process a registered function'''
        obs = h.observation(instrument_name, current_state, spec_win, fm_sv,
                            **kwargs)
        if(obs is None):
            return (False, None)
        return (True, obs)
                 
__all__ = ["ObservationHandleSet", "ObservationHandle", "mpy_radiance_from_observation_list"]
