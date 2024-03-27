from .refractor_uip import RefractorUip
from .cost_function import CostFunction
from .priority_handle_set import PriorityHandleSet
from .uip_updater import (StateVectorUpdateUip, MaxAPosterioriSqrtConstraintUpdateUip)
from .current_state import CurrentState
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
    creates a ForwardModel and Observation for that instrument.'''
    def create_state_vector(self, rf_uip : RefractorUip, **kwargs):
        '''Create the full StateVector for all the species in rf_uip.
        '''
        sv = rf.StateVector()
        for species_name in rf_uip.jacobian_all:
            pstart, plen = rf_uip.state_vector_species_index(species_name)
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

class ForwardModelHandle(object, metaclass=abc.ABCMeta):
    '''Base class for ForwardModelHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.

    Note ForwardModelHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''

    def notify_update_target(self, measurement_id : dict, filter_list : dict):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass
        
    @abc.abstractmethod
    def forward_model(self, instrument_name : str, rf_uip : RefractorUip,
                   obs : 'MusesObservation',
                   svhandle: StateVectorHandleSet,
                   obs_rad=None, meas_err=None, **kwargs):
        '''Return ForwardModel if we can process the given
        instrument_name, or None if we can't. Add any StateVectorHandle
        to the passed in set.

        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''
        raise NotImplementedError()
    
class ForwardModelHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and RefractorUip, and
    creates a ForwardModel and Observation for that instrument.

    Note ForwardModelHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''
    def notify_update_target(self, measurement_id : dict, filter_list : dict):
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(measurement_id, filter_list)
        
    def forward_model(self, instrument_name : str, rf_uip : RefractorUip,
                   obs : 'MusesObservation',
                   svhandle : StateVectorHandleSet,
                   obs_rad=None, meas_err=None,
                   **kwargs):
        '''Create a ForwardModel for the given instrument.
        
        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''

        return self.handle(instrument_name, rf_uip, obs, svhandle,
                           obs_rad=obs_rad, meas_err=meas_err,
                           **kwargs)
    
    def handle_h(self, h : ForwardModelHandle, instrument_name : str,
                 rf_uip : RefractorUip,
                 obs : 'MusesObservation',
                 svhandle : StateVectorHandleSet,
                 obs_rad=None, meas_err=None,
                 **kwargs):
        '''Process a registered function'''
        fm = h.forward_model(instrument_name, rf_uip, obs, svhandle,
                             obs_rad=obs_rad,meas_err=meas_err,**kwargs)
        if(fm is None):
            return (False, None)
        return (True, fm)

class ObservationHandle(object, metaclass=abc.ABCMeta):
    '''Base class for ObservationHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.

    Note ObservationHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''
    def notify_update_target(self, measurement_id : dict, filter_list : dict):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def observation(self, instrument_name : str, rs: 'RetrievalStategy',
                    svhandle: 'StateVectorHandleSet', **kwargs):
        '''Return Observation if we can process the given instrument_name, or
        None if we can't. Add and StateVectorHandle needed to the passed in set.

        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''
        raise NotImplementedError()

class ObservationHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and a RetrievalStategy, and
    creates an Observation for that instrument.

    Note ObservationHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''
    def notify_update_target(self, measurement_id : dict, filter_list : dict):
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(measurement_id, filter_list)
        
    def observation(self, instrument_name : str, rs: 'RetrievalStategy',
                    svhandle : StateVectorHandleSet,
                    **kwargs):
        '''Create an Observation for the given instrument.
        
        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''
        return self.handle(instrument_name, rs, svhandle,
                           **kwargs)
    
    def handle_h(self, h : ObservationHandle, instrument_name : str,
                 rs: 'RetrievalStategy',
                 svhandle : StateVectorHandleSet,
                 **kwargs):
        '''Process a registered function'''
        obs = h.observation(instrument_name, rs, svhandle,**kwargs)
        if(obs is None):
            return (False, None)
        return (True, obs)
                 
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
        self.filter_list = None
        self.rs = None

    def update_target(self, measurement_id : dict, filter_list : dict, rs : 'RetrievalStategy'):
        '''Set up for processing a target.

        Note we separate this out from the cost_function creator
        because we want to allow internal caching based on the
        sounding - e.g., read the input file only once. Ignoring
        performance, functionally this is just like two extra arguments
        passed to cost_function.

        We take measure_id, which should act like a dict to give us
        the various sounding id information we need. The canonical
        version of this is the "preferences" part of a MeasurementID
        file, but we purposely just take a generic dict like structure
        so you can run this with requiring a MeasurementID file,
        e.g. for running stand alone tests.

        Similarly, we take filter_list, which acts like dict. The keys
        are the instruments we have for this target. These should be in
        the order desired for the cost function residual (note as of python
        3.6 the standard dict maintains insertion order - so the ordering
        should be automatic). This then returns a list of filters for that
        instrument.

        The filter_list should be the *complete* set of filters
        needed, so the union of the filter for all the retrieval
        steps. The canonical way to get that is the
        StrategyTable.filter_list_all(), but we purposely just take a
        generic dict like structure so you can run this without requiring
        a StrategyTable, e.g. for running stand alone tests.
        '''
        self.measurement_id = measurement_id
        self.filter_list = filter_list
        # TODO We want this to go away. But short term leave this here so we can
        # get the code cleaned up in pieces.
        self.rs = rs
        
        # Hopefully these will go away
        self.o_airs = None
        self.o_cris = None
        self.o_omi = None
        self.o_tropomi = None
        self.o_tes = None
        self.o_oco2 = None
        self._created_o = False
        self._radiance = None
        self.forward_model_handle_set.notify_update_target(self.measurement_id, self.filter_list)
        self.observation_handle_set.notify_update_target(self.measurement_id, self.filter_list)

    def create_o_obs(self):
        if(self._created_o):
            return
        (self.o_airs, self.o_cris, self.o_omi, self.o_tropomi, self.o_tes, self.o_oco2,
         _) = mpy.script_retrieval_setup_ms(self.rs.strategy_table.strategy_table_dict, False)
        self._created_o = True
        
    def _read_rad(self, state_info, instrument_name_all):

        '''This is a placeholder, we want to get this stuff pushed down into
        the handle for each instrument, probably into the various Observable classes.
        But for now we have this centralized so we can call the existing muses-py code.
        '''
        if(self._radiance is not None):
            return
        with self.rs.chdir_run_dir():
            self.create_o_obs()
            self._create_radiance(state_info, instrument_name_all)

    def _create_radiance(self, state_info, instrument_name_all):
        '''Read the radiance data. We can  perhaps move this into a Observation class
        by instrument.

        Note that this also creates the magic files Radiance_OMI_.pkl and
        Radiance_TROPOMI_.pkl. It would be nice if can rework that.
        '''
        logger.info(f"Instruments: {len(instrument_name_all)} {instrument_name_all}")
        obsrad = None
        for instrument_name in instrument_name_all:
            logger.info(f"Reading radiance: {instrument_name}")
            if instrument_name == 'OMI':
                result = mpy.get_omi_radiance(state_info.state_info_dict['current']['omi'], copy.deepcopy(self.o_omi))
                radiance = result['normalized_rad']
                nesr = result['nesr']
                my_filter = result['filter']
                frequency = result['wavelength']
                fname = f'{self.rs.run_dir}/Input/Radiance_OMI_.pkl'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                pickle.dump(self.o_omi, open(fname, "wb"))
            if instrument_name == 'TROPOMI':
                result = mpy.get_tropomi_radiance(state_info.state_info_dict['current']['tropomi'], copy.deepcopy(self.o_tropomi))

                radiance = result['normalized_rad']
                nesr = result['nesr']
                my_filter = result['filter']
                frequency = result['wavelength']
                fname = f'{self.rs.run_dir}/Input/Radiance_TROPOMI_.pkl'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                pickle.dump(self.o_tropomi, open(fname, "wb"))

            if instrument_name == 'AIRS':
                radiance = self.o_airs['radiance']['radiance']
                frequency = self.o_airs['radiance']['frequency']
                nesr = self.o_airs['radiance']['NESR']
                my_filter = mpy.radiance_get_filter_array(self.o_airs['radiance'])
            if instrument_name == 'CRIS':
                # The o_cris dictionary uses all uppercase keys.
                radiance = self.o_cris['radiance'.upper()]
                frequency = self.o_cris['frequency'.upper()]
                nesr = self.o_cris['nesr'.upper()]
                my_filter = mpy.radiance_get_filter_array(self.o_cris['radianceStruct'.upper()])
            if instrument_name == 'OCO2':
                radiance = self.o_oco2['radianceStruct']['radiance']
                frequency = self.o_oco2['radianceStruct']['frequency']
                nesr = self.o_oco2['radianceStruct']['NESR']
                my_filter = mpy.radiance_get_filter_array(self.o_oco2['radianceStruct'])

            if instrument_name == 'TES':
                radiance = self.o_tes['radianceStruct']['radiance']
                frequency = self.o_tes['radianceStruct']['frequency']
                nesr = self.o_tes['radianceStruct']['NESR']
                my_filter = mpy.radiance_get_filter_array(self.o_tes['radianceStruct'])

            # Add the first radiance if this is the first time in the loop.
            if(obsrad is None):
                obsrad = mpy.radiance_data(radiance, nesr, [-1], frequency, my_filter, instrument_name, None)
            else:
                filtersIn = np.asarray(['' for ii in range(0, len(frequency))])
                obsrad = mpy.radiance_add_filter(obsrad, radiance, nesr, [-1], frequency, my_filter, instrument_name)
        self._radiance = obsrad

    def radiance(self, state_info, instrument_name_all):
        '''I'm not 100% sure if this can go way or not, but state_initial_update
        depends on the radiance. For now, allow access to this.

        We may want to pull the ForwardModel and Observation apart, it might make
        sense to have the instrument Observation for such things as the radiance
        used in creating our RetrievalState. For now, just allow access this so
        we can push this out of RetrievalStrategy even if we shuffle around where
        this comes from.'''
        self._read_rad(state_info, instrument_name_all)
        return self._radiance

    def _create_radiance_step(self):
        # Note, I think we might replace this just with our SpectralWindow stuff,
        # along with an Observation class
        self._read_rad(self.rs.state_info, self.rs.instrument_name_all)
        radianceStepIn = self._radiance
        radianceStepIn = mpy.radiance_set_windows(radianceStepIn, self.rs.windows)

        if np.all(np.isfinite(radianceStepIn['radiance'])) == False:
            raise RuntimeError('ERROR! radiance NOT FINITE!')

        if np.all(np.isfinite(radianceStepIn['NESR'])) == False:
            raise RuntimeError('ERROR! radiance error NOT FINITE!')
        return radianceStepIn

    def _rf_uip_func_wrap(self):
        if(self._rf_uip is None):
            self._rf_uip = self._rf_uip_func()
        return self._rf_uip
        

    def cost_function(self,
                      instrument_name_list : "list[str]",
                      current_state : 'CurrentState',
                      spec_win : "list[rf.SpectralWindowRange]",
                      rf_uip_func,
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

        This Also sets self.radiance_step_in
        - this is a bit awkward but I think we may replace radiance_steo_in. Right now this is
        only used in RetrievalStrategyStep.'''
        # Keep track of this, in case we create one so we know to attach this to the state vector
        self._rf_uip = None
        self._rf_uip_func = rf_uip_func
        args = self._forward_model(
            instrument_name_list, current_state, spec_win,
            self._rf_uip_func_wrap,
            do_systematic=do_systematic, jacobian_speciesIn=jacobian_speciesIn,
            obs_list=obs_list, fix_apriori_size=fix_apriori_size,
            **kwargs)
        if(self.rs is not None):
            self.radiance_step_in = self._create_radiance_step()
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
                       current_state : 'CurrentState',
                       spec_win : "list[rf.SpectralWindowRange]",
                       rf_uip_func,
                       obs_list=None,
                       fix_apriori_size=False,
                       **kwargs):
        # TODO Right now always have rf_uip. We should extend our forward models to
        # just tell us if the uip was created. But for now, always have this
        rf_uip = rf_uip_func()
        ret_info = { 
            'sqrt_constraint': current_state.sqrt_constraint,
            'const_vec': current_state.apriori,
        }
        # Note, we are trying to decouple the rf_uip from everywhere. Right now this
        # function is only called by RetrievalStrategyStep.create_cost_function which
        # is set up for rf_uip to None, so we can start moving this out. Right now, if the
        # rf_uip is set, we make sure the rf_uip gets updated when the CostFunction.parameters
        # get updated. But this is only needed if we have a ForwardModel with a rf_uip.
        self.obs_list = []
        self.state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        if(obs_list is not None):
            self.obs_list = obs_list
        else:
            for instrument_name in rf_uip.instrument:
                obs = self.observation_handle_set.observation(instrument_name, self.rs,
                                                              self.state_vector_handle_set,
                                                              **kwargs)
                self.obs_list.append(obs)
                
        self.fm_list = []
        for i, instrument_name in enumerate(rf_uip.instrument):
                
            fm =  self.forward_model_handle_set.forward_model(instrument_name,
                                  rf_uip, self.obs_list[i], self.state_vector_handle_set,
                                  obs_rad=None, meas_err=None,**kwargs)
            self.fm_list.append(fm)
        sv = self.state_vector_handle_set.create_state_vector(rf_uip,
                               **kwargs)
        bmatrix = rf_uip.basis_matrix
        if(not fix_apriori_size):
            # Normally, we get the apriori and constraint for our current state
            sv_apriori =  current_state.apriori
            sv_sqrt_constraint = current_state.sqrt_constraint.transpose()
        else:
            # This handles when we call with a RefractorUip but without a
            # ret_info, or otherwise don't have a apriori and sqrt_constraint. We
            # create dummy data of the right size.
            # This isn't something we encounter in our normal processing,
            # this is more to support old testing
            if(bmatrix is not None):
                sv_apriori = np.zeros((bmatrix.shape[0],))
                sv_sqrt_constraint=np.eye(bmatrix.shape[0])
            else:
                sv_apriori = np.zeros((sv.observer_claimed_size,))
                sv_sqrt_constraint=np.eye(sv.observer_claimed_size)

        return (self.fm_list, self.obs_list, sv, sv_apriori, sv_sqrt_constraint,
                bmatrix)

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
        forward.

        '''
        # Fake the input for the normal cost_function function
        def uip_func():
            return rf_uip
        cstate = CurrentState()
        if(ret_info):
            fix_apriori_size=False
            cstate.sqrt_constraint = ret_info["sqrt_constraint"]
            cstate.apriori = ret_info["const_vec"]
        else:
            fix_apriori_size=True
            cstate.sqrt_constraint = np.eye(1)
            cstate.apriori = np.zeros((1,))
        # Todo, fill this in
        spec_win = None
        return self.cost_function(rf_uip.instrument, cstate, spec_win, uip_func,
                                  obs_list=obs_list,
                                  fix_apriori_size=fix_apriori_size, **kwargs)
    
__all__ = ["StateVectorHandle", "StateVectorHandleSet",
           "ForwardModelHandle", "ForwardModelHandleSet",
           "CostFunctionCreator", "ObservationHandleSet", "ObservationHandle"]
        
        
        
