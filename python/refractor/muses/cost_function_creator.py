from .refractor_uip import RefractorUip
from .cost_function import CostFunction
from .priority_handle_set import PriorityHandleSet
from .uip_updater import (StateVectorUpdateUip, MaxAPosterioriSqrtConstraintUpdateUip)
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
    creates a FowardModel and Observation for that instrument.'''
    def create_state_vector(self, rf_uip : RefractorUip,
                            use_full_state_vector=True, **kwargs):
        '''Create the full StateVector for all the species in rf_uip.

        For the retrieval, we use the "Retrieval State Vector".
        However, for testing it can be useful to use the "Full State Vector".
        See "Tropospheric Emission Spectrometer: Retrieval Method and Error
        Analysis" (IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,
        VOL. 44, NO. 5, MAY 2006) section III.A.1 for a discussion of this.
        Lower level muses-py functions work with the "Full State Vector", so
        it is useful to have the option of supporting this. Set
        use_full_state_vector to True to use the full state vector.

        If we have use_full_state_vector False, we also attach an observer
        that calls update_uip when the state vector is updated. We can't do
        that if use_full_state_vector is True, because muses-py only takes
        the retrieval vector, not the full state vector.
        '''
        sv = rf.StateVector()
        for species_name in rf_uip.jacobian_all:
            pstart, plen = rf_uip.state_vector_species_index(species_name,
                          use_full_state_vector=use_full_state_vector)
            self.add_sv(sv, species_name, pstart, plen, **kwargs)
        if(not use_full_state_vector):
            sv.add_observer_and_keep_reference(StateVectorUpdateUip(rf_uip))
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
    that a class is intended for this.

    Note InstrumentHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''

    def notify_update_target(self, measurement_id : dict, filter_list : dict):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass
        
    @abc.abstractmethod
    def fm_and_obs(self, instrument_name : str, rf_uip : RefractorUip,
                   obs : 'MusesObservation',
                   svhandle: StateVectorHandleSet,
                   use_full_state_vector=True,
                   obs_rad=None, meas_err=None, **kwargs):
        '''Return ForwardModel and Observation if we can process the given
        instrument_name, or (None, None) if we can't. Add any StateVectorHandle
        to the passed in set.

        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''
        raise NotImplementedError()
    
class InstrumentHandleSet(PriorityHandleSet):
    '''This takes  the instrument name and RefractorUip, and
    creates a FowardModel and Observation for that instrument.

    Note InstrumentHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''
    def notify_update_target(self, measurement_id : dict, filter_list : dict):
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(measurement_id, filter_list)
        
    def fm_and_obs(self, instrument_name : str, rf_uip : RefractorUip,
                   obs : 'MusesObservation',
                   svhandle : StateVectorHandleSet,
                   use_full_state_vector=True,
                   obs_rad=None, meas_err=None,
                   **kwargs):
        '''Create a ForwardModel and Observation for the given instrument.
        
        The StateVectorHandleSet svhandle is modified by having any
        StateVectorHandle added to it.'''

        return self.handle(instrument_name, rf_uip, obs, svhandle,
                           use_full_state_vector=use_full_state_vector,
                           obs_rad=obs_rad, meas_err=meas_err,
                           **kwargs)
    
    def handle_h(self, h : InstrumentHandle, instrument_name : str,
                 rf_uip : RefractorUip,
                 obs : 'MusesObservation',
                 svhandle : StateVectorHandleSet,
                 use_full_state_vector=True,
                 obs_rad=None, meas_err=None,
                 **kwargs):
        '''Process a registered function'''
        fm, obs = h.fm_and_obs(instrument_name, rf_uip, obs, svhandle,
                               use_full_state_vector=use_full_state_vector,
                               obs_rad=obs_rad,meas_err=meas_err,**kwargs)
        if(fm is None):
            return (False, None)
        return (True, (fm, obs))

class ObservationHandle(object, metaclass=abc.ABCMeta):
    '''Base class for InstrumentHandle. Note we use duck typing, so you
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.

    Note InstrumentHandle can assume that they are called for the same target, until
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
    def __init__(self, rs : 'Optional(RetrievalStategy)' = None):
        self.instrument_handle_set = copy.deepcopy(InstrumentHandleSet.default_handle_set())
        self.observation_handle_set = copy.deepcopy(ObservationHandleSet.default_handle_set())
        self.measurement_id = None
        self.filter_list = None

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
        self.instrument_handle_set.notify_update_target(self.measurement_id, self.filter_list)
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


    def cost_function(self, 
                      do_systematic=False,
                      jacobian_speciesIn=None,
                      **kwargs):
        '''Return cost function for the RetrievalStrategy. Also sets self.radiance_step_in
        - this is a bit awkward but I think we may replace radiance_steo_in. Right now this is
        only used in RetrievalStrategyStep.'''
        args, rf_uip, self.radiance_step_in = self._fm_and_obs_rs(
            do_systematic=do_systematic, jacobian_speciesIn=jacobian_speciesIn,
            **kwargs)
        cfunc = CostFunction(*args)
        # If we have an UIP, then update this when the parameters get updated.
        # Note the rf_uip.basis_matrix is None handles the degenerate case of when we
        # have no parameters, for example for RetrievalStrategyStepBT. Any time we
        # have parameters, the basis_matrix shouldn't be None.
        if(rf_uip is not None and rf_uip.basis_matrix is not None):
            cfunc.max_a_posteriori.add_observer_and_keep_reference(MaxAPosterioriSqrtConstraintUpdateUip(rf_uip))
        cfunc.parameters = rf_uip.current_state_x
        return cfunc

    def _fm_and_obs_rs(self,
                      do_systematic=False,
                      use_full_state_vector=True,
                      jacobian_speciesIn=None,
                      **kwargs):
        radianceStepIn = self._create_radiance_step()
        if(do_systematic):
            retrieval_info = self.rs.retrievalInfo.retrieval_info_obj
            rinfo = mpy.ObjectView({
                'parameterStartFM': retrieval_info.parameterStartSys,
                'parameterEndFM' : retrieval_info.parameterEndSys,
                'species': retrieval_info.speciesSys,
                'n_species': retrieval_info.n_speciesSys,
                'speciesList': retrieval_info.speciesListSys,
                'speciesListFM': retrieval_info.speciesListSys,
                'mapTypeListFM': mpy.constraint_get_maptype(self.rs.error_analysis.error_current, retrieval_info.speciesListSys),
                'initialGuessListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                'constraintVectorListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                'initialGuessList': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                'n_totalParametersFM': retrieval_info.n_totalParametersSys
            })
        else:
            rinfo = self.rs.retrievalInfo
        rf_uip = RefractorUip.create_uip(self.rs.state_info, self.rs.strategy_table,
                                         self.rs.windows, rinfo,
                                         self.o_airs, self.o_tes,
                                         self.o_cris, self.o_omi, self.o_tropomi,
                                         self.o_oco2,
                                         jacobian_speciesIn=jacobian_speciesIn)
        ret_info = { 
            'obs_rad': radianceStepIn["radiance"],
            'meas_err':radianceStepIn["NESR"],            
            'sqrt_constraint': (mpy.sqrt_matrix(self.rs.retrievalInfo.apriori_cov)).transpose(),
            'const_vec': self.rs.retrievalInfo.apriori,
        }
        # Note, we are trying to decouple the rf_uip from everywhere. Right now this
        # function is only called by RetrievalStrategyStep.create_cost_function which
        # is set up for rf_uip to None, so we can start moving this out. Right now, if the
        # rf_uip is set, we make sure the rf_uip gets updated when the CostFunction.parameters
        # get updated. But this is only needed if we have a ForwardModel with a rf_uip.
        self.obslist = []
        self.state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        for instrument_name in rf_uip.instrument:
            obs = self.observation_handle_set.observation(instrument_name, self.rs,
                                                          self.state_vector_handle_set,
                                                          **kwargs)
            self.obslist.append(obs)
        return (self._fm_and_obs(rf_uip, ret_info,
                               use_full_state_vector=use_full_state_vector,
                                **kwargs), rf_uip, radianceStepIn)

    def cost_function_from_uip(self, rf_uip : RefractorUip,
                               obslist,
                               ret_info : dict,
                               **kwargs):
        '''Create a cost function from a RefractorUip and a ret_info. Note that this is really
        just for backwards testing, we are trying to get away from using the RefractorUip because
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
        self.state_vector_handle_set = copy.deepcopy(StateVectorHandleSet.default_handle_set())
        # Pass obs to a bit it kludge. We'll come back and rework this, but this is
        # just jury rigged to get stuff working after we reworked the MusesObservation
        # stuff
        self.obslist = obslist
        if(ret_info is None):
            return CostFunction(*self._fm_and_fake_obs(rf_uip, **kwargs))
        return CostFunction(*self._fm_and_obs(rf_uip, ret_info, **kwargs))
    
    def _fm_and_obs(self, rf_uip : RefractorUip,
                   ret_info : dict,
                   use_full_state_vector=True,
                   fix_apriori_size=False,
                   identity_basis_matrix=False,
                   **kwargs):

        '''This returns a list of ForwardModel and Observation that goes
        with the supplied rf_uip. We also return a StateVector that has
        all the pieces of the ForwardModel and Observation objects
        attached. We also return the apriori for the StateVector and what
        muses-py calls the sqrt_constraint (note that despite the name,
        sqrt_constraint *isn't* actually the sqrt of the constraint matrix - 
        see the description of MaxAPosterioriSqrtConstraint in
        refractor.framework for explanation of this), and the basis matrix.
        '''
        fm_list = []
        obs_list = []
        obs_rad = ret_info["obs_rad"]
        meas_err = ret_info["meas_err"]
        sv_apriori = ret_info["const_vec"]
        sv_sqrt_constraint = ret_info["sqrt_constraint"].transpose()
        for i, instrument_name in enumerate(rf_uip.instrument):
                
            fm, obs =  self.instrument_handle_set.fm_and_obs(instrument_name,
                                  rf_uip, self.obslist[i], self.state_vector_handle_set,
                                  use_full_state_vector=use_full_state_vector,
                                  obs_rad=obs_rad, meas_err=meas_err,**kwargs)
            fm_list.append(fm)
            # Temp, we use the new obs list if we have a value, fall back to old if not.
            obs_list.append(self.obslist[i])
        sv = self.state_vector_handle_set.create_state_vector(rf_uip,
                               use_full_state_vector=use_full_state_vector,
                               **kwargs)
        if(identity_basis_matrix):
            bmatrix = None
        else:
            bmatrix = rf_uip.basis_matrix
        if(fix_apriori_size):
            sv_apriori = np.zeros((sv.observer_claimed_size,))
            sv_sqrt_constraint=np.eye(sv.observer_claimed_size)
        return (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint,
                bmatrix)
    
    def _fm_and_fake_obs(self, rf_uip: RefractorUip,
                        use_full_state_vector=True,
                        identity_basis_matrix=True,
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
        fake_ret_info["sqrt_constraint"] = np.eye(1)
        (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint, bmatrix) = \
            self._fm_and_obs(rf_uip, fake_ret_info,
                            use_full_state_vector=use_full_state_vector,
                            identity_basis_matrix=identity_basis_matrix,
                            **kwargs)
        sv_apriori = np.zeros((sv.observer_claimed_size,))
        sv_sqrt_constraint=np.eye(sv.observer_claimed_size)
        return (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint, bmatrix)
    
__all__ = ["StateVectorHandle", "StateVectorHandleSet",
           "InstrumentHandle", "InstrumentHandleSet",
           "CostFunctionCreator", "ObservationHandleSet", "ObservationHandle"]
        
        
        
