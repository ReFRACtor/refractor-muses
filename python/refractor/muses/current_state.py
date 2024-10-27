from . import muses_py as mpy
import refractor.framework as rf
import numpy as np
import abc

class CurrentState(object, metaclass=abc.ABCMeta):
    '''There are a number of "states" floating around
    py-retrieve/ReFRACtor, and it can be a little confusing if you
    don't know what you are looking at.

    A good reference is section III.A.1 of "Tropospheric Emission
    Spectrometer: Retrieval Method and Error Analysis" (IEEE
    TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY
    2006). This describes the "retrieval state vector" and the "full
    state vector"

    In addition there is an intermediate set that isn't named in the
    paper.  In the py-retrieve code, this gets referred to as the
    "forward model state vector". And internally objects may map
    things it a "object state".

    A brief description of these:

    1. The "retrieval state vector" is what we use in our Solver for a
    retrieval step. This is the parameters passed to our CostFunction.

    2. The "forward model state vector" has the same content as the
    "retrieval state vector", but it is specified on a finer number of
    levels used by the forward model.

    3. The "full state vector" is all the parameters the various
    objects making up our ForwardModel and Observation needs. This
    includes a number of things held fixed in a particular retrieval
    step.

    4. The "object state" is the subset of the "forward model
    state vector" needed by a particular object in our ForwardModel or
    Observation, mapped to what the object needs. E.g., the forward
    model state vector is in log(vmr), but for the actual object we
    translate this to vmr.

    Each of these vectors are made up of a number of StateElement,
    each with a fixed string used as a name to identify it.  We look
    up the StateElement by these fixed names. Note that the
    StateElement name value may have different values depending on the
    context - so it might have fewer levels in a retrieval state
    vector vs forward model state vector. Note that muses-py often but
    not always refers to these as "species". We use the more general
    name "StateElement" because these aren't always gas species.

    A few examples, might illustrate this

    One of the things that might be retrieved is log(vmr) for O3. In
    the "retrieval state vector" this has 25 log(vmr) values (for the
    25 levels). In the "forward model state vector" this as 64
    log(vmr) values (for the 64 levels the FM is run on).  In the
    "object state" for the rf.AbsorberVmr part of the FowardModel is
    64 vmr values (so log(vmr) converted to vmr needed for
    calculation).

    For the tropomi ForwardModel, a component object is a
    rf.GroundLambertian which has a polynomial with values
    "TROPOMISURFACEALBEDOBAND3", "TROPOMISURFACEALBEDOSLOPEBAND3",
    "TROPOMISURFACEALBEDOSLOPEORDER2BAND3". We might be retrieving
    only a subset of these values, holding the other fixed. So in this
    case the "retrieval state vector" might have only 2 entries for
    "TROPOMISURFACEALBEDOBAND3" and "TROPOMISURFACEALBEDOSLOPEBAND3",
    the "forward model state vector" would also only have 2 entries
    (since there aren't any levels involved, there is no difference
    between the retrieval state and the forward model state).  The
    "full state vector" would have 3 values, since we add in the part
    that is being held fixed.

    In all cases, we handle converting from one type of state vector
    to the other with a rf.StateMapping. The
    rf.MaxAPosterioriSqrtConstraint object in our CostFunction has a
    mapping going to and from the "retrieval state vector" to the
    "forward model state vector". The various pieces of the
    ForwardModel and Observation have mapping from "forward model
    state vector" to "full state vector", and the various objects
    handle mappings from "full state vector" to "object state".

    For a normal retrieval, we get all the information needed from
    our StateInfo. But for testing it can be useful to have other implementations,
    include a hard coded set of values (for small unit tests) or a RefractorUip
    (to compare against old py-retrieve runs where we captured the UIP).
    This class gives the interface needed by the other classes, as well as implementing
    some stuff that doesn't really depend on where we are getting the information.
    '''
    def __init__(self):
        # Cache these values, they don't normally change.
        self._fm_sv_loc = None
        self._fm_state_vector_size = None

    def clear_cache(self):
        '''Clear cache, if an update has occurred'''
        self._fm_sv_loc = None
        self._fm_state_vector_size = None

    @property
    def fm_sv_loc(self) -> 'dict[str,int]':
        '''Dict that gives the starting location in the forward model state vector for a
        particular state element name (state elements not being retrieved don't
        get listed here)'''
        if(self._fm_sv_loc is None):
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_name in self.retrieval_state_element:
                plen = len(self.full_state_value(state_element_name))
                self._fm_sv_loc[state_element_name] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def fm_state_vector_size(self) -> int:
        '''Full size of the forward model state vector.'''
        if(self._fm_state_vector_size is None):
            # Side effect of fm_sv_loc is filling in fm_state_vector_size
            _ = self.fm_sv_loc
        # As a convention, py-retrieve uses a dummy state vector of length 1 if
        # we aren't actually retrieving anything. This should probably get cleaned up
        # at some point, there isn't actually anything wrong with a zero size state
        # vector. But for now, support this convention
        if(self._fm_state_vector_size == 0):
            return 1
        return self._fm_state_vector_size
    
    def object_state(self, state_element_name_list : 'list[str]') -> (np.array, rf.StateMapping):
        '''Return a set of coefficients and a rf.StateMapping to get the full state values
        used by an object. The object passes in the list of state element names it uses.
        In general only a (possibly empty) subset of the state elements are actually retrieved.
        This gets handled by the StateMapping, which might have a component like
        rf.StateMappingAtIndexes to handle the subset that is in the fm_state_vector.'''
        # TODO put in handling of log/linear
        coeff = np.concatenate([self.full_state_value(nm) for nm in state_element_name_list])
        rlist = self.retrieval_state_element
        rflag = np.concatenate(
            [np.full((len(self.full_state_value(nm)),), nm in rlist, dtype=bool)
             for nm in state_element_name_list])
        mp = rf.StateMappingAtIndexes(rflag)
        return (coeff, mp)
    
    def add_fm_state_vector_if_needed(self, fm_sv : rf.StateVector,
                                      state_element_name_list : 'list[str]',
                                      obj_list : 'list[rf.SubStateVectorObserver]'):
        '''This takes an object and a list of the state element names that object
        uses. This then adds the object to the forward model state vector if
        some of the elements are being retrieved.  This is a noop if none of the
        state elements are being retrieved. So objects don't need to try to figure
        out if they are in the retrieved set or not, then can just call this function
        to try adding themselves.'''
        pstart = None
        for sname in state_element_name_list:
            if sname in self.fm_sv_loc:
                ps, _ = self.fm_sv_loc[sname]
                if(pstart is None or ps < pstart):
                    pstart = ps
        if(pstart is not None):
            fm_sv.observer_claimed_size = pstart
            for obj in obj_list:
                fm_sv.add_observer(obj)

    @abc.abstractproperty
    def retrieval_state_element(self) -> 'list[str]':
        '''Return list of state elements we are retrieving.'''
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_value(self, state_element_name) -> np.array:
        '''Return the full state value for the given state element name.
        Just as a convention we always return a np.array, so if there is only one value
        put that in a length 1 np.array.'''
        raise NotImplementedError

    @property
    def step_directory(self) -> str:
        '''Return the step directory. This is a bit odd, but it is needed by
        MusesOpticalDepthFile. Since the current state depends on the step we are
        using, it isn't ridiculous to have this here. However if we find a better
        home for or better still remove the need for this that would be good.'''
        raise NotImplementedError
    
                
class CurrentStateUip(CurrentState):
    '''Implementation of CurrentState that uses a RefractorUip'''
    def __init__(self, rf_uip: 'RefractorUip'):
        super().__init__()
        self.rf_uip = rf_uip
        self.initial_guess = rf_uip.current_state_x
        self.basis_matrix = rf_uip.basis_matrix

    # We don't have the other gas species working yet. Short term, just have a
    # different implementation of fm_sv_loc. We should sort this out at some point.
    @property
    def fm_sv_loc(self):
        if(self._fm_sv_loc is None):
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for species_name in self.retrieval_state_element:
                pstart, plen = self.rf_uip.state_vector_species_index(species_name)
                self._fm_sv_loc[species_name] = (pstart, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def retrieval_state_element(self) -> 'list[str]':
        return self.rf_uip.jacobian_all

    @property
    def step_directory(self) -> str:
        return self.rf_uip.step_directory
    
    def full_state_value(self, state_element_name) -> np.array:
        '''Return the full state value for the given state element name.
        Just as a convention we always return a np.array, so if there is only one value
        put that in a length 1 np.array.'''
        # We've extracted this logic out from update_uip
        o_uip = mpy.ObjectView(self.rf_uip.uip)
        if state_element_name == 'TSUR':
            return np.array([o_uip.surface_temperature,])
        elif state_element_name == 'EMIS':
            return np.array(o_uip.emissivity['value'])
        elif state_element_name == 'PTGANG': 
            return np.array([o_uip.obs_table['pointing_angle']])
        elif state_element_name == 'RESSCALE':
            return np.array([o_uip.res_scale])
        elif state_element_name == 'CLOUDEXT':
            return np.array(o_uip.cloud['extinction'])
        elif state_element_name == 'PCLOUD':
            return np.array([o_uip.cloud['pressure']])
        elif state_element_name == 'OMICLOUDFRACTION':
            return np.array([o_uip.omiPars['cloud_fraction']])
        elif state_element_name == 'OMISURFACEALBEDOUV1':
            return np.array([o_uip.omiPars['surface_albedo_uv1']])
        elif state_element_name == 'OMISURFACEALBEDOUV2':
            return np.array([o_uip.omiPars['surface_albedo_uv2']])
        elif state_element_name == 'OMISURFACEALBEDOSLOPEUV2':
            return np.array([o_uip.omiPars['surface_albedo_slope_uv2']])
        elif state_element_name == 'OMINRADWAVUV1':
            return np.array([o_uip.omiPars['nradwav_uv1']])
        elif state_element_name == 'OMINRADWAVUV2':
            return np.array([o_uip.omiPars['nradwav_uv2']])
        elif state_element_name == 'OMIODWAVUV1':
            return np.array([o_uip.omiPars['odwav_uv1']])
        elif state_element_name == 'OMIODWAVUV2':
            return np.array([o_uip.omiPars['odwav_uv2']])
        elif state_element_name == 'OMIODWAVSLOPEUV1':
            return np.array([o_uip.omiPars['odwav_slope_uv1']])
        elif state_element_name == 'OMIODWAVSLOPEUV2':
            return np.array([o_uip.omiPars['odwav_slope_uv2']])
        elif state_element_name == 'OMIRINGSFUV1':
            return np.array([o_uip.omiPars['ring_sf_uv1']])
        elif state_element_name == 'OMIRINGSFUV2':
            return np.array([o_uip.omiPars['ring_sf_uv2']])
        elif state_element_name == 'TROPOMICLOUDFRACTION':
            return np.array([o_uip.tropomiPars['cloud_fraction']])
        elif state_element_name == 'TROPOMISURFACEALBEDOBAND1':
            return np.array([o_uip.tropomiPars['surface_albedo_BAND1']])
        elif state_element_name == 'TROPOMISURFACEALBEDOBAND2':
            return np.array([o_uip.tropomiPars['surface_albedo_BAND2']])
        elif state_element_name == 'TROPOMISURFACEALBEDOBAND3':
            return np.array([o_uip.tropomiPars['surface_albedo_BAND3']])
        elif state_element_name == 'TROPOMISURFACEALBEDOBAND7':
            return np.array([o_uip.tropomiPars['surface_albedo_BAND7']])
        elif state_element_name == 'TROPOMISURFACEALBEDOBAND3TIGHT':
            return np.array([o_uip.tropomiPars['surface_albedo_BAND3']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEBAND2':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_BAND2']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEBAND3':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_BAND3']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEBAND7':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_BAND7']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEBAND3TIGHT':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_BAND3']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEORDER2BAND2':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_order2_BAND2']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEORDER2BAND3':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_order2_BAND3']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEORDER2BAND7':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_order2_BAND7']])
        elif state_element_name == 'TROPOMISURFACEALBEDOSLOPEORDER2BAND3TIGHT':
            return np.array([o_uip.tropomiPars['surface_albedo_slope_order2_BAND3']])
        elif state_element_name == 'TROPOMISOLARSHIFTBAND1':
            return np.array([o_uip.tropomiPars['solarshift_BAND1']])
        elif state_element_name == 'TROPOMISOLARSHIFTBAND2':
            return np.array([o_uip.tropomiPars['solarshift_BAND2']])
        elif state_element_name == 'TROPOMISOLARSHIFTBAND3':
            return np.array([o_uip.tropomiPars['solarshift_BAND3']])
        elif state_element_name == 'TROPOMISOLARSHIFTBAND7':
            return np.array([o_uip.tropomiPars['solarshift_BAND7']])
        elif state_element_name == 'TROPOMIRADIANCESHIFTBAND1':
            return np.array([o_uip.tropomiPars['radianceshift_BAND1']])
        elif state_element_name == 'TROPOMIRADIANCESHIFTBAND2':
            return np.array([o_uip.tropomiPars['radianceshift_BAND2']])
        elif state_element_name == 'TROPOMIRADIANCESHIFTBAND3':
            return np.array([o_uip.tropomiPars['radianceshift_BAND3']])
        elif state_element_name == 'TROPOMIRADIANCESHIFTBAND7':
            return np.array([o_uip.tropomiPars['radianceshift_BAND7']])
        elif state_element_name == 'TROPOMIRADSQUEEZEBAND1':
            return np.array([o_uip.tropomiPars['radsqueeze_BAND1']])
        elif state_element_name == 'TROPOMIRADSQUEEZEBAND2':
            return np.array([o_uip.tropomiPars['radsqueeze_BAND2']])
        elif state_element_name == 'TROPOMIRADSQUEEZEBAND3':
            return np.array([o_uip.tropomiPars['radsqueeze_BAND3']])
        elif state_element_name == 'TROPOMIRADSQUEEZEBAND7':
            return np.array([o_uip.tropomiPars['radsqueeze_BAND7']])
        elif state_element_name == 'TROPOMIRINGSFBAND1':
            return np.array([o_uip.tropomiPars['ring_sf_BAND1']])
        elif state_element_name == 'TROPOMIRINGSFBAND2':
            return np.array([o_uip.tropomiPars['ring_sf_BAND2']])
        elif state_element_name == 'TROPOMIRINGSFBAND3':
            return np.array([o_uip.tropomiPars['ring_sf_BAND3']])
        elif state_element_name == 'TROPOMIRINGSFBAND7':
            return np.array([o_uip.tropomiPars['ring_sf_BAND7']])
        elif state_element_name == 'TROPOMIRESSCALEO0BAND2':
            return np.array([o_uip.tropomiPars['resscale_O0_BAND2']])
        elif state_element_name == 'TROPOMIRESSCALEO1BAND2':
            return np.array([o_uip.tropomiPars['resscale_O1_BAND2']])
        elif state_element_name == 'TROPOMIRESSCALEO2BAND2':
            return np.array([o_uip.tropomiPars['resscale_O2_BAND2']])
        elif state_element_name == 'TROPOMIRESSCALEO0BAND3':
            return np.array([o_uip.tropomiPars['resscale_O0_BAND3']])
        elif state_element_name == 'TROPOMIRESSCALEO1BAND3':
            return np.array([o_uip.tropomiPars['resscale_O1_BAND3']])
        elif state_element_name == 'TROPOMIRESSCALEO2BAND3':
            return np.array([o_uip.tropomiPars['resscale_O2_BAND3']])
        elif state_element_name == 'TROPOMITEMPSHIFTBAND3':
            return np.array([o_uip.tropomiPars['temp_shift_BAND3']])
        elif state_element_name == 'TROPOMIRESSCALEO0BAND7':
            return np.array([o_uip.tropomiPars['resscale_O0_BAND7']])
        elif state_element_name == 'TROPOMIRESSCALEO1BAND7':
            return np.array([o_uip.tropomiPars['resscale_O1_BAND7']])
        elif state_element_name == 'TROPOMIRESSCALEO2BAND7':
            return np.array([o_uip.tropomiPars['resscale_O2_BAND7']])
        elif state_element_name == 'TROPOMITEMPSHIFTBAND7':
            return np.array([o_uip.tropomiPars['temp_shift_BAND7']])
        elif state_element_name == 'TROPOMITEMPSHIFTBAND3TIGHT':
            return np.array([o_uip.tropomiPars['temp_shift_BAND3']])
        elif state_element_name == 'TROPOMICLOUDSURFACEALBEDO':
            return np.array([o_uip.tropomiPars['cloud_Surface_Albedo']])
        # Check if it is a column
        try:
            return self.rf_uip.atmosphere_column(state_element_name)
        except ValueError:
            pass
        raise RuntimeError(f"Don't recognize {state_element_name}")
        
class CurrentStateDict(CurrentState):
    '''Implementation of CurrentState that just takes a dictionary of state elements
    and list of retrieval elements.'''
    def __init__(self, state_element_dict : dict, retrieval_element : list):
        '''This takes a dictionary from state element name to value, and a list of
        retrieval elements. This is useful for creating unit tests that don't depend
        on other objects.

        Note both self.state_element_dict and self.retrieval_element can be updated
        if desired, if for whatever reason we want to add/tweak the data.'''
        super().__init__()
        self._state_element_dict = state_element_dict
        self._retrieval_element = retrieval_element

    @property
    def state_element_dict(self) -> dict:
        return self._state_element_dict

    @state_element_dict.setter
    def state_element_dict(self, val : dict):
        self._state_element_dict = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_state_element(self)  -> 'list[str]':
        return self._retrieval_element

    @retrieval_state_element.setter
    def retrieval_state_element(self, val : 'list[str]'):
        self._retrieval_element = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    def full_state_value(self, state_element_name) -> np.array:
        '''Return the full state value for the given state element name.
        Just as a convention we always return a np.array, so if there is only one value
        put that in a length 1 np.array.'''
        v = self.state_element_dict[state_element_name]
        if(isinstance(v, np.ndarray)):
            return v
        elif(isinstance(v, list)):
            return np.array(v)
        return np.array([v,])

class CurrentStateStateInfo(CurrentState):
    '''Implementation of CurrentState that uses our StateInfo. This is the way
    the actual full retrieval works.'''
    def __init__(self, state_info : 'StateInfo', retrieval_info : 'RetrievalInfo',
                 step_directory : str,
                 retrieval_state_element_override=None, do_systematic=False):
        '''I think we'll want to get some of the logic in
        RetrievalInfo into this class, I'm not sure that we want this
        as separate. But for now, include this as an argument.

        The retrieval_state_element_override is an odd argument, it
        overrides the retrieval_state_element in RetrievalInfo with a
        different set. It isn't clear why this is handled this way -
        why doesn't RetrievalInfo just figure out the right retrieval_state_element list?
        But for now, do it the same way as py-retrieve. This seems to only be used in the
        RetrievalStrategyStepBT - I'm guessing this was a kludge put in to support
        this retrieval step.

        In addition, the CurrentState can also we used when we are calculating the
        "systematic" jacobian. This create a StateVector with a different set of state elements.
        This isn't used to do a retrieval, but rather to just calculate a jacobian.
        If do_systematic is set to True, we use this values instead.
        '''
        super().__init__()
        self._state_info = state_info
        self._retrieval_info = retrieval_info
        self.retrieval_state_element_override = retrieval_state_element_override
        self.do_systematic = do_systematic
        self._step_directory = step_directory

    @property
    def step_directory(self) -> str:
        return self._step_directory
        
    @property
    def state_info(self) -> 'StateInfo':
        return self._state_info

    @state_info.setter
    def state_info(self, val : 'StateInfo'):
        self._state_info = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_info(self) -> 'RetrievalInfo':
        return self._retrieval_info

    @retrieval_info.setter
    def retrieval_info(self, val : 'RetrievalInfo'):
        self._retrieval_info = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()
        
    @property
    def retrieval_state_element(self)  -> 'list[str]':
        if(self.retrieval_state_element_override is not None):
            return self.retrieval_state_element_override
        if(self.do_systematic):
            return self.retrieval_info.species_names_sys
        return self.retrieval_info.species_names

    @property
    def fm_sv_loc(self) -> 'dict[str,int]':
        '''Dict that gives the starting location in the forward model state vector for a
        particular state element name (state elements not being retrieved don't
        get listed here)'''
        if(self._fm_sv_loc is None):
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_name in self.retrieval_state_element:
                if(self.do_systematic):
                    plen = self.retrieval_info.species_list_sys.count(state_element_name)
                else:
                    plen = self.retrieval_info.species_list_fm.count(state_element_name)
                    
                # As a convention, if plen is 0 py-retrieve pads this to 1, although
                # the state vector isn't actually used - it does get set. I think this
                # is to avoid having a 0 size state vector. We should perhaps clean this
                # up as some point, there isn't anything wrong with a zero size state
                # vector (although this might have been a problem with IDL). But for
                # now, use the py-retrieve convention. This can generally
                # only happen if we have retrieval_state_element_override set, i.e., we
                # are doing RetrievalStrategyStepBT.
                if(plen == 0):
                    plen = 1
                self._fm_sv_loc[state_element_name] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    def full_state_value(self, state_element_name) -> np.array:
        '''Return the full state value for the given state element name.
        Just as a convention we always return a np.array, so if there is only one value
        put that in a length 1 np.array.'''
        selem = self.state_info.state_element(state_element_name)
        return selem.value
    
__all__ = ["CurrentState", "CurrentStateUip", "CurrentStateDict", "CurrentStateStateInfo"]    
        
        

