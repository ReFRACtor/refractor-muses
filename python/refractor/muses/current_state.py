from __future__ import annotations
from . import fake_muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy as np
import abc
from pathlib import Path
from copy import copy, deepcopy
import typing
from .identifier import StateElementIdentifier
from .refractor_uip import AttrDictAdapter
from scipy.linalg import block_diag  # type: ignore
from .retrieval_array import (
    RetrievalGridArray,
    FullGridArray,
    FullGridMappedArray,
    FullGridMappedArrayFromRetGrid,
    RetrievalGrid2dArray,
    FullGrid2dArray,
)

if typing.TYPE_CHECKING:
    from .refractor_uip import RefractorUip
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MeasurementId
    from .muses_strategy import CurrentStrategyStep
    from .muses_strategy import MusesStrategy
    from .observation_handle import ObservationHandleSet
    from .state_info import StateElement, StateElementHandleSet
    from .record_and_play_func import CurrentStateRecordAndPlay
    from .sounding_metadata import SoundingMetadata


class PropagatedQA:
    """There are a few parameters that get propagated from one step to
    the next. Not sure exactly what this gets looked for, it look just
    like flags copied from one step to the next. But pull this
    together into one place so we can track this.

    TODO Note that we might just make this appear like any other
    StateElementOld, but at least for now leave this as separate
    because that is how muse-py does this.

    """

    def __init__(self) -> None:
        self.propagated_qa: dict[str, int] = {"TATM": 1, "H2O": 1, "O3": 1}

    @property
    def tatm_qa(self) -> int:
        return self.propagated_qa["TATM"]

    @property
    def h2o_qa(self) -> int:
        return self.propagated_qa["H2O"]

    @property
    def o3_qa(self) -> int:
        return self.propagated_qa["O3"]

    def update(
        self, retrieval_state_element_id: list[StateElementIdentifier], qa_flag: int
    ) -> None:
        """Update the QA flags for items that we retrieved."""
        for state_element_id in retrieval_state_element_id:
            if str(state_element_id) in self.propagated_qa:
                self.propagated_qa[str(state_element_id)] = qa_flag


class CurrentState(object, metaclass=abc.ABCMeta):
    """There are a number of "states" floating around
    py-retrieve/ReFRACtor, and it can be a little confusing if you
    don't know what you are looking at.

    A good reference is section III.A.1 of "Tropospheric Emission
    Spectrometer: Retrieval Method and Error Analysis" (IEEE
    TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY
    2006) (https://ieeexplore.ieee.org/document/1624609). This
    describes the "retrieval state vector" and the "full state vector"

    In addition there is an intermediate set that isn't named in the
    paper.  In the py-retrieve code, this gets referred to as the
    "forward model state vector". And internally objects may map
    things to a "object state".

    A brief description of these:

    1. The "retrieval state vector" is what we use in our Solver for a
       retrieval step. This is the parameters passed to our
       CostFunction. Note that for something like a log(vmr) variable,
       this is in the unmapped log(vmr) representation

    2. The "forward model state vector" has the same content as the
       "retrieval state vector", but it is specified on a finer number
       of levels used by the forward model. It is also unmapped (e.g.,
       it might be log(vmr)).

    3. The "full state vector" is all the parameters the various
       objects making up our ForwardModel and Observation needs. This
       includes a number of things held fixed in a particular
       retrieval step. This is a super set of the "forward model state vector",
       so it has all the contents of the forward model plus extra stuff needed
       by the model but not being varies as part of the CostFunction.
       This is unmapped (e.g., it might be log(vmr)).

    4. the "full state vector mapped" is the "full state vector" but with
       the mapping applied - so a log retrieval item is mapped to vmr.

    5. The "object state" is the subset of the "forward model state
       vector" needed by a particular object in our ForwardModel or
       Observation, mapped to what the object needs. E.g., the forward
       model state vector is in log(vmr), but for the actual object we
       translate this to vmr.

    We store this data in a number of StateElement, with a
    StateElementIdentifier to label it. Note that muses-py often but
    not always refers to these as "species". We use the more general
    name "StateElement" because these aren't always gas species.

    A few examples, might illustrate this:

    One of the things that might be retrieved is log(vmr) for O3. In
    the "retrieval state vector" this has 25 log(vmr) values (for the
    25 levels). In the "forward model state vector" this as 64
    log(vmr) values (for the 64 levels the FM is run on).  In the
    "full state vector mapped" this is 64 vmr values (so log(vmr)
    converted to vmr needed for calculation). In the "object state"
    for the rf.AbsorberVmr part of the FowardModel this is the same 64
    values - just labeling the subset needed by AbsorberVmr

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

    For a normal retrieval, we get all the information needed from our
    StateInfo. But for testing it can be useful to have other
    implementations, include a hard coded set of values (for small
    unit tests) or a RefractorUip (to compare against old py-retrieve
    runs where we captured the UIP).  This class gives the interface
    needed by the other classes, as well as implementing some stuff
    that doesn't really depend on where we are getting the
    information.

    """

    # Can check against old state element stuff, if the StateInfo has this.
    # True means check if StateInfo has a old current state, False skips this
    # check evne if the old current state is available.
    check_old_state_element_value = True
    # check_old_state_element_value = False

    def __init__(self) -> None:
        # Cache these values, they don't normally change.
        self._fm_sv_loc: dict[StateElementIdentifier, tuple[int, int]] | None = None
        self._fm_state_vector_size = -1
        self._sys_sv_loc: dict[StateElementIdentifier, tuple[int, int]] | None = None
        self._sys_state_vector_size = -1
        self._retrieval_sv_loc: dict[StateElementIdentifier, tuple[int, int]] | None = (
            None
        )
        self._retrieval_state_vector_size = -1
        # Temp
        self._posteriori_slice: dict[StateElementIdentifier, slice] = {}
        self._previous_posteriori_cov_fm = np.zeros((0, 0))
        self.record: None | CurrentStateRecordAndPlay = None

    # TODO Replace this awkward interface. Only used in a couple of places, should
    # have a simpler interface.
    def current_state_override(
        self,
        do_systematic: bool,
        retrieval_state_element_override: None | list[StateElementIdentifier],
    ) -> CurrentState:
        """Create a variation of the current state that either does a systematic
        jacobian and/or overrides the retrieval_state_element_override. This is actually
        a bit awkward, but it is how muses-py was set up. This is only used in a few places,
        so we'll go ahead and use the same logic here."""
        raise NotImplementedError()

    def height(self) -> rf.ArrayWithUnit_double_1:
        """We already have height objects for our forward model, but this particular
        value gets written out as metadata in a few places. Calculate this the
        same way. We can possibly remove this in the future, but for now do this.
        """
        (results, _) = mpy.compute_altitude_pge(
            self.state_value("pressure"),
            self.state_value("TATM"),
            self.state_value("H2O"),
            self.sounding_metadata.surface_altitude.convert("m").value,
            self.sounding_metadata.latitude.value,
            None,
            True,
        )
        return rf.ArrayWithUnit_double_1(results["altitude"] / 1000.0, "km")

    @property
    def propagated_qa(self) -> PropagatedQA:
        raise NotImplementedError()

    def propagated_qa_update(
        self, retrieval_state_element_id: list[StateElementIdentifier], qa_flag: int
    ) -> None:
        if self.record is not None:
            self.record.record(
                "propagated_qa_update", retrieval_state_element_id, qa_flag
            )
        self.propagated_qa.update(retrieval_state_element_id, qa_flag)

    def brightness_temperature_data(self, step: int) -> dict[str, float | None] | None:
        raise NotImplementedError()

    def set_brightness_temperature_data(
        self, step: int, val: dict[str, float | None]
    ) -> None:
        raise NotImplementedError()

    def clear_cache(self) -> None:
        """Clear cache, if an update has occurred"""
        self._fm_sv_loc = None
        self._fm_state_vector_size = -1
        self._sys_sv_loc = None
        self._sys_state_vector_size = -1
        self._retrieval_sv_loc = None
        self._retrieval_state_vector_size = -1

    @property
    def initial_guess(self) -> RetrievalGridArray:
        """Initial guess, on the retrieval grid."""
        raise NotImplementedError()

    # TODO Do we actually need this?
    @property
    def initial_guess_full(self) -> FullGridArray:
        """Return the initial guess for the forward model grid.  This
        isn't independent, it is directly calculated from the
        initial_guess and basis_matrix. But convenient to supply this
        (mostly as a help in unit testing).

        """
        raise NotImplementedError()

    def update_full_state_element(
        self,
        state_element_id: StateElementIdentifier,
        value_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        next_constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        next_step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        raise NotImplementedError()

    # TODO See if we can remove this. I think this boils down to using a SpectralWindow
    # on a few frequency state elements, but I'm not sure. This is really an odd sort
    # of thing to carry around here
    @property
    def updated_fm_flag(self) -> FullGridArray:
        """This is array of boolean flag indicating which parts of the forward
        model state vector got updated when we called notify_solution. A 1 means
        it was updated, a 0 means it wasn't. This is used in the ErrorAnalysis."""
        raise NotImplementedError()

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        """Constraint matrix"""
        raise NotImplementedError()

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        """Sqrt matrix from covariance"""
        return (mpy.sqrt_matrix(self.constraint_matrix)).transpose()

    def constraint_vector(self, fix_negative: bool = True) -> RetrievalGridArray:
        """Apriori value"""
        raise NotImplementedError()

    # Is this needed? Should be it be mapped or not?
    @property
    def constraint_vector_full(self) -> FullGridArray:
        """Apriori value"""
        raise NotImplementedError()

    @property
    def true_value(self) -> RetrievalGridArray:
        """True value"""
        raise NotImplementedError()

    # Is this needed? Should be it be mapped or not?
    @property
    def true_value_full(self) -> FullGridArray:
        """True value"""
        raise NotImplementedError()

    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to full model vector.
        We don't always have this, so we return None if there isn't a basis matrix.
        """
        raise NotImplementedError()

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from basis matrix"""
        raise NotImplementedError()

    @property
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        """Return StateMapping going from RetrievalGridArray to FullGridArray.
        This is done by a basis_matrix in muses-py, but we are trying to move to a more
        general StateMapping."""
        bmatrix = self.basis_matrix
        if bmatrix is None:
            return rf.StateMappingLinear()
        return rf.StateMappingBasisMatrix(bmatrix.transpose())

    def fm_sv_slice(self, sid: StateElementIdentifier) -> slice | None:
        """Slice object needed  to subset the forward model state vector to
        the values for the StateElement. As a convention, this can be called
        with StateElement that aren't being retrieved and we return None."""
        if sid in self.fm_sv_loc:
            p, plen = self.fm_sv_loc[sid]
            return slice(p, p + plen)
        return None

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Dict that gives the starting location in the forward model
        state vector and length for a particular state element name
        (state elements not being retrieved don't get listed here)
        """
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_id in self.retrieval_state_element_id:
                plen = len(self.state_value(state_element_id))
                self._fm_sv_loc[state_element_id] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    def sys_sv_slice(self, sid: StateElementIdentifier) -> slice | None:
        """Slice object needed to subset the systematic forward model
        state vector to the values for the StateElement. As a
        convention, this can be called with StateElement that aren't
        being retrieved and we return None.

        """
        if sid in self.sys_sv_loc:
            p, plen = self.sys_sv_loc[sid]
            return slice(p, p + plen)
        return None

    @property
    def sys_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Dict that gives the starting location in the systematic forward model
        state vector and length for a particular state element name
        (state elements not being retrieved don't get listed here)
        """
        if self._sys_sv_loc is None:
            self._sys_sv_loc = {}
            self._sys_state_vector_size = 0
            for state_element_id in self.systematic_state_element_id:
                plen = len(self.state_value(state_element_id))
                self._sys_sv_loc[state_element_id] = (self._sys_state_vector_size, plen)
                self._sys_state_vector_size += plen
        return self._sys_sv_loc

    def retrieval_sv_slice(self, sid: StateElementIdentifier) -> slice | None:
        """Slice object needed  to subset the retrieval state vector to
        the values for the StateElement. As a convention, this can be called
        with StateElement that aren't being retrieved and we return None."""
        if sid in self.retrieval_sv_loc:
            p, plen = self.retrieval_sv_loc[sid]
            return slice(p, p + plen)
        return None

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Like fm_sv_loc, but for the retrieval state vactor (rather than the
        forward model state vector). If we don't have a basis_matrix, these are the
        same. With a basis_matrix, the total length of the fm_sv_loc is the
        basis_matrix column size, and retrieval_vector_loc is the smaller basis_matrix
        row size."""
        raise NotImplementedError()

    def state_mapping(
        self, state_element_id: StateElementIdentifier | str
    ) -> rf.StateMapping:
        """StateMapping used by the forward model (so taking the FullGridArray
        to FullGridMappedArray)"""
        return self.state_element(state_element_id).state_mapping

    # TODO Are these actually needed?
    def pressure_list(
        self, state_element_id: StateElementIdentifier | str
    ) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the retrieval state vector
        levels (generally smaller than the pressure_list_fm).
        """
        raise NotImplementedError()

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the forward model state
        vector levels (generally larger than the pressure_list).
        """
        raise NotImplementedError()

    @property
    def forward_model_state_vector_element_list(self) -> list[StateElementIdentifier]:
        """List of StateElementIdentifier for each entry in the
        forward model state vector. This is the full size of the
        forward model state vector, so in general a
        StateElementIdentifier may be listed multiple times.

        """
        res = [
            StateElementIdentifier("None"),
        ] * self.fm_state_vector_size
        for sid, (pstart, plen) in self.fm_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                sid,
            ] * plen
        return res

    @property
    def systematic_model_state_vector_element_list(
        self,
    ) -> list[StateElementIdentifier]:
        """List of StateElementIdentifier for each entry in the
        systematic forward model state vector. This is the full size of the
        systematic forward model state vector, so in general a
        StateElementIdentifier may be listed multiple times.

        """
        res = [
            StateElementIdentifier("None"),
        ] * self.sys_state_vector_size
        for sid, (pstart, plen) in self.sys_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                sid,
            ] * plen
        return res

    @property
    def retrieval_state_vector_element_list(self) -> list[StateElementIdentifier]:
        """List of StateElementIdentifier for each entry in the
        retrieval state vector. This is the full size of the retrieval
        state vector, so in general a StateElementIdentifier may be
        listed multiple times.

        """
        res = [
            StateElementIdentifier("None"),
        ] * self.retrieval_state_vector_size
        for sid, (pstart, plen) in self.retrieval_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                sid,
            ] * plen
        return res

    @property
    def fm_state_vector_size(self) -> int:
        """Full size of the forward model state vector."""
        if self._fm_state_vector_size < 0:
            # Side effect of fm_sv_loc is filling in fm_state_vector_size
            _ = self.fm_sv_loc
        return self._fm_state_vector_size

    @property
    def sys_state_vector_size(self) -> int:
        """Full size of the systematic forward model state vector."""
        if self._sys_state_vector_size < 0:
            # Side effect of sys_sv_loc is filling in fm_state_vector_size
            _ = self.sys_sv_loc
        return self._sys_state_vector_size

    @property
    def retrieval_state_vector_size(self) -> int:
        """Full size of the retrieval state vector."""
        if self._retrieval_state_vector_size < 0:
            # Side effect of retrieval_sv_loc is filling in fm_state_vector_size
            _ = self.retrieval_sv_loc
        return self._retrieval_state_vector_size

    def object_state(
        self, state_element_id_list: list[StateElementIdentifier]
    ) -> tuple[FullGridArray, rf.StateMapping]:
        """Return a set of coefficients and a rf.StateMapping to get
        the full state values used by an object. The object passes in
        the list of state element names it uses.  In general only a
        (possibly empty) subset of the state elements are actually
        retrieved.  This gets handled by the StateMapping, which might
        have a component like rf.StateMappingAtIndexes to handle the
        subset that is in the fm_state_vector.

        """
        # TODO put in handling of log/linear
        coeff = np.concatenate([self.state_value(nm) for nm in state_element_id_list])
        rlist = self.retrieval_state_element_id
        rflag = np.concatenate(
            [
                np.full((len(self.state_value(nm)),), nm in rlist, dtype=bool)
                for nm in state_element_id_list
            ]
        )
        mp = rf.StateMappingAtIndexes(rflag)
        return (coeff.view(FullGridArray), mp)

    def add_fm_state_vector_if_needed(
        self,
        fm_sv: rf.StateVector,
        state_element_id_list: list[StateElementIdentifier],
        obj_list: list[rf.SubStateVectorObserver],
    ) -> None:
        """This takes an object and a list of the state element names
        that object uses. This then adds the object to the forward
        model state vector if some of the elements are being
        retrieved.  This is a noop if none of the state elements are
        being retrieved. So objects don't need to try to figure out if
        they are in the retrieved set or not, then can just call this
        function to try adding themselves.

        """
        pstart = None
        for sname in state_element_id_list:
            if sname in self.fm_sv_loc:
                ps, _ = self.fm_sv_loc[sname]
                if pstart is None or ps < pstart:
                    pstart = ps
        if pstart is not None:
            fm_sv.observer_claimed_size = pstart
            for obj in obj_list:
                fm_sv.add_observer(obj)

    @abc.abstractproperty
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements we are retrieving."""
        raise NotImplementedError()

    @abc.abstractproperty
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that are in the systematic list (used by the
        ErrorAnalysis)."""
        raise NotImplementedError()

    @abc.abstractproperty
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that make up the full state, generally a
        larger list than retrieval_state_element_id"""
        raise NotImplementedError()

    def state_desc(self) -> str:
        """Return a description of the full state."""
        res = ""
        for selem in self.full_state_element_id:
            res += f"{str(selem)}:\n{self.state_value(selem)}\n"
        return res

    @abc.abstractproperty
    def sounding_metadata(self) -> SoundingMetadata:
        """Return the sounding metadata. It isn't clear if this really
        belongs in CurrentState or not, but there isn't another
        obvious place for this so for now we'll have this here.

        Perhaps this can migrate to the MuseObservation or MeasurementId if we decide
        that makes more sense.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_element(
        self, state_element_id: StateElementIdentifier | str
    ) -> StateElement:
        """Return the StateElement for the state_element_id. I'm not sure if we want to
        have this exposed or not, but there is a bit of useful information we have in
        each StateElement. We can have this exposed for
        now, and revisit it if we end up deciding this is too much coupling. There are
        only a few spots that use full_state_element vs something like full_state_value,
        so we will just need to revisit those few spots if this becomes an issue."""
        raise NotImplementedError()

    @abc.abstractmethod
    def state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier | str
    ) -> np.ndarray | None:
        """Return the spectral domain (as nm) for the given state_element_id, or None if
        there isn't an associated frequency for the given state_element_id"""
        raise NotImplementedError()

    @abc.abstractmethod
    def state_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        """Return the full state value for the given state element
        name.  Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        Also, the value returned is generally the *same*
        np.ndarray as used internally. Generally this is fine, the values tend
        to get used right away so there is no reason to return a copy. However
        if you are stashing the value for an internal state or something like that,
        you will want to make a copy of the returned value so it doesn't mysteriously
        change underneath you when the StateElement is updated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> FullGridMappedArray:
        """Return the initial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        Also, the value returned is generally the *same*
        np.ndarray as used internally. Generally this is fine, the values tend
        to get used right away so there is no reason to return a copy. However
        if you are stashing the value for an internal state or something like that,
        you will want to make a copy of the returned value so it doesn't mysteriously
        change underneath you when the StateElement is updated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        """Return the true value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        If we don't have a true value, return None

        Also, the value returned is generally the *same*
        np.ndarray as used internally. Generally this is fine, the values tend
        to get used right away so there is no reason to return a copy. However
        if you are stashing the value for an internal state or something like that,
        you will want to make a copy of the returned value so it doesn't mysteriously
        change underneath you when the StateElement is updated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        """Return the initialInitial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        Also, the value returned is generally the *same*
        np.ndarray as used internally. Generally this is fine, the values tend
        to get used right away so there is no reason to return a copy. However
        if you are stashing the value for an internal state or something like that,
        you will want to make a copy of the returned value so it doesn't mysteriously
        change underneath you when the StateElement is updated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        """Return the constraint vector of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        Also, the value returned is generally the *same*
        np.ndarray as used internally. Generally this is fine, the values tend
        to get used right away so there is no reason to return a copy. However
        if you are stashing the value for an internal state or something like that,
        you will want to make a copy of the returned value so it doesn't mysteriously
        change underneath you when the StateElement is updated.
        """
        raise NotImplementedError()

    def state_constraint_vector_fmprime(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArrayFromRetGrid:
        """Map state_constraint_vector to the retrieval grid, and back to
        the forward model grid. See discussion of this in FullGridMappedArrayFromRetGrid.
        """
        raise NotImplementedError()

    @property
    def Sa(self) -> FullGrid2dArray:
        """This combines the retrieval state element apriori_cov_fm and cross terms into
        a apriori_cov_fm of those elements. This S_a in the paper

        Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis
        V. ERROR CHARACTERIZATION
        (https://ieeexplore.ieee.org/document/1624609)
        """
        return self.apriori_cov_fm(self.retrieval_state_element_id)

    def setup_previous_aposteriori_cov_fm(
        self,
        covariance_state_element_name: list[StateElementIdentifier],
        current_strategy_step: CurrentStrategyStep,
    ) -> None:
        """Kludge, pass in list of elements. This will go away, but right now
        we do this. Also the update_initial_guess is needed by the old state element wrappers,
        this can go away with new ones"""
        pstart = 0
        for sid in covariance_state_element_name:
            selem = deepcopy(self.state_element(sid))
            if hasattr(selem, "update_initial_guess"):
                selem.update_initial_guess(current_strategy_step)
            plen = selem.apriori_cov_fm.shape[0]
            self._posteriori_slice[sid] = slice(pstart, pstart + plen)
            pstart += plen
        self._previous_posteriori_cov_fm = self.apriori_cov_fm(
            covariance_state_element_name, current_strategy_step
        )

    def previous_aposteriori_cov_fm(
        self, list_state_element_id: list[StateElementIdentifier]
    ) -> FullGrid2dArray:
        # Select only the elements requested
        v = np.zeros((self._previous_posteriori_cov_fm.shape[0]), dtype=bool)
        for sid in list_state_element_id:
            v[self._posteriori_slice[sid]] = True
        return self._previous_posteriori_cov_fm[:, v][v, :].copy().view(FullGrid2dArray)

    @property
    def Sb(self) -> FullGrid2dArray:
        """previous_aposteriori_cov_fm for the systematic_state_element_id"""
        return self.previous_aposteriori_cov_fm(self.systematic_state_element_id).view(
            FullGrid2dArray
        )

    @property
    def error_current_values(self) -> FullGrid2dArray:
        """previous_aposteriori_cov_fm for the retrieval_state_element_id. This is closely
        related to Sx, but has a different name in ErrorAnalysis"""
        return self.previous_aposteriori_cov_fm(self.retrieval_state_element_id)

    def update_previous_aposteriori_cov_fm(
        self, Sx: np.ndarray, off_diagonal_sys: np.ndarray | None
    ) -> None:
        """Update the previous_aposteriori_cov_fm for this retrieval step. Comes from
        ErrorAnalysis. We have the previous_aposteriori_cov_fm for retrieval_state_element_id,
        and the retrieval_state_element_id x systematic_state_element_id off diagonal matrix."""
        if self.record:
            self.record.record(
                "update_previous_aposteriori_cov_fm", Sx, off_diagonal_sys
            )
        # Zero out all the cross terms with retrieval_state_element_id
        v1 = np.zeros((self._previous_posteriori_cov_fm.shape[0]), dtype=bool)
        for sid in self.retrieval_state_element_id:
            v1[self._posteriori_slice[sid]] = True
        self._previous_posteriori_cov_fm[v1, :] = 0
        self._previous_posteriori_cov_fm[:, v1] = 0
        # Set Sx
        self._previous_posteriori_cov_fm[np.ix_(v1, v1)] = Sx
        # Populate the retrieval_state_element_id, systematic_state_element_id cross terms
        if off_diagonal_sys is not None:
            v2 = np.zeros((self._previous_posteriori_cov_fm.shape[0]), dtype=bool)
            for sid in self.systematic_state_element_id:
                v2[self._posteriori_slice[sid]] = True
            self._previous_posteriori_cov_fm[np.ix_(v2, v1)] = off_diagonal_sys
            self._previous_posteriori_cov_fm[np.ix_(v1, v2)] = (
                off_diagonal_sys.transpose()
            )

    def apriori_cov_fm(
        self,
        list_state_element_id: list[StateElementIdentifier],
        current_strategy_step: CurrentStrategyStep | None = None,
    ) -> FullGrid2dArray:
        """Return apriori covariance for the given list of state elements, including cross
        terms. When the list is the retrieval_state_element_id this is S_a in the paper

        Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis
        V. ERROR CHARACTERIZATION
        (https://ieeexplore.ieee.org/document/1624609)
        """
        # TODO deepcopy is temp, only needed for old state elements
        if current_strategy_step is not None:
            selem_list = [
                deepcopy(self.state_element(sname)) for sname in list_state_element_id
            ]
            for selem in selem_list:
                if hasattr(selem, "update_initial_guess"):
                    selem.update_initial_guess(current_strategy_step)
        else:
            selem_list = [self.state_element(sname) for sname in list_state_element_id]

        # Make block diagonal covariance.
        species_list = []
        matrix_list = []
        for selem in selem_list:
            matrix = selem.apriori_cov_fm
            species_list.extend([str(selem.state_element_id)] * matrix.shape[0])
            matrix_list.append(matrix)

        res = block_diag(*matrix_list)
        # TODO Replace this with cross term state elements when we put those
        # into place
        # Off diagonal blocks for covariance.
        for i, selem1 in enumerate(selem_list):
            for selem2 in selem_list[i + 1 :]:
                matrix2 = selem1.apriori_cross_covariance_fm(selem2)
                r1 = np.array(species_list) == str(selem1.state_element_id)
                r2 = np.array(species_list) == str(selem2.state_element_id)
                if matrix2 is not None:
                    res[np.ix_(r1, r2)] = matrix2
                    res[np.ix_(r2, r1)] = matrix2.transpose()
        return res.view(FullGrid2dArray)

    @abc.abstractmethod
    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        """Return the covariance of the apriori value of the given state element identification.

        Also, the value returned is generally the *same*
        np.ndarray as used internally. Generally this is fine, the values tend
        to get used right away so there is no reason to return a copy. However
        if you are stashing the value for an internal state or something like that,
        you will want to make a copy of the returned value so it doesn't mysteriously
        change underneath you when the StateElement is updated.
        """
        raise NotImplementedError()

    @property
    def step_directory(self) -> Path:
        """Return the step directory. This is a bit odd, but it is
        needed by MusesOpticalDepthFile. Since the current state
        depends on the step we are using, it isn't ridiculous to have
        this here. However if we find a better home for or better
        still remove the need for this that would be good.

        """
        raise NotImplementedError()

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        raise NotImplementedError()

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        pass

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        """Have updated the target we are processing."""
        pass

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        """Called when MusesStrategy has gone to the next step."""
        pass

    def notify_step_solution(self, xsol: RetrievalGridArray) -> None:
        """Called with MusesStrategy has found a solution for the current step."""
        pass


class CurrentStateUip(CurrentState):
    """Implementation of CurrentState that uses a RefractorUip"""

    def __init__(self, rf_uip: RefractorUip, ret_info: dict | None = None):
        """Get the CurrentState from a RefractorUip and ret_info. Note
        that this is just for backwards testing, we don't use the UIP
        in our current processing but rather something like
        CurrentStateStateInfo.

        The RefractorUip doesn't have everything we need, specifically
        we don't have the apriori and sqrt_constraint. We can get this
        from a ret_info, if available.  For testing we don't always
        have ret_info. This is fine if we don't actually need the
        apriori and sqrt_constraint. We still need a value for this,
        so if ret_info is None we return arrays of all zeros of the
        right size.

        """
        super().__init__()
        self.rf_uip = rf_uip
        self._initial_guess = rf_uip.current_state_x
        self._basis_matrix = rf_uip.basis_matrix
        self.ret_info = ret_info

    @property
    def initial_guess(self) -> RetrievalGridArray:
        return copy(self._initial_guess).view(RetrievalGridArray)

    @property
    def initial_guess_full(self) -> FullGridArray:
        """Return the initial guess for the forward model grid.  This
        isn't independent, it is directly calculated from the
        initial_guess and basis_matrix. But convenient to supply this
        (mostly as a help in unit testing).

        """
        raise NotImplementedError()

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        # Don't think we need this. We can calculate something frm
        # sqrt_constraint if needed, but for now just leave
        # unimplemented
        raise NotImplementedError()

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        if self.ret_info:
            return self.ret_info["sqrt_constraint"].view(RetrievalGridArray)
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.eye(len(self.initial_guess)).view(RetrievalGridArray)

    def constraint_vector(self, fix_negative: bool = True) -> RetrievalGridArray:
        if self.ret_info:
            return self.ret_info["const_vec"].view(RetrievalGridArray)
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.zeros((len(self.initial_guess),)).view(RetrievalGridArray)

    @property
    def constraint_vector_full(self) -> FullGridArray:
        if self.ret_info:
            return self.ret_info["const_vec"].view(FullGridArray)
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.zeros((len(self.initial_guess),)).view(FullGridArray)

    @property
    def basis_matrix(self) -> np.ndarray | None:
        return self._basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        raise NotImplementedError()

    # We don't have the other gas species working yet. Short term,
    # just have a different implementation of fm_sv_loc. We should
    # sort this out at some point.
    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for species_name in self.retrieval_state_element_id:
                pstart, plen = self.rf_uip.state_vector_species_index(str(species_name))
                self._fm_sv_loc[species_name] = (pstart, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for species_name in self.retrieval_state_element_id:
                pstart, plen = self.rf_uip.state_vector_species_index(
                    str(species_name), use_full_state_vector=False
                )
                self._retrieval_sv_loc[species_name] = (pstart, plen)
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        return [StateElementIdentifier(i) for i in self.rf_uip.jacobian_all]

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        raise NotImplementedError()

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        # I think we could come up with something here if needed, but for now
        # just punt on this
        raise NotImplementedError()

    @property
    def step_directory(self) -> Path:
        return self.rf_uip.step_directory

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        raise NotImplementedError()

    def state_element(
        self, state_element_id: StateElementIdentifier | str
    ) -> StateElement:
        raise NotImplementedError()

    def state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier | str
    ) -> np.ndarray | None:
        raise NotImplementedError()

    def state_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        # We've extracted this logic out from update_uip
        o_uip = AttrDictAdapter(self.rf_uip.uip)
        res = None
        if str(state_element_id) == "TSUR":
            res = np.array(
                [
                    o_uip.surface_temperature,
                ]
            )
        elif str(state_element_id) == "EMIS":
            res = np.array(o_uip.emissivity["value"])
        elif str(state_element_id) == "PTGANG":
            res = np.array([o_uip.obs_table["pointing_angle"]])
        elif str(state_element_id) == "RESSCALE":
            res = np.array([o_uip.res_scale])
        elif str(state_element_id) == "CLOUDEXT":
            res = np.array(o_uip.cloud["extinction"])
        elif str(state_element_id) == "PCLOUD":
            res = np.array([o_uip.cloud["pressure"]])
        elif str(state_element_id) == "OMICLOUDFRACTION":
            res = np.array([o_uip.omiPars["cloud_fraction"]])
        elif str(state_element_id) == "OMISURFACEALBEDOUV1":
            res = np.array([o_uip.omiPars["surface_albedo_uv1"]])
        elif str(state_element_id) == "OMISURFACEALBEDOUV2":
            res = np.array([o_uip.omiPars["surface_albedo_uv2"]])
        elif str(state_element_id) == "OMISURFACEALBEDOSLOPEUV2":
            res = np.array([o_uip.omiPars["surface_albedo_slope_uv2"]])
        elif str(state_element_id) == "OMINRADWAVUV1":
            res = np.array([o_uip.omiPars["nradwav_uv1"]])
        elif str(state_element_id) == "OMINRADWAVUV2":
            res = np.array([o_uip.omiPars["nradwav_uv2"]])
        elif str(state_element_id) == "OMIODWAVUV1":
            res = np.array([o_uip.omiPars["odwav_uv1"]])
        elif str(state_element_id) == "OMIODWAVUV2":
            res = np.array([o_uip.omiPars["odwav_uv2"]])
        elif str(state_element_id) == "OMIODWAVSLOPEUV1":
            res = np.array([o_uip.omiPars["odwav_slope_uv1"]])
        elif str(state_element_id) == "OMIODWAVSLOPEUV2":
            res = np.array([o_uip.omiPars["odwav_slope_uv2"]])
        elif str(state_element_id) == "OMIRINGSFUV1":
            res = np.array([o_uip.omiPars["ring_sf_uv1"]])
        elif str(state_element_id) == "OMIRINGSFUV2":
            res = np.array([o_uip.omiPars["ring_sf_uv2"]])
        elif str(state_element_id) == "TROPOMICLOUDFRACTION":
            res = np.array([o_uip.tropomiPars["cloud_fraction"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND1":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND1"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND2":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND3":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND7":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND3TIGHT":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND2":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND3":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND7":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND3TIGHT":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND2":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND3":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND7":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND3TIGHT":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND1":
            res = np.array([o_uip.tropomiPars["solarshift_BAND1"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND2":
            res = np.array([o_uip.tropomiPars["solarshift_BAND2"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND3":
            res = np.array([o_uip.tropomiPars["solarshift_BAND3"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND7":
            res = np.array([o_uip.tropomiPars["solarshift_BAND7"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND1":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND1"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND2":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND2"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND3":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND3"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND7":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND7"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND1":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND1"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND2":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND2"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND3":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND3"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND7":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND7"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND1":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND1"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND2":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND2"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND3":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND3"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND7":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND2":
            res = np.array([o_uip.tropomiPars["resscale_O0_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND2":
            res = np.array([o_uip.tropomiPars["resscale_O1_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND2":
            res = np.array([o_uip.tropomiPars["resscale_O2_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND3":
            res = np.array([o_uip.tropomiPars["resscale_O0_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND3":
            res = np.array([o_uip.tropomiPars["resscale_O1_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND3":
            res = np.array([o_uip.tropomiPars["resscale_O2_BAND3"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND3":
            res = np.array([o_uip.tropomiPars["temp_shift_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND7":
            res = np.array([o_uip.tropomiPars["resscale_O0_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND7":
            res = np.array([o_uip.tropomiPars["resscale_O1_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND7":
            res = np.array([o_uip.tropomiPars["resscale_O2_BAND7"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND7":
            res = np.array([o_uip.tropomiPars["temp_shift_BAND7"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND3TIGHT":
            res = np.array([o_uip.tropomiPars["temp_shift_BAND3"]])
        elif str(state_element_id) == "TROPOMICLOUDSURFACEALBEDO":
            res = np.array([o_uip.tropomiPars["cloud_Surface_Albedo"]])
        if res is not None:
            return res.view(FullGridMappedArray)
        # Check if it is a column
        try:
            return self.rf_uip.atmosphere_column(str(state_element_id)).view(
                FullGridMappedArray
            )
        except ValueError:
            pass
        raise RuntimeError(f"Don't recognize {state_element_id}")

    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        raise NotImplementedError()

    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        raise NotImplementedError()


class CurrentStateDict(CurrentState):
    """Implementation of CurrentState that just takes a dictionary of
    state elements and list of retrieval elements.

    """

    def __init__(
        self,
        state_element_dict: dict[
            StateElementIdentifier | str, np.ndarray | list[float] | float
        ],
        retrieval_element: list[StateElementIdentifier | str],
    ):
        """This takes a dictionary from state element name to value,
        and a list of retrieval elements. This is useful for creating
        unit tests that don't depend on other objects.

        Note both self.state_element_dict and self.retrieval_element
        can be updated if desired, if for whatever reason we want to
        add/tweak the data.

        """
        super().__init__()
        self._state_element_dict = {
            (
                k
                if isinstance(k, StateElementIdentifier)
                else StateElementIdentifier(k)
            ): v
            for (k, v) in state_element_dict.items()
        }
        self._retrieval_element = [
            i if isinstance(i, StateElementIdentifier) else StateElementIdentifier(i)
            for i in retrieval_element
        ]

    @property
    def state_element_dict(self) -> dict:
        return self._state_element_dict

    @state_element_dict.setter
    def state_element_dict(self, val: dict) -> None:
        self._state_element_dict = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        return self._retrieval_element

    @retrieval_state_element_id.setter
    def retrieval_state_element_id(self, val: list[StateElementIdentifier]) -> None:
        self._retrieval_element = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        raise NotImplementedError()

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        return list(self.state_element_dict.keys())

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        raise NotImplementedError()

    def state_element(
        self, state_element_id: StateElementIdentifier | str
    ) -> StateElement:
        raise NotImplementedError()

    def state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier | str
    ) -> np.ndarray | None:
        raise NotImplementedError()

    def state_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        v = self.state_element_dict[state_element_id]
        if isinstance(v, np.ndarray):
            return v.view(FullGridMappedArray)
        elif isinstance(v, list):
            return np.array(v).view(FullGridMappedArray)
        return np.array(
            [
                v,
            ]
        ).view(FullGridMappedArray)

    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        raise NotImplementedError()

    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        raise NotImplementedError()


__all__ = [
    "CurrentState",
    "CurrentStateUip",
    "CurrentStateDict",
    "PropagatedQA",
]
