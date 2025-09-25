# Might end up breaking this file up, for now have all the stuff here
from __future__ import annotations
import refractor.framework as rf  # type: ignore
import refractor.muses.muses_py as mpy  # type: ignore
from .creator_handle import CreatorHandle, CreatorHandleSet
from .identifier import StateElementIdentifier, StrategyStepIdentifier
from .retrieval_array import (
    RetrievalGridArray,
    FullGridMappedArray,
    FullGridMappedArrayFromRetGrid,
    RetrievalGrid2dArray,
    FullGrid2dArray,
    FullGridArray,
)
from .current_state import CurrentState
import abc
from loguru import logger
import numpy as np
import numpy.testing as npt
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .cost_function_creator import CostFunctionStateElementNotify
    from .sounding_metadata import SoundingMetadata
    from .state_info import StateInfo


class StateElement(object, metaclass=abc.ABCMeta):
    """This handles a single StateElement, identified by
    a StateElementIdentifier (e.g., the name "TATM" or
    "pressure".

    A StateElement has a current value, a apriori value, a
    retrieval_initial_fm (at the start of a full multi-step retrieval)
    and step_initial_fm (at the start the current retrieval step).
    In addition, we possibly have "true" value, e.g, we know the answer
    because we are simulating data or something like that. Not
    every state element has a "true" value, is None in those cases.

    See the documentation at the start of CurrentState for a discussion
    of the various state vectors.

    As a convention, we always return the values as a np.ndarray, even
    for single scalar values. This just saves on needing to have lots
    of "if ndarray else if scalar" code. A scalar is returned as a
    np.ndarray of size 1. Also, the value returned is generally the *same*
    np.ndarray as used internally. Generally this is fine, the values tend
    to get used right away so there is no reason to return a copy. However
    if you are stashing the value for an internal state or something like that,
    you will want to make a copy of the returned value so it doesn't mysteriously
    change underneath you when the StateElement is updated.

    Note that we have a separate apriori_cov_fm and constraint_matrix. Most of
    the time these aren't actually independent, for a MaxAPosteriori type cost
    function the constraint matrix is just apriori_cov.

    However the uses of these are different. The constraint matrix is used to
    regularize the solver, it is added as augmented terms to the cost function
    as a penalty for moving away from the constraint_vector. When the constraint
    matrix is apriori covariance this is a maximum a posteriori problem, which
    is a common step used in the retrieval strategy. However we can also just
    use an ad hoc constraint providing e.g. smoothness (see II.B of the paper
    mentioned below). For example the ig_refine step after getting the OMI
    or TropOMI cloud fraction in a brightness temperature uses a tighter constraint
    than the apriori covariance.

    The apriori_cov_fm is used in the error analysis, and really should be the
    apriori covariance matrix in all cases (this is the state elements portion of
    S_a, in the terminology of the paper listed below).

    In addition, we use the a posterior from previous steps in the error analysis of
    the current retrieval step. We maintain this as previous_posteriori_cov_fm, which is
    a portion of S_b for that StateElement.

    See the paper:

    "Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis"
    (https://ieeexplore.ieee.org/document/1624609).

    We have two state mappings, one that goes between the retrieval
    state vector and the forward model state vector, and a second that
    is used in the forward model. These are separated because
    MaxAPosterioriSqrtConstraint needs to be able to go to the forward model
    state vector separately (used by the error analysis). In py-retrieve,
    state_mapping_retrieval_to_fm was the basis matrix, and state_mapping
    were the mapTypeList. But we use the more generate rf.StateMapping here so
    we aren't restricted to just these two types of mappings (e.g., we
    might want to do a shape retrieval like OCO-2 does).

    A few of the state elements have cross coupling, for example H2O and HDO. We
    purposely have all the handling of that *outside* this StateElement, in a
    CrossStateElement. So for the purposes of these state elements, we *always*
    handle things as if the other term wasn't available. In particular, we return
    the constraint_matrix as if we only had H2O. We have any updates/changes handled
    in the CrossStateElement. This just makes the logic simpler - we don't need to somehow
    maintain information about a cross term in this StateElement - we have a separate class
    for that. See CurrentStateStateInfo.constraint_matrix for an example of this.

    The StateElement gets notified when various things happen in a retrieval. These
    are:

    1. notify_start_retrieval called at the very start of a retrieval. So this can
       do things like update the value and step_initial_fm to the retrieval_initial_fm.
    2. notify_start_step called at the start of a retrieval step. So this can do
       things like update the value and step_initial_fm to the next_step_initial_fm
       determined in the previous step
    3. notify_step_solution called when a retrieval step reaches a solution. This can
       determine the step_initial_fm for the next step.
    4. notify_parameter_update is called when the rf.StateVector in a rf.ForwardModel is updated
       (e.g., the CostFunction gets updated parameters). This is somewhat redundant with
       notify_step_solution, but it allows the StateElement to update its value for each
       change to the rf.StateVector occurs. We don't have any code right now that actually
       makes use of this, but it just seems more consistent with the normal "updating
       a state vector updates objects that use that state" semantics in ReFRACtor.

    This is the general approach used in most of our code (and very different from a
    typical structured program like IDL). Things external to this class don't tell it
    how to maintain its state, instead we just tell it when things happen and the
    classes decide what to do with that information. So we have "notify_step_solution" rather
    than "update _value_fm to this value".
    """

    def __init__(self, state_element_id: StateElementIdentifier):
        self._state_element_id = state_element_id
        # Used to get notify_parameter_update called with CostFunction parameter
        # gets changed
        self._cost_function_notify_helper: CostFunctionStateElementNotify | None = None

    def __getstate__(self) -> dict[str, Any]:
        # Don't include CostFunctionStateElementNotify when pickling
        state = self.__dict__.copy()
        state["_cost_function_notify_helper"] = None
        return state

    @property
    def metadata(self) -> dict[str, Any]:
        """Some StateElement have extra metadata. There is really only one example
        now, emissivity has camel_distance and prior_source. It isn't clear the best
        way to handle this, but the current design just returns a dictionary with
        any extra metadata values. We can perhaps rework this if needed in the future.
        For most StateElement this will just be a empty dict."""
        res: dict[str, Any] = {}
        return res

    @property
    def should_fix_negative(self) -> bool:
        """For some StateElement, it doesn't make sense to have negative values on the
        retrieval grid. An example of this is VMR (with a linear mapping), where negative
        VMR never makes sense.

        We leave the actual values alone, but in some cases we can correct for this. For
        example, it makes sense to have the constraint_vector used in the solver replace
        these values - we don't want the regularization we add to our cost function to
        try pulling values negative.

        So have the StateElement indicate if it shouldn't be negative - but by design it
        doesn't actually do anything about this. Instead we handle this at a higher level, to
        make it explicit that we are changing values.
        """
        return False

    @property
    def state_element_id(self) -> StateElementIdentifier:
        return self._state_element_id

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        """For StateElementWithFrequency, this returns the frequency associated
        with it. For all other StateElement, just return None."""
        return None

    @property
    def basis_matrix(self) -> np.ndarray:
        """Basis matrix going from retrieval vector to forward model
        vector. Would be nice to replace this with a general
        rf.StateMapping, but for now this is assumed in a lot of
        muses-py code."""
        return np.eye(self.value_fm.shape[0])

    @property
    def map_to_parameter_matrix(self) -> np.ndarray:
        """Go the other direction from the basis matrix, going from
        the forward model vector the retrieval vector."""
        return np.eye(self.value_fm.shape[0])

    @abc.abstractproperty
    def forward_model_sv_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractproperty
    def sys_sv_length(self) -> int:
        raise NotImplementedError()

    @property
    def retrieved_this_step(self) -> bool:
        raise NotImplementedError()

    @property
    def pressure_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise). Note unlike other things on
        RetrievalGridArray, this is always in pressure units (so we don't take
        the log if self.state_mapping is log)."""
        if self.pressure_list_fm is None:
            return None
        return self.pressure_list_fm.to_ret(
            self.state_mapping_retrieval_to_fm, rf.StateMappingLinear()
        )

    @property
    def value_ret(self) -> RetrievalGridArray:
        return self.value_fm.to_ret(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def value_full(self) -> FullGridArray:
        return self.value_fm.to_full(self.state_mapping)

    @property
    def step_initial_ret(self) -> RetrievalGridArray:
        return self.step_initial_fm.to_ret(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def step_initial_ret_to_fmprime(self) -> FullGridMappedArrayFromRetGrid:
        """Because the retrieval grid has fewer levels than the forward model grid,
        you in general have different values if you start at the forward model grid
        vs. starting with constraint_vector_ret and mapping to FullGridMappedArray.
        I'm not sure how much it matters, but various calculations in muses-py
        uses this second version. Supply this."""
        return self.step_initial_ret.to_fmprime(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def step_initial_full(self) -> FullGridArray:
        return self.step_initial_fm.to_full(self.state_mapping)

    @property
    def retrieval_initial_ret(self) -> RetrievalGridArray:
        return self.retrieval_initial_fm.to_ret(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def retrieval_initial_full(self) -> FullGridArray:
        return self.retrieval_initial_fm.to_full(self.state_mapping)

    @property
    def constraint_vector_ret(self) -> RetrievalGridArray:
        return self.constraint_vector_fm.to_ret(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def constraint_vector_fmprime(self) -> FullGridMappedArrayFromRetGrid:
        """Because the retrieval grid has fewer levels than the forward model grid,
        you in general have different values if you start at the forward model grid
        vs. starting with constraint_vector_ret and mapping to FullGridMappedArray.
        I'm not sure how much it matters, but various calculations in muses-py
        uses this second version. Supply this."""
        return self.constraint_vector_fm.to_fmprime(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def constraint_vector_full(self) -> FullGridArray:
        return self.constraint_vector_fm.to_full(self.state_mapping)

    @property
    def true_value_ret(self) -> RetrievalGridArray | None:
        if self.true_value_fm is None:
            return None
        return self.true_value_fm.to_ret(
            self.state_mapping_retrieval_to_fm, self.state_mapping
        )

    @property
    def true_value_full(self) -> FullGridArray | None:
        if self.true_value_fm is None:
            return None
        return self.true_value_fm.to_full(self.state_mapping)

    @abc.abstractproperty
    def value_fm(self) -> FullGridMappedArray:
        """Current value of StateElement.

        This is the mapped state from state_mapping (so VMR rather than
        log(VMR) for a log mapping)"""
        raise NotImplementedError()

    @abc.abstractproperty
    def constraint_vector_fm(self) -> FullGridMappedArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        """Constraint matrix, generally the same as apriori_cov, although see the
        discussion in StateElement class about how this might be different.

        Note that this should be the constraint matrix without considering any cross
        terms. The cross terms are handled separately by CrossStateElement. In particular,
        this matrix might get replaced by a cross term (e.g., instead of reading H2O.asc
        we read H2O_H2O.asc) - however this class doesn't need to worry about this. It
        should just always return the non-cross term informed constraint matrix and we
        separately look at CrossStateElement to see if anything needs to be modified."""
        raise NotImplementedError()

    @abc.abstractproperty
    def apriori_cov_fm(self) -> FullGrid2dArray:
        """Apriori Covariance"""
        raise NotImplementedError()

    def apriori_cross_covariance_fm(
        self, selem2: StateElement
    ) -> FullGrid2dArray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    @abc.abstractproperty
    def step_initial_fm(self) -> FullGridMappedArray:
        """Value StateElement had at the start of the retrieval step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_initial_fm(self) -> FullGridMappedArray:
        """Value StateElement had at the start of the retrieval."""
        raise NotImplementedError()

    @abc.abstractproperty
    def state_mapping(self) -> rf.StateMapping:
        """StateMapping used by the forward model (so taking the FullGridArray
        and mapping to the internal object state)"""
        raise NotImplementedError()

    @abc.abstractproperty
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        """StateMapping used to go between the RetrievalGridArray and
        FullGridArray (e.g., the basis matrix in muses-py)"""
        raise NotImplementedError()

    @property
    def true_value_fm(self) -> FullGridMappedArray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        return None

    @abc.abstractproperty
    def updated_fm_flag(self) -> FullGridMappedArray:
        """This is array of boolean flag indicating which parts of the forward
        model state vector got updated when we called notify_solution. A 1 means
        it was updated, a 0 means it wasn't. This is used in the ErrorAnalysis."""
        raise NotImplementedError()

    @property
    def pressure_list_fm(self) -> FullGridMappedArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise). This is the levels that
        the apriori_cov_fm are on."""
        return None

    @abc.abstractmethod
    def update_state_element(
        self,
        value_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        next_constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        next_step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        """Update the value of the StateElement. This function updates
        each of the various values passed in.  A value of 'None' (the
        default) means skip updating that part of the StateElement.
        """
        raise NotImplementedError()

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        """Called with the subset of parameters for this StateElement
        when the cost function changes, or a solution has been found."""
        pass

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        """Called at the start of a retrieval (before the first step)."""
        pass

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        """Called each time at the start of a retrieval step."""
        pass

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        """Called when a retrieval step has a solution.

        A note, the step_initial_fm shouldn't be updated yet. classes that use the results want
        to have the initial guess for *this* step, not what will be used in the next. Instead
        the initial guess should be updated in notify_start_step. We do save what the next
        initial guess should be, assuming it was updated.

        We pass in the slice needed to get this StateElement values, or None if we aren't
        actually retrieving this particular StateElement. It just is more natural to have
        something outside this class maintain this information (e.g the CurrentState), since
        this depends on access to all the StateElement in the StateInfo.
        """
        pass


class StateElementHandle(CreatorHandle):
    """Return StateElement objects, for a given StateElementIdentifier

    Note StateElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        state_info: StateInfo | None,
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        raise NotImplementedError()


class StateElementHandleSet(CreatorHandleSet):
    """This maps a StateElementIdentifier to a StateElement object that handles it.

    Note StatElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def __init__(self) -> None:
        super().__init__("state_element")

    def state_element(self, state_element_id: StateElementIdentifier) -> StateElement:
        return self.handle(state_element_id)

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        state_info: StateInfo | None,
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(
                    measurement_id,
                    retrieval_config,
                    strategy,
                    observation_handle_set,
                    sounding_metadata,
                    state_info,
                )


class StateElementImplementation(StateElement):
    """A very common implementation of a StateElement just populates member variables for
    the value, apriori, etc. This class handles this common case, derived classes should
    fill in handling the various notify functions. Note we allow various values to be "None",
    this is useful for derived classes that might fill something in lazily."""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        value_fm: FullGridMappedArray,
        constraint_vector_fm: FullGridMappedArray,
        apriori_cov_fm: FullGrid2dArray | None,
        constraint_matrix: RetrievalGrid2dArray | None,
        state_mapping_retrieval_to_fm: rf.StateMapping | None = rf.StateMappingLinear(),
        state_mapping: rf.StateMapping | None = rf.StateMappingLinear(),
        initial_value_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
        spectral_domain: rf.SpectralDomain | None = None,
        selem_wrapper: Any | None = None,
    ) -> None:
        super().__init__(state_element_id)
        self._value_fm = value_fm.astype(np.float64).view(FullGridMappedArray)
        self._retrieval_start_constraint_fm = constraint_vector_fm.astype(
            np.float64, copy=True
        ).view(FullGridMappedArray)
        self._constraint_vector_fm = constraint_vector_fm.astype(
            np.float64, copy=True
        ).view(FullGridMappedArray)
        if constraint_matrix is not None:
            self._constraint_matrix: RetrievalGrid2dArray | None = (
                constraint_matrix.astype(np.float64, copy=True).view(
                    RetrievalGrid2dArray
                )
            )
        else:
            self._constraint_matrix = None
        if apriori_cov_fm is not None:
            self._apriori_cov_fm: FullGrid2dArray | None = apriori_cov_fm.astype(
                np.float64, copy=True
            ).view(FullGrid2dArray)
        else:
            self._apriori_cov_fm = None
        self._state_mapping = state_mapping
        self._state_mapping_retrieval_to_fm = state_mapping_retrieval_to_fm
        self._pressure_list_fm: FullGridMappedArray | None = None
        if initial_value_fm is not None:
            self._step_initial_fm = initial_value_fm.astype(np.float64, copy=True).view(
                FullGridMappedArray
            )
        else:
            self._step_initial_fm = value_fm.astype(np.float64, copy=True).view(
                FullGridMappedArray
            )

        self._retrieval_initial_fm = self._step_initial_fm.astype(
            np.float64, copy=True
        ).view(FullGridMappedArray)
        if true_value_fm is not None:
            self._true_value_fm: FullGridMappedArray | None = true_value_fm.astype(
                np.float64, copy=True
            ).view(FullGridMappedArray)
        else:
            self._true_value_fm = None
        self._updated_fm_flag: FullGridMappedArray | None = None
        if apriori_cov_fm is not None:
            self._updated_fm_flag = (
                np.zeros((apriori_cov_fm.shape[0],))
                .astype(bool)
                .view(FullGridMappedArray)
            )
        self._retrieved_this_step = False
        self._initial_guess_not_updated = False
        self._next_step_initial_fm: FullGridMappedArray | None = None
        self._next_constraint_vector_fm: FullGridMappedArray | None = None
        self._metadata: dict[str, Any] = {}
        self._spectral_domain = spectral_domain
        # Temp, until we have tested everything out
        self._sold = selem_wrapper
        if self._sold is not None and hasattr(self._sold, "update_initial_guess"):
            self.update_initial_guess = self._update_initial_guess
        self.is_bt_ig_refine = False
        self._current_strategy_step: StrategyStepIdentifier | None = None

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        # Skip if we aren't actually retrieving. This fits with the hokey way that
        # muses-py handles the BT and systematic jacobian steps. We should clean this
        # up an some point, this is all unnecessarily obscure
        # TODO - Fix this logic
        if not self._retrieved_this_step:
            return
        # if self._value_fm is not None and self.value.shape[0] != param_subset.shape[0]:
        #    raise RuntimeError(
        #        f"param_subset doesn't match value size {param_subset.shape[0]} vs {self.value.shape[0]}"
        #    )
        # Short term skip so we can compare to old state element
        if False:
            self._value_fm = param_subset.view(RetrievalGridArray).to_fm(
                self.state_mapping_retrieval_to_fm, self.state_mapping
            )

    def _update_initial_guess(self, current_strategy_step: CurrentStrategyStep) -> None:
        if self._sold is None:
            raise RuntimeError("This shouldn't happen")
        self._sold.update_initial_guess(current_strategy_step)

    def _check_result(
        self,
        res: float | np.ndarray | None,
        func_name: str,
        exclude_negative: bool = False,
    ) -> None:
        """Function to check against the old state element. This will go away at
        some point, but for now it is useful for spotting problems. No error if
        we don't have the old state element, we just skip the check. Also we
        have variable in CurrentState. This is also temporary, we are doing this
        to work around issues with unit tests using data that isn't in sync. This
        should go away when we regenerate the saved state data to not use the
        old state elements, but we'll hold off on that until the end."""
        if (
            res is None
            or self._sold is None
            or not CurrentState.check_old_state_element_value
        ):
            return
        res2 = getattr(self._sold, func_name)
        if res2 is None:
            raise RuntimeError("res2 should not be None")
        if isinstance(res, np.ndarray) and len(res.shape) == 1 and len(res2.shape) == 2:
            res2 = res2[0, :]
        if isinstance(res, np.ndarray) and len(res.shape) == 2 and len(res2.shape) == 1:
            res2 = res2[np.newaxis, :]
        if isinstance(res, np.ndarray):
            # For some values, the old code truncated negative values.
            # We handle this outside of StateElement. Exclude these points,
            # they can be different and this isn't a problem
            if exclude_negative and self.should_fix_negative:
                if np.count_nonzero(res > 0) > 0:
                    npt.assert_allclose(res[res > 0], res2[res > 0], 1e-12)
            else:
                npt.assert_allclose(res, res2, 1e-12)
            # Special case, some of the basis matrix that happen to be integer
            # are left as int64. No real reason, but also no harm
            if res.dtype != bool and res2.dtype != np.int64:
                assert res.dtype == res2.dtype
        else:
            assert res == res2

    @property
    def should_fix_negative(self) -> bool:
        # Not sure of the exact logic here. We can rework this if needed, perhaps put
        # in a general linear constraint library? But for now, we use the simple logic
        # of 1) if we are something that has pressure levels and 2) we use a linear mapping
        # then we say we shouldn't go negative.
        # This is probably too simple in general, but this should be at least a reasonable
        # stand in here until/unless we need something more sophisticated
        if (
            isinstance(self.state_mapping, rf.StateMappingLinear)
            and self._pressure_list_fm is not None
        ):
            return True
        return False

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def basis_matrix(self) -> np.ndarray:
        if isinstance(self.state_mapping_retrieval_to_fm, rf.StateMappingBasisMatrix):
            res = self.state_mapping_retrieval_to_fm.basis_matrix.transpose().view()
        else:
            res = np.eye(self.value_fm.shape[0])
        res.flags.writeable = False
        # Only present in sold when we are retrieving
        if self._retrieved_this_step:
            self._check_result(res, "basis_matrix")
        return res

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def map_to_parameter_matrix(self) -> np.ndarray:
        if isinstance(self.state_mapping_retrieval_to_fm, rf.StateMappingBasisMatrix):
            res = self.state_mapping_retrieval_to_fm.inverse_basis_matrix.transpose().view()
        else:
            res = np.eye(self.value_fm.shape[0])
        res.flags.writeable = False
        # Only present in sold when we are retrieving
        if self._retrieved_this_step:
            self._check_result(res, "map_to_parameter_matrix")
        return res

    @property
    def fm_update_flag(self) -> np.ndarray:
        """For some of the StateElement, we set up the state_mapping_retrieval_to_fm so
        that we map identically to 0. This is a way to hold some of the forward model
        elements constant. This is currently just done for EMIS and CLOUDEXT.

        Note this is pretty much the same idea and updated_fm_flag, however we keep these
        distinct. This is more to match the old muses-py behavior, somewhat confusingly these
        flags don't always have the same values.

        update_fm_flag is just used in the ErrorAnalysis to hold error estimates constant.
        fm_update_flag is used to determine which part of value_fm get updated. These should
        be the same, but currently aren't.

        TODO - Figure out how to make these the same.
        """
        # We just find rows in the basis_matrix that are all zero.
        return np.max(np.abs(self.basis_matrix), axis=0) >= 1e-10

    @property
    def forward_model_sv_length(self) -> int:
        if self._retrieved_this_step:
            res = self.apriori_cov_fm.shape[0]
        else:
            # By convention muses-py uses a size 1 all zero size if we aren't actually
            # retrieving this. This fits with the hokey way that
            # muses-py handles the BT and systematic jacobian steps. We should clean this
            # up an some point, this is all unnecessarily obscure
            res = 1
        self._check_result(res, "forward_model_sv_length")
        return res

    @property
    def sys_sv_length(self) -> int:
        res = self.apriori_cov_fm.shape[0]
        self._check_result(res, "sys_sv_length")
        return res

    @property
    def value_fm(self) -> FullGridMappedArray:
        res = self._value_fm.view()
        res.flags.writeable = False
        # For CLOUDEXT, we determined that although there are differences with
        # self._sold.value_fm this doesn't actually matter. So skip check for
        # this particular state element
        if self.state_element_id not in (StateElementIdentifier("CLOUDEXT"),):
            self._check_result(res, "value_fm")
        return res

    @property
    def constraint_vector_ret(self) -> RetrievalGridArray:
        res = super().constraint_vector_ret.view()
        res.flags.writeable = False
        self._check_result(res, "constraint_vector_ret", exclude_negative=True)
        return res

    @property
    def constraint_vector_fm(self) -> FullGridMappedArray:
        res = self._constraint_vector_fm.view()
        res.flags.writeable = False
        # Note, the old state element has constraint_vector_fm sometimes mapped,
        # sometimes not. Not sure why, but not worth tracking down. We check
        # constraint_vector_ret which is the only thing that actually matters.
        # We could track that down, but this code is actually right as long
        # as constraint_vector_ret agrees
        # self._check_result(res, "constraint_vector_fm")
        return res

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        if self._constraint_matrix is None:
            raise RuntimeError("_constraint_matrix shouldn't be None")
        res = self._constraint_matrix.view()
        res.flags.writeable = False
        self._check_result(res, "constraint_matrix")
        return res

    @property
    def apriori_cov_fm(self) -> FullGrid2dArray:
        if self._apriori_cov_fm is None:
            raise RuntimeError("_apriori_cov_fm shouldn't be None")
        res = self._apriori_cov_fm.view()
        res.flags.writeable = False
        self._check_result(res, "apriori_cov_fm")
        return res

    @property
    def retrieval_initial_fm(self) -> FullGridMappedArray:
        res = self._retrieval_initial_fm.view()
        res.flags.writeable = False
        self._check_result(res, "retrieval_initial_fm")
        return res

    @property
    def step_initial_fm(self) -> FullGridMappedArray:
        res = self._step_initial_fm.view()
        res.flags.writeable = False
        self._check_result(res, "step_initial_fm")
        return res

    @property
    def true_value_fm(self) -> FullGridMappedArray | None:
        if self._true_value_fm is not None:
            res: FullGridMappedArray | None = self._true_value_fm.view()
            assert res is not None
            res.flags.writeable = False
        else:
            res = None
        return res

    @property
    def state_mapping(self) -> rf.StateMapping:
        return self._state_mapping

    @property
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        return self._state_mapping_retrieval_to_fm

    @property
    def metadata(self) -> dict[str, Any]:
        """Some StateElement have extra metadata. There is really only one example
        now, emissivity has camel_distance and prior_source. It isn't clear the best
        way to handle this, but the current design just returns a dictionary with
        any extra metadata values. We can perhaps rework this if needed in the future.
        For most StateElement this will just be a empty dict."""
        res = self._metadata
        return res

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        return self._spectral_domain

    @property
    def pressure_list_fm(self) -> FullGridMappedArray | None:
        if self._pressure_list_fm is not None:
            res: FullGridMappedArray | None = self._pressure_list_fm.view()
            assert res is not None
            res.flags.writeable = False
        else:
            res = None
        self._check_result(res, "pressure_list_fm")
        return res

    @property
    def retrieved_this_step(self) -> bool:
        """Return true if the current step retrieves this StateElement, false otherwise"""
        return self._retrieved_this_step

    def update_state_element(
        self,
        value_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        next_constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        next_step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        if value_fm is not None:
            self._value_fm = value_fm.astype(np.float64, copy=True).view(
                FullGridMappedArray
            )
        if constraint_vector_fm is not None:
            self._constraint_vector_fm = constraint_vector_fm.astype(
                np.float64, copy=True
            ).view(FullGridMappedArray)
        if next_constraint_vector_fm is not None:
            self._next_constraint_vector_fm = next_constraint_vector_fm.astype(
                np.float64, copy=True
            ).view(FullGridMappedArray)
        if step_initial_fm is not None:
            self._step_initial_fm = step_initial_fm.astype(np.float64, copy=True).view(
                FullGridMappedArray
            )
        if next_step_initial_fm is not None:
            self._next_step_initial_fm = next_step_initial_fm.astype(
                np.float64, copy=True
            ).view(FullGridMappedArray)
        if retrieval_initial_fm is not None:
            self._retrieval_initial_fm = retrieval_initial_fm.astype(
                np.float64, copy=True
            ).view(FullGridMappedArray)
        if true_value_fm is not None:
            self._true_value = true_value_fm.astype(np.float64, copy=True).view(
                FullGridMappedArray
            )
        # We don't update sold here. This gets handled one level higher, in
        # current_state_state_info.py

    @property
    def updated_fm_flag(self) -> FullGridMappedArray:
        # This is what muses-py does. I'm not sure about the logic here - what if we
        # update something and it just doesn't move? But none the less, match what
        # muses-py does. Also - what should the logic be if value_fm is zero?
        res = np.abs((self.step_initial_fm - self.value_fm) / self.value_fm) > 1e-6
        # Special case if nothing moves - assumption is that we started at the
        # the "true" value.
        if np.count_nonzero(res) == 0:
            res[:] = True
        self._check_result(res, "updated_fm_flag")
        return res.view(FullGridMappedArray)

    def tes_levels(
        self, retrieval_levels: np.ndarray, pressure_input: np.ndarray
    ) -> np.ndarray:
        """This is a mapping from the "retrieval_levels" found in the
        OSP species files to the levels used in the generation of the
        basis matrix. This is a mapping from the input pressure levels
        to the forward model pressure levels
        """
        if self.pressure_list_fm is None:
            raise RuntimeError("Need pressure_list_fm")
        res = mpy.supplier_retrieval_levels_tes(
            retrieval_levels, pressure_input, self.pressure_list_fm
        )
        # Filter out any levels out of range, and convert to 0 based
        res = np.array([i - 1 for i in res if i <= self.pressure_list_fm.shape[0]])
        return res

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        self._value_fm = self._retrieval_initial_fm.copy()
        self._step_initial_fm = self._retrieval_initial_fm.copy()

        # TODO Should constraint_vector_fm be allowed to be different
        # from retrieval_initial_fm?  This is the case currently in
        # muses-py, and is certainly mathematically allowed. But it might make
        # more sense to not have our constraint regularization try to pull the
        # state vector from its initial value. It isn't clear if this was
        # intended with muses-py, or just an accident in the code.

        self._constraint_vector_fm = self._retrieval_start_constraint_fm.copy()
        self._next_step_initial_fm = None
        self._next_constraint_vector_fm = None

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        self._current_strategy_step = current_strategy_step.strategy_step
        # Update the initial value if we have a setting from the previous step
        if self._next_step_initial_fm is not None:
            self._step_initial_fm = self._next_step_initial_fm
            self._next_step_initial_fm = None
        if self._next_constraint_vector_fm is not None:
            self._constraint_vector_fm = self._next_constraint_vector_fm
            self._next_constraint_vector_fm = None
        self._value_fm = self._step_initial_fm.copy()
        self._retrieved_this_step = (
            self.state_element_id in current_strategy_step.retrieval_elements
        )
        self._initial_guess_not_updated = (
            self.state_element_id
            in current_strategy_step.retrieval_elements_not_updated
        )

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_fm = None

        if retrieval_slice is not None:
            # Note we really are replacing value_fm as a FullGridMappedArray with
            # the results from the RetrievalGridArray solution. This is what we want
            # to do after a retrieval step. We have the "fprime" here just to make
            # that explicit that this is what we are intending on doing here. Sp
            # to_fmprime returns a FullGridMappedArrayFromRetGrid, but we then use
            # that as a FullGridMappedArray.

            res = (
                xsol[retrieval_slice]
                .view(RetrievalGridArray)
                .to_fmprime(self.state_mapping_retrieval_to_fm, self.state_mapping)
                .view(FullGridMappedArray)
            )
            self._value_fm = res
            if not self._initial_guess_not_updated:
                self._next_step_initial_fm = self._value_fm.copy()


class StateElementFillValueHandle(StateElementHandle):
    """There are a few state element (like OMICLOUDFRACTION) that get created even
    when we don't have the instrument data. These should just return a StateElement
    with fill values. This handle is for these."""

    def __init__(
        self,
        sid: StateElementIdentifier,
    ) -> None:
        self.sid = sid

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None
        logger.debug(f"Creating StateElementFillValue for {state_element_id}")
        fill = np.array(
            [
                -999.0,
            ]
        ).view(FullGridMappedArray)
        fill_2d = np.array(
            [
                [
                    -999.0,
                ]
            ]
        )
        return StateElementImplementation(
            self.sid,
            fill,
            fill,
            fill_2d.view(FullGrid2dArray),
            fill_2d.view(RetrievalGrid2dArray),
        )


class StateElementFixedValueHandle(StateElementHandle):
    """Create state element from static values, rather than getting this from somewhere else."""

    def __init__(
        self,
        sid: StateElementIdentifier,
        constraint_vector_fm: FullGridMappedArray,
        apriori_cov_fm: FullGrid2dArray,
    ) -> None:
        self.sid = sid
        self.constraint_vector_fm = constraint_vector_fm
        self.apriori_cov_fm = apriori_cov_fm

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None
        logger.debug(f"Creating StateElementFixedValue for {state_element_id}")
        return StateElementImplementation(
            self.sid,
            self.constraint_vector_fm,
            self.constraint_vector_fm,
            self.apriori_cov_fm,
            np.linalg.inv(self.apriori_cov_fm).view(RetrievalGrid2dArray),
        )


class StateElementWithCreate(StateElementImplementation):
    """While we can have the StateElementHandle determine all the
    things needed to create a StateElement, often it makes sense to
    have the StateElement just handle this. At the same time, we don't
    want "fat" interfaces where our relatively low level StateElement
    objects depend on lot of high level classes.

    So we use a level of indirection, we have a "create" function that
    uses these high level classes (so we aren't passing in a zillion
    arguments). The create function then determines what is actually needed to
    pass generally to a __init__ function.

    Just as a convenience, we pass all the arguments as keyword arguments, just
    so classes can easily ignore arguments it doesn't need.

    Note StateElement classes aren't required to use this interface,
    only if this makes sense. The classes can derive from StateElement
    or StateElementImplementation instead of StateElementWithCreate,
    and then just supply their own StateElementHandle.

    Note we generally create objects that have *not* been through
    retrieval_initial_fm_from_cycle, we do that step after creating this.
    So for a lot of classes we match the value_fm at the start of the old muses-py
    retrieval, but before it cycles through all the steps. The class indicates
    if it needs to go through retrieval_initial_fm_from_cycle through the function
    need_retrieval_initial_fm_from_cycle.
    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        value_fm: FullGridMappedArray,
        constraint_vector_fm: FullGridMappedArray,
        apriori_cov_fm: FullGrid2dArray | None,
        constraint_matrix: RetrievalGrid2dArray | None,
        state_mapping_retrieval_to_fm: rf.StateMapping | None = rf.StateMappingLinear(),
        state_mapping: rf.StateMapping | None = rf.StateMappingLinear(),
        initial_value_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
        spectral_domain: rf.SpectralDomain | None = None,
        selem_wrapper: Any | None = None,
    ) -> None:
        super().__init__(
            state_element_id,
            value_fm,
            constraint_vector_fm,
            apriori_cov_fm,
            constraint_matrix,
            state_mapping_retrieval_to_fm,
            state_mapping,
            initial_value_fm,
            true_value_fm,
            spectral_domain,
            selem_wrapper,
        )
        self._need_retrieval_initial_fm_from_cycle: bool | None = None

    @classmethod
    def create(
        cls,
        sid: StateElementIdentifier | None = None,
        measurement_id: MeasurementId | None = None,
        retrieval_config: RetrievalConfiguration | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        sounding_metadata: SoundingMetadata | None = None,
        state_info: StateInfo | None = None,
        selem_wrapper: Any | None = None,
        **kwargs: Any,
    ) -> Self | None:
        pass

    def need_retrieval_initial_fm_from_cycle(self) -> bool:
        """Return True if we need to have retrieval_initial_fm_from_cycle run for
        this StateElement."""
        if self._need_retrieval_initial_fm_from_cycle is None:
            self._need_retrieval_initial_fm_from_cycle = (
                self.pressure_list_fm is not None or self.spectral_domain is not None
            )
        return self._need_retrieval_initial_fm_from_cycle

    def notify_done_retrieval_initial_fm_from_cycle(self) -> None:
        """Called when we have run
        MusesStrategy.retrieval_initial_fm_from_cycle, just to mark
        this as done. This automatically gets called by
        MusesStrategy.retrieval_initial_fm_from_cycle.
        """
        self._need_retrieval_initial_fm_from_cycle = False


class StateElementWithCreateHandle(StateElementHandle):
    """A lot of StateElementHandle just call the __init__ function of a class to
    create the object. This generic handle works for this common case."""

    def __init__(
        self,
        sid: StateElementIdentifier | None,
        obj_cls: type[StateElementWithCreate],
        include_old_state_info: bool = False,
        **kwargs: Any,
    ):
        """Create handler for the given StateElementIdentifier, using
        the given class.

        Optionally we can include the old state info
        StateElementOldWrapper and using to verify the
        StateElement. We did that during initial development. We don't
        generally do this now, but it can be useful if we need to
        diagnose a problem. Since we have the mechanics in place
        already, we still support this.  This will likely get phased
        out at some point when it isn't worth maintaining any longer.

        Any extra keywords supplied get passed through to the create
        function, so you can add specific stuff needed (e.g., a
        band_id or other arguments).
        """
        self.sid = sid
        self.obj_cls = obj_cls
        self._hold: Any | None = None
        self.include_old_state_info = include_old_state_info
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.extra_kwargs = kwargs

    @property
    def hold(self) -> Any:
        # Extra level of indirection to handle cycle in including old_py_retrieve_wrapper
        if self._hold is None and self.include_old_state_info:
            from refractor.old_py_retrieve_wrapper import (
                state_element_old_wrapper_handle,
            )

            self._hold = state_element_old_wrapper_handle
        return self._hold

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        state_info: StateInfo | None,
    ) -> None:
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config
        self.strategy = strategy
        self.observation_handle_set = observation_handle_set
        self.sounding_metadata = sounding_metadata
        self.state_info = state_info

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if self.sid is not None and state_element_id != self.sid:
            return None

        sold = (
            self.hold.state_element(state_element_id) if self.hold is not None else None
        )
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        res = self.obj_cls.create(
            sid=state_element_id,
            measurement_id=self.measurement_id,
            retrieval_config=self.retrieval_config,
            strategy=self.strategy,
            observation_handle_set=self.observation_handle_set,
            sounding_metadata=self.sounding_metadata,
            state_info=self.state_info,
            selem_wrapper=sold,
            **self.extra_kwargs,
        )
        if res is None:
            return None
        if res.need_retrieval_initial_fm_from_cycle():
            self.strategy.retrieval_initial_fm_from_cycle(res, self.retrieval_config)
        logger.debug(f"Creating {self.obj_cls.__name__} for {state_element_id}")
        return res


__all__ = [
    "StateElement",
    "StateElementImplementation",
    "StateElementHandle",
    "StateElementHandleSet",
    "StateElementFillValueHandle",
    "StateElementFixedValueHandle",
    "StateElementWithCreate",
    "StateElementWithCreateHandle",
]
