# Might end up breaking this file up, for now have all the stuff here
from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .creator_handle import CreatorHandle, CreatorHandleSet
from .identifier import StateElementIdentifier, RetrievalType
from .current_state import (
    RetrievalGridArray,
    FullGridMappedArray,
    RetrievalGrid2dArray,
    FullGrid2dArray,
    FullGridArray,
    CurrentState
)
import abc
from loguru import logger
import numpy as np
import numpy.testing as npt
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .state_element_old_wrapper import (
        StateElementOldWrapper,
    )
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .cost_function_creator import CostFunctionStateElementNotify
    from .current_state import SoundingMetadata


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
        return np.eye(1)

    @abc.abstractproperty
    def forward_model_sv_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractproperty
    def sys_sv_length(self) -> int:
        raise NotImplementedError()

    @property
    def altitude_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        return None

    @property
    def altitude_list_fm(self) -> FullGridMappedArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        return None

    @property
    def pressure_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        if self.pressure_list_fm is None:
            return None
        return self.pressure_list_fm.to_ret(
            self.state_mapping_retrieval_to_fm, self.state_mapping
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

    @property
    def value_str(self) -> str | None:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like value, but we
        return a str instead. For most StateElement, this returns "None"
        instead which indicates we don't have a str value.

        It isn't clear that this is the best interface, on the other hand these
        str values don't have an obvious other place to go. And these value are
        at least in the spirit of other StateElement. So at least for now we
        will support this, possibly reworking this in the future.
        """
        return None

    @abc.abstractproperty
    def constraint_vector_fm(self) -> FullGridMappedArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        """Constraint matrix, generally the inverse of apriori_cov, although see the
        discussion in StateElement class about how this might be different."""
        raise NotImplementedError()

    def constraint_cross_covariance(
        self, selem2: StateElement
    ) -> RetrievalGrid2dArray | None:
        """Return the constraint cross matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

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
        current_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
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
        pass

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
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
                )


class StateElementImplementation(StateElement):
    """A very common implementation of a StateElement just populates member variables for
    the value, apriori, etc. This class handles this common case, derived classes should
    fill in handling the various notify functions. Note we allow various values to be "None",
    this is useful for derived classes that might fill something in lazily."""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        value_fm: FullGridMappedArray | None,
        constraint_vector_fm: FullGridMappedArray | None,
        apriori_cov_fm: FullGrid2dArray | None,
        constraint_matrix: RetrievalGrid2dArray | None,
        state_mapping_retrieval_to_fm: rf.StateMapping | None = rf.StateMappingLinear(),
        state_mapping: rf.StateMapping | None = rf.StateMappingLinear(),
        initial_value_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
        selem_wrapper: StateElementOldWrapper | None = None,
        value_str: str | None = None,
        copy_on_first_use: bool = False,
    ) -> None:
        super().__init__(state_element_id)
        self._value_fm = value_fm
        self._constraint_vector_fm = constraint_vector_fm
        self._constraint_matrix = constraint_matrix
        self._apriori_cov_fm = apriori_cov_fm
        self._state_mapping = state_mapping
        self._state_mapping_retrieval_to_fm = state_mapping_retrieval_to_fm
        self._step_initial_fm = (
            initial_value_fm if initial_value_fm is not None else constraint_vector_fm
        )
        self._retrieval_initial_fm: FullGridMappedArray | None = None
        if self._step_initial_fm is not None:
            self._retrieval_initial_fm = self._step_initial_fm.copy()
        self._true_value_fm = true_value_fm
        self._value_str = value_str
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
        # Temp, until we have tested everything out
        self._sold = selem_wrapper
        self._copy_on_first_use = copy_on_first_use
        if self._sold is not None and hasattr(self._sold, "update_initial_guess"):
            self.update_initial_guess = self._update_initial_guess

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        # Skip if we aren't actually retrieving. This fits with the hokey way that
        # muses-py handles the BT and systematic jacobian steps. We should clean this
        # up an some point, this is all unnecessarily obscure
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

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def basis_matrix(self) -> np.ndarray:
        if isinstance(self.state_mapping_retrieval_to_fm, rf.StateMappingBasisMatrix):
            res = self.state_mapping_retrieval_to_fm.basis_matrix.transpose()
        else:
            res = np.eye(self.value_fm.shape[0])
        if self._sold is not None and CurrentState.check_old_state_element_value:
            res2 = self._sold.basis_matrix
            if res2 is None:
                raise RuntimeError("res2 should not be None")
            npt.assert_allclose(res, res2)
            # Special case, some of the basis matrix that happen to be integer
            # are left as int64. No real reason, but also no harm
            if res2.dtype != np.int64:
                assert res.dtype == res2.dtype
        return res

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def map_to_parameter_matrix(self) -> np.ndarray:
        if isinstance(self.state_mapping_retrieval_to_fm, rf.StateMappingBasisMatrix):
            res = self.state_mapping_retrieval_to_fm.inverse_basis_matrix.transpose()
        else:
            res = np.eye(self.value_fm.shape[0])
        if self._sold is not None and CurrentState.check_old_state_element_value:
            res2 = self._sold.map_to_parameter_matrix
            if res2 is None:
                raise RuntimeError("res2 should not be None")
            npt.assert_allclose(res, res2, atol=1e-12)
            # Special case, some of the basis matrix that happen to be integer
            # are left as int64. No real reason, but also no harm
            if res2.dtype != np.int64:
                assert res.dtype == res2.dtype
        return res

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
        if self._sold is not None:
            res2 = self._sold.forward_model_sv_length
            assert res == res2
        return res

    @property
    def sys_sv_length(self) -> int:
        res = self.apriori_cov_fm.shape[0]
        if self._sold is not None:
            res2 = self._sold.sys_sv_length
            assert res == res2
        return res

    @property
    def value_fm(self) -> FullGridMappedArray:
        if (
            self._value_fm is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            self._value_fm = self._sold.value_fm.copy()
        if self._value_fm is None:
            raise RuntimeError("_value_fm shouldn't be None")
        res = self._value_fm
        if self._sold is not None and CurrentState.check_old_state_element_value:
            try:
                res2 = self._sold.value_fm
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
            except NotImplementedError:
                pass
        return res

    @property
    def constraint_vector_fm(self) -> FullGridMappedArray:
        if (
            self._constraint_vector_fm is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            return self._sold.constraint_vector_fm
            self._constraint_vector_fm = self._sold.constraint_vector_fm
        if self._constraint_vector_fm is None:
            raise RuntimeError("_constraint_vector_fm shouldn't be None")
        res = self._constraint_vector_fm
        if self._sold is not None and CurrentState.check_old_state_element_value:
            try:
                res2 = self._sold.constraint_vector_fm
            except (AssertionError, RuntimeError):
                res2 = None
            if res2 is not None:
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
        return res

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        if (
            self._constraint_matrix is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            return self._sold.constraint_matrix
            self._constraint_matrix = self._sold.constraint_matrix.copy()
        if self._constraint_matrix is None:
            raise RuntimeError("_constraint_matrix shouldn't be None")
        res = self._constraint_matrix
        if self._sold is not None and CurrentState.check_old_state_element_value:
            res2 = self._sold.constraint_matrix
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    def constraint_cross_covariance(
        self, selem2: StateElement
    ) -> RetrievalGrid2dArray | None:
        if self._sold is not None:
            res = self._sold.constraint_cross_covariance(selem2)
            if res is None:
                return None
            return res.view(RetrievalGrid2dArray)
        return None

    @property
    def apriori_cov_fm(self) -> FullGrid2dArray:
        if (
            self._apriori_cov_fm is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            return self._sold.apriori_cov_fm
            self._apriori_cov_fm = self._sold.apriori_cov_fm.copy()
        if self._apriori_cov_fm is None:
            raise RuntimeError("_apriori_cov_fm shouldn't be None")
        res = self._apriori_cov_fm
        if self._sold is not None:
            try:
                res2 = self._sold.apriori_cov_fm
            except AssertionError:
                res2 = None
            if res2 is not None and CurrentState.check_old_state_element_value:
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
        return res

    @property
    def retrieval_initial_fm(self) -> FullGridMappedArray:
        if (
            self._retrieval_initial_fm is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            self._retrieval_initial_fm = self._sold.retrieval_initial_fm.copy()
        if self._retrieval_initial_fm is None:
            raise RuntimeError("_retrieval_initial_fm shouldn't be None")
        res = self._retrieval_initial_fm
        if self._sold is not None and CurrentState.check_old_state_element_value:
            res2 = self._sold.retrieval_initial_fm
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def step_initial_fm(self) -> FullGridMappedArray:
        if (
            self._step_initial_fm is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            self._step_initial_fm = self._sold.step_initial_fm.copy()
        if self._step_initial_fm is None:
            raise RuntimeError("_step_initial_fm shouldn't be None")
        res = self._step_initial_fm
        if self._sold is not None and CurrentState.check_old_state_element_value:
            res2 = self._sold.step_initial_fm
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def true_value_fm(self) -> FullGridMappedArray | None:
        if self._sold is not None:
            self._true_value_fm = self._sold.true_value_fm
        res = self._true_value_fm
        return res

    @property
    def state_mapping(self) -> rf.StateMapping:
        if (
            self._state_mapping is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            self._state_mapping = self._sold.state_mapping
        return self._state_mapping

    @property
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        if (
            self._state_mapping_retrieval_to_fm is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            self._state_mapping_retrieval_to_fm = (
                self._sold.state_mapping_retrieval_to_fm
            )
        return self._state_mapping_retrieval_to_fm

    # These are placeholders, need to fill in
    @property
    def metadata(self) -> dict[str, Any]:
        """Some StateElement have extra metadata. There is really only one example
        now, emissivity has camel_distance and prior_source. It isn't clear the best
        way to handle this, but the current design just returns a dictionary with
        any extra metadata values. We can perhaps rework this if needed in the future.
        For most StateElement this will just be a empty dict."""
        if self._sold is not None:
            return self._sold.metadata
        res: dict[str, Any] = {}
        return res

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        if self._sold is None:
            raise RuntimeError("Not implemented yet")
        return self._sold.spectral_domain

    @property
    def value_str(self) -> str | None:
        if (
            self._value_str is None
            and self._copy_on_first_use
            and self._sold is not None
        ):
            self._value_str = self._sold.value_str
        res = self._value_str
        if self._sold is not None:
            res2 = self._sold.value_str
            assert res == res2
        return res

    @property
    def pressure_list(self) -> RetrievalGridArray | None:
        # TODO - I think this can be removed, but we'll want to
        # check. Default is just to convert pressure_list_fm, which I
        # think is right - but double check
        if self._sold is None:
            return None
        return self._sold.pressure_list

    @property
    def pressure_list_fm(self) -> FullGridMappedArray | None:
        if self._sold is None:
            return None
        return self._sold.pressure_list_fm

    def update_state_element(
        self,
        current_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        if current_fm is not None:
            self._value_fm = current_fm
        if constraint_vector_fm is not None:
            # TODO get rid of None tests here
            if self._constraint_vector_fm is not None:
                self._constraint_vector_fm = constraint_vector_fm
        if step_initial_fm is not None:
            self._step_initial_fm = step_initial_fm
        if retrieval_initial_fm is not None:
            self._retrieval_initial_fm = retrieval_initial_fm
        if true_value_fm is not None:
            self._true_value = true_value_fm
        if self._sold is not None:
            self._sold.update_state_element(
                current_fm,
                constraint_vector_fm,
                step_initial_fm,
                retrieval_initial_fm,
                true_value_fm,
            )

    @property
    def updated_fm_flag(self) -> FullGridMappedArray:
        # We don't yet have support for the items that depend on frequency, so
        # use the old state element stuff until we get this working
        if self._sold is not None:
            return self._sold.updated_fm_flag
        res = np.zeros((self.apriori_cov_fm.shape[0],), dtype=bool)
        if self._retrieved_this_step:
            res[:] = True
        if self._sold is not None and CurrentState.check_old_state_element_value:
            res2 = self._sold.updated_fm_flag
            npt.assert_allclose(res, res2)
        return res.view(FullGridMappedArray)

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if self._sold is not None:
            self._sold.notify_start_retrieval(current_strategy_step, retrieval_config)
        # The value and step initial guess should be set to the retrieval initial value
        if self._retrieval_initial_fm is not None:
            self._value_fm = self._retrieval_initial_fm.copy()
            self._step_initial_fm = self._retrieval_initial_fm.copy()
        # This is to support testing. We currently have a way of populate StateInfoOld when
        # we restart a step, but not StateInfo. Longer term we will fix this, but short term
        # just propagate any values in selem_wrapper to this class
        if self._sold:
            # self._value_fm = self._sold.value_fm
            # self._step_initial_fm = self._sold.value_fm
            pass
        self._next_step_initial_fm = None

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        if self._sold is not None:
            self._sold.notify_start_step(
                current_strategy_step,
                retrieval_config,
                skip_initial_guess_update,
            )
        # handling for cloudEffExt, see below. We should move this into it's own
        # class, but for now just plop this in here
        if self.state_element_id in (
            StateElementIdentifier("cloudEffExt"),
            StateElementIdentifier("CLOUDEXT"),
        ):
            self.is_bt_ig_refine = (
                current_strategy_step.retrieval_type == RetrievalType("bt_ig_refine")
            )
            # Also, the basis matrix changes for CLOUDEXT from one step to the next
            if self._sold is not None:
                self._state_mapping_retrieval_to_fm = (
                    self._sold.state_mapping_retrieval_to_fm
                )

        # Update the initial value if we have a setting from the previous step
        if self._next_step_initial_fm is not None:
            self._step_initial_fm = self._next_step_initial_fm
            self._next_step_initial_fm = None
        # Set value to initial value
        if self._step_initial_fm is not None:
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
        # We've already called notify_parameter_update, so no need to update
        # self._value here
        if self._sold is not None:
            self._sold.notify_step_solution(xsol, retrieval_slice)
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_fm = None

        # We have some odd logic in StateElementOld for cloudEffExt for the
        # bt_ig_refine step. We will need to duplicate this, but short term
        # just punt and use the old value. Note that we end up at about
        # line 264, there cloudEffExt is set to an average value, but we will
        # need to work through this. Probably special handling on
        # notify_step_solution for cloudEffExt, for bt_ig_refine step.
        # Also CLOUDEXT seems to be a sort of alias for cloudEffExt, but we don't
        # have that fully supported yet - should probably add that
        if (
            self.state_element_id
            in (
                StateElementIdentifier("cloudEffExt"),
                StateElementIdentifier("CLOUDEXT"),
            )
            and self.is_bt_ig_refine
            and self._sold is not None
        ):
            self._value_fm = self._sold._current_state_old.state_value("cloudEffExt")
            if self.state_element_id == StateElementIdentifier("CLOUDEXT"):
                # Bizarrely, different dim for cloudEffExt vs CLOUDEXT
                self._value_fm = self._value_fm[0, :].view(FullGridMappedArray)
            self._next_step_initial_fm = self._value_fm.copy()
        elif retrieval_slice is not None:
            self._value_fm = (
                xsol[retrieval_slice]
                .view(RetrievalGridArray)
                .to_fm(self.state_mapping_retrieval_to_fm, self.state_mapping)
            )
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


__all__ = [
    "StateElement",
    "StateElementImplementation",
    "StateElementHandle",
    "StateElementHandleSet",
    "StateElementFillValueHandle",
    "StateElementFixedValueHandle",
]
